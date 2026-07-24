"""Correction — 物理约束纠错流水线。

4 阶段流水线：候选生成 → Viterbi 全局最优路径选择 → 应用修正 → 级联填充。
Viterbi DP 替代了旧 LCS 局部评分机制，实现联合全局优化。
支持 GUI 和无头 CLI 共用同一实现。
"""
from __future__ import annotations
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ocr_engine import (
    extract_speed_value, build_speed_candidates, Flag,
)
from viterbi import viterbi_correct
from config import (MPS_TO_KMH,
    FILL_MAX_PASSES,
    CORRECTION_MIN_DIFF,
    REOCR_HEIGHTS,
    VITERBI_CONTEXT_WINDOW,
    SMOOTHNESS_THRESHOLD_MULT,
    ACCEL_ANOMALY_THRESHOLD, MAX_SUGGESTED_FRAMES,
    PROBLEM_MIN_SEGMENT_LEN, MAX_PARTIAL_WILDCARDS,
    LCS_CONFIDENCE_MIN_SCORE,
)

if TYPE_CHECKING:
    from rapidocr import RapidOCR

logger = logging.getLogger("RaceVideoToLog.correction")


def _find_neighbor_trusted(i: int, n: int, rows: list) -> tuple[int | None, int | None]:
    """Find nearest left and right HIGH_TRUST/PINNED frame indices for frame i."""
    la = None
    for j in range(i - 1, -1, -1):
        if Flag.is_trusted(rows[j][3]) and rows[j][2] >= 0:
            la = j
            break
    ra = None
    for j in range(i + 1, n):
        if Flag.is_trusted(rows[j][3]) and rows[j][2] >= 0:
            ra = j
            break
    return la, ra


def expand_partial(pattern: str, max_speed: float) -> list[int]:
    """Generate values matching a partial digit pattern. 'x' = any digit 0-9.

    Supports any number of x's. All-x patterns (e.g. 'xxx') return empty
    since they provide no constraint. Patterns with >3 x's also return
    empty to avoid combinatorial explosion.
    The caller always has re-OCR + interpolation as fallback candidates.
    """
    import itertools
    x_count = pattern.count('x') + pattern.count('X')
    if x_count == 0:
        val = float(pattern)
        return [val] if val <= max_speed else []
    # No constraint = skip (all-x or too many x's)
    if x_count == len(pattern) or x_count > MAX_PARTIAL_WILDCARDS:
        return []
    results = []
    pattern_lower = pattern.lower()
    for digits in itertools.product('0123456789', repeat=x_count):
        di = 0
        chars = []
        for ch in pattern_lower:
            if ch == 'x':
                chars.append(digits[di]); di += 1
            else:
                chars.append(ch)
        val = float(''.join(chars))
        if val <= max_speed:
            results.append(val)
    return results


def _auto_expand_digits(raw_text: str, max_speed_kmh: float) -> list[int]:
    """根据 OCR 读数自动生成所有可能的数字候选。

    - 1-2 位数: 插入 x + 逐位替换为 x
    - 3 位数: 逐位替换为 x（如 211 → x11/2x1/21x，展开出所有 3 位数候选）
    - ≥4 位数: 不扩展（假定完整）
    """
    if not raw_text or not raw_text.isdigit():
        return []
    digits = raw_text
    if len(digits) >= 4:
        return []
    candidates: list[int] = []
    try:
        candidates.append(int(digits))
    except ValueError:
        pass
    # 插入 x（处理缺失位）：仅对 1-2 位数
    if len(digits) <= 2:
        for pos in range(len(digits) + 1):
            pattern = digits[:pos] + "x" + digits[pos:]
            for v in expand_partial(pattern, max_speed_kmh):
                if v not in candidates:
                    candidates.append(v)
    # 逐位替换为 x（处理误读位）：1-3 位数都做
    for pos in range(len(digits)):
        pattern = digits[:pos] + "x" + digits[pos + 1:]
        for v in expand_partial(pattern, max_speed_kmh):
            if v not in candidates:
                candidates.append(v)
    return candidates


def _re_ocr_frame(crop_bgr: "np.ndarray", ocr: "RapidOCR", max_speed_kmh: float,
                    timing: dict | None = None, cache: dict | None = None) -> set:
    """阶段 2：对单帧尝试多种预处理高度的重 OCR，有缓存。

    尝试 3 种高度（24, 32, 48），取所有不同结果的并集。
    不同高度下 OCR 模型可能产生不同读数，提高候选覆盖率。

    Args:
        cache: 可选的外部缓存字典（图像哈希 → 候选值集合）。
                绑定到 Pipeline 实例生命周期以避免内存泄漏。
    """
    cache = cache if cache is not None else {}
    if crop_bgr is not None and crop_bgr.size > 0:
        raw = crop_bgr.data.tobytes() if hasattr(crop_bgr, 'data') else crop_bgr.tobytes()
        cache_key = hash(raw[:256])
    else:
        cache_key = None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]

    candidates = set()
    if crop_bgr is None or crop_bgr.size == 0:
        return candidates

    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return candidates

    for target_h in REOCR_HEIGHTS:
        scale = target_h / h if h > 0 else 1.0
        proc = cv2.resize(crop_bgr, (max(1, int(w * scale)), target_h))
        res = ocr(proc)
        sv, rt, _conf = extract_speed_value(res)
        if sv is not None and sv <= max_speed_kmh:
            candidates.add(int(sv))

    if cache_key is not None:
        cache[cache_key] = candidates
    return candidates


def _interp_candidate(i: int, rows: list, pinned_set: set, times: list, max_speed_kmh: float, fps: float = 1.0) -> float | None:
    """计算帧 i 在左右高信帧间的线性插值估计。"""
    n = len(rows)
    la, ra = _find_neighbor_trusted(i, n, rows)
    if la is not None and ra is not None:
        lv = rows[la][2]; rv = rows[ra][2]
        lt = rows[la][0] / fps; rt = rows[ra][0] / fps
        total_dt = max(rt - lt, 1e-3)
        frac = (times[i] - lt) / total_dt
        val = lv + (rv - lv) * frac
        if 0 <= val <= max_speed_kmh:
            return round(val)
    return None


def _fill_unrecoverable(rows: list, pinned_set: set, error_set: set, times: list, max_speed_kmh: float, max_accel_mps2: float, fps: float = 1.0,
                        progress_fn: "Callable | None" = None,
                        notes: dict[int, str] | None = None,
                        profile_values: list[float] | None = None) -> None:
    """阶段 4：对无法通过 Viterbi 修复的帧，以最近高信帧为基准插值。

    左右均使用 HIGH_TRUST/PINNED 帧作为约束，防止 fill 链式累积。
    当无高信邻居时，回退到宽窗口中值剖面值（若提供）。
    """
    n = len(rows)
    sorted_errors = sorted(i for i in error_set if i not in pinned_set and not Flag.is_trusted(rows[i][3]))
    total = len(sorted_errors)
    progress_done = 0
    for i in sorted_errors:
        if profile_values is not None and i < len(profile_values) and profile_values[i] > 0:
            pv = profile_values[i]
            la = None
            for j in range(i - 1, -1, -1):
                if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
                    la = j; break
            ra = None
            for j in range(i + 1, n):
                if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
                    ra = j; break

            lo, hi = 0.0, max_speed_kmh
            if la is not None:
                lv = rows[la][2]; lt = rows[la][0] / fps
                left_dt = max(times[i] - lt, 1e-3)
                lo = max(lo, lv - max_accel_mps2 * left_dt * MPS_TO_KMH)
                hi = min(hi, lv + max_accel_mps2 * left_dt * MPS_TO_KMH)
            if ra is not None:
                rv = rows[ra][2]; rt = rows[ra][0] / fps
                right_dt = max(rt - times[i], 1e-3)
                lo = max(lo, rv - max_accel_mps2 * right_dt * MPS_TO_KMH)
                hi = min(hi, rv + max_accel_mps2 * right_dt * MPS_TO_KMH)
            val = round(max(lo, min(hi, pv)))
        else:
            # ── No profile: interpolate between trusted neighbors ──
            la = None
            for j in range(i - 1, -1, -1):
                if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
                    la = j; break
            ra = None
            for j in range(i + 1, n):
                if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
                    ra = j; break

            if la is not None and ra is not None:
                lv = rows[la][2]; lt = rows[la][0] / fps
                rv = rows[ra][2]; rt = rows[ra][0] / fps
                left_dt = max(times[i] - lt, 1e-3)
                right_dt = max(rt - times[i], 1e-3)
                left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
                right_max_dv = max_accel_mps2 * right_dt * MPS_TO_KMH
                lo = max(0.0, lv - left_max_dv)
                hi = min(max_speed_kmh, lv + left_max_dv)
                lo = max(lo, rv - right_max_dv)
                hi = min(hi, rv + right_max_dv)
                interp = lv + (rv - lv) * (left_dt / max(left_dt + right_dt, 1e-3))
                val = round(max(lo, min(hi, interp)))
            elif la is not None:
                # Single trusted neighbor + max_accel envelope
                lv = rows[la][2]; lt = rows[la][0] / fps
                left_dt = max(times[i] - lt, 1e-3)
                left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
                lo = max(0.0, lv - left_max_dv)
                hi = min(max_speed_kmh, lv + left_max_dv)
                val = round(max(lo, min(hi, lv)))
            else:
                continue
        rows[i][2] = int(val)
        if rows[i][3] == Flag.RAW:
            rows[i][3] = Flag.FILL_INTERP
        if notes is not None:
            notes[i] = f"fill: {val:.0f}"

        progress_done += 1
        if progress_fn:
            progress_fn(progress_done, total)


def _smoothness_pass(rows: list, times: list, max_speed_kmh: float,
                     max_accel_mps2: float, fps: float = 1.0,
                     notes: dict[int, str] | None = None,
                     wide_profile: list[float] | None = None) -> int:
    """Auto-mode: profile-anchored spike removal.

    Frames that deviate from the wide median profile by more than threshold
    are pulled toward the profile. Small clusters (≤5 frames) are treated as
    OCR errors; larger clusters are legitimate speed changes and are preserved.

    Overrides ANY flag (including HIGH_TRUST).
    """
    n = len(rows)
    if n < 3 or wide_profile is None:
        return 0

    max_dv = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else 8.0
    threshold = max_dv * SMOOTHNESS_THRESHOLD_MULT

    smoothed = 0
    for _pass in range(10):
        fixed_this_pass = 0
        for i in range(n):
            v_cur = rows[i][2]
            if v_cur < 0:
                continue
            pv = wide_profile[i] if i < len(wide_profile) else v_cur
            if pv <= 0:
                continue
            if abs(v_cur - pv) <= threshold:
                continue

            # Cluster size check: small clusters are OCR errors
            cluster_size = 1
            for j in range(i-1, -1, -1):
                if rows[j][2] >= 0 and abs(rows[j][2] - (wide_profile[j] if j < len(wide_profile) else rows[j][2])) > threshold:
                    cluster_size += 1
                else:
                    break
            for j in range(i+1, n):
                if rows[j][2] >= 0 and abs(rows[j][2] - (wide_profile[j] if j < len(wide_profile) else rows[j][2])) > threshold:
                    cluster_size += 1
                else:
                    break
            if cluster_size > 5:
                continue

            # Pull toward profile, clamped by acceleration
            target = pv
            lo, hi = 0.0, max_speed_kmh
            for j in range(i-1, -1, -1):
                if rows[j][2] >= 0 and abs(rows[j][2] - (wide_profile[j] if j < len(wide_profile) else rows[j][2])) <= threshold:
                    lv = rows[j][2]; lt = rows[j][0] / fps
                    dt_l = max(times[i] - lt, 1e-3)
                    lo = max(lo, lv - max_accel_mps2 * dt_l * MPS_TO_KMH)
                    hi = min(hi, lv + max_accel_mps2 * dt_l * MPS_TO_KMH)
                    break
            for j in range(i+1, n):
                if rows[j][2] >= 0 and abs(rows[j][2] - (wide_profile[j] if j < len(wide_profile) else rows[j][2])) <= threshold:
                    rv = rows[j][2]; rt = rows[j][0] / fps
                    dt_r = max(rt - times[i], 1e-3)
                    lo = max(lo, rv - max_accel_mps2 * dt_r * MPS_TO_KMH)
                    hi = min(hi, rv + max_accel_mps2 * dt_r * MPS_TO_KMH)
                    break

            new_val = round(max(lo, min(hi, target)))
            old_val = rows[i][2]
            if abs(new_val - old_val) < 0.5:
                continue
            rows[i][2] = int(new_val)
            if notes is not None:
                notes[i] = f"smooth: {old_val:.0f}→{new_val:.0f}"
            smoothed += 1
            fixed_this_pass += 1
        if fixed_this_pass == 0:
            break
    return smoothed


# ═══════════════════════════════════════════════════════════════
# 主纠错流水线 (Viterbi-based)
# ═══════════════════════════════════════════════════════════════

def correct_with_trust(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR",
                            max_speed_kmh: float, max_accel_mps2: float,
                            log_fn: "Callable | None" = None,
                            progress_fn: "Callable | None" = None,
                            skip_fill: bool = False,
                            timing: dict | None = None,
                            partial_corrections: dict[int, str] | None = None,
                            reocr_cache: dict | None = None,
                            light_mode: bool = False,
                            reocr_only: bool = False,
                            notes: dict[int, str] | None = None, pinned: set[int] | None = None,
                            split_results: dict[int, str] | None = None,
                                fps: float = 1.0) -> list:
    """4 阶段物理约束纠错流水线 (Viterbi 全局最优路径选择)。

    以 pinned 帧（用户手动修正）为硬约束（固定不变），
    Viterbi DP 对所有非信任帧同时做全局最优选择。

    Args:
        reocr_cache: 可选的重 OCR 缓存字典，绑定到 Pipeline 实例生命周期。
        light_mode: 轻量模式 — 仅 re-OCR 候选，不填充（pass1 用）。
        reocr_only: 仅 re-OCR 候选生成（不含混淆/扩展/插值），但保留完整填充。
    Returns: 修改后的 rows（原地修改）
    """
    pinned = pinned or set()
    n = len(rows)
    for pi in pinned:
        if rows[pi][3] < Flag.HIGH_TRUST:
            rows[pi][3] = Flag.PINNED
    pinned_set = pinned  # user-verified frames treated as ground truth

    times = [r[0] / fps for r in rows]
    cache: dict = reocr_cache if reocr_cache is not None else {}

    if log_fn:
        mode_str = " (light)" if light_mode else ""
        log_fn(f"Correction{mode_str}: {n} rows, {len(pinned_set)} pinned, Viterbi DP")

    # ── Stage 0: 双窗口中值剖面 ──
    # Narrow (0.5s): Viterbi 观测代价用，对孤立离群值敏感
    # Wide (5s): 候选生成用，抗一致性孤岛污染，为短读数帧提供可靠的百位候选
    def _median_filter_np(y: "np.ndarray", window: int) -> "np.ndarray":
        half = window // 2
        result = np.zeros(len(y), dtype=float)
        for i in range(len(y)):
            lo = max(0, i - half)
            hi = min(len(y), i + half + 1)
            result[i] = float(np.median(y[lo:hi]))
        return result

    _vals = np.array([r[2] for r in rows], dtype=float)
    if n >= 2:
        _dt = times[1] - times[0]
    else:
        _dt = 1.0 / max(fps, 1.0)
    _med_win = max(5, int(0.5 / _dt + 0.5))
    if _med_win % 2 == 0:
        _med_win += 1
    _med_win = min(_med_win, n - 2)
    if n >= 3:
        # Force window to be at least 3 and odd
        _win = max(3, _med_win)
        if _win % 2 == 0:
            _win += 1
        median_profile = _median_filter_np(_vals, min(_win, n)).tolist()
    else:
        median_profile = _vals.tolist()

    # ── Wide profile (5s): robust to consistency islands, used for candidate generation ──
    _wide_win = max(31, int(2.5 / max(_dt, 0.01)))
    if _wide_win % 2 == 0:
        _wide_win += 1
    _wide_win = min(_wide_win, n)
    if n >= _wide_win:
        wide_profile = _median_filter_np(_vals, _wide_win).tolist()
    else:
        wide_profile = median_profile  # fallback to narrow

    # ── Stage 1: 识别可疑帧 + 生成候选 ──
    # 可疑帧（剖面偏离/短文本）：完整候选生成
    # 上下文帧（可疑帧附近）：仅原始OCR值作为候选，但参与Viterbi段
    real_suspect, context_only = _identify_suspect_frames(rows, observations, median_profile, pinned_set, n, times, max_accel_mps2)
    suspect_frames = set(real_suspect) | set(context_only)

    candidates_by_frame: dict[int, list[float]] = {}
    total_suspect = len(real_suspect)
    progress_done = 0
    only_reocr = light_mode or reocr_only

    for fi in real_suspect:
        # Use wide profile for candidate ranking — robust against consistency islands
        mp_ref = wide_profile[fi] if fi < len(wide_profile) else None
        cands = _generate_candidates(
            fi, rows, observations, raw_frames, ocr, pinned_set,
            times, max_speed_kmh, cache, split_results,
            only_reocr=only_reocr, fps=fps, median_ref=mp_ref,
        )
        if cands:
            candidates_by_frame[fi] = cands

        progress_done += 1
        if progress_fn and total_suspect > 0:
            progress_fn(progress_done, total_suspect)

    # Context-only frames: raw value as sole candidate
    for fi in context_only:
        raw_v = rows[fi][2]
        if 0 <= raw_v <= max_speed_kmh:
            candidates_by_frame[fi] = [raw_v]
        else:
            candidates_by_frame[fi] = [0.0]

    # For non-suspect non-trusted frames: raw value as sole candidate
    # This ensures Viterbi processes all frames and produces confidence scores
    for i in range(n):
        if i not in pinned_set and not Flag.is_trusted(rows[i][3]) and i not in candidates_by_frame:
            raw_v = rows[i][2]
            if 0 <= raw_v <= max_speed_kmh:
                candidates_by_frame[i] = [raw_v]
            else:
                candidates_by_frame[i] = [0.0]

    if log_fn:
        n_with_cands = len(candidates_by_frame)
        total_cands = sum(len(c) for c in candidates_by_frame.values())
        log_fn(f"  Stage 1: {len(real_suspect)} suspect + {len(context_only)} context frames, "
               f"{n_with_cands} total with candidates ({total_cands} total)")

    # ── Stage 2: Viterbi DP 全局最优路径选择 ──
    # Combine pinned_set with any pre-existing HIGH_TRUST frames for anchoring
    trusted_set = pinned_set.copy()
    for i in range(n):
        if Flag.is_trusted(rows[i][3]):
            trusted_set.add(i)

    viterbi_result = viterbi_correct(
        rows, candidates_by_frame, trusted_set, times,
        max_speed_kmh, max_accel_mps2, median_profile=median_profile,
    )

    if log_fn:
        log_fn(f"  Stage 2: Viterbi found {len(viterbi_result['error_set'])} errors, "
               f"{len(viterbi_result['corrected'])} corrections")

    # ── Stage 3: 应用 Viterbi 修正 ──
    fixed = _apply_viterbi_corrections(
        rows, viterbi_result, pinned_set, notes,
        only_reocr=only_reocr, log_fn=log_fn,
    )
    if log_fn:
        log_fn(f"  Stage 3: {fixed} corrections applied")

    # ── Stage 4: Fill unrecoverable (auto mode: aggressive profile-based fill) ──
    if skip_fill or light_mode:
        if log_fn:
            log_fn(f"  Stage 4: skipped ({'light_mode' if light_mode else 'skip_fill'})")
    else:
        remaining_errors, wide_profile_fill = _find_remaining_errors(rows, viterbi_result, trusted_set, pinned_set, observations)
        if remaining_errors:
            fill_pass = 0
            while fill_pass < FILL_MAX_PASSES and remaining_errors:
                _fill_unrecoverable(rows, pinned_set, remaining_errors, times,
                                    max_speed_kmh, max_accel_mps2, fps,
                                    progress_fn=progress_fn, notes=notes,
                                    profile_values=wide_profile_fill)
                if log_fn:
                    log_fn(f"  Stage 4 pass {fill_pass+1}: filled {len(remaining_errors)} unrecoverable frames")
                fill_pass += 1
                remaining_errors, wide_profile_fill = _find_remaining_errors(rows, viterbi_result, trusted_set, pinned_set, observations)
                if not remaining_errors:
                    break
        elif log_fn:
            log_fn(f"  Stage 4: no remaining errors to fill")

    # ── Stage 5: HIGH_TRUST marking (after fill: corrected values get trusted) ──
    confidence = viterbi_result['confidence']
    trust_threshold = 70.0
    n_trusted = 0
    _max_dv = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else 8.0
    for ci in confidence:
        i = ci['index']
        if i in trusted_set or rows[i][3] != Flag.RAW:
            continue
        v = rows[i][2]
        if v < 0:
            continue
        if ci['score'] < trust_threshold and i in candidates_by_frame and len(candidates_by_frame[i]) > 1:
            continue
        left_ok = True
        for j in range(i - 1, max(-1, i - 4), -1):
            if j < 0: break
            nbr_v = rows[j][2]
            if nbr_v < 0: continue
            nbr_rt = observations[j].raw_text if j < len(observations) else ''
            if nbr_rt and len(nbr_rt) < 3: continue
            if rows[j][3] == Flag.REOCR_AUTO: continue
            if abs(v - nbr_v) > _max_dv: left_ok = False
            break
        if not left_ok: continue
        right_ok = True
        for j in range(i + 1, min(n, i + 4)):
            nbr_v = rows[j][2]
            if nbr_v < 0: continue
            nbr_rt = observations[j].raw_text if j < len(observations) else ''
            if nbr_rt and len(nbr_rt) < 3: continue
            if rows[j][3] == Flag.REOCR_AUTO: continue
            if abs(nbr_v - v) > _max_dv: right_ok = False
            break
        if not right_ok: continue
        if i < len(wide_profile) and wide_profile[i] > 0:
            if abs(v - wide_profile[i]) > max(4.0, wide_profile[i] * 0.02):
                continue
        rows[i][3] = Flag.HIGH_TRUST
        n_trusted += 1
    if log_fn:
        log_fn(f"  Stage 5: {n_trusted} new HIGH_TRUST")

    # ── Stage 6: Smoothness enforcement (auto mode only) ──
    if not skip_fill and not light_mode:
        n_smoothed = _smoothness_pass(rows, times, max_speed_kmh, max_accel_mps2, fps, notes, wide_profile)
        if log_fn and n_smoothed > 0:
            log_fn(f"  Stage 6: smoothed {n_smoothed} profile-deviating spikes")

    # Store confidence for compute_confidence() to use
    _store_confidence_cache(rows, confidence)

    return rows


def _identify_suspect_frames(
    rows: list, observations: list, median_profile: list[float],
    pinned_set: set[int], n: int,
    times: list[float] | None = None,
    max_accel_mps2: float = 50.0,
) -> tuple[list[int], list[int]]:
    """Identify frames that may need correction.

    Returns: (real_suspect, context_only) where:
    - real_suspect: frames with profile deviation, short text, or physically
      impossible transitions — gets full candidates
    - context_only: frames near suspects — raw-only candidate

    Criteria for real_suspect:
    - Median profile deviation > threshold
    - Short OCR text (1-2 digits)
    - Physically impossible acceleration vs neighbors (catches 116→216 errors)
    """
    real_suspect: set[int] = set()
    window = VITERBI_CONTEXT_WINDOW

    for i in range(n):
        if i in pinned_set or Flag.is_trusted(rows[i][3]):
            continue
        v = rows[i][2]
        if v < 0 or v > 400:
            real_suspect.add(i)
            continue

        # Profile deviation check
        if i < len(median_profile):
            mp = median_profile[i]
            if mp > 0 and abs(v - mp) > max(4.0, mp * 0.02):
                real_suspect.add(i)
                continue

        # Short text check
        if i < len(observations):
            rt = observations[i].raw_text
            if rt and len(rt) < 3:
                real_suspect.add(i)
                continue

        # ── Physics check: flag frames that are inconsistent with BOTH neighbors ──
        # A single-neighbor violation means the neighbor is wrong, not this frame.
        # Both-neighbor violation means this frame is the outlier.
        if times is not None and i > 0 and i < n - 1:
            dt = times[i] - times[i-1] if i > 0 else 1.0
            if dt > 0:
                max_dv = max_accel_mps2 * dt * MPS_TO_KMH * 2
                bad_count = 0
                for ni in (i-1, i+1):
                    if 0 <= ni < n and rows[ni][2] > 0 and v > 0:
                        if abs(v - rows[ni][2]) > max_dv:
                            bad_count += 1
                if bad_count >= 2:
                    real_suspect.add(i)

    # Expand context window around real suspects
    context_only: set[int] = set()
    for fi in real_suspect:
        for di in range(-window, window + 1):
            if di == 0:
                continue
            ni = fi + di
            if (0 <= ni < n and ni not in pinned_set
                    and not Flag.is_trusted(rows[ni][3])
                    and ni not in real_suspect):
                context_only.add(ni)

    return sorted(real_suspect), sorted(context_only)


def _generate_candidates(
    fi: int, rows: list, observations: list, raw_frames: list,
    ocr: "RapidOCR", pinned_set: set, times: list,
    max_speed_kmh: float, reocr_cache: dict,
    split_results: dict[int, str] | None,
    only_reocr: bool = False, fps: float = 1.0,
    median_ref: float | None = None,
) -> list[float]:
    """Generate candidate speed values for a single frame.

    Candidates come from: re-OCR, split OCR, confusion chars, digit expansion, interpolation.
    Limited to VITERBI_MAX_CANDIDATES total. Re-OCR and raw values always survive truncation.
    """
    from config import VITERBI_MAX_CANDIDATES
    raw_val = rows[fi][2]
    mp = median_ref if median_ref is not None and median_ref > 0 else raw_val

    # Protected: raw value + re-OCR results (always survive)
    protected: list[float] = []
    protected_set: set[float] = set()
    if 0 <= raw_val <= max_speed_kmh:
        protected.append(raw_val)
        protected_set.add(raw_val)

    oid = min(fi, len(observations) - 1)
    obs = observations[oid]

    # Re-OCR candidates (protected)
    if fi < len(raw_frames):
        reocr_set = _re_ocr_frame(raw_frames[fi][1], ocr, max_speed_kmh,
                                   cache=reocr_cache)
        for cv in sorted(reocr_set):
            if 0 <= cv <= max_speed_kmh and cv not in protected_set:
                protected.append(cv)
                protected_set.add(cv)

    # Split OCR candidates (protected)
    if split_results and fi in split_results:
        try:
            split_val = int(split_results[fi])
            if 0 <= split_val <= max_speed_kmh and split_val not in protected_set:
                protected.append(split_val)
                protected_set.add(split_val)
        except ValueError:
            pass

    # Other candidates (ranked, may be truncated)
    other: list[float] = []
    other_set: set[float] = set()

    # Short text (<3 digits): missing-digit frames where OCR is unreliable.
    # These get full expansion + profile candidate to recover the missing digit.
    is_short = obs.raw_text and len(obs.raw_text) < 3

    # ── Hundreds-digit candidates for 3-digit frames ──
    # OCR may misread the hundreds digit: "117" when truth is 217.
    # Generate ALL possible hundreds-digit variants: base, 100+base, 200+base...
    # e.g., raw=116 → 16, 116, 216, 316. Viterbi picks via transition cost.
    if not is_short and obs.raw_text and obs.raw_text.isdigit() and len(obs.raw_text) == 3 and raw_val > 0:
        base = int(obs.raw_text) % 100
        for hundreds in range(0, int(max_speed_kmh) + 1, 100):
            alt = hundreds + base
            if 0 <= alt <= max_speed_kmh and alt not in protected_set:
                protected.append(alt)
                protected_set.add(alt)

    # ── Median profile candidate: ONLY for short-text frames ──
    if is_short and mp > 0:
        mp_rounded = round(mp)
        if mp_rounded <= max_speed_kmh and mp_rounded not in protected_set:
            protected.append(mp_rounded)
            protected_set.add(mp_rounded)

    if not only_reocr or is_short:
        # Confusion character candidates
        for cv in build_speed_candidates(obs.raw_text, max_speed_kmh):
            if cv not in protected_set and cv not in other_set:
                other.append(cv)
                other_set.add(cv)

        # Digit expansion candidates
        for cv in _auto_expand_digits(obs.raw_text, max_speed_kmh):
            if cv not in protected_set and cv not in other_set:
                other.append(cv)
                other_set.add(cv)

    if not only_reocr:
        # Interpolation candidate
        interp_val = _interp_candidate(fi, rows, pinned_set, times, max_speed_kmh, fps=fps)
        if interp_val is not None and interp_val not in protected_set and interp_val not in other_set:
            other.append(interp_val)
            other_set.add(interp_val)

    # Truncate other candidates by rank
    # Short-text frames: keep ALL expansion candidates. They're rare (~0.5% of frames)
    # and the Viterbi transition cost will select the physically correct one.
    # Truncation risks dropping the correct hundreds-digit expansion (121, 221).
    # 3-digit frames: raw OCR is trustworthy, limit to VITERBI_MAX_CANDIDATES.
    remaining = VITERBI_MAX_CANDIDATES - len(protected)
    if remaining > 0 and len(other) > remaining and not is_short:
        def _rank(v: float) -> float:
            d_raw = abs(v - raw_val) / max(1.0, abs(raw_val)) if raw_val > 0 else abs(v - raw_val)
            d_mp = abs(v - mp) / max(1.0, mp)
            return d_raw + d_mp
        other.sort(key=_rank)
        other = other[:remaining]

    return protected + other


def _apply_viterbi_corrections(
    rows: list, viterbi_result: dict, pinned_set: set,
    notes: dict[int, str] | None,
    only_reocr: bool = False,
    log_fn: "Callable | None" = None,
) -> int:
    """Apply Viterbi corrections to rows, respecting pinned/trusted constraints."""
    corrected = viterbi_result['corrected']
    fixed = 0

    for fi, new_val in sorted(corrected.items()):
        if fi in pinned_set:
            continue
        if Flag.is_trusted(rows[fi][3]):
            continue  # Don't override trusted frames

        old_val = rows[fi][2]
        if abs(new_val - old_val) <= CORRECTION_MIN_DIFF:
            continue

        rows[fi][2] = new_val
        if rows[fi][3] == Flag.RAW:
            rows[fi][3] = Flag.REOCR_AUTO
        if notes is not None:
            notes[fi] = f"viterbi: {old_val:.0f}→{new_val:.0f}"
        fixed += 1

    return fixed


def _find_remaining_errors(
    rows: list, viterbi_result: dict, trusted_set: set, pinned_set: set,
    observations: list | None = None,
) -> tuple[set[int], list[float]]:
    """Find frames that Viterbi missed — using wide-median-profile deviation.

    Only targets short-text frames (1-2 digit OCR): the wide profile can be
    polluted by consistency islands, so 3-digit frames that deviate are more
    likely to be correct than the profile.

    Returns: (error_set, wide_profile) where wide_profile can be passed to
             _fill_unrecoverable as fallback values.
    """
    n = len(rows)
    if n < 5:
        return set(), []

    # ── Wide median profile (5-second window, robust to large consistency islands) ──
    speeds = [r[2] for r in rows]
    import numpy as np
    if n >= 2:
        dt_est = rows[1][0] - rows[0][0]
        if dt_est <= 0:
            dt_est = 1.0
    else:
        dt_est = 1.0
    wide_half = max(31, int(2.5 / max(dt_est, 0.01)))  # ~5s window, min 63 frames
    if wide_half % 2 == 0:
        wide_half += 1
    half = wide_half // 2

    vals = np.array(speeds, dtype=float)
    wide_profile = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        wide_profile[i] = float(np.median(vals[lo:hi]))

    # ── Find frames with large deviation from wide profile ──
    # Only fill short-text frames (1-2 digit OCR): their raw value is
    # unreliable and the correct value is likely in the profile.
    # 3-digit frames are generally trustworthy — if they deviate from the
    # profile, the profile is likely polluted by a consistency island.
    error_set: set[int] = set()
    for i in range(n):
        if i in trusted_set or i in pinned_set:
            continue
        if Flag.is_trusted(rows[i][3]):
            continue
        if observations is not None and i < len(observations):
            rt = observations[i].raw_text
            if rt and len(rt) >= 3:
                continue  # 3-digit frame: trust the OCR
        v = rows[i][2]
        ref = wide_profile[i]
        if v < 0 or ref <= 0:
            continue
        if abs(v - ref) > max(4.0, ref * 0.02):
            error_set.add(i)

    return error_set, wide_profile.tolist()


# ═══════════════════════════════════════════════════════════════
# Viterbi confidence cache (attached to rows for GUI access)
# ═══════════════════════════════════════════════════════════════

_viterbi_confidence_cache: dict[int, list[dict]] = {}

def _store_confidence_cache(rows: list, confidence: list[dict]) -> None:
    """Store Viterbi confidence on rows for later retrieval by compute_confidence."""
    # Use id(rows) as key since rows is the same list object throughout the pipeline
    _viterbi_confidence_cache[id(rows)] = confidence

def _get_confidence_cache(rows: list) -> list[dict] | None:
    """Retrieve stored Viterbi confidence for the given rows list."""
    return _viterbi_confidence_cache.get(id(rows))


# ═══════════════════════════════════════════════════════════════
# 置信度评分 — 用于聚焦人工审核
# ═══════════════════════════════════════════════════════════════

def compute_confidence(rows: list, observations: list, max_speed: float,
                        max_accel: float, pinned: set[int] | None = None,
                        fps: float = 1.0) -> list[dict]:
    """基于 Viterbi 路径代价的置信度评分 (0-100)。

    优先使用 Viterbi 缓存的置信度（如果 correct_with_trust 已经运行过）。
    如果缓存不可用，回退到基于中值剖面偏差的简单评分。

    - 100 分: trusted 帧或 Viterbi 路径认为当前值最优
    - 70+ 分: 高置信（路径代价低）
    - 30-70 分: 存疑
    - < 30 分: 错误（路径代价高或剖面偏离大）
    """
    n = len(rows)

    # Try to use cached Viterbi confidence first
    cached = _get_confidence_cache(rows)
    if cached is not None and len(cached) == n:
        # Viterbi already ran — use its confidence directly
        return cached

    # Fallback: simple median-profile-based confidence
    # (used when correction is skipped entirely)
    speeds = np.array([r[2] for r in rows], dtype=float)
    half = 7
    median_profile = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        median_profile[i] = float(np.median(speeds[lo:hi]))

    flags = [r[3] for r in rows]
    confidences = []
    for i in range(n):
        cur = rows[i][2]
        if cur < 0 or cur > max_speed:
            confidences.append({
                'index': i, 'score': 0.0, 'is_corrected': Flag.is_corrected(flags[i]),
                'speed': cur, 'reason': '速度超出范围',
            })
            continue

        mp = median_profile[i]
        if mp > 0:
            deviation = abs(cur - mp) / max(1.0, mp)
            if Flag.is_trusted(flags[i]):
                reason = '锚点帧(可信)'
                score = 100.0
            elif deviation < 0.01:
                reason = '正常'
                score = 90.0
            elif deviation < 0.03:
                reason = '轻微偏离剖面'
                score = 70.0
            elif deviation < 0.06:
                reason = '中度偏离剖面'
                score = 45.0
            else:
                reason = '显著偏离剖面'
                score = 20.0
        else:
            reason = '正常'
            score = 70.0

        confidences.append({
            'index': i, 'score': round(score, 1),
            'is_corrected': Flag.is_corrected(flags[i]),
            'speed': cur, 'reason': reason,
        })

    return confidences


def find_problem_segments(confidences: list[dict], min_score: float = LCS_CONFIDENCE_MIN_SCORE,
                            min_segment_len: int = PROBLEM_MIN_SEGMENT_LEN) -> list[dict]:
    """将低置信度连续帧聚合成问题段。

    min_score 默认 30（对应低置信度帧）。
    低于此分数的帧才被视为"需要人工审核"的问题帧。

    Returns: [{start, end, count, avg_score, min_score, frames, reason, suggested}]
    """
    segments = []
    i = 0
    while i < len(confidences):
        if confidences[i]['score'] < min_score:
            start = i
            reasons = set()
            while i < len(confidences) and confidences[i]['score'] < min_score:
                r = confidences[i]['reason']
                if r and r != '正常':
                    reasons.add(r)
                i += 1
            count = i - start
            if count >= min_segment_len:
                seg_frames = confidences[start:i]
                scores = [f['score'] for f in seg_frames]
                segments.append({
                    'start': start, 'end': i - 1, 'count': count,
                    'avg_score': round(sum(scores) / len(scores), 1),
                    'min_score': min(scores),
                    'reason': ', '.join(sorted(reasons)[:3]) if reasons else '低置信度',
                })
        i += 1

    # ── 增强建议帧：段首尾 + 最低置信度 + 加速度异常点 ──
    for seg in segments:
        suggested = {seg['start'], seg['end'] - 1 if seg['end'] > seg['start'] else seg['start']}
        # 最低置信度帧
        seg_frames = confidences[seg['start']:seg['end']]
        if seg_frames:
            suggested.add(min(seg_frames, key=lambda f: f['score'])['index'])
        # 加速度异常帧（与前后邻帧差值 > 阈值）
        for fi in range(seg['start'] + 1, seg['end']):
            if fi > 0 and fi + 1 < len(confidences):
                v = confidences[fi].get('speed', 0)
                v_prev = confidences[fi - 1].get('speed', 0)
                v_next = confidences[fi + 1].get('speed', 0)
                if v > 0 and v_prev > 0 and abs(v - v_prev) > ACCEL_ANOMALY_THRESHOLD:
                    suggested.add(fi)
                if v > 0 and v_next > 0 and abs(v_next - v) > ACCEL_ANOMALY_THRESHOLD:
                    suggested.add(fi + 1)
            if len(suggested) >= MAX_SUGGESTED_FRAMES:  # 最多 8 个建议帧
                break
        seg['suggested'] = sorted(suggested)

    segments.sort(key=lambda s: s['avg_score'])
    return segments
