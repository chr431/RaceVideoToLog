"""Correction — Phase 2: error correction based on confidence scores.

Receives per-frame confidence from Phase 1 (error_detection), interprets
the scores, and applies corrections using dense-lattice Viterbi + fill +
smoothness.

Single correction mode (auto, smoothness-first).  The manual mode was
removed in v2.12.1 — auto accuracy is high and two-mode maintenance was
not worth the cost.
"""
from __future__ import annotations
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ocr_engine import extract_speed_value, build_speed_candidates, Flag
# v2.12: 稠密格点连续 DP 替换离散候选 trellis —— 候选缺真值是旧 Viterbi
# 97% 的失败模式（只能挑最不坏的错误候选）；稠密能合成任意整数。同签名
# 换核，调用点不变。守卫见 viterbi_dense.py（单候选钉死 + min-obs）。
from viterbi_dense import dense_viterbi as viterbi_correct
from monitor import STAGE
from config import (
    MPS_TO_KMH,
    FILL_MAX_PASSES, CORRECTION_MIN_DIFF,
    AUTO_CORRECT_THRESHOLD,
    AUTO_SMOOTH_CLUSTER_MAX, AUTO_SMOOTH_DEVIATION_MULT,
    VITERBI_TRUSTED_BOUNDARY_CONFIDENCE, CORRECTION_MAX_ROUNDS,
    MAX_PARTIAL_WILDCARDS, VITERBI_MAX_CANDIDATES,
    SMOOTHNESS_MAX_ITERATIONS,
    AUTO_ALIGN_DIFF_MIN_KMH, AUTO_ALIGN_DIFF_MAX_KMH,
    AUTO_ALIGN_NUDGE_FACTOR, AUTO_ALIGN_MIN_CHANGE_KMH,
    FORCE_MEDIAN_MAX_ITERATIONS, FORCE_MEDIAN_NUDGE_FACTOR,
    FORCE_MEDIAN_THRESHOLD_MULT, FORCE_MEDIAN_MIN_CHANGE_KMH,
    CANDIDATE_POSTFILTER_ABS_MIN,
    CANDIDATE_HUNDREDS_MAX_DIFF,
    ACCEL_SCORE_ISLAND_INTERIOR,
    ZERO_CHANGE_TARGET_CONF, ZERO_CHANGE_NEIGHBOR_CONF, ZERO_CHANGE_DIFF_THRESHOLD,
    REF_GUARD_ABS_MIN, INTERP_ANCHOR_CONF_MIN,
    DISTANT_INTERP_MIN_TIME, DISTANT_INTERP_ISLAND_THRESHOLD,
    FORCE_MEDIAN_WINDOW_TIME,
    REF_INTERP_MAX_KMH_DIFF, REF_MIN_DIFF,
    VITERBI_POST_TRUST_THRESHOLD, TRUST_WINDOW_FALLBACK_MAX_DV,
    TRUST_WINDOW_TIME, FILL_CONFIDENCE_THRESHOLD, FILL_CANDIDATE_MAX_DIFF,
    FINAL_CONF_BLEND_PHASE1, FINAL_CONF_BLEND_VITERBI,
)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

logger = logging.getLogger("RaceVideoToLog.correction")

# PIL convert('L') 灰度权重（零变化约束二值化用）
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ═══════════════════ Helpers ═══════════════════

def _find_neighbor_trusted(i: int, n: int, rows: list) -> tuple[int | None, int | None]:
    la = None
    for j in range(i - 1, -1, -1):
        if Flag.is_trusted(rows[j][3]) and rows[j][2] >= 0:
            la = j; break
    ra = None
    for j in range(i + 1, n):
        if Flag.is_trusted(rows[j][3]) and rows[j][2] >= 0:
            ra = j; break
    return la, ra


def _interp_candidate(i: int, rows: list, times: list,
                      max_speed_kmh: float, fps: float = 1.0) -> float | None:
    n = len(rows)
    la, ra = _find_neighbor_trusted(i, n, rows)
    if la is not None and ra is not None:
        lv, rv = rows[la][2], rows[ra][2]
        lt, rt = rows[la][0] / fps, rows[ra][0] / fps
        total_dt = max(rt - lt, 1e-3)
        frac = (times[i] - lt) / total_dt
        val = lv + (rv - lv) * frac
        if 0 <= val <= max_speed_kmh:
            return round(val)
    return None


def _zc_pre_pass(rows: list, raw_frames: list, conf_by_idx: dict[int, float]) -> int:
    """零变化约束（holding 拉伸版）：连续小 diff 段 = 显示未变段，段内必须同值。

    单边硬证据：diff < T ⇒ 必然未变（数字变了像素必变）；diff >= T ⇒ 未知
    （模糊/运动/光照都可能，不用于判定"变了"）。故只把「小 diff 拉伸段」
    作为硬等值约束：段内任一高置信帧（>= NEIGHBOR_CONF）定值 → 整段低置信帧
    （< TARGET_CONF）强制取该值。比 pairwise 强在：一个锚点修复整段，不受
    链式传播断链影响。段内无高置信帧（如整段误读）则不动作，交给 ref/interp。

    惰性扩展：只算含低置信帧的拉伸涉及的边 diff（缓存），效率可控。
    返回修正帧数。"""
    if not raw_frames:
        return 0
    crops: dict[int, object] = {}
    for _fi, _crop in raw_frames:
        if _crop is not None:
            crops[_fi] = _crop
    # 固定阈值校准：采样 crops 的 Otsu 阈值取中位数（速度数字颜色大致恒定）。
    # 逐帧 Otsu 会被模糊帧直方图干扰（阈值漂移翻转背景像素），固定阈值
    # 更快且更稳（实测分离度相当：test5 未变 max 0.89→0.63%）。
    thresh = _zc_calibrate_thresh(crops)
    n = len(rows)
    if n < 2:
        return 0

    frac_cache: dict[int, float] = {}

    def _frac(edge: int) -> float:
        """边 (edge, edge+1) 的二值化 diff（缓存）。"""
        v = frac_cache.get(edge)
        if v is not None:
            return v
        a = crops.get(rows[edge][0])
        b = crops.get(rows[edge + 1][0])
        if a is None or b is None or a.shape != b.shape:
            v = 1.0
        else:
            ga = (a.astype(np.float32) @ _GRAY_WEIGHTS).astype(np.uint8)
            gb = (b.astype(np.float32) @ _GRAY_WEIGHTS).astype(np.uint8)
            v = float(((ga > thresh) != (gb > thresh)).mean())
        frac_cache[edge] = v
        return v

    fixed = 0
    visited: set[int] = set()
    for i in range(n):
        if conf_by_idx.get(i, 50) >= ZERO_CHANGE_TARGET_CONF:
            continue  # 目标必须低置信（可疑）
        if i in visited:
            continue
        # 扩展 holding 拉伸 [lo, hi]：相邻边 diff 均 < 阈值
        lo = i
        while lo > 0 and _frac(lo - 1) < ZERO_CHANGE_DIFF_THRESHOLD:
            lo -= 1
        hi = i
        while hi < n - 1 and _frac(hi) < ZERO_CHANGE_DIFF_THRESHOLD:
            hi += 1
        for k in range(lo, hi + 1):
            visited.add(k)
        # 段内高置信锚点（值须一致）
        anchors = [k for k in range(lo, hi + 1)
                   if conf_by_idx.get(k, 50) >= ZERO_CHANGE_NEIGHBOR_CONF
                   and rows[k][2] >= 0]
        if not anchors:
            continue
        avals = {rows[k][2] for k in anchors}
        if len(avals) != 1:
            continue  # 锚点值冲突 → 不动作（避免误判）
        av = next(iter(avals))
        for k in range(lo, hi + 1):
            if conf_by_idx.get(k, 50) >= ZERO_CHANGE_TARGET_CONF:
                continue  # 不覆盖高置信帧
            if abs(rows[k][2] - av) > 0.5:
                rows[k][2] = av
                rows[k][3] = Flag.ZERO_CHANGE
                fixed += 1
    return fixed


def _zc_calibrate_thresh(crops: dict, n_samples: int = 50) -> int:
    """采样 crops 的 Otsu 阈值，取中位数作为固定二值化阈值。"""
    if not crops:
        return 127
    keys = sorted(crops)
    step = max(1, len(keys) // n_samples)
    ths = []
    for k in keys[::step][:n_samples]:
        g = (crops[k].astype(np.float32) @ _GRAY_WEIGHTS).astype(np.uint8)
        ths.append(_otsu_thresh(g))
    return int(np.median(ths))


def _otsu_thresh(gray: np.ndarray) -> int:
    """Otsu 二值化阈值（最大化类间方差）。"""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = int(gray.size)
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    var_max = -1.0
    best = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        vb = w_b * w_f * (m_b - m_f) ** 2
        if vb > var_max:
            var_max = vb
            best = t
    return best


def _local_interp(i: int, rows: list, observations: list, times: list,
                  max_speed_kmh: float, fps: float = 1.0,
                  min_distance: int = 0,
                  exclude_flags: set[int] | None = None,
                  max_accel_mps2: float = 0.0,
                  conf_by_idx: dict[int, float] | None = None) -> float | None:
    """Interpolation using nearest valid neighbors.

    Anchors are always physically validated: a candidate anchor must be
    consistent with its immediate neighbors (±2 frames within max_accel) —
    a single wrong OCR value (e.g. 2744=17 vs neighbors 170) must not
    become an interpolation anchor.  exclude_flags additionally skips
    anchors carrying those flag values (e.g. auto-corrected frames whose
    values are guesses, not observations).

    conf_by_idx (Phase-1 confidence): when provided, anchors must ALSO be
    high-confidence (>= INTERP_ANCHOR_CONF_MIN) or already trusted.  The
    physics-only check is insufficient: a cluster of consecutive misreads
    is internally "physically consistent" and passes it, then poisons the
    interpolation (test5 1600: misread 1601=7 became the right anchor,
    local ref collapsed to garbage; distant ref then crossed a real
    speed curve and landed 6 km/h off).  Confidence catches these — the
    abs signal has wider context than ±2 frames, so a wrong cluster is
    low-confidence even when locally smooth."""
    n = len(rows)
    dv_frame = 0.0
    if n >= 2:
        dt_frame = times[1] - times[0] if times[1] > times[0] else 1.0 / max(fps, 1.0)
        dv_frame = max_accel_mps2 * dt_frame * MPS_TO_KMH

    def _valid_anchor(j: int) -> bool:
        if not (j < len(observations) and observations[j] and rows[j][2] > 0):
            return False
        if exclude_flags and rows[j][3] in exclude_flags:
            return False
        if conf_by_idx is not None:
            if not (conf_by_idx.get(j, 0) >= INTERP_ANCHOR_CONF_MIN
                    or Flag.is_trusted(rows[j][3])):
                return False
        if dv_frame > 0:
            # 锚点与其紧邻有效帧（±2 帧）的差必须在物理范围内
            for k in (j - 1, j + 1):
                if 0 <= k < n and rows[k][2] >= 0:
                    dt = abs(times[j] - times[k]) if times[k] != times[j] else 1.0 / max(fps, 1.0)
                    if abs(rows[j][2] - rows[k][2]) > max_accel_mps2 * dt * MPS_TO_KMH * 2:
                        return False
        return True

    la = None
    for j in range(i - 1 - min_distance, -1, -1):
        if _valid_anchor(j):
            la = j; break
    ra = None
    for j in range(i + 1 + min_distance, n):
        if _valid_anchor(j):
            ra = j; break
    if la is None or ra is None:
        return None
    lv, rv = rows[la][2], rows[ra][2]
    lt, rt = times[la], times[ra]
    total_dt = max(rt - lt, 1e-3)
    frac = (times[i] - lt) / total_dt
    val = lv + (rv - lv) * frac
    if 0 <= val <= max_speed_kmh:
        return round(val)
    return None


def expand_partial(pattern: str, max_speed: float) -> list[int]:
    import itertools
    x_count = pattern.count('x') + pattern.count('X')
    if x_count == 0:
        val = float(pattern)
        return [int(val)] if val <= max_speed else []
    if x_count == len(pattern) or x_count > MAX_PARTIAL_WILDCARDS:
        return []
    results = []
    for digits in itertools.product('0123456789', repeat=x_count):
        di = 0; chars = []
        for ch in pattern.lower():
            if ch == 'x': chars.append(digits[di]); di += 1
            else: chars.append(ch)
        val = float(''.join(chars))
        if val <= max_speed: results.append(val)
    return results


def _auto_expand_digits(raw_text: str, max_speed_kmh: float) -> list[int]:
    if not raw_text or not raw_text.isdigit():
        return []
    digits = raw_text
    if len(digits) >= 4: return []
    candidates: list[int] = []
    try: candidates.append(int(digits))
    except ValueError: pass
    if len(digits) <= 2:
        for pos in range(len(digits) + 1):
            pattern = digits[:pos] + "x" + digits[pos:]
            for v in expand_partial(pattern, max_speed_kmh):
                if v not in candidates: candidates.append(v)
    for pos in range(len(digits)):
        pattern = digits[:pos] + "x" + digits[pos + 1:]
        for v in expand_partial(pattern, max_speed_kmh):
            if v not in candidates: candidates.append(v)
    return candidates


# ═══════════════════ Candidate generation ═══════════════════

def _generate_candidates(fi: int, rows: list, observations: list, raw_frames: list,
                         pinned_set: set, times: list,
                         max_speed_kmh: float,
                         split_results: dict[int, str] | None,
                         fps: float,
                         max_accel_mps2: float | None = None,
                         max_width: int = 0) -> list[float]:
    raw_val = rows[fi][2]
    if max_accel_mps2 is None:
        max_accel_mps2 = 0.0  # 未指定 = 不施加加速度约束
    protected: list[float] = []
    protected_set: set[float] = set()

    if 0 <= raw_val <= max_speed_kmh:
        protected.append(raw_val)
        protected_set.add(raw_val)

    obs = observations[min(fi, len(observations) - 1)]

    if split_results and fi in split_results:
        try:
            sv = int(split_results[fi])
            if 0 <= sv <= max_speed_kmh and sv not in protected_set:
                protected.append(sv); protected_set.add(sv)
        except ValueError: pass

    if obs.raw_text and obs.raw_text.isdigit() and raw_val > 0:
        base = int(obs.raw_text) % 100
        for hundreds in range(0, int(max_speed_kmh) + 1, 100):
            alt = hundreds + base
            if 0 <= alt <= max_speed_kmh and alt not in protected_set:
                protected.append(alt); protected_set.add(alt)

    other: list[float] = []
    other_set: set[float] = set()

    needs_interp = raw_val < 0
    if needs_interp:
        interp_val = _interp_candidate(fi, rows, times, max_speed_kmh, fps=fps)
        if interp_val is None:
            interp_val = _local_interp(fi, rows, observations, times, max_speed_kmh, fps=fps, max_accel_mps2=max_accel_mps2)
        if interp_val is not None and interp_val not in protected_set and interp_val not in other_set:
            if raw_val < 0:
                protected.append(interp_val); protected_set.add(interp_val)
            else:
                other.append(interp_val); other_set.add(interp_val)

    remaining = VITERBI_MAX_CANDIDATES - len(protected)
    if remaining > 0 and len(other) > remaining:
        def _rank(v): return abs(v - raw_val) / max(1.0, abs(raw_val)) if raw_val > 0 else abs(v - raw_val)
        other.sort(key=_rank)
        other = other[:remaining]

    return protected + other


# ═══════════════════ Fill unrecoverable ═══════════════════

def _fill_unrecoverable(rows: list, pinned_set: set, error_set: set, times: list,
                        max_speed_kmh: float, max_accel_mps2: float, fps: float = 1.0,
                        progress_fn: "Callable | None" = None,
                        notes: dict[int, str] | None = None,
                        candidates_by_frame: dict[int, list[float]] | None = None) -> None:
    """Fill frames that Viterbi could not recover."""
    n = len(rows)
    sorted_errors = sorted(i for i in error_set if i not in pinned_set and not Flag.is_trusted(rows[i][3]))
    total = len(sorted_errors)
    for idx, i in enumerate(sorted_errors):
        raw_v = rows[i][2]
        la = None
        for j in range(i - 1, -1, -1):
            if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
                la = j; break
        if la is None: continue
        lv = rows[la][2]; lt = rows[la][0] / fps
        ra = None
        for j in range(i + 1, n):
            if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
                ra = j; break
        left_dt = max(times[i] - lt, 1e-3)
        left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
        lo, hi = 0.0, max_speed_kmh
        interp = lv
        if ra is not None:
            rv = rows[ra][2]; rt = rows[ra][0] / fps
            right_dt = max(rt - times[i], 1e-3)
            right_max_dv = max_accel_mps2 * right_dt * MPS_TO_KMH
            lo = max(lo, lv - left_max_dv, rv - right_max_dv)
            hi = min(hi, lv + left_max_dv, rv + right_max_dv)
            interp = lv + (rv - lv) * (left_dt / max(left_dt + right_dt, 1e-3))
        else:
            lo = max(lo, lv - left_max_dv)
            hi = min(hi, lv + left_max_dv)
        if hi < lo:  # 锚点物理不可达（急刹）：收敛到可达区间中点，避免越界
            val = int((lo + hi) / 2 + 0.5)
        else:
            # 候选优先：re-OCR 读出的正确值（物理可达且接近插值时）优于纯插值猜测
            # 距离保护：re-OCR 错误候选（如 132 vs 插值 112）不得覆盖合理插值
            cands = candidates_by_frame.get(i) if candidates_by_frame else None
            val = None
            if cands:
                in_range = [c for c in cands
                            if lo - 0.5 <= c <= hi + 0.5
                            and abs(c - interp) <= FILL_CANDIDATE_MAX_DIFF]
                if in_range:
                    # 选最接近插值点的候选（re-OCR 值优先，插值作为 tie-break）
                    val = min(in_range, key=lambda c: abs(c - interp))
            if val is None:
                val = int(max(lo, min(hi, interp)) + 0.5)  # +0.5: 避免 banker's rounding
        rows[i][2] = int(val)
        if rows[i][3] == Flag.RAW:
            rows[i][3] = Flag.FILL_INTERP
        if notes is not None: notes[i] = f"fill: {val:.0f}"
        if progress_fn: progress_fn(idx + 1, total)


# ═══════════════════ Smoothness pass ═══════════════════

def _smoothness_pass(rows: list, times: list, max_speed_kmh: float,
                     max_accel_mps2: float, fps: float = 1.0,
                     notes: dict[int, str] | None = None) -> int:
    n = len(rows)
    if n < 3: return 0
    max_dv = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else 8.0
    threshold = max_dv * AUTO_SMOOTH_DEVIATION_MULT
    smoothed = 0
    for _pass in range(SMOOTHNESS_MAX_ITERATIONS):
        fixed_this_pass = 0
        for i in range(n):
            v_cur = rows[i][2]
            if v_cur < 0: continue
            bad = 0
            for ni in (i - 1, i + 1):
                if 0 <= ni < n and rows[ni][2] > 0:
                    if abs(v_cur - rows[ni][2]) > threshold: bad += 1
            if bad < 2: continue
            cluster_size = 1
            for j in range(i - 1, -1, -1):
                if rows[j][2] < 0: break
                nbr_bad = sum(1 for nj in (j-1, j+1) if 0 <= nj < n and rows[nj][2] > 0 and abs(rows[j][2] - rows[nj][2]) > threshold)
                if nbr_bad >= 2: cluster_size += 1
                else: break
            for j in range(i + 1, n):
                if rows[j][2] < 0: break
                nbr_bad = sum(1 for nj in (j-1, j+1) if 0 <= nj < n and rows[nj][2] > 0 and abs(rows[j][2] - rows[nj][2]) > threshold)
                if nbr_bad >= 2: cluster_size += 1
                else: break
            if cluster_size > AUTO_SMOOTH_CLUSTER_MAX: continue
            left_val = None
            for j in range(i - 1, -1, -1):
                if rows[j][2] >= 0 and abs(rows[j][2] - v_cur) <= threshold:
                    left_val = rows[j][2]; break
            right_val = None
            for j in range(i + 1, n):
                if rows[j][2] >= 0 and abs(rows[j][2] - v_cur) <= threshold:
                    right_val = rows[j][2]; break
            if left_val is not None and right_val is not None: target = (left_val + right_val) / 2.0
            elif left_val is not None: target = left_val
            elif right_val is not None: target = right_val
            else: continue
            lo, hi = 0.0, max_speed_kmh
            for j in range(i - 1, -1, -1):
                if rows[j][2] >= 0 and abs(rows[j][2] - v_cur) <= threshold:
                    lv = rows[j][2]; lt = rows[j][0] / fps
                    dt_l = max(times[i] - lt, 1e-3)
                    lo = max(lo, lv - max_accel_mps2 * dt_l * MPS_TO_KMH)
                    hi = min(hi, lv + max_accel_mps2 * dt_l * MPS_TO_KMH)
                    break
            for j in range(i + 1, n):
                if rows[j][2] >= 0 and abs(rows[j][2] - v_cur) <= threshold:
                    rv = rows[j][2]; rt = rows[j][0] / fps
                    dt_r = max(rt - times[i], 1e-3)
                    lo = max(lo, rv - max_accel_mps2 * dt_r * MPS_TO_KMH)
                    hi = min(hi, rv + max_accel_mps2 * dt_r * MPS_TO_KMH)
                    break
            new_val = round(max(lo, min(hi, target)))
            if abs(new_val - v_cur) < 0.5: continue
            rows[i][2] = int(new_val)
            if notes is not None: notes[i] = f"smooth: {v_cur:.0f}→{new_val:.0f}"
            smoothed += 1; fixed_this_pass += 1
        if fixed_this_pass == 0: break
    return smoothed


# ═══════════════════ Auto-mode SG alignment ═══════════════════

def _auto_align_pass(rows: list, observations: list, times: list,
                     max_speed_kmh: float, max_accel_mps2: float,
                     fps: float, confidence_scores: list,
                     notes: dict[int, str] | None = None) -> int:
    """Auto mode only: nudge frames with moderate SG deviation toward
    local interpolation. Catches systematic small OCR offsets (5-15 km/h)
    that are too small for Viterbi but still meaningful for curve quality."""
    n = len(rows)
    if n < 3:
        return 0
    max_dv_per_frame = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else 4.0
    corrected = 0

    for i in range(n):
        if Flag.is_trusted(rows[i][3]) or rows[i][2] < 0:
            continue
        # 跳过已被更强阶段修正的帧（fill / re-OCR / partial）：
        # 这些帧的 align 锚点须排除修正帧 → 锚点退到岛两端，线性插值在
        # 阶梯坡上滞后（实测 test4 align 61/62 把 fill 已写对的正确值拉偏）。
        # fine-tune 只应作用于原始 RAW 帧的小偏移，不应对抗强阶段输出。
        if Flag.is_corrected(rows[i][3]):
            continue
        sigs = confidence_scores[i].get('signals', {}) if i < len(confidence_scores) else {}
        conf = confidence_scores[i].get('score', 50) if i < len(confidence_scores) else 50
        # 跳过 Phase 1 高置信帧（conf≥70 即被判定可信）：align 只应修正
        # 可疑帧的小偏移；可信帧若有偏移，启发式 nudge 在阶梯坡上只会因
        # 插值滞后而改错（实测 test4 RAW 帧 align 5 毁 0 修）。
        if conf >= 70:
            continue
        acc = sigs.get('accel', 100)
        # Skip island interiors (very low accel score) — need stronger correction
        if acc <= ACCEL_SCORE_ISLAND_INTERIOR + 5:
            continue

        cur_v = rows[i][2]
        # 参考插值排除所有自动修正帧（FILL/REOCR_AUTO/PARTIAL）：
        # 对齐基准只能来自未修正的原始值或信任锚点，避免被污染的修正值二次传播
        interp = _local_interp(i, rows, observations, times, max_speed_kmh, fps=fps, max_accel_mps2=max_accel_mps2,
                               exclude_flags={Flag.FILL_INTERP, Flag.REOCR_AUTO,
                                              Flag.PARTIAL_AUTO})
        if interp is None:
            # Try trusted-neighbor interpolation as fallback
            interp = _interp_candidate(i, rows, times, max_speed_kmh, fps=fps)
        if interp is None:
            continue
        diff = interp - cur_v
        if abs(diff) < AUTO_ALIGN_DIFF_MIN_KMH or abs(diff) > AUTO_ALIGN_DIFF_MAX_KMH:
            continue

        # Acceleration-constrained nudge toward interpolation
        lo, hi = 0.0, max_speed_kmh
        for j in range(i - 1, -1, -1):
            if rows[j][2] >= 0:
                dt = max(times[i] - times[j], 1e-3)
                dv_limit = max_accel_mps2 * dt * MPS_TO_KMH
                lo = max(lo, rows[j][2] - dv_limit)
                hi = min(hi, rows[j][2] + dv_limit)
                break
        for j in range(i + 1, n):
            if rows[j][2] >= 0:
                dt = max(times[j] - times[i], 1e-3)
                dv_limit = max_accel_mps2 * dt * MPS_TO_KMH
                lo = max(lo, rows[j][2] - dv_limit)
                hi = min(hi, rows[j][2] + dv_limit)
                break

        # Nudge toward interpolation (80% to stay somewhat conservative)
        target = cur_v + diff * AUTO_ALIGN_NUDGE_FACTOR
        new_val = round(max(lo, min(hi, target)))
        if abs(new_val - cur_v) < AUTO_ALIGN_MIN_CHANGE_KMH:
            continue  # Too small a change, skip

        rows[i][2] = int(new_val)
        if notes is not None:
            notes[i] = (notes.get(i, '') + f" align: {cur_v:.0f}→{new_val:.0f}").strip()
        corrected += 1

    return corrected


# ═══════════════════ Force-SG smoothing (final pass) ═══════════════════

def _force_median_smooth(rows: list, times: list, max_speed_kmh: float,
                     max_accel_mps2: float, fps: float = 1.0,
                     notes: dict[int, str] | None = None) -> int:
    """Median-filter smoothing pass.

    Goal: minimize max_dv (frame-to-frame speed change), not to match truth.
    自动模式：不处理已修正帧 —— 强阶段（Viterbi/fill/re-OCR）输出已是
    最终值；median 窗口排除 FILL 值（避免互相确认）后，在阶梯坡岛上 median
    被拉到非岛 RAW 锚点，反而把正确的 fill 值重新拉偏（实测 test4 0修7毁、
    test6 0修9毁）。

    Uses iterative time-window sliding median: if the center value deviates
    from the local median by more than max_dv_per_frame * threshold_mult, it gets
    nudged toward the median (acceleration-clamped)."""
    n = len(rows)
    if n < 5:
        return 0
    max_dv_per_frame = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else 4.0
    threshold = max_dv_per_frame * FORCE_MEDIAN_THRESHOLD_MULT
    total_smoothed = 0
    dry_passes = 0  # consecutive passes with very few changes

    # 窗口值来源：全帧平滑（排除 FILL 猜测值，避免互相确认）
    window_ok = (lambda r: r[3] != Flag.FILL_INTERP)

    for _pass in range(FORCE_MEDIAN_MAX_ITERATIONS):
        changed = 0
        for i in range(2, n - 2):
            v = rows[i][2]
            if v < 0:
                continue
            # 不处理已修正帧（见函数 docstring）
            if Flag.is_corrected(rows[i][3]):
                continue
            # Time-based median window
            med_look = max(1, int(FORCE_MEDIAN_WINDOW_TIME / max((times[1] - times[0]) if n >= 2 else 1/30, 1e-3)))
            neighbors = [rows[j][2] for j in range(i - med_look, i + med_look + 1)
                         if 0 <= j < n and rows[j][2] >= 0 and window_ok(rows[j])]
            if len(neighbors) < 3:
                continue
            neighbors.sort()
            median = neighbors[len(neighbors) // 2]

            if abs(v - median) < 0.5:
                continue

            # Only nudge if beyond threshold, toward median
            diff = median - v
            if abs(diff) <= threshold:
                continue

            # Acceleration clamp
            lo, hi = 0.0, max_speed_kmh
            if i > 0 and rows[i - 1][2] >= 0:
                dt_left = max(times[i] - times[i - 1], 1e-3)
                dv_limit = max_accel_mps2 * dt_left * MPS_TO_KMH
                lo = max(lo, rows[i - 1][2] - dv_limit)
                hi = min(hi, rows[i - 1][2] + dv_limit)
            if i < n - 1 and rows[i + 1][2] >= 0:
                dt_right = max(times[i + 1] - times[i], 1e-3)
                dv_limit = max_accel_mps2 * dt_right * MPS_TO_KMH
                lo = max(lo, rows[i + 1][2] - dv_limit)
                hi = min(hi, rows[i + 1][2] + dv_limit)

            target = v + diff * FORCE_MEDIAN_NUDGE_FACTOR  # Partial nudge
            new_val = round(max(lo, min(hi, target)))
            if abs(new_val - v) < FORCE_MEDIAN_MIN_CHANGE_KMH:
                continue

            rows[i][2] = int(new_val)
            if notes is not None:
                notes[i] = (notes.get(i, '') + f" forceSG: {v:.0f}→{new_val:.0f}").strip()
            changed += 1

        total_smoothed += changed

        # Adaptive convergence: stop when two consecutive passes fix very
        # few frames — remaining deviations are either too small to matter
        # or inside large error islands where local median can't help.
        if changed <= 2:
            dry_passes += 1
            if dry_passes >= 2:
                break
        else:
            dry_passes = 0

    return total_smoothed


# ═══════════════════ Main correction pipeline ═══════════════════

def correct_errors(rows: list, observations: list, raw_frames: list,
                   confidence_scores: list[dict],
                   times: list[float], max_speed_kmh: float, max_accel_mps2: float,
                   pinned: set[int] | None = None,
                   split_results: dict[int, str] | None = None,
                   fps: float = 1.0, log_fn: "Callable | None" = None,
                   progress_fn: "Callable | None" = None,
                   notes: dict[int, str] | None = None,
                   max_width: int = 0,
                   ) -> tuple[list, list[dict]]:
    pinned = pinned or set()
    n = len(rows)
    for pi in pinned:
        if rows[pi][3] < Flag.HIGH_TRUST: rows[pi][3] = Flag.PINNED
    pinned_set = pinned
    correct_threshold = AUTO_CORRECT_THRESHOLD

    conf_by_idx: dict[int, float] = {}
    for c in confidence_scores: conf_by_idx[c['index']] = c['score']

    # ── 零变化约束（思路 2）──
    # 相邻帧 ROI 差分 < 阈值 + 邻帧高置信 → 目标帧 = 邻帧值。修复"显示未变
    # 但单帧 OCR 误读"（clear 视频未变帧 mean diff ~0.03 vs 变帧 ~30，分离度好）。
    # 只作用于低置信目标帧（conf < ZERO_CHANGE_TARGET_CONF），从不破坏高置信帧。
    n_zc = _zc_pre_pass(rows, raw_frames, conf_by_idx)
    if log_fn and n_zc > 0:
        log_fn(f"  Zero-change: {n_zc} frames forced equal to holding display")

    if log_fn:
        log_fn(f"Correction (Phase 2, auto): threshold={correct_threshold}, "
               f"{n} rows, {len(pinned_set)} pinned")

    correction_frames: set[int] = set()
    for i in range(n):
        if i in pinned_set or Flag.is_trusted(rows[i][3]): continue
        if Flag.is_corrected(rows[i][3]): continue  # 零变化约束已修正 → 不再重纠正
        if conf_by_idx.get(i, 50) < correct_threshold:
            correction_frames.add(i)

    candidates_by_frame: dict[int, list[float]] = {}
    total_correct = len(correction_frames)

    with STAGE.stage("corr.candidates"):
        for idx, fi in enumerate(sorted(correction_frames)):
            cands = _generate_candidates(fi, rows, observations, raw_frames, pinned_set,
                times, max_speed_kmh, split_results,
                fps=fps,
                max_accel_mps2=max_accel_mps2, max_width=max_width)
            if cands: candidates_by_frame[fi] = cands
            if progress_fn and total_correct > 0: progress_fn(idx + 1, total_correct)

    with STAGE.stage("corr.cheap"):
        n_cheap = 0
        for i in range(n):
            if i in pinned_set or Flag.is_trusted(rows[i][3]) or i in candidates_by_frame: continue
            conf = conf_by_idx.get(i, 50)
            if conf >= VITERBI_TRUSTED_BOUNDARY_CONFIDENCE: continue
            raw_val = rows[i][2]
            obs = observations[min(i, len(observations) - 1)]
            cands: list[float] = []; cands_set: set[float] = set()
            if 0 <= raw_val <= max_speed_kmh:
                cands.append(raw_val); cands_set.add(raw_val)
            if obs.raw_text and obs.raw_text.isdigit() and raw_val > 0:
                base = int(obs.raw_text) % 100
                for hundreds in range(0, int(max_speed_kmh) + 1, 100):
                    alt = hundreds + base
                    if 0 <= alt <= max_speed_kmh and alt not in cands_set:
                        cands.append(alt); cands_set.add(alt)
            if split_results and i in split_results:
                try:
                    sv = int(split_results[i])
                    if 0 <= sv <= max_speed_kmh and sv not in cands_set:
                        cands.append(sv); cands_set.add(sv)
                except (ValueError, TypeError): pass
            for cv in _auto_expand_digits(obs.raw_text, max_speed_kmh):
                if cv not in cands_set and 0 <= cv <= max_speed_kmh:
                    cands.append(cv); cands_set.add(cv)
            if len(cands) > 1:
                candidates_by_frame[i] = cands; n_cheap += 1

    if log_fn:
        n_with = len(candidates_by_frame)
        total_c = sum(len(c) for c in candidates_by_frame.values())
        log_fn(f"  Candidates: {n_with} frames ({n_cheap} cheap), {total_c} total")

    # ── Post-filter: remove dangerous hundreds-digit variants ──
    # For frames that are internally consistent (abs>=85), a hundreds-digit
    # variant that differs by >=100 from the raw value is almost certainly
    # wrong. Removing it prevents Viterbi from choosing a path that looks
    # "cheaper" due to lower transition cost from nearby wrong frames.
    # Example: raw=168 (3-digit), abs=100 → remove candidate 68 (168-100)
    #   to prevent false correction.
    n_filtered = 0
    n_island_cands = 0  # Track frames with distant-interp candidates
    with STAGE.stage("corr.postfilter"):
        for fi in list(candidates_by_frame.keys()):
            sigs = confidence_scores[fi].get('signals', {}) if fi < len(confidence_scores) else {}
            abs_s = sigs.get('abs', 50)
            raw_v = rows[fi][2]
            if raw_v <= 0: continue
            if abs_s >= CANDIDATE_POSTFILTER_ABS_MIN:
                old_cands = candidates_by_frame[fi]
                new_cands = [c for c in old_cands if abs(c - raw_v) < CANDIDATE_HUNDREDS_MAX_DIFF]
                if len(new_cands) < len(old_cands):
                    n_filtered += 1
                    candidates_by_frame[fi] = new_cands if len(new_cands) >= 1 else [raw_v]
            # Track frames where distant interpolation was added
            a = sigs.get('accel', 100)
            if a <= ACCEL_SCORE_ISLAND_INTERIOR + 5 and abs_s >= CANDIDATE_POSTFILTER_ABS_MIN:
                n_island_cands += 1
    if log_fn and n_filtered > 0:
        log_fn(f"  Filtered hundreds variants from {n_filtered} consistent 3-digit frames")

    # Build reference values for ALL non-trusted frames.
    # (历史：手动模式曾限定 conf<40，遗留 conf 40-70 帧无参考保护，Viterbi
    # 转移代价把它们拉进错误斜坡 —— 2731-cluster / 1115-cluster。已统一覆盖。)
    #
    # Guard 1: skip internally consistent frames (abs>=85 AND accel>=70) —
    # raw OCR almost certainly correct.
    # Guard 2 (REF_MIN_DIFF): if raw already agrees with interpolation (<3),
    # the frame is self-consistent — no reference (else Viterbi would be
    # pulled off a correct raw by a ±1-2 off interpolation).
    with STAGE.stage("corr.reference"):
        reference_values: dict[int, float] = {}
        conf_by_idx_ref = {c['index']: c['score'] for c in confidence_scores}
        for i in range(n):
            if i in pinned_set or Flag.is_trusted(rows[i][3]): continue
            sigs = confidence_scores[i].get('signals', {}) if i < len(confidence_scores) else {}
            abs_s = sigs.get('abs', 50)
            a = sigs.get('accel', 100)
            if abs_s >= REF_GUARD_ABS_MIN and a >= 70:
                # abs 自洽 + accel 正常 → trust OCR, skip interpolation
                continue
            raw_v = rows[i][2]
            # 本地插值锚点必须高置信/可信（INTERP_ANCHOR_CONF_MIN）：
            # 物理自洽的误读簇（如 test5 1601=7 相对 ±2 邻居一致）不能当锚点，
            # 否则本地插值塌成垃圾、distant 跳岛又跨过真实速度曲线（错 6 km/h）。
            ref = _local_interp(i, rows, observations, times, max_speed_kmh, fps=fps,
                                max_accel_mps2=max_accel_mps2, conf_by_idx=conf_by_idx_ref)
            # For suspected island interiors (accel <= 15) where the conf-gated
            # local search finds NO reliable anchors, the nearest neighbors may
            # all be wrong. Try distant interpolation that skips past the island
            # to reach correct anchor frames. Only when local is None: if local
            # found good anchors, its interpolation is the better estimate and
            # must not be overridden (distant spans real accel/decel curves).
            # （历史：手动模式曾禁用 distant —— 893/1708/1709 被拉到错误
            # distant ref。自动模式实测无此回归。）
            if ref is None and a <= ACCEL_SCORE_ISLAND_INTERIOR + 5 and raw_v > 0:
                dt_frame = (times[1] - times[0]) if n >= 2 else 1/60
                min_frames = max(1, int(DISTANT_INTERP_MIN_TIME / dt_frame))
                distant = _local_interp(i, rows, observations, times, max_speed_kmh,
                                        fps=fps, min_distance=min_frames,
                                        max_accel_mps2=max_accel_mps2,
                                        conf_by_idx=conf_by_idx_ref)
                if distant is not None and abs(distant - raw_v) > DISTANT_INTERP_ISLAND_THRESHOLD:
                    # Island detected: local cluster differs from distant anchors
                    ref = distant
                    # Add distant interp as candidate so Viterbi has a
                    # correct option to choose from.
                    if i in candidates_by_frame:
                        if distant not in candidates_by_frame[i]:
                            candidates_by_frame[i].append(distant)
                    elif distant != raw_v:
                        candidates_by_frame[i] = [raw_v, distant]
            if ref is None:
                ref = _interp_candidate(i, rows, times, max_speed_kmh, fps=fps)
            # Safety: if interpolation is far from raw OCR (>50 km/h), the
            # interpolation likely crosses a real speed change. Skip it to
            # prevent false corrections (e.g., 168→68 across a speed jump).
            # Exception: island interiors (sg <= 20) where large discrepancies
            # are expected and indicate the raw value is wrong.
            if ref is not None and raw_v > 0 and abs(ref - raw_v) > REF_INTERP_MAX_KMH_DIFF and a > ACCEL_SCORE_ISLAND_INTERIOR + 10:
                ref = None
            # Guard 2: raw already agrees with interpolation → frame is
            # self-consistent; a reference would only pull it ±1-2 off.
            if ref is not None and raw_v > 0 and abs(ref - raw_v) < REF_MIN_DIFF:
                ref = None
            if ref is not None: reference_values[i] = ref

    trusted_set = pinned_set.copy()
    for i in range(n):
        if Flag.is_trusted(rows[i][3]): trusted_set.add(i)

    _dv = max_accel_mps2 * (times[1] - times[0]) * MPS_TO_KMH if n >= 2 else TRUST_WINDOW_FALLBACK_MAX_DV
    total_fixed, total_trusted = 0, 0
    viterbi_conf: dict[int, float] = {}

    # ── When island candidates are present, limit to 1 round ──
    # Multi-round Viterbi can undo island corrections (r1 fixes 113→213,
    # r2 sees transition tension with uncorrected neighbors & reverts).
    # （历史：手动模式单轮 —— r1 2747→11 后 r2 信任传播把错误斜坡互相确认。）
    max_rounds = 1 if n_island_cands > 0 else CORRECTION_MAX_ROUNDS
    if log_fn and max_rounds == 1:
        log_fn(f"  Island mode: max 1 Viterbi round")

    # ── 自洽帧锚定 raw（单候选）— 两种模式统一 ──
    # raw 与物理插值一致（差 < REF_MIN_DIFF）说明该帧 raw 高可信；
    # 若不锚定，Viterbi 转移代价会把它们拉进错误斜坡
    # （实测 2749 raw=171 被改 71，随后 2750-2760 链式斜坡）。
    # 锚定在 Viterbi 前一次性完成（候选不随 round 变化）。
    with STAGE.stage("corr.anchor"):
        for _i in range(n):
            _c = candidates_by_frame.get(_i)
            if not _c or len(_c) <= 1:
                continue
            _rv = rows[_i][2]
            if _rv <= 0:
                continue
            _ip = _local_interp(_i, rows, observations, times, max_speed_kmh,
                                fps=fps, max_accel_mps2=max_accel_mps2)
            if _ip is not None and abs(_ip - _rv) < REF_MIN_DIFF:
                candidates_by_frame[_i] = [_rv]

    with STAGE.stage("corr.viterbi"):
        for round_num in range(max_rounds):
            viterbi_result = viterbi_correct(rows, candidates_by_frame, confidence_scores,
                times, max_speed_kmh, max_accel_mps2, trusted_indices=trusted_set,
                reference_values=reference_values)

            round_fixed = 0
            for fi, new_val in sorted(viterbi_result['corrected'].items()):
                if fi in pinned_set or Flag.is_trusted(rows[fi][3]): continue
                old_val = rows[fi][2]
                # 自动模式保留小步长（平滑优先）；±1 噪声微调由阈值保护
                min_diff = CORRECTION_MIN_DIFF
                if abs(new_val - old_val) <= min_diff: continue
                rows[fi][2] = new_val
                if rows[fi][3] == Flag.RAW: rows[fi][3] = Flag.REOCR_AUTO
                if notes is not None: notes[fi] = f"viterbi(r{round_num+1}): {old_val:.0f}→{new_val:.0f}"
                round_fixed += 1
            total_fixed += round_fixed

            viterbi_conf = {c['index']: c['score'] for c in viterbi_result['confidence']}
            round_trusted = 0
            for i in range(n):
                if i in trusted_set or rows[i][3] != Flag.RAW: continue
                v = rows[i][2]
                if v < 0: continue
                vc = viterbi_conf.get(i, 50)
                if vc < VITERBI_POST_TRUST_THRESHOLD and i in candidates_by_frame and len(candidates_by_frame[i]) > 1: continue
                trust_look = max(1, int(TRUST_WINDOW_TIME / max((times[1] - times[0]) if n >= 2 else 1/30, 1e-3)))
                # 验证必须包含 REOCR_AUTO 邻居：跳过它们会留下验证空洞，
                # 使严重错值（如 2744=17 vs 邻居 170）被错误标 HT，随后成为
                # fill 的插值锚点并污染整段（2731-2761 斜坡的根因）。
                left_ok = True
                for j in range(i - 1, max(-1, i - 1 - trust_look), -1):
                    if j < 0: break
                    nbr_v = rows[j][2]
                    if nbr_v < 0: continue
                    if abs(v - nbr_v) > (i - j) * _dv: left_ok = False; break
                if not left_ok: continue
                right_ok = True
                for j in range(i + 1, min(n, i + 1 + trust_look)):
                    nbr_v = rows[j][2]
                    if nbr_v < 0: continue
                    if abs(nbr_v - v) > (j - i) * _dv: right_ok = False; break
                if not right_ok: continue
                rows[i][3] = Flag.HIGH_TRUST; trusted_set.add(i); round_trusted += 1
            total_trusted += round_trusted

            if log_fn:
                log_fn(f"  Round {round_num+1}: {len(viterbi_result['error_set'])} errors, "
                       f"{round_fixed} fixed, {round_trusted} new HT")
            if round_fixed == 0 and round_trusted == 0: break

    with STAGE.stage("corr.fill"):
        remaining_errors: set[int] = set()
        for i in range(n):
            if i in trusted_set or i in pinned_set or Flag.is_trusted(rows[i][3]): continue
            if conf_by_idx.get(i, 50) < FILL_CONFIDENCE_THRESHOLD: remaining_errors.add(i)
        if remaining_errors:
            for fill_pass in range(FILL_MAX_PASSES):
                if not remaining_errors: break
                _fill_unrecoverable(rows, pinned_set, remaining_errors, times,
                    max_speed_kmh, max_accel_mps2, fps, progress_fn=progress_fn, notes=notes,
                    candidates_by_frame=candidates_by_frame)
                if log_fn: log_fn(f"  Fill pass {fill_pass+1}: {len(remaining_errors)} frames")
                remaining_errors = {i for i in range(n)
                    if i not in trusted_set and i not in pinned_set
                    and not Flag.is_trusted(rows[i][3]) and conf_by_idx.get(i, 50) < FILL_CONFIDENCE_THRESHOLD}
        elif log_fn: log_fn("  Fill: no remaining errors")

    with STAGE.stage("corr.smoothness"):
        n_smoothed = _smoothness_pass(rows, times, max_speed_kmh, max_accel_mps2, fps, notes)
    if log_fn and n_smoothed > 0: log_fn(f"  Smoothness: {n_smoothed} spikes smoothed")

    # ── SG-guided alignment（参考排除 FILL 帧）──
    with STAGE.stage("corr.align"):
        n_aligned = _auto_align_pass(rows, observations, times, max_speed_kmh,
                                     max_accel_mps2, fps, confidence_scores, notes)
    if log_fn and n_aligned > 0:
        log_fn(f"  Auto-align: {n_aligned} frames nudged")

    # ── Median smoothing（全帧，最小化 max_dv）──
    with STAGE.stage("corr.force_median"):
        n_forced = _force_median_smooth(rows, times, max_speed_kmh, max_accel_mps2, fps, notes)
    if log_fn and n_forced > 0:
        log_fn(f"  Force-SG: {n_forced} frame-nudges applied")

    with STAGE.stage("corr.conf_blend"):
        for c in confidence_scores:
            i = c['index']
            c['is_corrected'] = Flag.is_corrected(rows[i][3])
            if i in viterbi_conf:
                c['score'] = round(c['score'] * FINAL_CONF_BLEND_PHASE1 + viterbi_conf[i] * FINAL_CONF_BLEND_VITERBI, 1)

    return rows, confidence_scores
