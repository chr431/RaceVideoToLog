"""Correction — Phase 2: error correction based on confidence scores.

Receives per-frame confidence from Phase 1 (error_detection), interprets
the scores, and applies corrections using Viterbi DP + fill + smoothness.

Both modes share the same full pipeline (fill, smoothness, auto-align).
Mode differences are expressed by ModeProfile (auto: smoothness-first,
manual: precision-first d=0 protection).
"""
from __future__ import annotations
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ocr_engine import extract_speed_value, build_speed_candidates, Flag
from viterbi import viterbi_correct
from config import (
    MPS_TO_KMH,
    FILL_MAX_PASSES, CORRECTION_MIN_DIFF, MANUAL_CORRECTION_MIN_DIFF,
    MANUAL_CORRECT_THRESHOLD, AUTO_CORRECT_THRESHOLD,
    AUTO_SMOOTH_CLUSTER_MAX, AUTO_SMOOTH_DEVIATION_MULT,
    VITERBI_TRUSTED_BOUNDARY_CONFIDENCE, CORRECTION_MAX_ROUNDS,
    MAX_PARTIAL_WILDCARDS, VITERBI_MAX_CANDIDATES,
    SMOOTHNESS_MAX_ITERATIONS,
    AUTO_ALIGN_DIFF_MIN_KMH, AUTO_ALIGN_DIFF_MAX_KMH,
    AUTO_ALIGN_NUDGE_FACTOR, AUTO_ALIGN_MIN_CHANGE_KMH,
    FORCE_MEDIAN_MAX_ITERATIONS, FORCE_MEDIAN_NUDGE_FACTOR,
    FORCE_MEDIAN_THRESHOLD_MULT, FORCE_MEDIAN_MIN_CHANGE_KMH,
    CANDIDATE_POSTFILTER_PHYSICS_MIN, CANDIDATE_POSTFILTER_LINEARITY_MIN,
    CANDIDATE_HUNDREDS_MAX_DIFF,
    ACCEL_SCORE_ISLAND_INTERIOR,
    REF_GUARD_PHYSICS_MIN, REF_GUARD_LINEARITY_MIN,
    DISTANT_INTERP_MIN_TIME, DISTANT_INTERP_ISLAND_THRESHOLD,
    FORCE_MEDIAN_WINDOW_TIME,
    REF_INTERP_MAX_KMH_DIFF, REF_MIN_DIFF,
    VITERBI_POST_TRUST_THRESHOLD, TRUST_WINDOW_FALLBACK_MAX_DV,
    TRUST_WINDOW_TIME, FILL_CONFIDENCE_THRESHOLD, FILL_CANDIDATE_MAX_DIFF,
    FINAL_CONF_BLEND_PHASE1, FINAL_CONF_BLEND_VITERBI,
)

if TYPE_CHECKING:
    from rapidocr import RapidOCR

logger = logging.getLogger("RaceVideoToLog.correction")


# ═══════════════════ 模式策略 ═══════════════════

@dataclass(frozen=True)
class ModeProfile:
    """模式策略：集中表达自动/手动模式的全部差异。

    auto   — 平滑优先：修正阈值宽（conf<70）、Viterbi 多轮、允许小步长
             微调、distant 参考、最终全帧中值平滑。
    manual — 精确优先（d=0）：只修 conf<40 的帧、单轮 Viterbi、
             拒绝 ±1 噪声微调、不用 distant 参考、只清理修正帧斜坡。
    """
    correct_threshold: int      # 进入纠正的置信度阈值
    viterbi_min_diff: float     # Viterbi 最小提交差值（噪声微调保护）
    max_rounds: int             # Viterbi 轮数上限
    distant_ref: bool           # 是否使用远距离插值参考
    force_sg_corrected: bool    # forceSG 只处理自动修正帧（True=手动）
    force_smooth: bool          # 是否执行最终中值平滑


_AUTO_PROFILE = ModeProfile(
    correct_threshold=AUTO_CORRECT_THRESHOLD,
    viterbi_min_diff=CORRECTION_MIN_DIFF,
    max_rounds=CORRECTION_MAX_ROUNDS,
    distant_ref=True,
    force_sg_corrected=False,
    force_smooth=True,
)

_MANUAL_PROFILE = ModeProfile(
    correct_threshold=MANUAL_CORRECT_THRESHOLD,
    viterbi_min_diff=MANUAL_CORRECTION_MIN_DIFF,
    max_rounds=1,
    distant_ref=False,
    force_sg_corrected=True,
    force_smooth=False,
)


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


def _interp_candidate(i: int, rows: list, pinned_set: set, times: list,
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


def _local_interp(i: int, rows: list, observations: list, times: list,
                  max_speed_kmh: float, fps: float = 1.0,
                  min_distance: int = 0,
                  exclude_flags: set[int] | None = None,
                  max_accel_mps2: float = 0.0) -> float | None:
    """Interpolation using nearest valid neighbors.

    Anchors are always physically validated: a candidate anchor must be
    consistent with its immediate neighbors (±2 frames within max_accel) —
    a single wrong OCR value (e.g. 2744=17 vs neighbors 170) must not
    become an interpolation anchor.  exclude_flags additionally skips
    anchors carrying those flag values (e.g. auto-corrected frames whose
    values are guesses, not observations)."""
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
        return [val] if val <= max_speed else []
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


def _multi_height_ocr(crop_bgr: "np.ndarray", ocr: "RapidOCR", max_speed_kmh: float,
                  cache: dict | None = None) -> set:
    cache = cache if cache is not None else {}
    if crop_bgr is not None and crop_bgr.size > 0:
        raw = crop_bgr.data.tobytes() if hasattr(crop_bgr, 'data') else crop_bgr.tobytes()
        cache_key = hash(raw[:256])
    else:
        cache_key = None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]
    candidates: set[int] = set()
    if crop_bgr is None or crop_bgr.size == 0:
        return candidates
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0: return candidates
    # Single-height re-OCR: multi-height (24,32,48) tested, no improvement
    # over using the main pipeline target_h=48 alone.  Removes ~0.5s latency.
    scale = 48 / h if h > 0 else 1.0
    from PIL import Image
    proc = np.array(
        Image.fromarray(crop_bgr).resize((max(1, int(w * scale)), 48), Image.LANCZOS))
    res = ocr(proc)
    sv, rt, _conf = extract_speed_value(res)
    if sv is not None and sv <= max_speed_kmh:
        candidates.add(int(sv))
    if cache_key is not None:
        cache[cache_key] = candidates
    return candidates


# ═══════════════════ Candidate generation ═══════════════════

def _generate_candidates(fi: int, rows: list, observations: list, raw_frames: list,
                         ocr: "RapidOCR", pinned_set: set, times: list,
                         max_speed_kmh: float, reocr_cache: dict,
                         split_results: dict[int, str] | None,
                         reocr_only: bool, fps: float,
                         confidence_score: float,
                         max_accel_mps2: float | None = None) -> list[float]:
    raw_val = rows[fi][2]
    protected: list[float] = []
    protected_set: set[float] = set()

    if 0 <= raw_val <= max_speed_kmh:
        protected.append(raw_val)
        protected_set.add(raw_val)

    obs = observations[min(fi, len(observations) - 1)]

    if fi < len(raw_frames):
        reocr_set = _multi_height_ocr(raw_frames[fi][1], ocr, max_speed_kmh, cache=reocr_cache)
        for cv in sorted(reocr_set):
            if 0 <= cv <= max_speed_kmh and cv not in protected_set:
                protected.append(cv); protected_set.add(cv)

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

    if not reocr_only:
        for cv in build_speed_candidates(obs.raw_text, max_speed_kmh):
            if cv not in protected_set and cv not in other_set:
                other.append(cv); other_set.add(cv)
        for cv in _auto_expand_digits(obs.raw_text, max_speed_kmh):
            if cv not in protected_set and cv not in other_set:
                other.append(cv); other_set.add(cv)
    needs_interp = (not reocr_only) or (raw_val < 0)
    if needs_interp:
        interp_val = _interp_candidate(fi, rows, pinned_set, times, max_speed_kmh, fps=fps)
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
                        candidates_by_frame: dict[int, list[float]] | None = None,
                        conservative: bool = False) -> None:
    """Fill frames that Viterbi could not recover.

    conservative=True (manual mode): only fill truly unreadable frames
    (raw<0 or raw>max_speed). Readable-but-wrong frames keep their raw
    value so outliers stay visible for manual correction — interpolation
    guesses trade small bias for a smooth-but-wrong ramp.
    """
    n = len(rows)
    sorted_errors = sorted(i for i in error_set if i not in pinned_set and not Flag.is_trusted(rows[i][3]))
    total = len(sorted_errors)
    for idx, i in enumerate(sorted_errors):
        raw_v = rows[i][2]
        if conservative and 0 <= raw_v <= max_speed_kmh:
            continue  # 可读帧保留原始值，离群留给人工
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
        sigs = confidence_scores[i].get('signals', {}) if i < len(confidence_scores) else {}
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
            pinned_set = {j for j in range(n) if Flag.is_trusted(rows[j][3])}
            interp = _interp_candidate(i, rows, pinned_set, times, max_speed_kmh, fps=fps)
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


# ═══════════════════ Force-SG smoothing (auto-mode final pass) ═══════════════════

def _force_median_smooth(rows: list, times: list, max_speed_kmh: float,
                     max_accel_mps2: float, fps: float = 1.0,
                     notes: dict[int, str] | None = None,
                     corrected_only: bool = False) -> int:
    """Median-filter smoothing pass.

    Auto mode: on ALL frames regardless of flags — goal is to minimize
    max_dv (frame-to-frame speed change), not to match truth values.
    Manual mode (corrected_only=True): only on auto-corrected frames
    (REOCR_AUTO/FILL), with the window restricted to RAW/trusted values —
    pulls Viterbi/fill ramps back to correct neighbors without touching
    correct raw frames.

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

    # 窗口值来源：手动（corrected_only）只留原始/信任锚点（清理斜坡时
    # 排除自动修正值，避免互相确认）；自动全帧平滑（排除 FILL 猜测值）。
    window_ok = ((lambda r: r[3] == Flag.RAW or Flag.is_trusted(r[3]))
                 if corrected_only else
                 (lambda r: r[3] != Flag.FILL_INTERP))

    for _pass in range(FORCE_MEDIAN_MAX_ITERATIONS):
        changed = 0
        for i in range(2, n - 2):
            v = rows[i][2]
            if v < 0:
                continue
            if corrected_only and not Flag.is_corrected(rows[i][3]):
                continue  # 手动模式：只处理 Viterbi/fill 修正过的帧
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
                   ocr: "RapidOCR", confidence_scores: list[dict],
                   times: list[float], max_speed_kmh: float, max_accel_mps2: float,
                   mode: str = "auto", pinned: set[int] | None = None,
                   reocr_cache: dict | None = None, reocr_only: bool = True,
                   split_results: dict[int, str] | None = None,
                   fps: float = 1.0, log_fn: "Callable | None" = None,
                   progress_fn: "Callable | None" = None,
                   notes: dict[int, str] | None = None,
                   ) -> tuple[list, list[dict]]:
    pinned = pinned or set()
    n = len(rows)
    for pi in pinned:
        if rows[pi][3] < Flag.HIGH_TRUST: rows[pi][3] = Flag.PINNED
    pinned_set = pinned
    cache: dict = reocr_cache if reocr_cache is not None else {}
    profile = _MANUAL_PROFILE if mode == "manual" else _AUTO_PROFILE
    correct_threshold = profile.correct_threshold

    conf_by_idx: dict[int, float] = {}
    for c in confidence_scores: conf_by_idx[c['index']] = c['score']

    if log_fn:
        log_fn(f"Correction (Phase 2, {mode} mode): threshold={correct_threshold}, "
               f"{n} rows, {len(pinned_set)} pinned")

    correction_frames: set[int] = set()
    for i in range(n):
        if i in pinned_set or Flag.is_trusted(rows[i][3]): continue
        if conf_by_idx.get(i, 50) < correct_threshold:
            correction_frames.add(i)

    candidates_by_frame: dict[int, list[float]] = {}
    total_correct = len(correction_frames)
    for idx, fi in enumerate(sorted(correction_frames)):
        cands = _generate_candidates(fi, rows, observations, raw_frames, ocr, pinned_set,
            times, max_speed_kmh, cache, split_results, reocr_only=reocr_only,
            fps=fps, confidence_score=conf_by_idx.get(fi, 50),
            max_accel_mps2=max_accel_mps2)
        if cands: candidates_by_frame[fi] = cands
        if progress_fn and total_correct > 0: progress_fn(idx + 1, total_correct)

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
    # For frames that are internally consistent (physics>=90 AND
    # linearity>=90), a hundreds-digit variant that differs by >=100
    # from the raw value is almost certainly wrong. Removing it
    # prevents Viterbi from choosing a path that looks "cheaper"
    # due to lower transition cost from nearby wrong frames.
    # Example: raw=168 (3-digit), physics=100, linearity=100 →
    #   remove candidate 68 (168-100) to prevent false correction.
    n_filtered = 0
    n_island_cands = 0  # Track frames with distant-interp candidates
    for fi in list(candidates_by_frame.keys()):
        sigs = confidence_scores[fi].get('signals', {}) if fi < len(confidence_scores) else {}
        p = sigs.get('physics', 50)
        l = sigs.get('linearity', 50)
        raw_v = rows[fi][2]
        if raw_v <= 0: continue
        if p >= CANDIDATE_POSTFILTER_PHYSICS_MIN and l >= CANDIDATE_POSTFILTER_LINEARITY_MIN:
            old_cands = candidates_by_frame[fi]
            new_cands = [c for c in old_cands if abs(c - raw_v) < CANDIDATE_HUNDREDS_MAX_DIFF]
            if len(new_cands) < len(old_cands):
                n_filtered += 1
                candidates_by_frame[fi] = new_cands if len(new_cands) >= 1 else [raw_v]
        # Track frames where distant interpolation was added
        a = sigs.get('accel', 100)
        if a <= ACCEL_SCORE_ISLAND_INTERIOR + 5 and p >= CANDIDATE_POSTFILTER_PHYSICS_MIN:
            n_island_cands += 1
    if log_fn and n_filtered > 0:
        log_fn(f"  Filtered hundreds variants from {n_filtered} consistent 3-digit frames")

    # Build reference values for ALL non-trusted frames (both modes).
    # Manual mode previously limited this to conf<40, which left conf 40-70
    # frames without reference protection — Viterbi's transition cost then
    # pulled them into wrong ramps (measured 2731-cluster / 1115-cluster).
    #
    # Guard 1: skip internally consistent frames (physics>=90 AND linearity>=90
    # AND accel>=70) — raw OCR almost certainly correct.
    # Guard 2 (REF_MIN_DIFF): if raw already agrees with interpolation (<3),
    # the frame is self-consistent — no reference (else Viterbi would be
    # pulled off a correct raw by a ±1-2 off interpolation).
    reference_values: dict[int, float] = {}
    for i in range(n):
        if i in pinned_set or Flag.is_trusted(rows[i][3]): continue
        sigs = confidence_scores[i].get('signals', {}) if i < len(confidence_scores) else {}
        p = sigs.get('physics', 50)
        l = sigs.get('linearity', 50)
        a = sigs.get('accel', 100)
        if p >= REF_GUARD_PHYSICS_MIN and l >= REF_GUARD_LINEARITY_MIN and a >= 70:
            # All signals healthy → trust OCR, skip interpolation
            continue
        raw_v = rows[i][2]
        ref = _local_interp(i, rows, observations, times, max_speed_kmh, fps=fps, max_accel_mps2=max_accel_mps2)
        # For suspected island interiors (accel <= 15), the nearest
        # neighbors may also be wrong. Try distant interpolation
        # that skips past the island to reach correct anchor frames.
        # Auto mode only: manual mode measured regressions (893/1708/1709
        # pulled to wrong distant refs even with anchor validity).
        if profile.distant_ref and a <= ACCEL_SCORE_ISLAND_INTERIOR + 5 and raw_v > 0:
            dt_frame = (times[1] - times[0]) if n >= 2 else 1/60
            min_frames = max(1, int(DISTANT_INTERP_MIN_TIME / dt_frame))
            distant = _local_interp(i, rows, observations, times, max_speed_kmh,
                                    fps=fps, min_distance=min_frames,
                                    max_accel_mps2=max_accel_mps2)
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
            ref = _interp_candidate(i, rows, pinned_set, times, max_speed_kmh, fps=fps)
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
    # Manual mode: single round only — round 2+ trust propagation lets
    # wrong ramps confirm each other (r1 2747→11, r2 marks 2748 HT=11,
    # then r2 ramps 2749-2758 toward it).
    max_rounds = 1 if n_island_cands > 0 else profile.max_rounds
    if log_fn and max_rounds == 1:
        log_fn(f"  Island/manual mode: max 1 Viterbi round")

    # ── 自洽帧锚定 raw（单候选）— 两种模式统一 ──
    # raw 与物理插值一致（差 < REF_MIN_DIFF）说明该帧 raw 高可信；
    # 若不锚定，Viterbi 转移代价会把它们拉进错误斜坡
    # （实测 2749 raw=171 被改 71，随后 2750-2760 链式斜坡）。
    # 锚定在 Viterbi 前一次性完成（候选不随 round 变化）。
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

    for round_num in range(max_rounds):
        viterbi_result = viterbi_correct(rows, candidates_by_frame, confidence_scores,
            times, max_speed_kmh, max_accel_mps2, trusted_indices=trusted_set,
            reference_values=reference_values)

        round_fixed = 0
        for fi, new_val in sorted(viterbi_result['corrected'].items()):
            if fi in pinned_set or Flag.is_trusted(rows[fi][3]): continue
            old_val = rows[fi][2]
            # 手动模式：±1 微调视为噪声（Viterbi 转移代价偏好平滑，会把
            # 正确的 raw 微调 1-2）。手动模式只提交 ≥2 km/h 的实质修正；
            # 自动模式保留小步长（平滑优先）。
            min_diff = profile.viterbi_min_diff
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

    n_smoothed = _smoothness_pass(rows, times, max_speed_kmh, max_accel_mps2, fps, notes)
    if log_fn and n_smoothed > 0: log_fn(f"  Smoothness: {n_smoothed} spikes smoothed")

    # ── SG-guided alignment (both modes; reference excludes FILL frames) ──
    n_aligned = _auto_align_pass(rows, observations, times, max_speed_kmh,
                                 max_accel_mps2, fps, confidence_scores, notes)
    if log_fn and n_aligned > 0:
        log_fn(f"  Auto-align: {n_aligned} frames nudged")

    # ── Median smoothing ──
    # Auto mode: aggressive, ALL frames (minimize max_dv).
    # Manual mode: conservative, only auto-corrected frames (clean up
    # Viterbi/fill ramps against RAW/trusted neighbors — protects d=0).
    if profile.force_smooth:
        n_forced = _force_median_smooth(rows, times, max_speed_kmh, max_accel_mps2, fps, notes,
                                        corrected_only=profile.force_sg_corrected)
        if log_fn and n_forced > 0:
            log_fn(f"  Force-SG: {n_forced} frame-nudges applied")

    for c in confidence_scores:
        i = c['index']
        c['is_corrected'] = Flag.is_corrected(rows[i][3])
        if i in viterbi_conf:
            c['score'] = round(c['score'] * FINAL_CONF_BLEND_PHASE1 + viterbi_conf[i] * FINAL_CONF_BLEND_VITERBI, 1)

    return rows, confidence_scores


# ═══════════════════ Backward-compat API ═══════════════════

def compute_confidence(rows: list, observations: list, max_speed: float,
                       max_accel: float, pinned: set[int] | None = None,
                       fps: float = 1.0) -> list[dict]:
    from error_detection import _signal_physics, _signal_ocr_conf
    n = len(rows)
    times = [r[0] / fps for r in rows]
    ocr_conf = _signal_ocr_conf(observations, n)
    physics = _signal_physics(rows, times, max_accel)
    confidences = []
    from config import COMPAT_CONF_OCR_WEIGHT, COMPAT_CONF_PHYSICS_WEIGHT
    for i in range(n):
        score = round(max(0.0, min(100.0, COMPAT_CONF_OCR_WEIGHT * ocr_conf[i] + COMPAT_CONF_PHYSICS_WEIGHT * physics[i])), 1)
        flags = [r[3] for r in rows]; cur = rows[i][2]
        if cur < 0 or cur > max_speed: reason = '速度超出范围'
        elif score >= 70: reason = '正常'
        elif score >= 30: reason = '存疑'
        else: reason = '错误'
        confidences.append({'index': i, 'score': score,
            'is_corrected': Flag.is_corrected(flags[i]), 'speed': cur, 'reason': reason})
    return confidences
