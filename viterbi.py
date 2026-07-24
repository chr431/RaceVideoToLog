"""Viterbi global optimal path selection — replaces LCS candidate scoring.

Finds the globally optimal sequence of speed values through per-frame candidate sets
using dynamic programming on a chain graph. Unlike LCS (which evaluates candidates
against potentially-wrong neighbor values), Viterbi jointly optimizes all frames
simultaneously.

Algorithm:
1. Split frame sequence into segments between trusted (PINNED/HIGH_TRUST) anchors
2. For each segment, run Viterbi DP with quadratic soft constraints
3. Trusted anchors are single-candidate hard constraints
4. Path deviation from raw OCR → error detection
5. Normalized path cost → confidence scores
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import (
    MPS_TO_KMH,
    VITERBI_OBS_WEIGHT, VITERBI_PROFILE_WEIGHT, VITERBI_ACCEL_WEIGHT,
    VITERBI_CONF_BONUS, VITERBI_TRUST_THRESHOLD,
    VITERBI_MAX_CANDIDATES, VITERBI_CONTEXT_WINDOW,
)
from ocr_engine import Flag


def _compute_median_profile(speeds: list[float], half_window: int = 7) -> list[float]:
    """Compute sliding median profile over speed values.

    Uses O(N*W) algorithm — acceptable for N≈6000, W≈15.
    The median is naturally robust to outliers, providing a clean
    global-trend reference for the Viterbi observation cost.
    """
    import numpy as np
    n = len(speeds)
    vals = np.array(speeds, dtype=float)
    profile = vals.copy()
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        profile[i] = float(np.median(vals[lo:hi]))
    return profile.tolist()


def viterbi_correct(
    rows: list,
    candidates_by_frame: dict[int, list[float]],
    trusted_set: set[int],
    times: list[float],
    max_speed_kmh: float,
    max_accel_mps2: float,
    median_profile: list[float] | None = None,
    obs_weight: float = VITERBI_OBS_WEIGHT,
    profile_weight: float = VITERBI_PROFILE_WEIGHT,
    accel_weight: float = VITERBI_ACCEL_WEIGHT,
    conf_bonus: float = VITERBI_CONF_BONUS,
    trust_threshold: float = VITERBI_TRUST_THRESHOLD,
) -> dict:
    """Find globally optimal speed values via Viterbi DP on candidate trellis.

    Segments the frame sequence at trusted anchor frames, then runs independent
    Viterbi DP on each segment. Trusted anchors are single-candidate hard constraints.

    Args:
        rows: [[frame_id, distance, speed, flag], ...]
        candidates_by_frame: {frame_idx: [candidate_values]} — only for non-trusted frames
        trusted_set: set of frame indices that are PINNED/HIGH_TRUST
        times: timestamps in seconds for each frame
        max_speed_kmh: maximum valid speed
        max_accel_mps2: maximum acceleration in m/s²
        median_profile: pre-computed sliding median profile (auto-computed if None)
        obs_weight: weight for deviation from raw OCR reading
        profile_weight: weight for deviation from median profile
        accel_weight: weight for acceleration constraint violation
        conf_bonus: OCR confidence bonus weight
        trust_threshold: confidence above this → HIGH_TRUST eligible

    Returns:
        {
            'corrected': {frame_idx: new_value} — frames to correct,
            'confidence': [0-1 score per frame],
            'error_set': {frame_idx} — frames where path ≠ original,
            'dp_cost': [per-frame normalized cost],
        }
    """
    n = len(rows)
    if n == 0:
        return {'corrected': {}, 'confidence': [], 'error_set': set(), 'dp_cost': []}

    # ── Median profile (global trend reference) ──
    if median_profile is None:
        speeds_raw = [r[2] for r in rows]
        median_profile = _compute_median_profile(speeds_raw)

    # ── Split into segments between trusted anchors ──
    segments = _split_segments(n, trusted_set)

    # ── Run Viterbi on each segment ──
    corrected: dict[int, float] = {}
    error_set: set[int] = set()
    dp_cost: list[float] = [-1.0] * n  # -1 = not processed (trusted or skipped)
    path_values: list[float | None] = [None] * n

    for seg_start, seg_end in segments:
        _viterbi_segment(
            seg_start, seg_end, rows, candidates_by_frame, trusted_set,
            times, max_speed_kmh, max_accel_mps2, median_profile,
            obs_weight, profile_weight, accel_weight, conf_bonus,
            corrected, error_set, dp_cost, path_values,
        )

    # ── Compute per-frame confidence from DP cost ──
    confidence = _compute_confidence_scores(n, dp_cost, path_values, rows, trusted_set, trust_threshold)

    return {
        'corrected': corrected,
        'confidence': confidence,
        'error_set': error_set,
        'dp_cost': dp_cost,
    }


def _split_segments(n: int, trusted_set: set[int]) -> list[tuple[int, int]]:
    """Split frame range [0, n) into segments between trusted anchors.

    Each segment starts and ends at a trusted frame (or boundary).
    Segments of length 1 (isolated trusted frames) are skipped.
    """
    # Collect trusted indices in sorted order, plus virtual anchors at boundaries
    anchors = sorted(i for i in range(n) if i in trusted_set)

    if not anchors:
        # No trusted frames: entire sequence is one segment with virtual anchors
        return [(0, n - 1)]

    segments: list[tuple[int, int]] = []

    # First segment: from frame 0 to first anchor
    if anchors[0] > 0:
        segments.append((0, anchors[0]))

    # Middle segments: between consecutive anchors
    for k in range(len(anchors) - 1):
        left, right = anchors[k], anchors[k + 1]
        if right - left > 1:  # skip adjacent anchors
            segments.append((left, right))

    # Last segment: from last anchor to end
    if anchors[-1] < n - 1:
        segments.append((anchors[-1], n - 1))

    return segments


def _viterbi_segment(
    seg_start: int, seg_end: int,
    rows: list,
    candidates_by_frame: dict[int, list[float]],
    trusted_set: set[int],
    times: list[float],
    max_speed_kmh: float,
    max_accel_mps2: float,
    median_profile: list[float],
    obs_weight: float,
    profile_weight: float,
    accel_weight: float,
    conf_bonus: float,
    corrected: dict[int, float],
    error_set: set[int],
    dp_cost: list[float],
    path_values: list[float | None],
) -> None:
    """Run Viterbi DP on a single segment [seg_start, seg_end].

    Modifies `corrected`, `error_set`, `dp_cost`, `path_values` in place.
    """
    n_seg = seg_end - seg_start + 1
    if n_seg < 2:
        return

    # ── Build candidate lists for each position in segment ──
    # seg_cands[k] = list of (value, tag, conf_bonus_amount) for position seg_start + k
    seg_cands: list[list[tuple[float, str, float]]] = []

    for k in range(n_seg):
        fi = seg_start + k
        if fi in trusted_set:
            # Single candidate: current value (hard constraint)
            v = rows[fi][2]
            if 0 <= v <= max_speed_kmh:
                seg_cands.append([(v, "trusted", 0.0)])
            else:
                seg_cands.append([(0.0, "trusted", 0.0)])
        else:
            cands = candidates_by_frame.get(fi, [])
            raw_v = rows[fi][2]
            # Always include raw OCR value as a candidate
            options: list[tuple[float, str, float]] = []
            seen = set()
            if 0 <= raw_v <= max_speed_kmh:
                options.append((raw_v, "current", 0.0))
                seen.add(raw_v)
            for cv in cands:
                if 0 <= cv <= max_speed_kmh and cv not in seen:
                    options.append((cv, "candidate", conf_bonus))
                    seen.add(cv)
            # Truncation should be handled by _generate_candidates already.
            # This is a safety net for edge cases.
            if len(options) > VITERBI_MAX_CANDIDATES:
                options = options[:VITERBI_MAX_CANDIDATES]
            if not options:
                options.append((0.0, "fallback", 0.0))
            seg_cands.append(options)

    # ── Forward DP ──
    # dp[k][idx] = (min_cost, prev_idx)
    dp: list[list[tuple[float, int]]] = []

    # Initialize first position
    dp0: list[tuple[float, int]] = []
    for idx, (v, tag, bonus) in enumerate(seg_cands[0]):
        if seg_start in trusted_set:
            cost = 0.0  # anchor: zero cost
        else:
            cost = _obs_cost(seg_start, v, rows, median_profile, obs_weight, profile_weight)
            cost -= bonus  # OCR confidence reward
        dp0.append((cost, -1))
    dp.append(dp0)

    # Fill DP table
    for k in range(1, n_seg):
        fi = seg_start + k
        dt = times[fi] - times[fi - 1] if fi > 0 and fi - 1 >= 0 else 1.0
        if dt <= 0:
            dt = 1.0 / 30.0  # fallback

        dpk: list[tuple[float, int]] = []
        for idx_w, (w, tag_w, bonus_w) in enumerate(seg_cands[k]):
            best_cost = float('inf')
            best_prev = -1
            for idx_v, (v, tag_v, bonus_v) in enumerate(seg_cands[k - 1]):
                trans = _trans_cost(v, w, dt, max_accel_mps2, accel_weight)
                total = dp[k - 1][idx_v][0] + trans
                if total < best_cost:
                    best_cost = total
                    best_prev = idx_v
            # Add observation cost (zero for trusted frames)
            if fi in trusted_set:
                obs = 0.0
            else:
                obs = _obs_cost(fi, w, rows, median_profile, obs_weight, profile_weight)
                obs -= bonus_w
            dpk.append((best_cost + obs, best_prev))
        dp.append(dpk)

    # ── Backtrack ──
    # Find best final candidate
    last_k = n_seg - 1
    best_final_idx = min(range(len(dp[last_k])), key=lambda idx: dp[last_k][idx][0])

    path: list[float] = [0.0] * n_seg
    cur_idx = best_final_idx
    for k in range(n_seg - 1, -1, -1):
        path[k] = seg_cands[k][cur_idx][0]
        cur_idx = dp[k][cur_idx][1]

    # ── Record results ──
    for k in range(n_seg):
        fi = seg_start + k
        if fi in trusted_set:
            dp_cost[fi] = 0.0
            path_values[fi] = path[k]
            continue

        optimal_val = path[k]
        raw_val = rows[fi][2]
        path_values[fi] = optimal_val

        # Normalize cost: divide by segment length and obs_weight for interpretability
        raw_cost = dp[k][min(range(len(dp[k])), key=lambda idx: dp[k][idx][0])][0]
        if isinstance(raw_cost, (int, float)) and raw_cost >= 0:
            dp_cost[fi] = raw_cost
        else:
            dp_cost[fi] = 0.0

        # Detect error: optimal path differs from original
        if abs(optimal_val - raw_val) > 0.5:
            corrected[fi] = optimal_val
            error_set.add(fi)


def _obs_cost(fi: int, v: float, rows: list, median_profile: list[float],
              obs_weight: float, profile_weight: float) -> float:
    """Observation cost: capped penalty for overriding OCR + profile deviation.

    The OCR-override penalty is SATURATED: deviating by 20 km/h or 200 km/h
    both represent "rejecting the OCR reading", so the penalty is capped.
    This prevents the observation cost from trapping Viterbi at wrong raw values
    when the correct value requires a large jump (e.g., 21→221 for a hundreds-
    digit error). The transition cost handles physical feasibility.
    """
    raw_v = rows[fi][2]
    # Relative deviation from OCR reading, capped at 100%
    if raw_v > 0:
        raw_ratio = abs(v - raw_v) / max(1.0, abs(raw_v))
    else:
        raw_ratio = abs(v - raw_v)
    obs = obs_weight * min(1.0, raw_ratio)  # Saturate at obs_weight

    # Deviation from median profile (global trend) — also capped
    if fi < len(median_profile):
        mp = median_profile[fi]
        if mp > 0:
            mp_ratio = abs(v - mp) / max(1.0, mp)
            obs += profile_weight * min(2.0, mp_ratio)  # Capped at 2× weight

    return obs


def _trans_cost(v: float, w: float, dt: float, max_accel_mps2: float,
                accel_weight: float) -> float:
    """Transition cost: exponential penalty for physically implausible acceleration.

    Design:
    - 0 to 0.7×max_dv: cost = 0 (comfortably feasible)
    - 0.7×max_dv to max_dv: slow growth (aggressive but possible)
    - Beyond max_dv: rapid exponential growth (physically impossible)
    - Ceiling very high: 100 km/h jumps completely dominate the decision

    The exponential shape matches the user's max_accel setting: users set
    max_accel higher than true physical limits as a safety margin, so the
    penalty should ramp up gently near max_accel and explode beyond it.
    """
    if dt <= 0:
        dt = 1.0 / 30.0
    max_dv = max_accel_mps2 * dt * MPS_TO_KMH
    dv = abs(w - v)

    # Quadratic penalty beyond max_dv:
    # Excess of 1 km/h → cost=1, excess of 100 km/h → cost=10000.
    # The quadratic shape makes moderate violations costly and severe
    # violations completely dominant — exactly what's needed.
    excess = dv - max_dv
    return accel_weight * excess * excess


def _compute_confidence_scores(
    n: int,
    dp_cost: list[float],
    path_values: list[float | None],
    rows: list,
    trusted_set: set[int],
    trust_threshold: float,
) -> list[dict]:
    """Convert per-frame DP costs to 0-100 confidence scores.

    Trusted frames always score 100.
    Other frames: confidence = 100 * exp(-normalized_cost)
    where normalized_cost is scaled so that typical costs map to interpretable scores.
    """
    # Find the max cost among non-trusted frames for normalization
    max_cost = 0.01
    for i in range(n):
        if i not in trusted_set and dp_cost[i] > max_cost:
            max_cost = dp_cost[i]

    confidence = []
    for i in range(n):
        if i in trusted_set:
            confidence.append({
                'index': i, 'score': 100.0,
                'is_corrected': Flag.is_corrected(rows[i][3]),
                'speed': rows[i][2], 'reason': '锚点帧(可信)',
            })
            continue

        cost = dp_cost[i]
        if cost < 0:
            # Not processed (shouldn't happen for non-trusted)
            confidence.append({
                'index': i, 'score': 50.0,
                'is_corrected': Flag.is_corrected(rows[i][3]),
                'speed': rows[i][2], 'reason': '未处理',
            })
            continue

        # Normalize and convert to confidence
        normalized = cost / max_cost  # 0 (best) to 1+ (worst)
        score = 100.0 * math.exp(-normalized)  # 100 → ~37 at cost=max_cost → ~0 at high cost

        cur = rows[i][2]
        if cur < 0 or cur > 400:  # max_speed not available, use reasonable default
            score = 0.0
            reason = '速度超出范围'
        elif score >= trust_threshold * 100:
            reason = '正常'
        elif score >= 30:
            reason = 'Viterbi存疑(临界的)'
        else:
            reason = 'Viterbi错误(代价高)'

        confidence.append({
            'index': i, 'score': round(score, 1),
            'is_corrected': Flag.is_corrected(rows[i][3]),
            'speed': cur, 'reason': reason,
        })

    return confidence
