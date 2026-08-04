"""Error Detection — Phase 1: multi-signal per-frame confidence scoring.

Computes continuous confidence scores [0, 100] for every frame using 4
independent signals. READ-ONLY: does not modify any values or flags.

Signals:
  1. OCR model confidence — from OcrEngine output
  2. Physics reachability  — both-neighbor check
  3. Local linearity       — median-of-pairs robust interpolation
  4. Acceleration spikes   — opposing spike pairs = consistency island
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


from config import (
    MPS_TO_KMH,
    ERROR_DETECT_OCR_CONF_WEIGHT,
    ERROR_DETECT_PHYSICS_WEIGHT,
    ERROR_DETECT_LINEARITY_WEIGHT,
    ERROR_DETECT_ACCEL_SPIKE_WEIGHT,
    ERROR_DETECT_FLOOR_CAP,
    PHYSICS_TIME_WINDOW, PHYSICS_DECAY_FACTOR,
    LINEARITY_DECAY_FACTOR, LINEARITY_TIME_WINDOW, LINEARITY_MAX_NEIGHBORS,
    ACCEL_SPIKE_VIOLATION_MULT, ACCEL_SPIKE_SEARCH_WINDOW,
    CONF_TIER_LOW_MAX, CONF_TIER_MEDIUM_MAX,
    ACCEL_SCORE_NORMAL, ACCEL_SCORE_NEAR_ONE, ACCEL_SCORE_SAME_DIR,
    ACCEL_SCORE_VIOLATION, ACCEL_SCORE_ISLAND_INTERIOR,
)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

logger = logging.getLogger("RaceVideoToLog.error_detection")


@dataclass
class ErrorReport:
    confidence: list[dict] = field(default_factory=list)


# ═══════════════════ Signal 1: OCR model confidence ═══════════════════

def _signal_ocr_conf(observations: list, n: int) -> list[float]:
    scores = []
    for i in range(n):
        if i < len(observations) and observations[i] is not None:
            conf = observations[i].confidence
            scores.append(round(conf * 100, 1) if conf > 0 else 50.0)
        else:
            scores.append(50.0)
    return scores


# ═══════════════════ Helper: nearest valid frame within time window ═══════════════════

def _find_nearest_valid(rows: list, times: list[float], i: int, direction: int,
                        time_window: float) -> int | None:
    """Return the nearest frame in *direction* with speed >= 0 within time_window seconds."""
    step = 1 if direction > 0 else -1
    j = i + step
    t_i = times[i]
    while 0 <= j < len(rows):
        if abs(times[j] - t_i) > time_window:
            break
        if rows[j][2] >= 0:
            return j
        j += step
    return None


# ═══════════════════ Signal 2: Physics reachability ═══════════════════

def _signal_physics(rows: list, times: list[float], max_accel_mps2: float,
                    time_window: float = PHYSICS_TIME_WINDOW) -> list[float]:
    """Physics reachability — continuous scoring, no text-length dependency.

    For each frame, finds the nearest valid-speed neighbour on each side
    within *time_window* seconds.  Score is based on how badly (if at all)
    the speed change exceeds the physics limit.

    v < 0      →   0.0  (invalid measurement)
    no neighbour →  50.0  (neutral — insufficient context)
    reachable  → 100.0  (perfect)
    excess     → 100 * exp(-excess_ratio * decay)  (continuous drop)
    """
    n = len(rows)
    scores = []
    for i in range(n):
        v = rows[i][2]
        if v < 0:
            scores.append(0.0)
            continue

        side_scores: list[float] = []
        for direction in (-1, 1):
            ni = _find_nearest_valid(rows, times, i, direction, time_window)
            if ni is None:
                continue
            nv = rows[ni][2]
            dt = abs(times[i] - times[ni])
            if dt <= 0:
                continue
            max_dv = max_accel_mps2 * dt * MPS_TO_KMH
            dv = abs(v - nv)
            if dv <= max_dv:
                side_scores.append(100.0)
            else:
                excess = (dv - max_dv) / max_dv
                side_scores.append(100.0 * math.exp(-excess * PHYSICS_DECAY_FACTOR))

        if not side_scores:
            scores.append(50.0)
        else:
            scores.append(round(min(side_scores), 1))

    return scores


# ═══════════════════ Signal 3: Local linearity (median-of-pairs) ═══════════════════

def _signal_linearity(rows: list, times: list[float],
                      time_window: float = LINEARITY_TIME_WINDOW,
                      max_neighbors: int = LINEARITY_MAX_NEIGHBORS) -> list[float]:
    """Robust linearity check using median-of-pairs interpolation.

    For each frame, finds valid-speed neighbours on each side within
    *time_window* seconds (capped at *max_neighbors* per side), linearly
    interpolates from all left×right pairs, takes the median expected
    value, and scores based on relative deviation.

    No reliability gate — the median naturally rejects outlier anchors.
    Works equally well at all speed ranges (no text-length dependency).
    """
    n = len(rows)
    scores = []
    for i in range(n):
        v = rows[i][2]
        if v < 0:
            scores.append(0.0)
            continue

        t_i = times[i]

        # Collect valid frames on each side within time_window
        left_frames: list[int] = []
        j = i - 1
        while j >= 0 and len(left_frames) < max_neighbors:
            if t_i - times[j] > time_window:
                break
            if rows[j][2] >= 0:
                left_frames.append(j)
            j -= 1

        right_frames: list[int] = []
        j = i + 1
        while j < n and len(right_frames) < max_neighbors:
            if times[j] - t_i > time_window:
                break
            if rows[j][2] >= 0:
                right_frames.append(j)
            j += 1

        if not left_frames or not right_frames:
            scores.append(50.0)
            continue

        # Expected values from all left-right pairs, median for robustness
        # （配对部分向量化：集合语义与原双层循环相同 —— span 过滤 +
        # expected>0 过滤 → 排序后中位数一致）
        lv = np.fromiter((rows[li][2] for li in left_frames),
                         dtype=np.float64, count=len(left_frames))
        lt = np.fromiter((times[li] for li in left_frames),
                         dtype=np.float64, count=len(left_frames))
        rv = np.fromiter((rows[ri][2] for ri in right_frames),
                         dtype=np.float64, count=len(right_frames))
        rt = np.fromiter((times[ri] for ri in right_frames),
                         dtype=np.float64, count=len(right_frames))
        span = rt[None, :] - lt[:, None]
        mask = span >= 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = (t_i - lt[:, None]) / span  # span==0 → nan，被 mask 排除
        expected = lv[:, None] + (rv[None, :] - lv[:, None]) * frac
        exp_flat = expected[mask]
        exp_flat = exp_flat[exp_flat > 0]

        if exp_flat.size == 0:
            scores.append(50.0)
            continue

        exp_flat.sort()
        expected = float(exp_flat[exp_flat.size // 2])  # median

        deviation = abs(v - expected) / expected
        score = 100.0 * math.exp(-deviation * LINEARITY_DECAY_FACTOR)
        scores.append(round(max(0.0, min(100.0, score)), 1))

    return scores


# ═══════════════════ Signal 4: Acceleration spike pairs ═══════════════════

def _signal_accel_spikes(rows: list, times: list[float], fps: float,
                         max_accel_mps2: float) -> list[float]:
    """Detect consistency islands via opposing acceleration spike pairs.

    A consistency island produces two opposing spikes:
    one at entry (correct→wrong), one at exit (wrong→correct).
    Frames between opposing spikes are inside the island.
    """
    n = len(rows)
    if n < 3:
        return [50.0] * n

    dt = times[1] - times[0] if n >= 2 else 1.0 / max(fps, 1.0)
    max_dv_frame = max_accel_mps2 * dt * MPS_TO_KMH

    diffs = [0.0] * n
    for i in range(1, n):
        vi, vp = rows[i][2], rows[i - 1][2]
        if vi > 0 and vp > 0:
            diffs[i] = vi - vp

    violation = [False] * n
    violation_sign = [0] * n
    threshold = max_dv_frame * ACCEL_SPIKE_VIOLATION_MULT
    for i in range(n):
        if abs(diffs[i]) > threshold:
            violation[i] = True
            violation_sign[i] = 1 if diffs[i] > 0 else -1

    look = ACCEL_SPIKE_SEARCH_WINDOW
    scores = []
    for i in range(n):
        if violation[i]:
            scores.append(ACCEL_SCORE_VIOLATION); continue

        left_start = max(0, i - look)
        right_end = min(n, i + look + 1)
        left_v = [violation_sign[j] for j in range(left_start, i) if violation[j]]
        right_v = [violation_sign[j] for j in range(i + 1, right_end) if violation[j]]
        has_left, has_right = len(left_v) > 0, len(right_v) > 0

        if has_left and has_right:
            if left_v[-1] != right_v[0]:
                scores.append(ACCEL_SCORE_ISLAND_INTERIOR)  # opposing spikes → island interior
            else:
                scores.append(ACCEL_SCORE_SAME_DIR)
        elif has_left or has_right:
            scores.append(ACCEL_SCORE_NEAR_ONE)  # near one spike
        else:
            scores.append(ACCEL_SCORE_NORMAL)  # normal
    return scores


# ═══════════════════ Main entry point ═══════════════════

def detect_errors(rows: list, observations: list, times: list[float],
                  max_accel_mps2: float, max_speed_kmh: float,
                  fps: float = 60.0) -> ErrorReport:
    n = len(rows)
    if n == 0:
        return ErrorReport()

    ocr_conf_scores = _signal_ocr_conf(observations, n)
    physics_scores = _signal_physics(rows, times, max_accel_mps2)
    linearity_scores = _signal_linearity(rows, times)
    accel_scores = _signal_accel_spikes(rows, times, fps, max_accel_mps2)
    total_w = (ERROR_DETECT_OCR_CONF_WEIGHT + ERROR_DETECT_PHYSICS_WEIGHT +
               ERROR_DETECT_LINEARITY_WEIGHT + ERROR_DETECT_ACCEL_SPIKE_WEIGHT)
    w_ocr = ERROR_DETECT_OCR_CONF_WEIGHT / total_w
    w_phy = ERROR_DETECT_PHYSICS_WEIGHT / total_w
    w_lin = ERROR_DETECT_LINEARITY_WEIGHT / total_w
    w_acc = ERROR_DETECT_ACCEL_SPIKE_WEIGHT / total_w

    confidence = []

    for i in range(n):
        score = (w_ocr * ocr_conf_scores[i] + w_phy * physics_scores[i] +
                 w_lin * linearity_scores[i] +
                 w_acc * accel_scores[i])
        score = round(max(0.0, min(100.0, score)), 1)

        # Worst-signal floor: any weak signal caps the combined score
        min_sig = min(physics_scores[i], linearity_scores[i], accel_scores[i])
        for threshold, cap in sorted(ERROR_DETECT_FLOOR_CAP.items()):
            if min_sig < threshold:
                score = min(score, cap)
                break

        if score < CONF_TIER_LOW_MAX: tier = "low"
        elif score < CONF_TIER_MEDIUM_MAX: tier = "medium"
        else: tier = "high"

        signals_sorted = sorted(
            [("ocr_conf", ocr_conf_scores[i]), ("physics", physics_scores[i]),
             ("linearity", linearity_scores[i]),
             ("accel", accel_scores[i])], key=lambda x: x[1])
        lowest_signal, lowest_val = signals_sorted[0]
        if score >= 70: reason = "正常"
        elif score < 30: reason = f"错误({lowest_signal}={lowest_val:.0f})"
        else: reason = f"存疑({lowest_signal}={lowest_val:.0f})"

        cur_v = rows[i][2]
        confidence.append({
            "index": i, "score": score, "tier": tier,
            "speed": cur_v, "reason": reason,
            "signals": {
                "ocr_conf": round(ocr_conf_scores[i], 1),
                "physics": round(physics_scores[i], 1),
                "linearity": round(linearity_scores[i], 1),
                "accel": round(accel_scores[i], 1),
            },
            "is_corrected": False,
        })

    return ErrorReport(confidence=confidence)
