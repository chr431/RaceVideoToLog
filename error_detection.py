"""Error Detection — Phase 1: multi-signal per-frame confidence scoring.

Computes continuous confidence scores [0, 100] for every frame using 4
independent signals. READ-ONLY: does not modify any values or flags.

Signals:
  1. OCR model confidence — from RapidOCR output
  2. Physics reachability  — both-neighbor check (reliable >=3-digit only)
  3. Local linearity       — deviation from reliable-neighbor interpolation
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
    ERROR_DETECT_CANDIDATE_THRESHOLD,
    PHYSICS_FALLBACK_DT, LINEARITY_NEIGHBOR_MAX_LOOK,
    LINEARITY_DECAY_FACTOR,
    ACCEL_SPIKE_VIOLATION_MULT, ACCEL_SPIKE_SEARCH_WINDOW,
    REOCR_CLUSTER_GAP_KMH,
    CONF_TIER_LOW_MAX, CONF_TIER_MEDIUM_MAX,
    ACCEL_SCORE_NORMAL, ACCEL_SCORE_NEAR_ONE, ACCEL_SCORE_SAME_DIR,
    ACCEL_SCORE_VIOLATION, ACCEL_SCORE_ISLAND_INTERIOR,
    REOCR_AGREE_1CLUSTER, REOCR_AGREE_2CLUSTER, REOCR_AGREE_3PLUS,
)

if TYPE_CHECKING:
    from rapidocr import RapidOCR

logger = logging.getLogger("RaceVideoToLog.error_detection")


@dataclass
class ErrorReport:
    confidence: list[dict] = field(default_factory=list)
    candidates: dict[int, list[float]] = field(default_factory=dict)
    n_total: int = 0
    n_low_conf: int = 0
    n_medium_conf: int = 0
    n_high_conf: int = 0
    n_candidate_frames: int = 0


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


# ═══════════════════ Signal 2: Physics reachability ═══════════════════

def _signal_physics(rows: list, observations: list, times: list[float],
                    max_accel_mps2: float) -> list[float]:
    n = len(rows)
    scores = []
    for i in range(n):
        v = rows[i][2]
        if v < 0:
            scores.append(0.0)
            continue
        reachable = 0
        total = 0
        for ni in (i - 1, i + 1):
            if 0 <= ni < n:
                rt = ""
                if ni < len(observations) and observations[ni] is not None:
                    rt = (observations[ni].raw_text or "")
                if len(rt) < 3:
                    continue
                nv = rows[ni][2]
                if nv > 0:
                    total += 1
                    dt = abs(times[i] - times[ni])
                    if dt <= 0:
                        dt = PHYSICS_FALLBACK_DT
                    max_dv = max_accel_mps2 * dt * MPS_TO_KMH
                    if abs(v - nv) <= max_dv:
                        reachable += 1
        if total == 0:
            scores.append(50.0)
        elif reachable == total:
            scores.append(100.0)
        elif reachable == 0:
            scores.append(5.0)
        else:
            scores.append(60.0)
    return scores


# ═══════════════════ Helper: find reliable neighbor ═══════════════════

def _find_reliable_neighbor(i: int, direction: int, rows: list, observations: list,
                            times: list[float], max_accel_mps2: float,
                            max_look: int = LINEARITY_NEIGHBOR_MAX_LOOK) -> int | None:
    step = 1 if direction > 0 else -1
    j = i + step
    v = rows[i][2]
    while 0 <= j < len(rows) and abs(j - i) <= max_look:
        nv = rows[j][2]
        if nv < 0:
            j += step; continue
        rt = ""
        if j < len(observations) and observations[j] is not None:
            rt = observations[j].raw_text or ""
        if len(rt) < 3:
            j += step; continue
        if v > 0 and nv > 0:
            dt = abs(times[i] - times[j])
            if dt <= 0: dt = 0.02
            max_dv = max_accel_mps2 * dt * MPS_TO_KMH
            if abs(v - nv) <= max_dv:
                return j
        j += step
    return None


# ═══════════════════ Signal 3: Local linearity ═══════════════════

def _signal_linearity(rows: list, observations: list, times: list[float],
                      max_accel_mps2: float) -> list[float]:
    n = len(rows)
    scores = []
    for i in range(n):
        v = rows[i][2]
        if v < 0:
            scores.append(0.0); continue
        la = _find_reliable_neighbor(i, -1, rows, observations, times, max_accel_mps2)
        ra = _find_reliable_neighbor(i, +1, rows, observations, times, max_accel_mps2)
        if la is None or ra is None:
            scores.append(50.0); continue
        lv, rv = rows[la][2], rows[ra][2]
        lt, rt = times[la], times[ra]
        span = max(rt - lt, 1e-3)
        frac = (times[i] - lt) / span
        expected = lv + (rv - lv) * frac
        if expected <= 0:
            scores.append(50.0); continue
        deviation = abs(v - expected) / expected
        scores.append(max(0.0, min(100.0, 100.0 * math.exp(-deviation * LINEARITY_DECAY_FACTOR))))
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
                  split_results: dict[int, str] | None = None,
                  fps: float = 60.0) -> ErrorReport:
    n = len(rows)
    if n == 0:
        return ErrorReport()

    ocr_conf_scores = _signal_ocr_conf(observations, n)
    physics_scores = _signal_physics(rows, observations, times, max_accel_mps2)
    linearity_scores = _signal_linearity(rows, observations, times, max_accel_mps2)
    accel_scores = _signal_accel_spikes(rows, times, fps, max_accel_mps2)
    total_w = (ERROR_DETECT_OCR_CONF_WEIGHT + ERROR_DETECT_PHYSICS_WEIGHT +
               ERROR_DETECT_LINEARITY_WEIGHT + ERROR_DETECT_ACCEL_SPIKE_WEIGHT)
    w_ocr = ERROR_DETECT_OCR_CONF_WEIGHT / total_w
    w_phy = ERROR_DETECT_PHYSICS_WEIGHT / total_w
    w_lin = ERROR_DETECT_LINEARITY_WEIGHT / total_w
    w_acc = ERROR_DETECT_ACCEL_SPIKE_WEIGHT / total_w

    confidence = []
    n_low, n_medium, n_high = 0, 0, 0
    candidate_frames: dict[int, list[float]] = {}

    for i in range(n):
        score = (w_ocr * ocr_conf_scores[i] + w_phy * physics_scores[i] +
                 w_lin * linearity_scores[i] +
                 w_acc * accel_scores[i])
        score = round(max(0.0, min(100.0, score)), 1)

        if score < CONF_TIER_LOW_MAX: tier = "low"; n_low += 1
        elif score < CONF_TIER_MEDIUM_MAX: tier = "medium"; n_medium += 1
        else: tier = "high"; n_high += 1

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

        if score < ERROR_DETECT_CANDIDATE_THRESHOLD:
            raw_v = rows[i][2]
            cands = []
            if 0 <= raw_v <= max_speed_kmh:
                cands.append(raw_v)
            if split_results and i in split_results:
                try:
                    sv = int(split_results[i])
                    if 0 <= sv <= max_speed_kmh and sv not in cands:
                        cands.append(sv)
                except (ValueError, TypeError): pass
            candidate_frames[i] = cands

    return ErrorReport(confidence=confidence, candidates=candidate_frames,
                       n_total=n, n_low_conf=n_low, n_medium_conf=n_medium,
                       n_high_conf=n_high, n_candidate_frames=len(candidate_frames))
