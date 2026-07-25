"""Error Detection — Phase 1: multi-signal per-frame confidence scoring.

Computes continuous confidence scores [0, 100] for every frame using 7
independent signals. READ-ONLY: does not modify any values or flags.

Signals:
  1. OCR model confidence — from RapidOCR output (0→100, nearly useless)
  2. Physics reachability  — both-neighbor check (reliable ≥3-digit only)
  3. Local linearity       — deviation from reliable-neighbor interpolation
  4. Re-OCR agreement      — consensus among all OCR readings
  5. Text length           — digit count as reliability indicator
  6. Acceleration spikes   — opposing spike pairs = consistency island
  7. SG/median deviation   — auxiliary broad-trend check
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
	ERROR_DETECT_REOCR_AGREE_WEIGHT,
	ERROR_DETECT_TEXT_LEN_WEIGHT,
	ERROR_DETECT_ACCEL_SPIKE_WEIGHT,
	ERROR_DETECT_SG_DEVIATION_WEIGHT,
	ERROR_DETECT_CANDIDATE_THRESHOLD,
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


# ═══════════════════ Signal 1: OCR confidence ═══════════════════

def _signal_ocr_conf(observations: list, n: int) -> list[float]:
	scores = []
	for i in range(n):
		if i < len(observations) and observations[i] is not None:
			conf = getattr(observations[i], "confidence", None)
			scores.append(round(conf * 100, 1) if (conf is not None and conf > 0) else 50.0)
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
						dt = 0.02
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
                            max_look: int = 120) -> int | None:
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
		scores.append(max(0.0, min(100.0, 100.0 * math.exp(-deviation * 3.0))))
	return scores


# ═══════════════════ Signal 5: Text length ═══════════════════

def _signal_text_len(observations: list, n: int) -> list[float]:
	scores = []
	for i in range(n):
		if i < len(observations) and observations[i] is not None:
			rt = observations[i].raw_text or ""
			digits = sum(1 for ch in rt if ch.isdigit())
			if digits >= 3: scores.append(100.0)
			elif digits == 2: scores.append(50.0)
			elif digits == 1: scores.append(20.0)
			else: scores.append(10.0)
		else:
			scores.append(10.0)
	return scores


# ═══════════════════ Signal 6: Acceleration spike pairs ═══════════════════

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
	threshold = max_dv_frame * 2.0
	for i in range(n):
		if abs(diffs[i]) > threshold:
			violation[i] = True
			violation_sign[i] = 1 if diffs[i] > 0 else -1

	look = 15
	scores = []
	for i in range(n):
		if violation[i]:
			scores.append(20.0); continue

		left_start = max(0, i - look)
		right_end = min(n, i + look + 1)
		left_v = [violation_sign[j] for j in range(left_start, i) if violation[j]]
		right_v = [violation_sign[j] for j in range(i + 1, right_end) if violation[j]]
		has_left, has_right = len(left_v) > 0, len(right_v) > 0

		if has_left and has_right:
			if left_v[-1] != right_v[0]:
				scores.append(10.0)  # opposing spikes → island interior
			else:
				scores.append(60.0)
		elif has_left or has_right:
			scores.append(50.0)  # near one spike
		else:
			scores.append(100.0)  # normal
	return scores


# ═══════════════════ Signal 7: SG/median deviation ═══════════════════

def _signal_sg_deviation(rows: list, observations: list, times: list[float],
                         fps: float) -> list[float]:
	"""Wide median profile deviation using only 3+ digit frames."""
	n = len(rows)
	if n < 5:
		return [50.0] * n

	speeds = np.array([r[2] if r[2] >= 0 else 0.0 for r in rows], dtype=float)
	reliable_mask = np.zeros(n, dtype=bool)
	for i in range(n):
		if i < len(observations) and observations[i] is not None:
			rt = observations[i].raw_text or ""
			if len(rt) >= 3:
				reliable_mask[i] = True

	wide_half = max(31, int(fps * 2.5) if fps > 0 else 150)
	wide_median = np.zeros(n, dtype=float)
	for i in range(n):
		lo = max(0, i - wide_half)
		hi = min(n, i + wide_half + 1)
		wv = speeds[lo:hi]
		wr = reliable_mask[lo:hi]
		if np.any(wr):
			wide_median[i] = float(np.median(wv[wr]))
		elif np.any(wv > 0):
			wide_median[i] = float(np.median(wv[wv > 0]))

	rel_threshold, abs_threshold = 0.03, 5.0
	scores = []
	for i in range(n):
		v, mp = speeds[i], wide_median[i]
		if v < 0 or mp <= 0:
			scores.append(50.0); continue
		deviation = abs(v - mp)
		is_deviating = deviation > max(abs_threshold, mp * rel_threshold)
		if not is_deviating:
			scores.append(100.0); continue

		cluster_size = 1
		for j in range(i - 1, -1, -1):
			vj, mpj = speeds[j], wide_median[j]
			if vj < 0 or mpj <= 0: break
			if abs(vj - mpj) > max(abs_threshold, mpj * rel_threshold):
				cluster_size += 1
			else: break
		for j in range(i + 1, n):
			vj, mpj = speeds[j], wide_median[j]
			if vj < 0 or mpj <= 0: break
			if abs(vj - mpj) > max(abs_threshold, mpj * rel_threshold):
				cluster_size += 1
			else: break

		if cluster_size <= 1: score = 70.0
		elif cluster_size <= 3: score = 40.0
		elif cluster_size <= 5: score = 30.0
		elif cluster_size <= 10: score = 15.0
		else: score = 5.0
		scores.append(score)
	return scores


# ═══════════════════ Signal 4: Re-OCR agreement ═══════════════════

def _signal_reocr_agree(rows: list, reocr_values_by_frame: dict[int, set[float]],
                        split_results: dict[int, str] | None, n: int) -> list[float]:
	scores = []
	for i in range(n):
		if i not in reocr_values_by_frame or not reocr_values_by_frame[i]:
			scores.append(50.0); continue
		raw_v = rows[i][2]
		all_readings = set(reocr_values_by_frame[i])
		if 0 <= raw_v <= 400:
			all_readings.add(raw_v)
		if split_results and i in split_results:
			try: all_readings.add(int(split_results[i]))
			except (ValueError, TypeError): pass
		if len(all_readings) <= 1:
			scores.append(100.0); continue
		sorted_vals = sorted(all_readings)
		clusters = 1
		for j in range(1, len(sorted_vals)):
			if sorted_vals[j] - sorted_vals[j - 1] > 2:
				clusters += 1
		if clusters == 1: scores.append(100.0)
		elif clusters == 2: scores.append(60.0)
		else: scores.append(20.0)
	return scores


# ═══════════════════ Main entry point ═══════════════════

def detect_errors(rows: list, observations: list, times: list[float],
                  max_accel_mps2: float, max_speed_kmh: float,
                  reocr_values_by_frame: dict[int, set[float]] | None = None,
                  split_results: dict[int, str] | None = None,
                  fps: float = 60.0) -> ErrorReport:
	n = len(rows)
	if n == 0:
		return ErrorReport()

	ocr_conf_scores = _signal_ocr_conf(observations, n)
	physics_scores = _signal_physics(rows, observations, times, max_accel_mps2)
	linearity_scores = _signal_linearity(rows, observations, times, max_accel_mps2)
	text_len_scores = _signal_text_len(observations, n)
	accel_scores = _signal_accel_spikes(rows, times, fps, max_accel_mps2)
	sg_scores = _signal_sg_deviation(rows, observations, times, fps)

	if reocr_values_by_frame is not None:
		reocr_scores = _signal_reocr_agree(rows, reocr_values_by_frame, split_results, n)
	else:
		reocr_scores = [50.0] * n

	total_w = (ERROR_DETECT_OCR_CONF_WEIGHT + ERROR_DETECT_PHYSICS_WEIGHT +
	           ERROR_DETECT_LINEARITY_WEIGHT + ERROR_DETECT_REOCR_AGREE_WEIGHT +
	           ERROR_DETECT_TEXT_LEN_WEIGHT + ERROR_DETECT_ACCEL_SPIKE_WEIGHT +
	           ERROR_DETECT_SG_DEVIATION_WEIGHT)
	w_ocr = ERROR_DETECT_OCR_CONF_WEIGHT / total_w
	w_phy = ERROR_DETECT_PHYSICS_WEIGHT / total_w
	w_lin = ERROR_DETECT_LINEARITY_WEIGHT / total_w
	w_reo = ERROR_DETECT_REOCR_AGREE_WEIGHT / total_w
	w_txt = ERROR_DETECT_TEXT_LEN_WEIGHT / total_w
	w_acc = ERROR_DETECT_ACCEL_SPIKE_WEIGHT / total_w
	w_sg = ERROR_DETECT_SG_DEVIATION_WEIGHT / total_w

	confidence = []
	n_low, n_medium, n_high = 0, 0, 0
	candidate_frames: dict[int, list[float]] = {}

	for i in range(n):
		score = (w_ocr * ocr_conf_scores[i] + w_phy * physics_scores[i] +
		         w_lin * linearity_scores[i] + w_reo * reocr_scores[i] +
		         w_txt * text_len_scores[i] + w_acc * accel_scores[i] +
		         w_sg * sg_scores[i])
		score = round(max(0.0, min(100.0, score)), 1)

		if score < 30: tier = "low"; n_low += 1
		elif score < 70: tier = "medium"; n_medium += 1
		else: tier = "high"; n_high += 1

		signals_sorted = sorted(
			[("ocr_conf", ocr_conf_scores[i]), ("physics", physics_scores[i]),
			 ("linearity", linearity_scores[i]), ("reocr_agree", reocr_scores[i]),
			 ("text_len", text_len_scores[i]), ("accel", accel_scores[i]),
			 ("sg_dev", sg_scores[i])], key=lambda x: x[1])
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
				"reocr_agree": round(reocr_scores[i], 1),
				"text_len": round(text_len_scores[i], 1),
				"accel": round(accel_scores[i], 1),
				"sg_dev": round(sg_scores[i], 1),
			},
			"is_corrected": False,
		})

		if score < ERROR_DETECT_CANDIDATE_THRESHOLD:
			raw_v = rows[i][2]
			cands = []
			if 0 <= raw_v <= max_speed_kmh:
				cands.append(raw_v)
			if reocr_values_by_frame and i in reocr_values_by_frame:
				for rv in sorted(reocr_values_by_frame[i]):
					if 0 <= rv <= max_speed_kmh and rv not in cands:
						cands.append(rv)
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
