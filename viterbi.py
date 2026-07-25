"""Viterbi global optimal path selection — DP on per-frame candidate trellis.

Finds the globally optimal sequence of speed values through per-frame
candidate sets using dynamic programming with soft constraints.

Key design:
- Observation cost: penalty for deviating from a reference value.
  For island-interior frames, the reference is interpolation (not raw OCR),
  so Viterbi prefers physics-based candidates over wrong raw values.
- Transition cost: quadratic penalty beyond max_accel × dt.
- Soft anchors: high-confidence frames reduce to single-candidate.
- Trusted indices: PINNED/HIGH_TRUST frames as hard segment boundaries.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import numpy as np

from config import (
	MPS_TO_KMH,
	VITERBI_OBS_WEIGHT, VITERBI_ACCEL_WEIGHT,
	VITERBI_MAX_CANDIDATES, VITERBI_SOFT_ANCHOR_CONFIDENCE,
)
from ocr_engine import Flag


def viterbi_correct(
	rows: list,
	candidates_by_frame: dict[int, list[float]],
	confidence_scores: list[dict],
	times: list[float],
	max_speed_kmh: float,
	max_accel_mps2: float,
	obs_weight: float = VITERBI_OBS_WEIGHT,
	accel_weight: float = VITERBI_ACCEL_WEIGHT,
	soft_anchor_threshold: int = VITERBI_SOFT_ANCHOR_CONFIDENCE,
	trusted_indices: set[int] | None = None,
	reference_values: dict[int, float] | None = None,
) -> dict:
	n = len(rows)
	if n == 0:
		return {'corrected': {}, 'confidence': [], 'error_set': set(), 'dp_cost': []}

	conf_by_idx: dict[int, float] = {}
	for c in confidence_scores:
		conf_by_idx[c['index']] = c['score']

	segments = _split_segments(n, conf_by_idx, soft_anchor_threshold, trusted_indices)

	corrected: dict[int, float] = {}
	error_set: set[int] = set()
	dp_cost: list[float] = [-1.0] * n
	path_values: list[float | None] = [None] * n

	for seg_start, seg_end in segments:
		_viterbi_segment(seg_start, seg_end, rows, candidates_by_frame, conf_by_idx,
			times, max_speed_kmh, max_accel_mps2, obs_weight, accel_weight,
			soft_anchor_threshold, corrected, error_set, dp_cost, path_values,
			reference_values=reference_values)

	confidence = _compute_confidence_scores(n, dp_cost, path_values, rows, conf_by_idx)

	return {'corrected': corrected, 'confidence': confidence,
	        'error_set': error_set, 'dp_cost': dp_cost}


def _split_segments(n: int, conf_by_idx: dict[int, float],
                    soft_anchor_threshold: int,
                    trusted_indices: set[int] | None = None) -> list[tuple[int, int]]:
	boundary_set: set[int] = set()
	for i in range(n):
		if conf_by_idx.get(i, 50) >= soft_anchor_threshold:
			boundary_set.add(i)
	if trusted_indices:
		boundary_set.update(trusted_indices)
	boundaries = sorted(boundary_set)

	if not boundaries:
		return [(0, n - 1)]

	segments: list[tuple[int, int]] = []
	if boundaries[0] > 0:
		segments.append((0, boundaries[0]))
	for k in range(len(boundaries) - 1):
		left, right = boundaries[k], boundaries[k + 1]
		if right - left > 1:
			segments.append((left, right))
	if boundaries[-1] < n - 1:
		segments.append((boundaries[-1], n - 1))
	return segments


def _viterbi_segment(
	seg_start: int, seg_end: int, rows: list,
	candidates_by_frame: dict[int, list[float]], conf_by_idx: dict[int, float],
	times: list[float], max_speed_kmh: float, max_accel_mps2: float,
	obs_weight: float, accel_weight: float, soft_anchor_threshold: int,
	corrected: dict[int, float], error_set: set[int],
	dp_cost: list[float], path_values: list[float | None],
	reference_values: dict[int, float] | None = None,
) -> None:
	n_seg = seg_end - seg_start + 1
	if n_seg < 2:
		return

	seg_cands: list[list[tuple[float, str]]] = []
	for k in range(n_seg):
		fi = seg_start + k
		conf = conf_by_idx.get(fi, 50)
		if conf >= soft_anchor_threshold:
			v = rows[fi][2]
			if 0 <= v <= max_speed_kmh:
				seg_cands.append([(v, "anchor")])
			else:
				seg_cands.append([(0.0, "anchor")])
		else:
			cands = candidates_by_frame.get(fi, [])
			raw_v = rows[fi][2]
			options: list[tuple[float, str]] = []
			seen: set[float] = set()
			if 0 <= raw_v <= max_speed_kmh:
				options.append((raw_v, "current"))
				seen.add(raw_v)
			for cv in cands:
				if 0 <= cv <= max_speed_kmh and cv not in seen:
					options.append((cv, "candidate"))
					seen.add(cv)
			if len(options) > VITERBI_MAX_CANDIDATES:
				def _sort_key(opt):
					v, tag = opt
					if tag == 'current': return (0, 0)
					if raw_v > 0 and (v % 100) == (raw_v % 100):
						return (1, abs(v - raw_v))
					return (2, abs(v - raw_v))
				options.sort(key=_sort_key)
				options = options[:VITERBI_MAX_CANDIDATES]
			if not options:
				options.append((0.0, "fallback"))
			seg_cands.append(options)

	dp: list[list[tuple[float, int]]] = []
	dp0: list[tuple[float, int]] = []
	for idx, (v, tag) in enumerate(seg_cands[0]):
		fi = seg_start
		conf = conf_by_idx.get(fi, 50)
		if conf >= soft_anchor_threshold:
			cost = 0.1
		else:
			ref = reference_values.get(fi) if reference_values else None
			cost = _obs_cost(fi, v, rows, obs_weight, ref_val=ref)
		dp0.append((cost, -1))
	dp.append(dp0)

	for k in range(1, n_seg):
		fi = seg_start + k
		dt = times[fi] - times[fi - 1] if fi > 0 and fi - 1 >= 0 else 1.0
		if dt <= 0: dt = 1.0 / 30.0
		dpk: list[tuple[float, int]] = []
		for idx_w, (w, tag_w) in enumerate(seg_cands[k]):
			best_cost = float('inf')
			best_prev = -1
			for idx_v, (v, tag_v) in enumerate(seg_cands[k - 1]):
				trans = _trans_cost(v, w, dt, max_accel_mps2, accel_weight)
				total = dp[k - 1][idx_v][0] + trans
				if total < best_cost:
					best_cost = total
					best_prev = idx_v
			conf = conf_by_idx.get(fi, 50)
			if conf >= soft_anchor_threshold:
				obs = 0.1
			else:
				ref = reference_values.get(fi) if reference_values else None
				obs = _obs_cost(fi, w, rows, obs_weight, ref_val=ref)
			dpk.append((best_cost + obs, best_prev))
		dp.append(dpk)

	last_k = n_seg - 1
	best_final_idx = min(range(len(dp[last_k])), key=lambda idx: dp[last_k][idx][0])
	path: list[float] = [0.0] * n_seg
	cur_idx = best_final_idx
	for k in range(n_seg - 1, -1, -1):
		path[k] = seg_cands[k][cur_idx][0]
		cur_idx = dp[k][cur_idx][1]

	for k in range(n_seg):
		fi = seg_start + k
		optimal_val = path[k]
		raw_val = rows[fi][2]
		path_values[fi] = optimal_val
		raw_cost = dp[k][min(range(len(dp[k])), key=lambda idx: dp[k][idx][0])][0]
		dp_cost[fi] = raw_cost if isinstance(raw_cost, (int, float)) and raw_cost >= 0 else 0.0
		if abs(optimal_val - raw_val) > 0.5:
			corrected[fi] = optimal_val
			error_set.add(fi)


def _obs_cost(fi: int, v: float, rows: list, obs_weight: float,
              ref_val: float | None = None) -> float:
	"""Penalty for deviating from reference value. If ref_val is provided
	(island-interior frames), it replaces raw OCR as the zero-cost point."""
	raw_v = rows[fi][2]
	effective_raw = ref_val if ref_val is not None and ref_val > 0 else raw_v
	if effective_raw > 0 and abs(v - effective_raw) < 0.5:
		return 0.0
	if effective_raw <= 0 and abs(v - effective_raw) < 0.5:
		return 0.0
	if effective_raw > 0:
		raw_ratio = abs(v - effective_raw) / max(1.0, abs(effective_raw))
	else:
		raw_ratio = abs(v - effective_raw) * 0.1
	return obs_weight * min(1.0, raw_ratio)


def _trans_cost(v: float, w: float, dt: float, max_accel_mps2: float,
                accel_weight: float) -> float:
	if dt <= 0: dt = 1.0 / 30.0
	max_dv = max_accel_mps2 * dt * MPS_TO_KMH
	dv = abs(w - v)
	excess = dv - max_dv
	if excess <= 0: return 0.0
	return accel_weight * excess * excess


def _compute_confidence_scores(n: int, dp_cost: list[float],
                               path_values: list[float | None], rows: list,
                               conf_by_idx: dict[int, float]) -> list[dict]:
	max_cost = 0.01
	for i in range(n):
		if i not in conf_by_idx or conf_by_idx[i] < 80:
			if dp_cost[i] > max_cost:
				max_cost = dp_cost[i]
	confidence = []
	for i in range(n):
		if i in conf_by_idx and conf_by_idx[i] >= 90:
			confidence.append({'index': i, 'score': conf_by_idx[i],
				'is_corrected': Flag.is_corrected(rows[i][3]),
				'speed': rows[i][2], 'reason': '锚点帧(高置信)'})
			continue
		cost = dp_cost[i]
		if cost < 0:
			confidence.append({'index': i, 'score': 50.0,
				'is_corrected': Flag.is_corrected(rows[i][3]),
				'speed': rows[i][2], 'reason': '未处理'})
			continue
		normalized = cost / max_cost
		score = 100.0 * math.exp(-normalized)
		cur = rows[i][2]
		if cur < 0 or cur > 400:
			score = 0.0; reason = '速度超出范围'
		elif score >= 80: reason = '正常'
		elif score >= 40: reason = 'Viterbi存疑(临界的)'
		else: reason = 'Viterbi错误(代价高)'
		confidence.append({'index': i, 'score': round(score, 1),
			'is_corrected': Flag.is_corrected(rows[i][3]),
			'speed': cur, 'reason': reason})
	return confidence
