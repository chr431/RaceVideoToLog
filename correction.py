"""Correction — 物理约束纠错流水线。

5 阶段流水线：LCS 错误检测 → 重OCR → LCS 最优选择 → 多轮迭代 → 级联填充。
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
	compute_lcs_scores, compute_lcs_scores_lr, lcs_detect_errors,
	_lcs_score_for_value,
)
from config import (MPS_TO_KMH, LCS_TRUST_HIGH,
	LCS_CONFIDENCE_MIN_SCORE, LCS_ERROR_LOW,
	LCS_INTERP_WEIGHT, LCS_NOVELTY_WEIGHT,
	PROFILE_TIME_WINDOW, PROFILE_MIN_WINDOW,
	PROFILE_ABS_TOLERANCE, PROFILE_PCT_TOLERANCE,
	CORRECTION_MAX_ROUNDS, FILL_MAX_PASSES,
	CORRECTION_ACCEPT_MIN_SCORE, CORRECTION_MIN_DIFF,
	INTERP_PROX_ABS, INTERP_PROX_PCT, REOCR_HEIGHTS,
	ACCEL_ANOMALY_THRESHOLD, MAX_SUGGESTED_FRAMES,
	PROBLEM_MIN_SEGMENT_LEN, MAX_PARTIAL_WILDCARDS)

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


def expand_partial(pattern: str, max_speed: float) -> list[float]:
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


def _auto_expand_digits(raw_text: str, max_speed_kmh: float) -> list[float]:
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
	candidates: list[float] = []
	try:
		candidates.append(float(digits))
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


def correct_with_trust(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR",
						 max_speed_kmh: float, max_accel_mps2: float, anchor_indices: set | None = None,  # deprecated, use pinned
						 log_fn: "Callable | None" = None,
						 progress_fn: "Callable | None" = None,
						 skip_fill: bool = False,
						 timing: dict | None = None,
						 partial_corrections: dict[int, str] | None = None,
						 reocr_cache: dict | None = None,
						 light_mode: bool = False,
						 notes: dict[int, str] | None = None, pinned: set[int] | None = None,
							 fps: float = 1.0) -> list:
	"""5 阶段物理约束纠错流水线。

	以 pinned 帧（用户手动修正）为硬约束（固定不变），
	LCS 自动检测的可信帧辅助约束，对其余帧进行错误检测、重OCR、最优选择和级联填充。

	Args:
	    reocr_cache: 可选的重 OCR 缓存字典，绑定到 Pipeline 实例生命周期。
	    light_mode: 轻量模式 — 仅重OCR + 原始值之间选择，不生成混淆/推断/插值候选，
	                不迭代，不级联填充。用于 pass1 人工审核前预处理。
	Returns: 修改后的 rows（原地修改）
	"""
	pinned = pinned or set()
	n = len(rows)
	for pi in (pinned or set()):
		if rows[pi][3] < Flag.HIGH_TRUST:
			rows[pi][3] = Flag.PINNED
	pinned_set = pinned  # user-verified frames treated as ground truth
	anchors = pinned_set  # kept as internal var name for backward compat
	times = [r[0] / fps for r in rows]
	cache: dict = reocr_cache if reocr_cache is not None else {}

	# ── 全局一致性参考剖面：用于 HIGH_TRUST 标记前的一致性验证 ──
	# 中值滤波天然抗离群值，能检测"一致性孤岛"——局部物理自洽但偏离全局趋势的误读
	# 相比 SG 滤波，中值滤波不会被少量离群值污染剖面

	def _median_filter_np(y: "np.ndarray", window: int) -> "np.ndarray":
		"""滑动中值滤波，O(N*W) 但 N≈6k, W≈15 时可接受。"""
		half = window // 2
		result = np.zeros(len(y), dtype=float)
		for i in range(len(y)):
			lo = max(0, i - half)
			hi = min(len(y), i + half + 1)
			result[i] = float(np.median(y[lo:hi]))
		return result

	_vals = np.array([r[2] for r in rows], dtype=float)
	# 基于时间的窗口：目标 0.5s，最少 5 帧
	# 使用 times 数组推导实际帧间隔（兼容不同 fps 和 div 参数）
	if n >= 2:
		_dt = times[1] - times[0]  # 连续行之间的秒数
	else:
		_dt = 1.0 / max(fps, 1.0)
	_med_win = max(PROFILE_MIN_WINDOW, int(PROFILE_TIME_WINDOW / _dt + 0.5))
	if _med_win % 2 == 0:
		_med_win += 1
	_med_win = min(_med_win, n - 2)
	if _med_win >= PROFILE_MIN_WINDOW and n >= _med_win:
		_ref_profile = _median_filter_np(_vals, _med_win)
	else:
		_ref_profile = _vals.copy()

	def _profile_trust_ok(idx: int) -> bool:
		"""检查帧 idx 的值与中值参考剖面的偏差是否在可接受范围内。"""
		v = rows[idx][2]
		ref_v = _ref_profile[idx]
		if v < 0 or ref_v <= 0:
			return True  # 无法判断时不过滤
		return abs(v - ref_v) <= max(PROFILE_ABS_TOLERANCE, ref_v * PROFILE_PCT_TOLERANCE)

	if log_fn:
		mode_str = " (light)" if light_mode else ""
		log_fn(f"Correction{mode_str}: {n} rows, {len(anchors)} trusted/pinned")

	# ── 阶段 1：错误检测 ──
	error_set, _scores_l, _scores_r = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2, fps=fps)
	if log_fn:
		log_fn(f"  Stage 1: detected {len(error_set)} errors")
	# 标记高信帧（两侧均 >= TRUST_HIGH + 中值参考剖面一致性）
	for i in range(len(_scores_l)):
		if (_scores_l[i] >= LCS_TRUST_HIGH and _scores_r[i] >= LCS_TRUST_HIGH
				and rows[i][3] == Flag.RAW and i not in error_set
				and _profile_trust_ok(i)):
			rows[i][3] = Flag.HIGH_TRUST
	if not error_set:
		return rows

	# ── 阶段 2+3：重 OCR + 最优选择（首轮）──
	fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
	                    anchors, times, max_speed_kmh, max_accel_mps2,
	                    progress_fn=progress_fn, timing=timing,
	                    partial_corrections=partial_corrections, reocr_cache=cache,
	                    light_mode=light_mode, notes=notes, fps=fps)
	if log_fn:
		log_fn(f"  Stage 2+3: fixed {fixed} frames in round 1")

	# ── Light mode: 一轮即止，剩余错误标记为待审核 ──
	if light_mode:
		error_set, _scores_l, _scores_r = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2, fps=fps)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Light: {len(error_set)} frames flagged for manual review")
		return rows

	# ── 阶段 4：多轮迭代 ──
	for rnd in range(2, CORRECTION_MAX_ROUNDS + 1):
		error_set, _scores_l, _scores_r = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2, fps=fps)
		if not error_set:
			break
		fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
		                    anchors, times, max_speed_kmh, max_accel_mps2,
		                    progress_fn=progress_fn, timing=timing,
		                    partial_corrections=partial_corrections, reocr_cache=cache,
		                    notes=notes, fps=fps)
		if log_fn:
			log_fn(f"  Stage 4 round {rnd}: {len(error_set)} errors, fixed {fixed}")

	# ── 阶段 5：迭代填充直到收敛（处理级联效应）──
	if skip_fill:
		error_set, _scores_l, _scores_r = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2, fps=fps)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Stage 5: {len(error_set)} frames flagged for manual review")
	else:
		fill_pass = 0
		while fill_pass < FILL_MAX_PASSES:
			error_set, _scores_l, _scores_r = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2, fps=fps)
			if not error_set:
				break
			_fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2, fps,
			                    progress_fn=progress_fn, notes=notes)
			if log_fn:
				log_fn(f"  Stage 5 pass {fill_pass+1}: filled {len(error_set)} unrecoverable frames")
			fill_pass += 1

	# 最终标记高信帧（两侧均 >= TRUST_HIGH + 中值参考剖面一致性）
	# 在修正后重新计算 中值参考剖面，避免被原始 OCR 离群值污染
	_corrected_vals = np.array([r[2] for r in rows], dtype=float)
	_ref_profile = _median_filter_np(_corrected_vals, _med_win)
	scores_l, scores_r = compute_lcs_scores_lr(rows, max_speed_kmh, max_accel_mps2, pinned=anchors, fps=fps)
	for i in range(len(scores_l)):
		if (scores_l[i] >= LCS_TRUST_HIGH and scores_r[i] >= LCS_TRUST_HIGH
				and rows[i][3] == Flag.RAW
				and _profile_trust_ok(i)):
			rows[i][3] = Flag.HIGH_TRUST
	return rows


# ── LCS 错误检测 ──

def _detect_errors(rows: list, anchors: set, times: list,
                   max_speed_kmh: float, max_accel_mps2: float,
                   fps: float = 1.0) -> tuple[set[int], list[float], list[float]]:
	"""阶段 1：LCS 左右分侧错误检测。

	Returns: (error_set, left_scores, right_scores)
	"""
	n = len(rows)
	error_set: set[int] = set()

	for i in range(n):
		if i in anchors:
			continue
		v = rows[i][2]
		if v < 0 or v > max_speed_kmh:
			error_set.add(i)

	scores_l, scores_r = compute_lcs_scores_lr(rows, max_speed_kmh, max_accel_mps2, pinned=anchors, fps=fps)
	lcs_errors, borderline = lcs_detect_errors(scores_l, scores_r)
	error_set.update(lcs_errors)
	error_set.update(borderline)
	error_set -= anchors

	return error_set, scores_l, scores_r


def _fix_errors(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR", error_set: set,
				anchors: set, times: list, max_speed_kmh: float, max_accel_mps2: float,
				progress_fn: "Callable | None" = None,
				timing: dict | None = None,
				partial_corrections: dict[int, str] | None = None,
				reocr_cache: dict | None = None,
				light_mode: bool = False,
				notes: dict[int, str] | None = None,
				fps: float = 1.0) -> int:
	"""阶段 2+3：对每个 error 帧重 OCR 获取备选，LCS 评分选最优值填入。"""
	n = len(rows)
	fixed = 0
	progress_done = 0
	# 按到最近 trusted 帧的距离排序：边界帧优先处理，修正可级联向内传播
	def _dist_to_trusted(fi: int) -> int:
		la, ra = _find_neighbor_trusted(fi, n, rows)
		d = n
		if la is not None: d = min(d, fi - la)
		if ra is not None: d = min(d, ra - fi)
		return d
	# 跳过已标记为高信/固定的帧（阻止级联错误传播）
	error_list = sorted((i for i in error_set if i not in anchors and not Flag.is_trusted(rows[i][3])),
	                    key=_dist_to_trusted)
	total = len(error_list)
	for i in error_list:
		has_partial = partial_corrections and i in partial_corrections
		interp_cand = _interp_candidate(i, rows, anchors, times, max_speed_kmh, fps=fps)
		oid = min(i, len(observations) - 1)
		reocr_set = _re_ocr_frame(raw_frames[i][1], ocr, max_speed_kmh,
		                          timing=timing, cache=reocr_cache)

		# ── 收集候选值 ──
		if has_partial:
			candidates = expand_partial(partial_corrections[i], max_speed_kmh)
		else:
			if light_mode:
				# 轻量模式：仅 re-OCR 值，不生成混淆/推断/插值候选
				candidates = list(reocr_set)
			else:
				candidates = list(reocr_set)
			# 混淆字符候选
			confusion_cands = build_speed_candidates(observations[oid].raw_text, max_speed_kmh)
			for c in confusion_cands:
				if c not in candidates:
					candidates.append(c)
			# 自动缺位扩展：OCR 读到 "21" → 生成所有可能的 2-3 位数
			for c in _auto_expand_digits(observations[oid].raw_text, max_speed_kmh):
				if c not in candidates:
					candidates.append(c)

		# 插值候选（light_mode 下不使用）
		if not light_mode and interp_cand is not None:
			candidates.append(interp_cand)

		# ── 选择最佳候选：候选 + 当前值 + 参考值 统一评分 ──
		if candidates:
			raw_val = rows[i][2]
			ref_value = interp_cand
			options: list[tuple[float, str]] = []
			for c in candidates:
				if 0 <= c <= max_speed_kmh:
					options.append((c, "candidate"))
			if interp_cand is not None and not light_mode:
				options.append((interp_cand, "interp"))
			if 0 <= raw_val <= max_speed_kmh:
				options.append((raw_val, "current"))

			if not options:
				progress_done += 1
				if progress_fn: progress_fn(progress_done, total)
				continue

			best_val = None; best_score = -1.0; best_tag = ""
			for val, tag in options:
				score = _lcs_score_for_value(i, val, rows, times,
				                              max_speed_kmh, max_accel_mps2,
				                              high_weight=anchors)
				# 参考值接近度加成
				if ref_value is not None and ref_value > 0:
					ref_prox = max(0.0, 1.0 - abs(val - ref_value) / max(INTERP_PROX_ABS, ref_value * INTERP_PROX_PCT))
					score += ref_prox * LCS_INTERP_WEIGHT
				# 新颖性
				if abs(val - raw_val) > 0.5:
					score += LCS_NOVELTY_WEIGHT
				if score > best_score:
					best_score = score; best_val = val; best_tag = tag

			if (best_tag != "current" and best_val is not None
					and abs(raw_val - best_val) > CORRECTION_MIN_DIFF and best_score > CORRECTION_ACCEPT_MIN_SCORE):
				if notes is not None:
					notes[i] = f"{best_tag}: {raw_val:.0f}→{best_val:.0f}"
				rows[i][2] = best_val
				if rows[i][3] == Flag.RAW:
					if best_tag == "interp":
						rows[i][3] = Flag.FILL_INTERP
					else:
						rows[i][3] = Flag.PARTIAL_AUTO if has_partial else Flag.REOCR_AUTO
				fixed += 1

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)
	return fixed


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
			candidates.add(float(sv))

	if cache_key is not None:
		cache[cache_key] = candidates
	return candidates


def _interp_candidate(i: int, rows: list, anchors: set, times: list, max_speed_kmh: float, fps: float = 1.0) -> float | None:
	"""计算帧 i 在左右高信帧间的线性插值估计。"""
	n = len(rows)
	la, ra = _find_neighbor_trusted(i, n, rows)
	if la is not None and ra is not None:
		lv = rows[la][2]; rv = rows[ra][2]
		lt = rows[la][0] / fps; rt = rows[ra][0] / fps
		total_dt = max(rt - lt, 0.001)
		frac = (times[i] - lt) / total_dt
		val = lv + (rv - lv) * frac
		if 0 <= val <= max_speed_kmh:
			return round(val)
	return None


def _fill_unrecoverable(rows: list, anchors: set, error_set: set, times: list, max_speed_kmh: float, max_accel_mps2: float, fps: float = 1.0,
						progress_fn: "Callable | None" = None,
						notes: dict[int, str] | None = None) -> None:
	"""阶段 5：对无法通过重 OCR 修复的帧，以最近高信帧为基准插值。

	左右均使用 HIGH_TRUST/PINNED 帧作为约束，防止 fill 链式累积。
	"""
	n = len(rows)
	# 跳过已标记为高信/固定的帧（阻止级联错误传播）
	sorted_errors = sorted(i for i in error_set if i not in anchors and not Flag.is_trusted(rows[i][3]))
	total = len(sorted_errors)
	progress_done = 0
	for i in sorted_errors:
		# 左侧最近高信帧（非前次 fill 值）
		la = None
		for j in range(i - 1, -1, -1):
			if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
				la = j; break
		if la is None:
			continue
		lv = rows[la][2]; lt = rows[la][0] / fps

		# 右侧最近高信帧
		ra = None
		for j in range(i + 1, n):
			if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
				ra = j; break

		left_dt = max(times[i] - lt, 0.001)
		left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
		if ra is not None:
			rv = rows[ra][2]; rt = rows[ra][0] / fps
			right_dt = max(rt - times[i], 0.001)
			right_max_dv = max_accel_mps2 * right_dt * MPS_TO_KMH
			lo = max(0.0, lv - left_max_dv)
			hi = min(max_speed_kmh, lv + left_max_dv)
			lo = max(lo, rv - right_max_dv)
			hi = min(hi, rv + right_max_dv)
			interp = lv + (rv - lv) * (left_dt / max(left_dt + right_dt, 0.001))
			val = round(max(lo, min(hi, interp)))
		else:
			# 无右侧高信帧 → 跳过（无法可靠约束）
			continue
		rows[i][2] = float(val)
		if rows[i][3] == Flag.RAW:
			rows[i][3] = Flag.FILL_INTERP
		if notes is not None:
			notes[i] = f"fill: {val:.0f}"

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)


# ═══════════════════════════════════════════════════════════════
# 置信度评分 — 用于聚焦人工审核
# ═══════════════════════════════════════════════════════════════

def compute_confidence(rows: list, observations: list, max_speed: float,
					   max_accel: float, pinned: set[int] | None = None,
					   fps: float = 1.0) -> list[dict]:
	"""LCS 置信度评分 (0-100)。

	直接用 compute_lcs_scores 的局部一致性分数，
	mapping: confidence = lcs_score × 100。

	- 100 分: 当前值与 0.5s 时间窗内所有邻居物理自洽
	- 0 分: 与所有邻居物理矛盾 / 速度超出范围
	- 70 分: LCS 0.7（correction 阶段的 borderline 阈值）
	- 30 分: LCS 0.3（correction 阶段的 error 阈值）

	pinned 帧在评分中获得 3× 权重。
	"""
	n = len(rows)
	scores_l, scores_r = compute_lcs_scores_lr(rows, max_speed, max_accel, pinned=pinned, fps=fps)
	flags = [r[3] for r in rows]

	confidences = []
	for i in range(n):
		lcs = (scores_l[i] + scores_r[i]) / 2.0
		cur = rows[i][2]

		if cur < 0 or cur > max_speed:
			confidences.append({
				'index': i, 'score': 0.0, 'is_corrected': Flag.is_corrected(flags[i]),
				'speed': cur, 'reason': '速度超出范围',
			})
			continue

		if lcs < LCS_ERROR_LOW:
			reason = 'LCS错误(物理矛盾)'
		elif lcs < LCS_TRUST_HIGH:
			reason = 'LCS存疑(临界的)'
		else:
			reason = '正常'

		confidences.append({
			'index': i, 'score': round(lcs * 100, 1),
			'is_corrected': Flag.is_corrected(flags[i]),
			'speed': cur, 'reason': reason,
		})

	return confidences


def find_problem_segments(confidences: list[dict], min_score: float = LCS_CONFIDENCE_MIN_SCORE,
						  min_segment_len: int = PROBLEM_MIN_SEGMENT_LEN) -> list[dict]:
	"""将低置信度连续帧聚合成问题段。

	min_score 默认 30（对应 LCS 0.3，即 correction 阶段的 error 阈值）。
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
					'suggested': sorted(set(
						[seg_frames[0]['index'], seg_frames[-1]['index']] +
						[min(seg_frames, key=lambda f: f['score'])['index']]
					)),
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
