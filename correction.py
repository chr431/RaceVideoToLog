"""Correction — 物理约束纠错流水线。

5 阶段流水线：错误检测 → 重OCR → 最优选择 → 多轮迭代 → 级联填充。
支持 GUI 和无头 CLI 共用同一实现。
"""
from __future__ import annotations
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ocr_engine import extract_speed_value, build_speed_candidates, Flag
from config import MPS_TO_KMH

if TYPE_CHECKING:
	from rapidocr import RapidOCR

logger = logging.getLogger("RaceVideoToLog.correction")

def _find_neighbor_anchors(i: int, n: int, anchors: set[int]) -> tuple[int | None, int | None]:
	"""Find nearest left and right anchor indices for frame i."""
	la = None
	for j in range(i - 1, -1, -1):
		if j in anchors:
			la = j
			break
	ra = None
	for j in range(i + 1, n):
		if j in anchors:
			ra = j
			break
	return la, ra


def _infer_partial_pattern(ocr_text: str, expected: float, max_speed: float) -> str | None:
	"""根据 OCR 文本和邻居插值估算，自动推断缺失数字的位置。

	例如 OCR 读到 "21"、邻居速度约 221 → 推断模式 "x21"（首位缺失）。
	OCR 读到 "20"、邻居速度约 200 → 推断模式 "20x"（末位缺失）。

	Returns: 如 "x21" 的模式字符串，或 None 表示无法可靠推断。
	"""
	if not ocr_text or not ocr_text.isdigit():
		return None
	if len(ocr_text) > 3:
		return None

	best_pattern = None
	best_diff = float("inf")

	# 在每个位置尝试插入 'x'（处理缺失数字）
	for i in range(len(ocr_text) + 1):
		pattern = ocr_text[:i] + "x" + ocr_text[i:]
		for v in expand_partial(pattern, max_speed):
			diff = abs(v - expected)
			if diff < best_diff:
				best_diff = diff
				best_pattern = pattern

	# 在每个位置替换为 'x'（处理误读数字）
	for i in range(len(ocr_text)):
		pattern = ocr_text[:i] + "x" + ocr_text[i + 1:]
		for v in expand_partial(pattern, max_speed):
			diff = abs(v - expected)
			if diff < best_diff:
				best_diff = diff
				best_pattern = pattern

	# 仅在与预期值相差 <20% 时才采纳
	if best_pattern and best_diff < expected * 0.2:
		return best_pattern
	return None


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
	if x_count == len(pattern) or x_count > 2:
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


def correct_with_anchors(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR",
						 max_speed_kmh: float, max_accel_mps2: float, anchor_indices: set,
						 log_fn: "Callable | None" = None,
						 progress_fn: "Callable | None" = None,
						 skip_fill: bool = False,
						 timing: dict | None = None,
						 partial_corrections: dict[int, str] | None = None,
						 reocr_cache: dict | None = None,
						 light_mode: bool = False) -> list:
	"""5 阶段物理约束纠错流水线。

	以 anchor_indices 中帧的速度为硬约束（固定不变），
	对其余帧进行错误检测、重OCR、最优选择和级联填充。

	Args:
	    reocr_cache: 可选的重 OCR 缓存字典，绑定到 Pipeline 实例生命周期。
	    light_mode: 轻量模式 — 仅重OCR + 原始值之间选择，不生成混淆/推断/插值候选，
	                不迭代，不级联填充。用于 pass1 人工审核前预处理。
	Returns: 修改后的 rows（原地修改）
	"""
	if len(anchor_indices) < 2:
		return rows

	n = len(rows)
	anchors = anchor_indices
	times = [r[0] for r in rows]
	cache: dict = reocr_cache if reocr_cache is not None else {}

	if log_fn:
		mode_str = " (light)" if light_mode else ""
		log_fn(f"Correction{mode_str}: {n} rows, {len(anchors)} anchors")

	# ── 阶段 1：错误检测 ──
	error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
	if log_fn:
		log_fn(f"  Stage 1: detected {len(error_set)} errors")
	if not error_set:
		return rows

	# ── 阶段 2+3：重 OCR + 最优选择（首轮）──
	fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
	                    anchors, times, max_speed_kmh, max_accel_mps2,
	                    progress_fn=progress_fn, timing=timing,
	                    partial_corrections=partial_corrections, reocr_cache=cache,
	                    light_mode=light_mode)
	if log_fn:
		log_fn(f"  Stage 2+3: fixed {fixed} frames in round 1")

	# ── Light mode: 一轮即止，剩余错误标记为待审核 ──
	if light_mode:
		error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Light: {len(error_set)} frames flagged for manual review")
		return rows

	# ── 阶段 4：多轮迭代 ──
	max_rounds = 3
	for rnd in range(2, max_rounds + 1):
		error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		if not error_set:
			break
		fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
		                    anchors, times, max_speed_kmh, max_accel_mps2,
		                    progress_fn=progress_fn, timing=timing,
		                    partial_corrections=partial_corrections, reocr_cache=cache)
		if log_fn:
			log_fn(f"  Stage 4 round {rnd}: {len(error_set)} errors, fixed {fixed}")

	# ── 阶段 5：迭代填充直到收敛（处理级联效应）──
	if skip_fill:
		error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Stage 5: {len(error_set)} frames flagged for manual review")
	else:
		fill_pass = 0
		while fill_pass < 10:
			error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
			if not error_set:
				break
			_fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2,
			                    progress_fn=progress_fn)
			if log_fn:
				log_fn(f"  Stage 5 pass {fill_pass+1}: filled {len(error_set)} unrecoverable frames")
			fill_pass += 1

	return rows


# ── 错误检测器（6 种独立策略）──

def _detect_neighbor_jump(i: int, v: float, n: int, raw_vals: list[float],
                           times: list[float], max_speed_kmh: float,
                           max_accel_mps2: float) -> bool:
	"""A. 邻帧跳变：与前后邻帧的加速度超限（需双向都失败）。"""
	fwd_fail = False
	bwd_fail = False
	if i > 0:
		prev_v = raw_vals[i - 1]
		if prev_v >= 0 and prev_v <= max_speed_kmh:
			dt = max(times[i] - times[i - 1], 0.001)
			max_dv = max_accel_mps2 * dt * MPS_TO_KMH * 1.2
			if abs(v - prev_v) > max_dv:
				if not (i + 1 < n and v == raw_vals[i + 1]
				        and times[i + 1] - times[i] < 0.15):
					fwd_fail = True
	if i + 1 < n:
		next_v = raw_vals[i + 1]
		if next_v >= 0 and next_v <= max_speed_kmh:
			dt = max(times[i + 1] - times[i], 0.001)
			max_dv = max_accel_mps2 * dt * MPS_TO_KMH * 1.2
			if abs(next_v - v) > max_dv:
				if not (i > 0 and v == raw_vals[i - 1]
				        and times[i] - times[i - 1] < 0.15):
					bwd_fail = True
	return fwd_fail and bwd_fail


def _detect_v_shape(i: int, v: float, n: int, raw_vals: list[float],
                    times: list[float], max_accel_mps2: float) -> bool:
	"""A2. V 字形：急减速后立即急加速（OCR 误读特征）。"""
	if i <= 0 or i + 1 >= n:
		return False
	prev_v = raw_vals[i - 1]
	next_v = raw_vals[i + 1]
	if prev_v <= 0 or next_v <= 0:
		return False
	dt_left = max(times[i] - times[i - 1], 0.001)
	dt_right = max(times[i + 1] - times[i], 0.001)
	accel_left = (v - prev_v) / dt_left
	accel_right = (next_v - v) / dt_right
	accel_limit = max_accel_mps2 * MPS_TO_KMH * 2.5
	if abs(accel_left) > accel_limit and accel_left * accel_right < 0:
		if not (i + 1 < n and v == raw_vals[i + 1]
		        and times[i + 1] - times[i] < 0.15):
			return True
	if abs(accel_right) > accel_limit and accel_right * accel_left < 0:
		if not (i > 0 and v == raw_vals[i - 1]
		        and times[i] - times[i - 1] < 0.15):
			return True
	return False


def _detect_cliff(i: int, v: float, n: int, raw_vals: list[float],
                  times: list[float], max_accel_mps2: float) -> bool:
	"""A3. 悬崖：单侧极端跳变 + 对侧平坦。"""
	if i <= 0 or i + 1 >= n:
		return False
	prev_v = raw_vals[i - 1]
	next_v = raw_vals[i + 1]
	if prev_v <= 0 or next_v <= 0:
		return False
	dt_left = max(times[i] - times[i - 1], 0.001)
	dt_right = max(times[i + 1] - times[i], 0.001)
	accel_left = (v - prev_v) / dt_left
	accel_right = (next_v - v) / dt_right
	cliff_limit = max_accel_mps2 * MPS_TO_KMH * 3.0
	if abs(accel_left) > cliff_limit and abs(accel_right) < cliff_limit * 0.3:
		return True
	if abs(accel_right) > cliff_limit and abs(accel_left) < cliff_limit * 0.3:
		return True
	return False


def _detect_anchor_trend(i: int, v: float, n: int, rows: list,
                          times: list[float], anchors: set[int],
                          max_accel_mps2: float) -> bool:
	"""B. 锚点趋势偏离：偏离锚点间线性插值过多。"""
	la, ra = _find_neighbor_anchors(i, n, anchors)
	if la is None or ra is None:
		return False
	lv = rows[la][2]; rv = rows[ra][2]
	lt = rows[la][0]; rt = rows[ra][0]
	total_dt = max(rt - lt, 0.001)
	frac = (times[i] - lt) / total_dt
	interp = lv + (rv - lv) * frac
	seg_dt = times[i] - lt
	threshold = max(5.0, 3.0 * max_accel_mps2 * max(seg_dt, 0.1) * MPS_TO_KMH)
	return abs(v - interp) > threshold


def _detect_isolated_spike(i: int, v: float, n: int, raw_vals: list[float],
                            times: list[float], max_accel_mps2: float) -> bool:
	"""C. 孤立离群：与两边都冲突但邻居彼此一致。"""
	if i < 2 or i + 2 >= n:
		return False
	left_v = (raw_vals[i - 1] if raw_vals[i - 1] >= 0
	          else (raw_vals[i - 2] if raw_vals[i - 2] >= 0 else None))
	right_v = (raw_vals[i + 1] if raw_vals[i + 1] >= 0
	           else (raw_vals[i + 2] if raw_vals[i + 2] >= 0 else None))
	if left_v is None or right_v is None:
		return False
	dt_cross = max(times[i + 2] - times[i - 2], 0.01)
	max_dv_cross = max_accel_mps2 * dt_cross * MPS_TO_KMH * 1.5
	if abs(right_v - left_v) > max_dv_cross:
		return False
	dt_left = max(times[i] - times[i - 1], 0.001)
	dt_right = max(times[i + 1] - times[i], 0.001)
	max_dv_l = max_accel_mps2 * dt_left * MPS_TO_KMH * 1.5
	max_dv_r = max_accel_mps2 * dt_right * MPS_TO_KMH * 1.5
	return abs(v - left_v) > max_dv_l and abs(right_v - v) > max_dv_r


def _detect_local_trend(i: int, v: float, n: int, raw_vals: list[float],
                         max_speed_kmh: float) -> bool:
	"""D. 局部趋势偏离：5 帧中位数偏离，且左右邻帧均接近中位数。"""
	if i < 2 or i + 2 >= n:
		return False
	window = []
	for j in range(max(0, i - 2), min(n, i + 3)):
		if j != i and raw_vals[j] >= 0 and raw_vals[j] <= max_speed_kmh:
			window.append(raw_vals[j])
	if len(window) < 3:
		return False
	window.sort()
	local_median = window[len(window) // 2]
	dev = abs(v - local_median)
	if dev <= 3.0:
		return False
	left_ok = (i >= 1 and raw_vals[i - 1] >= 0
	           and abs(raw_vals[i - 1] - local_median) < 2.0)
	right_ok = (i + 1 < n and raw_vals[i + 1] >= 0
	            and abs(raw_vals[i + 1] - local_median) < 2.0)
	return left_ok and right_ok


def _detect_stuck_value(i: int, v: float, n: int, raw_vals: list[float]) -> bool:
	"""E. 卡值：连续 ≥3 帧完全相同，但上下文显示速度在变化（OCR 重复输出同一错误值）。"""
	if i < 2 or i + 2 >= n:
		return False
	# 计算连续相同值的长度
	run_start = i
	while run_start > 0 and raw_vals[run_start - 1] == v:
		run_start -= 1
	run_end = i
	while run_end + 1 < n and raw_vals[run_end + 1] == v:
		run_end += 1
	if run_end - run_start + 1 < 3:
		return False
	# 检查跑段外的上下文是否在变化
	pre_val = None
	for j in range(run_start - 1, -1, -1):
		if raw_vals[j] > 0 and raw_vals[j] != v:
			pre_val = raw_vals[j]; break
	post_val = None
	for j in range(run_end + 1, n):
		if raw_vals[j] > 0 and raw_vals[j] != v:
			post_val = raw_vals[j]; break
	if pre_val is None or post_val is None:
		return False
	return abs(post_val - pre_val) > 5.0


def _detect_brief_excursion(i: int, v: float, n: int, raw_vals: list[float],
                             max_speed_kmh: float) -> bool:
	"""F. 短暂偏离：偏离 5 帧中位数 ≥5 km/h，且 ±3 帧内有帧贴近中位数。

	捕获"下探又弹回"模式（如 219→212→219），这类偏离虽在物理容限内，
	但表现为孤立瞬变，通常是 OCR 误读而非真实速度变化。
	"""
	if i < 2 or i + 2 >= n:
		return False
	# 5 帧中位数（排除自身）
	window = []
	for j in range(max(0, i - 2), min(n, i + 3)):
		if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh:
			window.append(raw_vals[j])
	if len(window) < 3:
		return False
	window.sort()
	median = window[len(window) // 2]
	if abs(v - median) < 5.0:
		return False
	# 前一方向：有帧贴近中位数
	pre_ok = False
	for j in range(i - 1, max(0, i - 4), -1):
		if raw_vals[j] > 0 and abs(raw_vals[j] - median) < 3.0:
			pre_ok = True; break
	# 后一方向：有帧贴近中位数
	post_ok = False
	for j in range(i + 1, min(n, i + 4)):
		if raw_vals[j] > 0 and abs(raw_vals[j] - median) < 3.0:
			post_ok = True; break
	return pre_ok and post_ok


def _detect_errors(rows: list, anchors: set, times: list,
                   max_speed_kmh: float, max_accel_mps2: float) -> set:
	"""阶段 1：错误检测。8 种独立检测器投票制标记异常帧。

	≥2 票 → 确定错误，1 票 + 锚点稀疏 → 错误，0 票 → 可信。

	A. 邻帧跳变  A2. V 字形  A3. 悬崖  E. 卡值  F. 短暂偏离
	B. 锚点趋势偏离  C. 孤立离群  D. 局部趋势偏离
	"""
	n = len(rows)
	raw_vals = [r[2] for r in rows]
	error_set: set[int] = set()

	for i in range(n):
		if i in anchors:
			continue
		v = raw_vals[i]
		if v < 0 or v > max_speed_kmh:
			error_set.add(i)
			continue

		# 收集所有检测器投票
		votes = 0
		if _detect_neighbor_jump(i, v, n, raw_vals, times, max_speed_kmh, max_accel_mps2):
			votes += 1
		if _detect_v_shape(i, v, n, raw_vals, times, max_accel_mps2):
			votes += 1
		if _detect_cliff(i, v, n, raw_vals, times, max_accel_mps2):
			votes += 1
		if _detect_anchor_trend(i, v, n, rows, times, anchors, max_accel_mps2):
			votes += 1
		if _detect_isolated_spike(i, v, n, raw_vals, times, max_accel_mps2):
			votes += 1
		if _detect_local_trend(i, v, n, raw_vals, max_speed_kmh):
			votes += 1
		if _detect_stuck_value(i, v, n, raw_vals):
			votes += 1
		if _detect_brief_excursion(i, v, n, raw_vals, max_speed_kmh):
			votes += 1

		if votes >= 2:
			error_set.add(i)
		elif votes == 1:
			# 单票需额外条件：锚点稀疏区域（距离最近锚点 > 30 帧）
			la, ra = _find_neighbor_anchors(i, n, anchors)
			gap = 999
			if la is not None and ra is not None:
				gap = ra - la
			elif la is not None:
				gap = n - la
			elif ra is not None:
				gap = ra
			if gap > 30:
				error_set.add(i)

	return error_set


def _fix_errors(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR", error_set: set,
				anchors: set, times: list, max_speed_kmh: float, max_accel_mps2: float,
				progress_fn: "Callable | None" = None,
				timing: dict | None = None,
				partial_corrections: dict[int, str] | None = None,
				reocr_cache: dict | None = None,
				light_mode: bool = False) -> int:
	"""阶段 2+3：对每个 error 帧重 OCR 获取备选，选最优值填入。"""
	fixed = 0
	progress_done = 0
	error_list = sorted(i for i in error_set if i not in anchors)
	total = len(error_list)
	for i in error_list:
		has_partial = partial_corrections and i in partial_corrections
		interp_cand = _interp_candidate(i, rows, anchors, times, max_speed_kmh)
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
			# 自动推断部分数字模式：OCR 读到 "21"、邻居约 221 → "x21"
			if interp_cand is not None:
				auto_pattern = _infer_partial_pattern(
					observations[oid].raw_text, interp_cand, max_speed_kmh)
				if auto_pattern:
					for c in reversed(expand_partial(auto_pattern, max_speed_kmh)):
						if c not in candidates:
							candidates.insert(0, c)

		# 插值候选（light_mode 下不使用）
		if not light_mode and interp_cand is not None:
			candidates.append(interp_cand)

		# ── 选择最佳候选 ──
		if candidates:
			raw_val = rows[i][2]
			# re-OCR 无结果且插值与当前值差异大 → 完整模式直接插值，light 模式跳过
			if len(reocr_set) <= 1 and interp_cand is not None and abs(interp_cand - raw_val) > 10.0:
				if not light_mode and abs(raw_val - interp_cand) > 0.5:
					rows[i][2] = interp_cand
					if rows[i][3] == Flag.RAW:
						rows[i][3] = Flag.PARTIAL_AUTO if has_partial else Flag.REOCR_AUTO
					fixed += 1
			else:
				best_val = None
				best_score = -1.0
				for cand in set(candidates):
					if not (0 <= cand <= max_speed_kmh):
						continue
					score = _score_candidate(cand, i, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2)
					if score > best_score:
						best_score = score
						best_val = cand

				if best_val is not None and best_score > 0.3 and abs(rows[i][2] - best_val) > 0.5:
					rows[i][2] = best_val
					if rows[i][3] == Flag.RAW:
						rows[i][3] = Flag.PARTIAL_AUTO if has_partial else Flag.REOCR_AUTO
					fixed += 1

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)
	return fixed


def _re_ocr_frame(crop_bgr: "np.ndarray", ocr: "RapidOCR", max_speed_kmh: float,
				  timing: dict | None = None, cache: dict | None = None) -> set:
	"""阶段 2：对单帧尝试 h=32 重 OCR（与主 OCR h=24 不同高度以产生不同结果），有缓存。

	Args:
	    cache: 可选的外部缓存字典（帧索引 → 候选值集合）。
	           绑定到 Pipeline 实例生命周期以避免内存泄漏。
	"""
	cache = cache if cache is not None else {}
	# 使用帧图像的轻量哈希作为缓存键（仅首 256 字节避免大数组拷贝）
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

	gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
	h, w = gray.shape[:2]
	if h <= 0 or w <= 0:
		return candidates

	# 变体 1: 灰度 h=32（与主 OCR h=24 不同，约 12% 概率产生不同结果）
	scale = 32.0 / h if h > 0 else 1.0
	proc = cv2.resize(gray, (max(1, int(w * scale)), 32))
	res = ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
	sv, rt = extract_speed_value(res)
	if sv is not None and sv <= max_speed_kmh:
		candidates.add(float(sv))

	if cache_key is not None:
		cache[cache_key] = candidates
	return candidates


def _interp_candidate(i: int, rows: list, anchors: set, times: list, max_speed_kmh: float) -> float | None:
	"""计算帧 i 在左右锚点间的线性插值估计。"""
	n = len(rows)
	la, ra = _find_neighbor_anchors(i, n, anchors)
	if la is not None and ra is not None:
		lv = rows[la][2]; rv = rows[ra][2]
		lt = rows[la][0]; rt = rows[ra][0]
		total_dt = max(rt - lt, 0.001)
		frac = (times[i] - lt) / total_dt
		val = lv + (rv - lv) * frac
		if 0 <= val <= max_speed_kmh:
			return val
	return None


def _score_candidate(val: float, i: int, rows: list, anchors: set, error_set: set, times: list, max_speed_kmh: float, max_accel_mps2: float) -> float:
	"""阶段 3：对候选值评分。权重根据到最近锚点的距离动态调整。

	锚点密集区（<10 帧）→ 锚点权重主导（0.5）；锚点稀疏区（>50 帧）→ 邻居和平滑权重上升。
	"""
	import math
	n = len(rows)

	# 1. neighbor_score
	neighbor_score = 0.0
	count = 0
	for j in range(i - 1, max(i - 4, -1), -1):
		if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
			continue
		dt = max(times[i] - times[j], 0.001)
		max_dv = max_accel_mps2 * dt * MPS_TO_KMH
		dv = abs(val - rows[j][2])
		neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
		count += 1
		break
	for j in range(i + 1, min(i + 5, n)):
		if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
			continue
		dt = max(times[j] - times[i], 0.001)
		max_dv = max_accel_mps2 * dt * MPS_TO_KMH
		dv = abs(rows[j][2] - val)
		neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
		count += 1
		break
	neighbor_score = neighbor_score / max(count, 1)

	# 2. anchor_score — 阈值按到最近锚点的时间缩放
	anchor_score = 0.0
	interp = _interp_candidate(i, rows, anchors, times, max_speed_kmh)
	if interp is not None:
		dev = abs(val - interp)
		# 时间缩放阈值：1 帧 (0.017s) 处 ~13 km/h，而非固定 252 km/h
		la2, ra2 = _find_neighbor_anchors(i, n, anchors)
		time_to_anchor = 999.0
		if la2 is not None:
			time_to_anchor = min(time_to_anchor, times[i] - times[la2])
		if ra2 is not None:
			time_to_anchor = min(time_to_anchor, times[ra2] - times[i])
		threshold = max(5.0, max_accel_mps2 * max(time_to_anchor, 0.05) * MPS_TO_KMH * 3.0)
		anchor_score = max(0.0, 1.0 - dev / threshold)

	# 3. smoothness_score
	smoothness_score = 0.5
	if i >= 1 and i + 1 < n:
		prev_v = None
		for j in range(i - 1, max(i - 3, -1), -1):
			if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
				prev_v = rows[j][2]; break
		next_v = None
		for j in range(i + 1, min(i + 4, n)):
			if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
				next_v = rows[j][2]; break
		if prev_v is not None and next_v is not None:
			expected = (prev_v + next_v) / 2.0
			dev2 = abs(val - expected)
			smoothness_score = max(0.0, 1.0 - dev2 / max(10.0, max_accel_mps2 * 1.8 * MPS_TO_KMH))

	# ── 动态权重：锚点权重随距离衰减 ──
	la, ra = _find_neighbor_anchors(i, n, anchors)
	dist_to_anchor = n  # default large
	if la is not None:
		dist_to_anchor = min(dist_to_anchor, i - la)
	if ra is not None:
		dist_to_anchor = min(dist_to_anchor, ra - i)
	# 锚点权重从 0.5 (dist=0) 衰减到 0.1 (dist→∞)，衰减常数 ~20 帧
	w_anchor = 0.1 + 0.4 * math.exp(-dist_to_anchor / 20.0)
	w_remaining = 1.0 - w_anchor
	w_neighbor = w_remaining * 0.55
	w_smoothness = w_remaining * 0.45

	return neighbor_score * w_neighbor + anchor_score * w_anchor + smoothness_score * w_smoothness


def _fill_unrecoverable(rows: list, anchors: set, error_set: set, times: list, max_speed_kmh: float, max_accel_mps2: float,
						progress_fn: "Callable | None" = None) -> None:
	"""阶段 5：对无法通过重 OCR 修复的帧，从左到右传播可信值。"""
	n = len(rows)
	sorted_errors = sorted(i for i in error_set if i not in anchors)
	total = len(sorted_errors)
	progress_done = 0
	for i in sorted_errors:
		# Left reference: any frame with valid speed (including previously filled)
		la = None
		for j in range(i - 1, -1, -1):
			if 0 <= rows[j][2] <= max_speed_kmh:
				la = j; break
		if la is None:
			continue
		lv = rows[la][2]; lt = rows[la][0]

		# Right reference: nearest anchor
		ra = None
		for j in range(i + 1, n):
			if j in anchors and 0 <= rows[j][2] <= max_speed_kmh:
				ra = j; break

		left_dt = max(times[i] - lt, 0.001)
		left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
		if ra is not None:
			rv = rows[ra][2]; rt = rows[ra][0]
			right_dt = max(rt - times[i], 0.001)
			right_max_dv = max_accel_mps2 * right_dt * MPS_TO_KMH
			# Range reachable from left anchor
			lo = max(0.0, lv - left_max_dv)
			hi = min(max_speed_kmh, lv + left_max_dv)
			# Also must be reachable from right anchor
			lo = max(lo, rv - right_max_dv)
			hi = min(hi, rv + right_max_dv)
			# Linear interp clamped to reachable range
			interp = lv + (rv - lv) * (left_dt / max(left_dt + right_dt, 0.001))
			val = max(lo, min(hi, interp))
		else:
			val = max(0.0, min(max_speed_kmh, lv + left_max_dv))
		rows[i][2] = val
		if rows[i][3] == Flag.RAW:
			rows[i][3] = Flag.FILL_INTERP

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)


# ═══════════════════════════════════════════════════════════════
# 置信度评分 — 用于聚焦人工审核
# ═══════════════════════════════════════════════════════════════

def compute_confidence(rows: list, observations: list, max_speed: float,
					   max_accel: float) -> list[dict]:
	"""计算每帧置信度 (0-100)，返回 [{index, score, is_corrected, reason}, ...]。

	评分维度:
	- OCR 偏差: 原始 OCR 值与纠错后值的差 (权重 0.3)
	- 邻帧加速度: 与前后帧的加速度是否超限 (权重 0.4)
	- 纠错标记: flag=1 惩罚 (权重 1.0, -30分)
	- 局部平滑: SG 滤波偏差 (权重 0.2)
	"""
	from ocr_engine import _savgol_filter_np
	n = len(rows)
	vals = [r[2] for r in rows]
	flags = [r[3] for r in rows]

	# SG 平滑曲线
	win = min(11, n - 2)
	if win >= 5:
		if win % 2 == 0:
			win += 1
		try:
			smoothed = _savgol_filter_np(np.array(vals), win, min(3, win - 1))
		except Exception:
			smoothed = vals
	else:
		smoothed = vals

	confidences = []
	for i in range(n):
		score = 100.0
		reasons = []

		cur = vals[i]
		if cur < 0 or cur > max_speed:
			confidences.append({'index': i, 'score': 0, 'is_corrected': Flag.is_corrected(flags[i]),
								'speed': cur, 'reason': '速度超出范围'})
			continue

		# OCR 偏差
		if i < len(observations):
			obs = observations[i]
			ocr_val = obs.raw_speed_kmh if obs.raw_speed_kmh >= 0 else None
			if ocr_val is not None and ocr_val > 0:
				dev = abs(ocr_val - cur) / max(max_speed, 1.0) * 100
				score -= 0.3 * dev
				if dev > 5:
					reasons.append(f'OCR偏差{dev:.0f}%')

		# 邻帧加速度
		if i > 0 and vals[i - 1] >= 0:
			dt = max(rows[i][0] - rows[i - 1][0], 0.001)
			accel = abs(cur - vals[i - 1]) / dt / MPS_TO_KMH
			if accel > max_accel:
				penalty = 0.4 * min(40, (accel / max_accel - 1) * 50)
				score -= penalty
				reasons.append(f'前向加速度{accel:.0f}m/s²')

		if i + 1 < n and vals[i + 1] >= 0:
			dt = max(rows[i + 1][0] - rows[i][0], 0.001)
			accel = abs(vals[i + 1] - cur) / dt / MPS_TO_KMH
			if accel > max_accel:
				penalty = 0.4 * min(40, (accel / max_accel - 1) * 50)
				score -= penalty
				r = f'后向加速度{accel:.0f}m/s²'
				if r not in reasons:
					reasons.append(r)

		# 纠错标记
		if Flag.is_corrected(flags[i]):
			score -= 30
			reasons.append('auto-corrected')

		# SG 平滑偏差
		if win >= 5:
			sg_dev = abs(cur - smoothed[i]) / max(max_speed, 1.0) * 100
			score -= 0.2 * sg_dev
			if sg_dev > 10:
				reasons.append(f'SG偏差{sg_dev:.0f}%')

		score = max(0.0, min(100.0, score))
		confidences.append({'index': i, 'score': round(score, 1),
							'is_corrected': Flag.is_corrected(flags[i]), 'speed': cur,
							'reason': reasons[0] if reasons else '正常'})

	return confidences


def find_problem_segments(confidences: list[dict], min_score: float = 70.0,
						  min_gap: int = 3, min_segment_len: int = 3) -> list[dict]:
	"""将低置信度连续帧聚合成问题段。

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
				if v > 0 and v_prev > 0 and abs(v - v_prev) > 10:
					suggested.add(fi)
				if v > 0 and v_next > 0 and abs(v_next - v) > 10:
					suggested.add(fi + 1)
			if len(suggested) >= 8:  # 最多 8 个建议帧
				break
		seg['suggested'] = sorted(suggested)

	segments.sort(key=lambda s: s['avg_score'])
	return segments
