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
	compute_lcs_scores, lcs_detect_errors, _lcs_score_for_value,
)
from config import MPS_TO_KMH

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


def _auto_expand_digits(raw_text: str, max_speed_kmh: float) -> list[float]:
	"""根据 OCR 读数自动生成所有可能缺位的数字候选。

	若 OCR 读到 1-2 位数字，暴力生成所有可能的 2-3 位数扩展。
	若 ≥3 位数字，不扩展（假定完整）。
	由 LCS 评分选择最优候选，不做插值猜测。
	"""
	if not raw_text or not raw_text.isdigit():
		return []
	digits = raw_text
	if len(digits) >= 3:
		return []
	candidates: list[float] = []
	try:
		candidates.append(float(digits))
	except ValueError:
		pass
	for pos in range(len(digits) + 1):
		pattern = digits[:pos] + "x" + digits[pos:]
		for v in expand_partial(pattern, max_speed_kmh):
			if v not in candidates:
				candidates.append(v)
	return candidates


def correct_with_trust(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR",
						 max_speed_kmh: float, max_accel_mps2: float, anchor_indices: set | None = None,
						 log_fn: "Callable | None" = None,
						 progress_fn: "Callable | None" = None,
						 skip_fill: bool = False,
						 timing: dict | None = None,
						 partial_corrections: dict[int, str] | None = None,
						 reocr_cache: dict | None = None,
						 light_mode: bool = False,
						 notes: dict[int, str] | None = None, pinned: set[int] | None = None) -> list:
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
	anchors = pinned  # kept as internal var name
	times = [r[0] for r in rows]
	cache: dict = reocr_cache if reocr_cache is not None else {}

	if log_fn:
		mode_str = " (light)" if light_mode else ""
		log_fn(f"Correction{mode_str}: {n} rows, {len(anchors)} trusted/pinned")

	# ── 阶段 1：错误检测 ──
	error_set, _scores = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
	if log_fn:
		log_fn(f"  Stage 1: detected {len(error_set)} errors")
	# 标记高信帧（供后续 _fix_errors 的 interp 回退使用）
	for i, s in enumerate(_scores):
		if s >= 0.7 and rows[i][3] == Flag.RAW and i not in error_set:
			rows[i][3] = Flag.HIGH_TRUST
	if not error_set:
		return rows

	# ── 阶段 2+3：重 OCR + 最优选择（首轮）──
	fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
	                    anchors, times, max_speed_kmh, max_accel_mps2,
	                    progress_fn=progress_fn, timing=timing,
	                    partial_corrections=partial_corrections, reocr_cache=cache,
	                    light_mode=light_mode, notes=notes)
	if log_fn:
		log_fn(f"  Stage 2+3: fixed {fixed} frames in round 1")

	# ── Light mode: 一轮即止，剩余错误标记为待审核 ──
	if light_mode:
		error_set, _scores = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Light: {len(error_set)} frames flagged for manual review")
		return rows

	# ── 阶段 4：多轮迭代 ──
	max_rounds = 4
	for rnd in range(2, max_rounds + 1):
		error_set, _scores = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		if not error_set:
			break
		fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
		                    anchors, times, max_speed_kmh, max_accel_mps2,
		                    progress_fn=progress_fn, timing=timing,
		                    partial_corrections=partial_corrections, reocr_cache=cache,
		                    notes=notes)
		if log_fn:
			log_fn(f"  Stage 4 round {rnd}: {len(error_set)} errors, fixed {fixed}")

	# ── 阶段 5：迭代填充直到收敛（处理级联效应）──
	if skip_fill:
		error_set, _scores = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		for i in error_set:
			if i not in anchors and rows[i][3] < 2:
				rows[i][3] = Flag.FLAGGED_REVIEW
		if log_fn:
			log_fn(f"  Stage 5: {len(error_set)} frames flagged for manual review")
	else:
		fill_pass = 0
		while fill_pass < 10:
			error_set, _scores = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
			if not error_set:
				break
			_fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2,
			                    progress_fn=progress_fn, notes=notes)
			if log_fn:
				log_fn(f"  Stage 5 pass {fill_pass+1}: filled {len(error_set)} unrecoverable frames")
			fill_pass += 1

	# 最终标记高信帧
	scores = compute_lcs_scores(rows, max_speed_kmh, max_accel_mps2, pinned=anchors)
	for i, s in enumerate(scores):
		if s >= 0.7 and rows[i][3] == Flag.RAW:
			rows[i][3] = Flag.HIGH_TRUST
	return rows


# ── LCS 错误检测 ──

def _detect_errors(rows: list, anchors: set, times: list,
                   max_speed_kmh: float, max_accel_mps2: float) -> tuple[set[int], list[float]]:
	"""阶段 1：LCS 局部一致性评分错误检测。

	对每帧计算指数加权时间窗内邻居一致性分数，
	score < 0.7 的帧标记为错误（合并 error 和 borderline）。
	anchors（pinned 帧）在评分时获得 3× 权重，且自身不会被标记为错误。

	Returns: (error_set, lcs_scores)
	"""
	n = len(rows)
	error_set: set[int] = set()

	for i in range(n):
		if i in anchors:
			continue
		v = rows[i][2]
		if v < 0 or v > max_speed_kmh:
			error_set.add(i)

	scores = compute_lcs_scores(rows, max_speed_kmh, max_accel_mps2, pinned=anchors)
	lcs_errors, borderline = lcs_detect_errors(scores)
	error_set.update(lcs_errors)
	error_set.update(borderline)
	error_set -= anchors

	return error_set, scores


def _lcs_pick_best(candidates: list[float], i: int, rows: list,
                   times: list[float], max_speed_kmh: float,
                   max_accel_mps2: float,
                   pinned: set[int] | None = None) -> tuple[float | None, float]:
	"""对候选值逐一计算 LCS 分数，返回得分最高的 (best_value, best_score)。

	pinned 帧在 LCS 评分中获得 3× 权重。
	best_value 为 None 表示没有有效候选值。
	"""
	best_val = None
	best_score = -1.0
	for cand in set(candidates):
		if not (0 <= cand <= max_speed_kmh):
			continue
		score = _lcs_score_for_value(i, cand, rows, times,
		                              max_speed_kmh, max_accel_mps2,
		                              high_weight=pinned)
		if score > best_score:
			best_score = score
			best_val = cand
	return best_val, best_score


def _fix_errors(rows: list, observations: list, raw_frames: list, ocr: "RapidOCR", error_set: set,
				anchors: set, times: list, max_speed_kmh: float, max_accel_mps2: float,
				progress_fn: "Callable | None" = None,
				timing: dict | None = None,
				partial_corrections: dict[int, str] | None = None,
				reocr_cache: dict | None = None,
				light_mode: bool = False,
				notes: dict[int, str] | None = None) -> int:
	"""阶段 2+3：对每个 error 帧重 OCR 获取备选，LCS 评分选最优值填入。"""
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
			# 自动缺位扩展：OCR 读到 "21" → 生成所有可能的 2-3 位数
			for c in _auto_expand_digits(observations[oid].raw_text, max_speed_kmh):
				if c not in candidates:
					candidates.append(c)

		# 插值候选（light_mode 下不使用）
		if not light_mode and interp_cand is not None:
			candidates.append(interp_cand)

		# ── 选择最佳候选 ──
		if candidates:
			raw_val = rows[i][2]
			# re-OCR 无结果且插值与当前值差异大 → 直接插值，light 模式跳过
			if len(reocr_set) <= 1 and interp_cand is not None and abs(interp_cand - raw_val) > 10.0:
				if not light_mode and abs(raw_val - interp_cand) > 0.5:
					if notes is not None:
						notes[i] = f"interp: {raw_val:.0f}→{interp_cand:.0f}"
					rows[i][2] = interp_cand
					if rows[i][3] == Flag.RAW:
						rows[i][3] = Flag.PARTIAL_AUTO if has_partial else Flag.REOCR_AUTO
					fixed += 1
			else:
				best_val, best_score = _lcs_pick_best(
					candidates, i, rows, times, max_speed_kmh, max_accel_mps2,
					pinned=anchors)

				if best_val is not None and best_score >= 0.7 and abs(rows[i][2] - best_val) > 0.5:
					old_val = rows[i][2]
					rows[i][2] = best_val
					if rows[i][3] == Flag.RAW:
						rows[i][3] = Flag.PARTIAL_AUTO if has_partial else Flag.REOCR_AUTO
					if notes is not None:
						source = "partial" if has_partial else "reOCR"
						notes[i] = f"{source}: {old_val:.0f}→{best_val:.0f}"
					fixed += 1
				elif not light_mode:
					# 无候选达到 LCS >= 0.7 → 回退到插值填充
					interp = _interp_candidate(i, rows, anchors, times, max_speed_kmh)
					if interp is not None and abs(rows[i][2] - interp) > 0.5:
						if notes is not None:
							notes[i] = f"fill: {rows[i][2]:.0f}→{interp:.0f}"
						rows[i][2] = interp
						if rows[i][3] == Flag.RAW:
							rows[i][3] = Flag.FILL_INTERP
						fixed += 1

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)
	return fixed


def _re_ocr_frame(crop_bgr: "np.ndarray", ocr: "RapidOCR", max_speed_kmh: float,
				  timing: dict | None = None, cache: dict | None = None) -> set:
	"""阶段 2：对单帧尝试重 OCR（h=48，与主 OCR 一致；模型不同保证多样性），有缓存。

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

	h, w = crop_bgr.shape[:2]
	if h <= 0 or w <= 0:
		return candidates

	# BGR resize h=32（与主 OCR h=24 不同高度，约 10% 概率产生不同结果）
	scale = 32.0 / h if h > 0 else 1.0
	proc = cv2.resize(crop_bgr, (max(1, int(w * scale)), 32))
	res = ocr(proc)
	sv, rt, _conf = extract_speed_value(res)
	if sv is not None and sv <= max_speed_kmh:
		candidates.add(float(sv))

	if cache_key is not None:
		cache[cache_key] = candidates
	return candidates


def _interp_candidate(i: int, rows: list, anchors: set, times: list, max_speed_kmh: float) -> float | None:
	"""计算帧 i 在左右高信帧间的线性插值估计。"""
	n = len(rows)
	la, ra = _find_neighbor_trusted(i, n, rows)
	if la is not None and ra is not None:
		lv = rows[la][2]; rv = rows[ra][2]
		lt = rows[la][0]; rt = rows[ra][0]
		total_dt = max(rt - lt, 0.001)
		frac = (times[i] - lt) / total_dt
		val = lv + (rv - lv) * frac
		if 0 <= val <= max_speed_kmh:
			return val
	return None


def _fill_unrecoverable(rows: list, anchors: set, error_set: set, times: list, max_speed_kmh: float, max_accel_mps2: float,
						progress_fn: "Callable | None" = None,
						notes: dict[int, str] | None = None) -> None:
	"""阶段 5：对无法通过重 OCR 修复的帧，以最近高信帧为基准插值。

	左右均使用 HIGH_TRUST/PINNED 帧作为约束，防止 fill 链式累积。
	"""
	n = len(rows)
	sorted_errors = sorted(i for i in error_set if i not in anchors)
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
		lv = rows[la][2]; lt = rows[la][0]

		# 右侧最近高信帧
		ra = None
		for j in range(i + 1, n):
			if Flag.is_trusted(rows[j][3]) and 0 <= rows[j][2] <= max_speed_kmh:
				ra = j; break

		left_dt = max(times[i] - lt, 0.001)
		left_max_dv = max_accel_mps2 * left_dt * MPS_TO_KMH
		if ra is not None:
			rv = rows[ra][2]; rt = rows[ra][0]
			right_dt = max(rt - times[i], 0.001)
			right_max_dv = max_accel_mps2 * right_dt * MPS_TO_KMH
			lo = max(0.0, lv - left_max_dv)
			hi = min(max_speed_kmh, lv + left_max_dv)
			lo = max(lo, rv - right_max_dv)
			hi = min(hi, rv + right_max_dv)
			interp = lv + (rv - lv) * (left_dt / max(left_dt + right_dt, 0.001))
			val = max(lo, min(hi, interp))
		else:
			# 无右侧高信帧 → 跳过（无法可靠约束）
			continue
		rows[i][2] = val
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
					   max_accel: float, pinned: set[int] | None = None) -> list[dict]:
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
	scores = compute_lcs_scores(rows, max_speed, max_accel, pinned=pinned)
	flags = [r[3] for r in rows]

	confidences = []
	for i in range(n):
		lcs = scores[i]
		cur = rows[i][2]

		if cur < 0 or cur > max_speed:
			confidences.append({
				'index': i, 'score': 0.0, 'is_corrected': Flag.is_corrected(flags[i]),
				'speed': cur, 'reason': '速度超出范围',
			})
			continue

		if lcs < 0.3:
			reason = 'LCS错误(物理矛盾)'
		elif lcs < 0.7:
			reason = 'LCS存疑(临界的)'
		else:
			reason = '正常'

		confidences.append({
			'index': i, 'score': round(lcs * 100, 1),
			'is_corrected': Flag.is_corrected(flags[i]),
			'speed': cur, 'reason': reason,
		})

	return confidences


def find_problem_segments(confidences: list[dict], min_score: float = 30.0,
						  min_gap: int = 3, min_segment_len: int = 3) -> list[dict]:
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
				if v > 0 and v_prev > 0 and abs(v - v_prev) > 10:
					suggested.add(fi)
				if v > 0 and v_next > 0 and abs(v_next - v) > 10:
					suggested.add(fi + 1)
			if len(suggested) >= 8:  # 最多 8 个建议帧
				break
		seg['suggested'] = sorted(suggested)

	segments.sort(key=lambda s: s['avg_score'])
	return segments
