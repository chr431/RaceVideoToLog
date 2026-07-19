"""统一处理流水线 — GUI 和 CLI 共用。
运行在调用者线程中（GUI 应在原生 threading.Thread 中调用以避免 QThread 性能损失）。
"""
from __future__ import annotations
import csv
import logging
import time as _time
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from rapidocr_onnxruntime import RapidOCR
from ocr_engine import (
	auto_select_anchors, clamp_region, compute_video_hash,
	extract_speed_value, SpeedObservation, Flag,
	SOURCE_TO_KMH, _parse_int_or_none,
	_reset_backend, _select_backend, _get_model_kwargs,
)
from config import MPS_TO_KMH
from correction import correct_with_anchors, compute_confidence, find_problem_segments
from gpu_setup import get_gpu_backend

logger = logging.getLogger("RaceVideoToLog.pipeline")


def _preprocess_standard(crop: np.ndarray, target_h: float, pad: float) -> np.ndarray:
	"""标准预处理：纯 resize + 填充。保留 BGR 通道以利用颜色信息。"""
	h, w = crop.shape[:2]
	th = max(8.0, float(target_h))
	scale = th / float(h) if h > 0 else 1.0
	if abs(scale - 1.0) > 0.02:
		resized = cv2.resize(crop, (max(1, int(w * scale)), int(th)))
	else:
		resized = crop
	pad_int = int(pad)
	if pad_int > 0:
		resized = cv2.copyMakeBorder(resized, pad_int, pad_int, pad_int, pad_int,
									 cv2.BORDER_REPLICATE)
	return resized


ProgressFn = Callable[[str, float], None]


class ProcessingPipeline:
	"""统一处理流水线。

	同时支持自动锚点模式和人工辅助（两段式）模式。
	通过 progress_cb 回调报告进度，与 UI 框架解耦。
	"""

	def __init__(self, video_path: str | Path, roi: tuple[int, int, int, int],
				 max_speed: float, max_accel: float,
				 frame_div: int, target_h: float, pad: float, buffer_size: int,
				 backend: str, ocr_model: str, speed_format: str,
				 frame_start: str = "", frame_end: str = "",
				 progress_cb: ProgressFn | None = None,
				 reocr_model: str | None = None):
		self._video_path = Path(video_path)
		self._roi = roi
		self._max_speed = max_speed
		self._max_accel = max_accel
		self._frame_div = frame_div
		self._target_h = target_h
		self._pad = pad
		self._buffer_size = buffer_size
		self._reocr_model = reocr_model  # None = 使用 ocr_model
		self._backend = backend
		self._ocr_model = ocr_model
		self._speed_format = speed_format
		self._frame_start = frame_start
		self._frame_end = frame_end
		self._progress = progress_cb

		# 状态
		self._ocr: RapidOCR | None = None
		self._reocr: RapidOCR | None = None  # 重 OCR 引擎
		self._raw_frames: list[tuple[float, np.ndarray]] = []
		self._observations: list[SpeedObservation] = []
		self._rows: list[list] = []
		self._anchor_indices: set[int] = set()
		self._segments: list[dict] = []
		# ── 性能计时 ──
		self._timing: dict[str, float] = {}
		# ── 重 OCR 缓存（绑定到 Pipeline 实例生命周期）──
		self._reocr_cache: dict[int, set[float]] = {}
		# ── 调试模式：在 CSV 中输出原始 OCR 文本 ──
		self._debug_raw_text: bool = False

	# ═══════════════ 公开接口 ═══════════════

	def run_auto(self, output_path: str | Path) -> None:
		"""自动锚点模式：完整流水线 → 写 CSV。"""
		t_total = _time.perf_counter()
		self._emit("加载 OCR 引擎...", 1.0)
		self._ensure_ocr()
		self._run_ocr()

		if not self._observations:
			raise RuntimeError("未识别到任何速度数据。")

		if self._max_speed <= 0:
			self._emit("跳过纠错（原始OCR输出）...", 95.0)
			self._rows = []
			for obs in self._observations:
				self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, Flag.RAW])
			self._integrate_distance()
			self._write_csv(self._rows, Path(output_path), auto_anchor=True)
		else:
			self._run_correction(Path(output_path), skip_fill=False)
		self._timing["total"] = _time.perf_counter() - t_total
		logger.info("流水线完成: 总计 %.1fs (%s)",
					 self._timing["total"],
					 ", ".join(f"{k}={v:.1f}s" for k, v in self._timing.items()))
		self._emit("完成", 100.0)

	def run_review_pass1(self, output_path: str | Path | None = None) -> tuple | None:
		"""人工辅助第 1 轮：OCR → 轻量纠错（仅重OCR）→ 置信度 → 问题段。

		轻量纠错：仅对错误帧重 OCR，在原始值和重 OCR 值之间选择更优者。
		不进行混淆候选、部分数字推断、插值填充，避免产生"物理合理但实际错误"的数据。
		这样问题段反映的是真正需要人工介入的帧。
		"""
		self._emit("加载 OCR 引擎...", 1.0)
		self._ensure_ocr()
		self._run_ocr()

		if not self._observations:
			raise RuntimeError("未识别到任何速度数据。")

		self._emit("锚点选择...", 85.0)
		self._anchor_indices = auto_select_anchors(
			self._observations, self._max_speed, max_accel_mps2=self._max_accel)
		if len(self._anchor_indices) < 3:
			raise RuntimeError("自动锚点选择失败：未找到足够的可靠帧。")

		# 构建初始 rows（原始 OCR + 锚点标记）
		self._rows = []
		for i, obs in enumerate(self._observations):
			self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh,
							   Flag.ANCHOR_AUTO if i in self._anchor_indices else Flag.RAW])

		# ── 轻量纠错：仅重 OCR，不混淆/推断/填充 ──
		self._emit("轻量纠错 (仅重OCR)...", 88.0)
		def _prog(done, total):
			if done % max(1, total // 3) != 0 and done != total:
				return
			self._emit(f"轻量纠错: {done}/{total} 帧", 88.0 + (done / max(total, 1)) * 4.0)
		self._rows = correct_with_anchors(
			self._rows, self._observations, self._raw_frames, self._reocr,
			self._max_speed, self._max_accel, self._anchor_indices,
			progress_fn=_prog, skip_fill=True, light_mode=True,
			reocr_cache=self._reocr_cache)

		self._emit("计算置信度...", 95.0)
		confidences = compute_confidence(self._rows, self._observations,
										 self._max_speed, self._max_accel)
		segments = find_problem_segments(confidences, min_segment_len=1)

		if segments:
			self._emit(f"发现 {len(segments)} 个问题段，等待人工审核...", 98.5)
			self._segments = segments
			return (list(self._rows), list(self._observations),
					list(self._raw_frames), confidences, segments)
		else:
			self._emit("未发现问题段，无需人工审核。", 98.5)
			self._integrate_distance()
			if output_path is not None:
				self._write_csv(self._rows, Path(output_path), auto_anchor=True)
			self._emit("完成", 100.0)
			return None

	def run_review_pass2(self, corrections: dict[int, float],
						 confirmed_segments: set[int],
						 output_path: str | Path,
						 partial_corrections: dict[int, str] | None = None) -> None:
		"""人工辅助第 2 轮：合并手动修正 → 再纠错 → 写 CSV。"""
		for fi, v in corrections.items():
			if 0 <= fi < len(self._rows):
				self._rows[fi][2] = v
				self._rows[fi][3] = Flag.ANCHOR_MANUAL
				self._anchor_indices.add(fi)

		for seg_start in confirmed_segments:
			for seg in self._segments:
				if seg["start"] == seg_start:
					for fi in range(seg["start"], seg["end"] + 1):
						if fi not in corrections and 0 <= fi < len(self._rows):
							self._rows[fi][3] = Flag.CONFIRMED_SEG
					# 仅将段首和段尾加入锚点（作为边界约束），
					# 而非所有帧，避免削弱纠错灵活性
					for fi in (seg["start"], seg["end"]):
						if fi not in self._anchor_indices:
							self._anchor_indices.add(fi)
					break

		self._correct(91.0, 7.0, skip_fill=False, reuse_anchors=True,
					  partial_corrections=partial_corrections)

		self._integrate_distance()
		self._write_csv(self._rows, Path(output_path), auto_anchor=False,
						manual_anchor=True)
		self._emit(f"人工审核完成 — {len(corrections)} 帧已修正，结果已保存。", 100.0)

	# ═══════════════ 内部 ═══════════════

	def _emit(self, msg: str, pct: float) -> None:
		if self._progress:
			self._progress(msg, pct)

	def _ensure_ocr(self) -> RapidOCR:
		if self._ocr is None:
			_reset_backend()
			self._backend_actual = _select_backend(self._backend)
			kw = _get_model_kwargs(self._ocr_model)
			self._ocr = RapidOCR(**(kw or {}))
			# 若指定了不同的重 OCR 模型，创建独立引擎
			_reocr_model = self._reocr_model or self._ocr_model
			if _reocr_model != self._ocr_model:
				kw2 = _get_model_kwargs(_reocr_model)
				self._reocr = RapidOCR(**(kw2 or {}))
			else:
				self._reocr = self._ocr
		return self._ocr

	def _correct(self, progress_base: float, progress_span: float,
				 skip_fill: bool, reuse_anchors: bool = False,
				 partial_corrections: dict[int, str] | None = None) -> None:
		"""共享纠错逻辑：锚点选择 → 构建 rows → correct_with_anchors。

		reuse_anchors=True 时使用 self._anchor_indices 和 self._rows 的已有值，
		跳过锚点选择和 rows 构建（用于 pass2 手动锚点场景）。
		"""
		if not reuse_anchors:
			self._emit("锚点选择 + 纠错...", progress_base)
			self._anchor_indices = auto_select_anchors(
				self._observations, self._max_speed, max_accel_mps2=self._max_accel)
			if len(self._anchor_indices) < 3:
				raise RuntimeError("自动锚点选择失败：未找到足够的可靠帧。")
			self._rows = []
			for i, obs in enumerate(self._observations):
				self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh,
								   Flag.ANCHOR_AUTO if i in self._anchor_indices else Flag.RAW])

		t0 = _time.perf_counter()
		self._emit("纠错: 检测误差...", progress_base + 1.0)
		corr_timing: dict[str, float] = {}
		def _prog(done: int, total: int) -> None:
			if done % max(1, total // 5) != 0 and done != total:
				return
			pct = done / max(total, 1)
			self._emit(f"纠错: {done}/{total} 帧", progress_base + 1.0 + pct * progress_span)
		self._rows = correct_with_anchors(
			self._rows, self._observations, self._raw_frames, self._reocr,
			self._max_speed, self._max_accel, self._anchor_indices,
			progress_fn=_prog, timing=corr_timing, skip_fill=skip_fill,
			partial_corrections=partial_corrections,
			reocr_cache=self._reocr_cache)
		self._timing["correction"] = _time.perf_counter() - t0
		if corr_timing.get("re_ocr", 0) > 0:
			logger.info("重OCR 耗时: %.2fs", corr_timing["re_ocr"])

	def _run_correction(self, output_path: Path, skip_fill: bool) -> None:
		"""自动锚点模式的完整纠错 + 写 CSV（调用 _correct 后继续）。"""
		self._correct(91.0, 6.0, skip_fill=skip_fill)
		t1 = _time.perf_counter()
		self._integrate_distance()
		self._write_csv(self._rows, output_path, auto_anchor=True)
		self._timing["integrate_write"] = _time.perf_counter() - t1

	def _run_ocr(self) -> None:
		"""解码 + OCR：producer 解码/预处理 → consumer 做 ONNX 推理。

		Queue 流水线重叠 I/O 与 GPU 推理。对于小 ROI 裁切 (~4ms/帧)，
		单 ONNX 会话已使 GPU 饱和，多消费者增加协调开销而无收益。
		ORT 1.27 支持多会话并发（不再 crash），需要时可启用。
		"""
		import threading
		from queue import Queue

		ocr = self._ocr
		speed_format = self._speed_format
		target_h = self._target_h
		pad = self._pad
		frame_step = max(1, self._frame_div)
		t_start = _time.perf_counter()

		cap = cv2.VideoCapture(str(self._video_path))
		total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		x1, y1, x2, y2 = clamp_region(*self._roi, w, h)
		f_start = _parse_int_or_none(self._frame_start)
		f_end = _parse_int_or_none(self._frame_end)
		_end_limit = f_end if f_end is not None else total_video_frames

		self._raw_frames = []
		buf_size = max(2, self._buffer_size * 2)
		q: Queue = Queue(maxsize=buf_size)
		errors: list[Exception] = []

		def _producer() -> None:
			try:
				fi = 0
				while fi < total_video_frames:
					if fi >= _end_limit:
						break
					if f_start is not None and fi < f_start:
						cap.grab(); fi += 1; continue
					if fi % frame_step != 0:
						cap.grab(); fi += 1; continue
					if not cap.grab():
						break
					ok, frame = cap.retrieve()
					if not ok or frame is None:
						break
					ts = fi / fps if fps > 0 else 0.0
					crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
					self._raw_frames.append((ts, crop))
					proc = _preprocess_standard(crop, target_h, pad)
					q.put((ts, proc))
					fi += 1
				q.put(None)
			except Exception as e:
				errors.append(e)
				q.put(None)
			finally:
				cap.release()

		t = threading.Thread(target=_producer, daemon=True)
		t.start()

		observations: list[SpeedObservation] = []
		done = 0
		est_total = (_end_limit - (f_start or 0)) // frame_step
		while True:
			item = q.get()
			if item is None:
				break
			ts, proc = item
			ocr_result, _ = ocr(proc)
			sv, rt = extract_speed_value(ocr_result)
			if sv is not None and rt is not None:
				observations.append(SpeedObservation(
					timestamp=ts,
					raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
					raw_text=rt))
			else:
				observations.append(SpeedObservation(ts, -1.0, ""))
			done += 1
			if done % 10 == 0 or done <= 3 or done == est_total:
				pct = 3.0 + (done / max(est_total, 1)) * 87.0
				self._emit(f"[{get_gpu_backend()}] OCR: {done}/{est_total}", pct)
		t.join()
		if errors:
			raise errors[0]
		self._observations = observations
		self._timing["ocr"] = _time.perf_counter() - t_start
		logger.info("OCR 完成: %d 帧, 耗时 %.1fs",
					 len(observations), self._timing["ocr"])

	def _integrate_distance(self) -> None:
		dist = 0.0; prev_t = prev_v = None
		for r in self._rows:
			v = r[2] / MPS_TO_KMH if r[2] >= 0 else 0.0
			if prev_t is not None and prev_v is not None:
				dt = r[0] - prev_t
				if dt > 0:
					dist += (prev_v + v) * 0.5 * dt
			prev_t, prev_v = r[0], v
			r[1] = dist

	def _write_csv(self, rows: list, output_path: Path,
				   auto_anchor: bool = True, manual_anchor: bool = False) -> None:
		vhash = compute_video_hash(self._video_path)
		r = self._roi
		tag = "manual_anchor" if manual_anchor else ("auto_anchor" if auto_anchor else "")
		# ── 统计信息 ──
		n_total = len(rows)
		n_anchors = sum(1 for row in rows if Flag.is_anchor(row[3]))
		n_corrected = sum(1 for row in rows if Flag.is_corrected(row[3]))
		timing_str = ", ".join(f"{k}={v:.1f}s" for k, v in self._timing.items())
		with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write("# RaceVideoToLog v2.3.0\n")
			fh.write(f"# video_hash={vhash}, video={self._video_path.name}\n")
			fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}"
					 f", frame_start={self._frame_start or ''}"
					 f", frame_end={self._frame_end or ''}\n")
			fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}"
					 f", div={self._frame_div}, target_h={self._target_h}"
					 f", pad={self._pad}, buffer={self._buffer_size}\n")
			fh.write(f"# backend={self._backend_actual}, model={self._ocr_model}")
			reocr_info = f", reocr_model={self._reocr_model}" if self._reocr_model and self._reocr_model != self._ocr_model else ""
			fh.write(f"{reocr_info}\n")
			if tag:
				fh.write(f"# {tag}=1\n")
			fh.write(f"# stats: total={n_total}, anchors={n_anchors},"
					 f" corrected={n_corrected}\n")
			if timing_str:
				fh.write(f"# timing: {timing_str}\n")
			w = csv.writer(fh)
			# 调试模式：增加 raw_text 列
			if self._debug_raw_text and self._observations:
				for i, row in enumerate(rows):
					raw_text = (self._observations[i].raw_text
								if i < len(self._observations) else "")
					w.writerow([f"{row[0]:.2f}", f"{row[1]:.2f}",
							   f"{row[2]:.2f}", str(row[3]), raw_text])
			else:
				for row in rows:
					w.writerow([f"{row[0]:.2f}", f"{row[1]:.2f}",
							   f"{row[2]:.2f}", str(row[3])])
