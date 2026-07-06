"""RaceVideoToLog PySide6 GUI — 主窗口 + 导出线程。

分离自 RaceVideoToLog.py，包含所有 GUI 相关逻辑。
"""
from __future__ import annotations

import csv
from pathlib import Path
import traceback

import cv2
import numpy as np

from PySide6.QtWidgets import (
	QMainWindow, QWidget, QStackedWidget,
	QDialog, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import (
	QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QShortcut,
)

import ocr_engine
from ocr_engine import *  # noqa: F403, F405
from gui_analysis import AnalysisTab
from gui_review import ReviewDialog

from qfluentwidgets import (setTheme, Theme,
	PushButton, PrimaryPushButton, LineEdit, ComboBox, CheckBox, RadioButton,
	BodyLabel, StrongBodyLabel, CaptionLabel, CardWidget, Slider, ProgressBar, CompactSpinBox, Pivot)





class _ExportThread(QThread):
	"""后台执行 OCR + 纠错 + CSV 写出。

	通过 Qt Signals 与 GUI 线程通信。
	"""
	_progress = Signal(str, float)   # (message, pct 0–100)
	_finished = Signal(str)           # mode: "auto" | "baseline" | "review"
	_review_data = Signal(list, list, list, list)  # rows, obs, conf, segs
	_error = Signal(str)              # error message
	_cancelled = Signal()

	def __init__(self, app: "RaceVideoToLogApp", output_path: Path,
			region: tuple, max_speed_kmh: float, max_accel_mps2: float,
			frame_div: int, target_h: float, pad_px: float, buffer_size: int,
			parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.app = app
		self._output_path = output_path
		self._region = region
		self._max_speed_kmh = max_speed_kmh
		self._max_accel_mps2 = max_accel_mps2
		self._frame_div = frame_div
		self._target_h = target_h
		self._pad_px = pad_px
		self._buffer_size = buffer_size
		self._cancel_flag = False

	def run(self) -> None:
		try:
			mode = self.app.correction_mode
			self._emit_progress("加载 OCR 引擎...", 2.0)
			ocr = self.app.get_ocr_engine()

			self._emit_progress("OCR 引擎就绪, 解码视频帧...", 5.0)
			raw_frames, total_frames = self._extract_frames()
			if total_frames == 0:
				self._error.emit("未从视频中读取到任何帧。")
				return

			observations = self._run_ocr(raw_frames, ocr, total_frames)
			if not observations:
				self._error.emit("未识别到任何速度数据。")
				return

			if mode == "auto":
				self._run_auto_anchor(observations, raw_frames, ocr, total_frames)
			else:
				self._run_focused_review(observations, raw_frames, ocr, total_frames)
		except _CancelExport:
			self._cancelled.emit()
		except Exception as exc:
			traceback.print_exc()
			self._error.emit(str(exc))

	def _check_cancel(self) -> None:
		if self._cancel_flag:
			raise _CancelExport()

	def _emit_progress(self, msg: str, pct: float) -> None:
		self._progress.emit(msg, pct)

	# ── 帧提取 ──

	def _extract_frames(self) -> tuple[list, int]:
		assert self.app.video_path is not None
		assert self.app.metadata is not None

		cap = cv2.VideoCapture(str(self.app.video_path))
		x1, y1, x2, y2 = self._region
		frame_step = max(1, self._frame_div)
		total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

		f_start = _parse_int_or_none(self.app.frame_start_edit.text())
		f_end = _parse_int_or_none(self.app.frame_end_edit.text())
		_end_limit = f_end if f_end is not None else total_video_frames

		raw_frames: list[tuple[float, np.ndarray]] = []
		fi = 0
		while fi < total_video_frames:
			self._check_cancel()
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
			if fi % max(1, frame_step * 200) == 0:
				pct = 5.0 + 15.0 * (fi / max(_end_limit, 1))
				self._emit_progress(f"解码视频: {fi}/{_end_limit} 帧", pct)
			timestamp = fi / self.app.metadata.fps if self.app.metadata.fps > 0 else 0.0
			crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
			raw_frames.append((timestamp, crop))
			fi += 1
		cap.release()
		return raw_frames, len(raw_frames)

	# ── OCR ──

	def _run_ocr(self, raw_frames: list, ocr: "RapidOCR", total: int) -> list:
		import threading
		from queue import Queue

		max_speed = self._max_speed_kmh
		speed_format = self.app.speed_format
		n = len(raw_frames)
		buf_size = max(2, self._buffer_size * 2)
		q: Queue = Queue(maxsize=buf_size)
		errors: list[Exception] = []

		def _producer() -> None:
			try:
				for ts, crop in raw_frames:
					proc = self._preprocess(crop, self._target_h, self._pad_px)
					q.put((ts, crop, proc))
				q.put(None)
			except Exception as e:
				errors.append(e)
				q.put(None)

		t = threading.Thread(target=_producer, daemon=True)
		t.start()

		observations: list[SpeedObservation] = []
		done = 0
		while True:
			self._check_cancel()
			item = q.get()
			if item is None:
				break
			ts, crop, proc = item
			ocr_result, _ = ocr(proc)
			sv, rt = extract_speed_value(ocr_result)
			if sv is None:
				proc_fb = self._preprocess_fb(crop, self._target_h, self._pad_px)
				ocr_result, _ = ocr(proc_fb)
				sv, rt = extract_speed_value(ocr_result)
			if sv is None:
				sv, rt = ocr_digital_fallback(ocr, crop, max_speed)
			if sv is not None and rt is not None:
				observations.append(SpeedObservation(
					timestamp=ts,
					raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
					raw_text=rt))
			else:
				observations.append(SpeedObservation(ts, -1.0, ""))
			done += 1
			if done % 10 == 0:
				pct = (done / total * 90.0) + 5.0
				self._emit_progress(
					f"[{ocr_engine._gpu_backend}] OCR: {done}/{total}", pct)
		t.join()
		if errors:
			raise errors[0]
		return observations

