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
	QApplication, QMainWindow, QWidget, QTabWidget,
	QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import (
	QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QShortcut,
)

import ocr_engine
from ocr_engine import *  # noqa: F403, F405
from gui_analysis import AnalysisTab

from qfluentwidgets import (setTheme, Theme, FluentIcon, IconWidget,
	PushButton, PrimaryPushButton, LineEdit, ComboBox, CheckBox, RadioButton,
	BodyLabel, StrongBodyLabel, CaptionLabel, CardWidget, Slider, ProgressBar)





class _ExportThread(QThread):
	"""后台执行 OCR + 纠错 + CSV 写出。

	通过 Qt Signals 与 GUI 线程通信。
	"""
	_progress = Signal(str, float)   # (message, pct 0–100)
	_finished = Signal(str)           # mode: "auto" | "baseline"
	_error = Signal(str)              # error message
	_cancelled = Signal()

	def __init__(self, app: "RaceVideoToLogApp", output_path: Path,
			region: tuple, max_speed_kmh: float, max_accel_mps2: float,
			frame_div: int, target_h: float, pad_px: float, num_workers: int,
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
		self._num_workers = num_workers
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
				self._error.emit("人工基准模式暂未迁移到 PySide6。")
				return
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
		observations: list[SpeedObservation] = []
		max_speed = self._max_speed_kmh
		speed_format = self.app.speed_format
		for idx, (ts, crop) in enumerate(raw_frames):
			self._check_cancel()
			proc = self._preprocess(crop, self._target_h, self._pad_px)
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
			if (idx + 1) % 10 == 0:
				pct = ((idx + 1) / total * 90.0) + 5.0
				self._emit_progress(f"[{ocr_engine._gpu_backend}] OCR: {idx + 1}/{total}", pct)
		return observations

	# ── 自动锚点 ──

	def _run_auto_anchor(self, observations: list, raw_frames: list,
			ocr: "RapidOCR", total_frames: int) -> None:
		self._emit_progress("正在自动识别可靠锚点...", 40.0)
		anchor_indices = auto_select_anchors(observations, self._max_speed_kmh,
			max_accel_mps2=self._max_accel_mps2)
		if len(anchor_indices) < 3:
			self._error.emit("自动锚点选择失败：未找到足够的可靠帧。")
			return

		rows = []
		for i, obs in enumerate(observations):
			rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh,
				2 if i in anchor_indices else 0])

		self._emit_progress("正在纠错...", 60.0)
		from correction import correct_with_anchors
		def _prog(done: int, total: int) -> None:
			pct = done / max(total, 1)
			self._emit_progress(f"物理纠错: {done}/{total} 帧", 60.0 + pct * 30.0)
		rows = correct_with_anchors(rows, observations, raw_frames, ocr,
			self._max_speed_kmh, self._max_accel_mps2, anchor_indices,
			progress_fn=_prog)

		# 积分距离
		dist = 0.0; prev_t = prev_v = None
		for r in rows:
			v = r[2] / 3.6
			if prev_t is not None and prev_v is not None:
				dt = r[0] - prev_t
				if dt > 0: dist += (prev_v + v) * 0.5 * dt
			prev_t, prev_v = r[0], v; r[1] = dist

		# 写出 CSV
		self._write_csv(rows)

		self._emit_progress("完成", 100.0)
		self._finished.emit("auto")

	def _write_csv(self, rows: list) -> None:
		assert self.app.video_path is not None
		vhash = compute_video_hash(self.app.video_path)
		with self._output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write("# RaceVideoToLog\n")
			fh.write(f"# video_hash={vhash}, video={self.app.video_path.name}\n")
			r = self._region
			fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self.app.speed_format}\n")
			fh.write(f"# max_speed={self._max_speed_kmh}, max_accel={self._max_accel_mps2}, "
				f"div={self._frame_div}, target_h={self._target_h}, pad={self._pad_px}, "
				f"backend={ocr_engine._gpu_backend}, model=v6_small, "
				f"workers={self._num_workers}, "
				f"frame_start={self.app.frame_start_edit.text() or ''}, "
				f"frame_end={self.app.frame_end_edit.text() or ''}, auto_anchor=1\n")
			w = csv.writer(fh)
			for r in rows:
				w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

	# ── 预处理 ──

	@staticmethod
	def _preprocess(crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		return _ExportThread._finish(gray, target_h, pad_px)

	@staticmethod
	def _preprocess_fb(crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return _ExportThread._finish(gray, target_h, pad_px)

	@staticmethod
	def _finish(gray: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		h, w = gray.shape[:2]
		th = max(8.0, float(target_h))
		scale = th / float(h) if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)))
		pad_int = int(pad_px)
		if pad_int > 0:
			gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int,
				cv2.BORDER_REPLICATE)
		return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ═══════════════════════ 主窗口 ═══════════════════════

class RaceVideoToLogApp(QMainWindow):
	"""RaceVideoToLog PySide6 主窗口。"""

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Race Video To Log")
		self.resize(1500, 920)
		self.setMinimumSize(1100, 760)

		# ── 状态变量 ──
		self.video_path: Path | None = None
		self.metadata: VideoMetadata | None = None
		self.first_frame_bgr: np.ndarray | None = None
		self.first_frame_qimg: QImage | None = None
		self._preview_cap: cv2.VideoCapture | None = None
		self._preview_frame_no: int = 0
		self._throttle_timer: QTimer | None = None
		self.ocr_engine: RapidOCR | None = None
		self.ocr_engines: list[RapidOCR] = []

		self._export_thread: _ExportThread | None = None
		self.correction_mode: str = "auto"
		self.speed_format: str = "km/h"
		self._debug_log: bool = False

		# 预览
		self._preview_pm: QPixmap | None = None
		self._drag_active: bool = False
		self._drag_start: tuple = (0, 0)
		self._preview_scale: float = 1.0
		self._preview_ox: float = 0.0
		self._preview_oy: float = 0.0

		self._build_ui()
		self._connect_signals()
		self._add_shortcuts()

	# ═══════════════════ 构建 UI ═══════════════════

	def _build_ui(self) -> None:
		# ── 中央内容 ──
		central = QWidget()
		self.setCentralWidget(central)
		self._update_background()
		root = QVBoxLayout(central)
		root.setContentsMargins(12, 8, 12, 6)
		root.setSpacing(0)

		# ── 顶栏：主题按钮 ──
		top_bar = QWidget()
		tbl = QHBoxLayout(top_bar); tbl.setContentsMargins(0, 0, 0, 4)
		tbl.addStretch()
		self._theme_btn = IconWidget(FluentIcon.BRIGHTNESS, top_bar)
		self._theme_btn.setFixedSize(24, 24)
		self._theme_btn.setToolTip("切换亮色/暗色主题")
		self._theme_btn.mousePressEvent = lambda e: self._toggle_theme()
		tbl.addWidget(self._theme_btn)
		root.addWidget(top_bar)

		# ── TabWidget ──
		self._tabs = QTabWidget()
		root.addWidget(self._tabs)

		# ── Tab 1: OCR 处理 ──
		self._ocr_tab = QWidget()
		self._tabs.addTab(self._ocr_tab, "OCR 处理")
		self._build_ocr_tab()

		# ── Tab 2: 数据分析 ──
		self._analysis_tab = AnalysisTab(self._tabs)

		# ── 底部状态栏 ──
		self._footer = QWidget()
		fl = QVBoxLayout(self._footer); fl.setContentsMargins(0, 6, 0, 0)
		self._status_label = BodyLabel("请选择视频并设置识别范围。")
		self._progress_bar = ProgressBar()
		self._progress_bar.setRange(0, 100); self._progress_bar.setValue(0)
		self._progress_bar.setTextVisible(True)
		fl.addWidget(self._status_label)
		fl.addWidget(self._progress_bar)
		root.addWidget(self._footer)

	def _build_ocr_tab(self) -> None:
		layout = QVBoxLayout(self._ocr_tab)
		layout.setContentsMargins(0, 6, 0, 0); layout.setSpacing(8)

		# Header
		hdr = QHBoxLayout()
		self._import_btn = PushButton("导入视频")
		hdr.addWidget(self._import_btn)
		self._file_label = BodyLabel("未导入视频")
		self._file_label.setWordWrap(True)
		hdr.addWidget(self._file_label, 1)
		self._export_btn = PrimaryPushButton("导出 CSV")
		hdr.addWidget(self._export_btn)
		self._cancel_btn = PushButton("取消")
		self._cancel_btn.setEnabled(False)
		hdr.addWidget(self._cancel_btn)
		layout.addLayout(hdr)

		# 视频信息 Card
		info = CardWidget()
		il = QHBoxLayout(info)
		self._dur_label = BodyLabel("-"); self._res_label = BodyLabel("-")
		self._fps_label = BodyLabel("-"); self._codec_label = BodyLabel("-")
		for t, l in [("时长", self._dur_label), ("分辨率", self._res_label),
				("帧率", self._fps_label), ("编码", self._codec_label)]:
			w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(0, 0, 0, 0)
			wl.addWidget(CaptionLabel(t)); wl.addWidget(l); il.addWidget(w)
		layout.addWidget(info)

		# 主内容
		main_w = QHBoxLayout(); main_w.setSpacing(12)

		# 左侧面板
		left = QWidget(); left.setFixedWidth(450)
		ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)
		self._build_left_panel(ll)
		main_w.addWidget(left)

		# 右侧 = 识别范围 + 预览
		right = QVBoxLayout(); right.setSpacing(8)
		self._build_right_panel(right)
		main_w.addLayout(right, 1)

		layout.addLayout(main_w, 1)

	def _build_left_panel(self, ll: QVBoxLayout) -> None:
		# 速度格式 Card
		fmt_card = CardWidget()
		gl = QVBoxLayout(fmt_card)
		gl.addWidget(StrongBodyLabel("速度格式"))
		r = QHBoxLayout()
		self._fmt_ms = RadioButton("m/s"); self._fmt_kmh = RadioButton("km/h")
		self._fmt_kmh.setChecked(True); self._fmt_mph = RadioButton("mile/h")
		r.addWidget(self._fmt_ms); r.addWidget(self._fmt_kmh)
		r.addWidget(self._fmt_mph); r.addStretch()
		gl.addLayout(r)
		gl.addWidget(CaptionLabel("输出统一转换为 km/h。"))

		cg = CardWidget()
		cl = QGridLayout(cg)
		cl.addWidget(BodyLabel("最大速度 (km/h)"), 0, 0)
		self.max_speed_edit = LineEdit(); self.max_speed_edit.setText("400"); self.max_speed_edit.setFixedWidth(60)
		cl.addWidget(self.max_speed_edit, 0, 1)
		cl.addWidget(BodyLabel("最大加速度 (m/s²)"), 0, 2)
		self.max_accel_edit = LineEdit(); self.max_accel_edit.setText("50"); self.max_accel_edit.setFixedWidth(60)
		cl.addWidget(self.max_accel_edit, 0, 3)
		gl.addWidget(cg)
		ll.addWidget(fmt_card)

		# 性能 Card
		perf_card = CardWidget()
		pl = QGridLayout(perf_card)
		pl.addWidget(StrongBodyLabel("性能"), 0, 0, 1, 4)
		pl.addWidget(BodyLabel("采样率 1/"), 1, 0)
		self.div_edit = LineEdit(); self.div_edit.setText("2"); self.div_edit.setFixedWidth(60)
		pl.addWidget(self.div_edit, 1, 1)
		pl.addWidget(BodyLabel("并行线程数"), 1, 2)
		self.workers_edit = LineEdit(); self.workers_edit.setText("4"); self.workers_edit.setFixedWidth(60)
		pl.addWidget(self.workers_edit, 1, 3)
		pl.addWidget(BodyLabel("OCR 高度 (px)"), 2, 0)
		self.target_h_edit = LineEdit(); self.target_h_edit.setText("24"); self.target_h_edit.setFixedWidth(60)
		pl.addWidget(self.target_h_edit, 2, 1)
		pl.addWidget(BodyLabel("边缘填充 (px)"), 2, 2)
		self.pad_edit = LineEdit(); self.pad_edit.setText("0"); self.pad_edit.setFixedWidth(60)
		pl.addWidget(self.pad_edit, 2, 3)
		pl.addWidget(BodyLabel("OCR 后端"), 3, 0)
		self.backend_combo = ComboBox()
		self.backend_combo.addItems(["自动", "CUDA", "CPU"]); self.backend_combo.setCurrentIndex(0)
		pl.addWidget(self.backend_combo, 3, 1)
		self.debug_cb = CheckBox("调试日志"); pl.addWidget(self.debug_cb, 3, 2, 1, 2)
		ll.addWidget(perf_card)

		# 纠错模式 Card
		mode_card = CardWidget()
		ml = QVBoxLayout(mode_card)
		ml.addWidget(StrongBodyLabel("纠错模式"))
		self.mode_auto = RadioButton("自动锚点纠错（全自动，推荐）")
		self.mode_auto.setChecked(True)
		self.mode_baseline = RadioButton("人工基准标注")
		ml.addWidget(self.mode_auto); ml.addWidget(self.mode_baseline)
		bf = QWidget(); bfl = QHBoxLayout(bf); bfl.setContentsMargins(20, 0, 0, 0)
		bfl.addWidget(BodyLabel("抽样频率 1/"))
		self.baseline_edit = LineEdit(); self.baseline_edit.setText("10"); self.baseline_edit.setFixedWidth(60)
		bfl.addWidget(self.baseline_edit)
		bfl.addWidget(CaptionLabel("(1=全部人工)")); bfl.addStretch()
		ml.addWidget(bf)
		ll.addWidget(mode_card)

		# 时间轴范围 Card
		time_card = CardWidget()
		tl = QGridLayout(time_card)
		tl.addWidget(StrongBodyLabel("时间轴范围"), 0, 0, 1, 6)
		tl.addWidget(BodyLabel("起始帧"), 1, 0)
		self.frame_start_edit = LineEdit(); self.frame_start_edit.setFixedWidth(80)
		tl.addWidget(self.frame_start_edit, 1, 1)
		bfs = PushButton("设为当前"); bfs.setFixedWidth(80)
		bfs.clicked.connect(lambda: self.frame_start_edit.setText(str(self._slider.value())))
		tl.addWidget(bfs, 1, 2)
		tl.addWidget(BodyLabel("结束帧"), 1, 3)
		self.frame_end_edit = LineEdit(); self.frame_end_edit.setFixedWidth(80)
		tl.addWidget(self.frame_end_edit, 1, 4)
		bfe = PushButton("设为当前"); bfe.setFixedWidth(80)
		bfe.clicked.connect(lambda: self.frame_end_edit.setText(str(self._slider.value())))
		tl.addWidget(bfe, 1, 5)
		ll.addWidget(time_card)

		ll.addStretch()

	def _build_right_panel(self, rl: QVBoxLayout) -> None:
		# 识别范围 Card
		roi_card = CardWidget()
		rgl = QGridLayout(roi_card)
		rgl.addWidget(StrongBodyLabel("识别范围（像素）"), 0, 0, 1, 4)
		self.roi_x1 = LineEdit(); self.roi_y1 = LineEdit()
		self.roi_x2 = LineEdit(); self.roi_y2 = LineEdit()
		for e in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
			e.setFixedWidth(90)
		rgl.addWidget(CaptionLabel("左上 X"), 1, 0); rgl.addWidget(self.roi_x1, 2, 0)
		rgl.addWidget(CaptionLabel("左上 Y"), 1, 1); rgl.addWidget(self.roi_y1, 2, 1)
		rgl.addWidget(CaptionLabel("右下 X"), 1, 2); rgl.addWidget(self.roi_x2, 2, 2)
		rgl.addWidget(CaptionLabel("右下 Y"), 1, 3); rgl.addWidget(self.roi_y2, 2, 3)
		rl.addWidget(roi_card)

		# 预览 Card
		pv = CardWidget()
		pvl = QVBoxLayout(pv)
		pvl.addWidget(StrongBodyLabel("识别范围预览"))
		self._preview_label = BodyLabel()
		self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self._preview_label.setMinimumSize(400, 300)
		self._preview_label.setStyleSheet("background-color: #111; border-radius: 6px;")
		self._preview_label.setMouseTracking(True)
		self._preview_label.mousePressEvent = self._on_pv_press    # type: ignore[method-assign]
		self._preview_label.mouseMoveEvent = self._on_pv_move       # type: ignore[method-assign]
		self._preview_label.mouseReleaseEvent = self._on_pv_release # type: ignore[method-assign]
		pvl.addWidget(self._preview_label, 1)

		sr = QHBoxLayout()
		self._slider = Slider(Qt.Orientation.Horizontal)
		self._slider.setRange(0, 1); self._slider.setValue(0)
		self._slider.valueChanged.connect(self._on_slider)
		sr.addWidget(self._slider, 1)
		self._frame_label = CaptionLabel("#0"); self._frame_label.setFixedWidth(60)
		sr.addWidget(self._frame_label)
		pvl.addLayout(sr)
		rl.addWidget(pv, 1)

	# ═══════════════════ 信号连接 + 快捷键 ═══════════════════

	def _connect_signals(self) -> None:
		self._import_btn.clicked.connect(self._import_video)
		self._export_btn.clicked.connect(self._export_csv)
		self._cancel_btn.clicked.connect(self._cancel_export)
		self._fmt_ms.clicked.connect(lambda: self._on_fmt("m/s"))
		self._fmt_kmh.clicked.connect(lambda: self._on_fmt("km/h"))
		self._fmt_mph.clicked.connect(lambda: self._on_fmt("mile/h"))
		self.mode_auto.clicked.connect(lambda: self._on_mode("auto"))
		self.mode_baseline.clicked.connect(lambda: self._on_mode("baseline"))
		self.backend_combo.currentIndexChanged.connect(self._on_backend)
		self._tabs.currentChanged.connect(self._on_tab)
		self.debug_cb.toggled.connect(lambda v: setattr(self, '_debug_log', v))

	def _add_shortcuts(self) -> None:
		QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
		QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
		QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self._step(10))
		QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self._step(-10))

	def _on_fmt(self, fmt: str) -> None: self.speed_format = fmt
	def _on_mode(self, mode: str) -> None: self.correction_mode = mode
	def _log(self, msg: str) -> None:
		if self._debug_log: print(f"[DEBUG] {msg}", flush=True)

	# ═══════════════════ 主题切换 ═══════════════════

	@staticmethod
	def _is_dark() -> bool:
		from qfluentwidgets import qconfig, Theme
		return qconfig.theme == Theme.DARK

	def _apply_theme(self) -> None:
		from qfluentwidgets import qconfig, Theme, isDarkTheme
		if qconfig.theme == Theme.DARK:
			setTheme(Theme.LIGHT)
		else:
			setTheme(Theme.DARK)
		self._update_background()
		# 同步 matplotlib 画布
		if hasattr(self, '_analysis_tab'):
			self._analysis_tab._sync_figure_theme()

	def _update_background(self) -> None:
		from qfluentwidgets import isDarkTheme
		dark = isDarkTheme()
		bg = '#1f1f1f' if dark else '#f5f5f5'
		from PySide6.QtGui import QPalette, QColor
		for w in (self, self.centralWidget()):
			p = w.palette()
			p.setColor(QPalette.ColorRole.Window, QColor(bg))
			w.setPalette(p)
			w.setAutoFillBackground(True)
		# Windows title bar
		import sys
		if sys.platform == 'win32':
			try:
				import ctypes
				hwnd = int(self.winId())
				DWMWA = 20
				val = ctypes.c_int(1 if dark else 0)
				ctypes.windll.dwmapi.DwmSetWindowAttribute(
					hwnd, DWMWA, ctypes.byref(val), ctypes.sizeof(val))
			except Exception:
				pass

	def _toggle_theme(self) -> None:
		self._apply_theme()

	# ═══════════════════ 视频导入 ═══════════════════

	def _import_video(self) -> None:
		path, _ = QFileDialog.getOpenFileName(self, "选择需要处理的视频", "",
			"视频文件 (*.mp4 *.mkv *.avi *.mov *.m4v *.wmv *.flv *.webm);;所有文件 (*.*)")
		if not path: return
		try: self._load_video(Path(path))
		except Exception as e:
			QMessageBox.critical(self, "导入失败", str(e))
			self._status_label.setText("导入失败。")

	def _load_video(self, path: Path) -> None:
		cap = cv2.VideoCapture(str(path))
		if not cap.isOpened(): raise RuntimeError("无法打开视频文件。")

		fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		fourcc = cap.get(cv2.CAP_PROP_FOURCC) or 0.0
		dur = fc / fps if fps > 0 else 0.0

		ok, frame = cap.read()
		if not ok or frame is None:
			cap.release(); raise RuntimeError("无法读取视频第一帧。")

		if self._preview_cap is not None: self._preview_cap.release()
		self._preview_cap = cap; self._preview_frame_no = 0

		self.video_path = path
		self.metadata = VideoMetadata(path=path, duration_sec=dur, width=w, height=h,
			fps=fps, codec=codec_from_fourcc(fourcc), frame_count=fc)
		self.first_frame_bgr = frame

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		hh, ww, ch = rgb.shape
		self.first_frame_qimg = QImage(rgb.data, ww, hh, ch * ww,
			QImage.Format.Format_RGB888).copy()

		self._file_label.setText(str(path))
		self._dur_label.setText(format_duration(dur))
		self._res_label.setText(f"{w} x {h}")
		self._fps_label.setText(f"{fps:.3f}" if fps > 0 else "Unknown")
		self._codec_label.setText(self.metadata.codec)
		self._status_label.setText("视频已载入，请输入识别范围并预览。")
		self._slider.setRange(0, fc - 1); self._slider.setValue(0)
		self._frame_label.setText(f"#{0}/{fc}")
		self._show_frame(0)

	# ═══════════════════ 预览 ═══════════════════

	def _on_pv_press(self, event) -> None:
		if not self.metadata or self.first_frame_qimg is None: return
		x, y = self._to_video(event.position().x(), event.position().y())
		self._drag_active = True; self._drag_start = (x, y)
		for e, v in [(self.roi_x1, x), (self.roi_y1, y), (self.roi_x2, x), (self.roi_y2, y)]:
			e.setText(str(v))

	def _on_pv_move(self, event) -> None:
		if not self._drag_active or not self.metadata: return
		x, y = self._to_video(event.position().x(), event.position().y())
		x1 = min(self._drag_start[0], x); y1 = min(self._drag_start[1], y)
		x2 = max(self._drag_start[0], x); y2 = max(self._drag_start[1], y)
		self.roi_x1.setText(str(x1)); self.roi_y1.setText(str(y1))
		self.roi_x2.setText(str(x2)); self.roi_y2.setText(str(y2))
		self._redraw()

	def _on_pv_release(self, event) -> None:
		self._drag_active = False

	def _to_video(self, wx: float, wy: float) -> tuple[int, int]:
		if not self.metadata or self._preview_scale <= 0: return 0, 0
		x = (wx - self._preview_ox) / self._preview_scale
		y = (wy - self._preview_oy) / self._preview_scale
		return (max(0, min(self.metadata.width - 1, int(x))),
			max(0, min(self.metadata.height - 1, int(y))))

	def _show_frame(self, frame_no: int) -> None:
		pm = None
		if frame_no > 0 and self._preview_cap is not None and self._seek(frame_no):
			ok, frame = self._preview_cap.retrieve()
			if ok and frame is not None:
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				h, w, ch = rgb.shape
				qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
				pm = QPixmap.fromImage(qimg)
		if pm is None and self.first_frame_qimg is not None:
			pm = QPixmap.fromImage(self.first_frame_qimg)
		if pm is not None:
			self._preview_pm = pm
			self._redraw()

	def _seek(self, target: int) -> bool:
		cap = self._preview_cap
		if cap is None: return False
		diff = target - self._preview_frame_no
		if 0 < diff <= 30:
			for _ in range(diff):
				if not cap.grab(): return False
			self._preview_frame_no = target; return True
		cap.set(cv2.CAP_PROP_POS_FRAMES, target)
		self._preview_frame_no = target; return True

	def _on_slider(self, value: int) -> None:
		if self.metadata:
			self._frame_label.setText(f"#{value}/{self.metadata.frame_count}")
		if self._throttle_timer:
			self._throttle_timer.stop()
		self._throttle_timer = QTimer(self)
		self._throttle_timer.setSingleShot(True)
		self._throttle_timer.timeout.connect(lambda: self._show_frame(value))
		self._throttle_timer.start(30)

	def _step(self, delta: int) -> None:
		if not self.metadata: return
		v = max(0, min(self.metadata.frame_count - 1, self._slider.value() + delta))
		self._slider.setValue(v)

	def _redraw(self) -> None:
		if self._preview_pm is None: return
		ls = self._preview_label.size()
		pw, ph = ls.width(), ls.height()
		if pw <= 0 or ph <= 0: return

		pm = self._preview_pm
		scale = min(pw / pm.width(), ph / pm.height())
		dw = max(1, int(pm.width() * scale)); dh = max(1, int(pm.height() * scale))
		scaled = pm.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation)
		self._preview_scale = scale
		self._preview_ox = (pw - dw) / 2.0; self._preview_oy = (ph - dh) / 2.0

		# ROI 框
		roi = self._get_roi()
		if roi is not None:
			painter = QPainter(scaled)
			x1, y1, x2, y2 = roi
			painter.setPen(QPen(QColor("#ff5050"), max(2, int(scale * 2))))
			painter.drawRect(int(x1 * scale), int(y1 * scale),
				int((x2 - x1) * scale), int((y2 - y1) * scale))
			painter.end()

		result = QPixmap(pw, ph); result.fill(QColor("#151515"))
		rp = QPainter(result)
		rp.drawPixmap(int(self._preview_ox), int(self._preview_oy), scaled)
		rp.end()
		self._preview_label.setPixmap(result)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		self._redraw()

	def _get_roi(self) -> tuple | None:
		try:
			x1 = int(self.roi_x1.text()); y1 = int(self.roi_y1.text())
			x2 = int(self.roi_x2.text()); y2 = int(self.roi_y2.text())
		except ValueError: return None
		if self.metadata:
			x1, x2 = sorted((max(0, min(self.metadata.width - 1, x1)),
				max(0, min(self.metadata.width - 1, x2))))
			y1, y2 = sorted((max(0, min(self.metadata.height - 1, y1)),
				max(0, min(self.metadata.height - 1, y2))))
		return (x1, y1, x2, y2)

	# ═══════════════════ OCR 引擎 ═══════════════════

	def _on_backend(self, _idx: int) -> None:
		_reset_backend(); self._release_engines()
		keys = ["auto", "cuda", "cpu"]; key = keys[self.backend_combo.currentIndex()]
		actual = _select_backend(key)
		if key == "cuda" and actual != "CUDA":
			QMessageBox.warning(self, "后端不可用", f"CUDA 不可用。\n已回退为 {actual}。")
		self._status_label.setText(f"OCR 后端: {'CUDA (GPU)' if actual == 'CUDA' else 'CPU'}")

	def get_ocr_engine(self) -> "RapidOCR":
		if self.ocr_engine is None: self.ocr_engine = self._create_ocr()
		return self.ocr_engine

	def _create_ocr(self) -> "RapidOCR":
		_reset_backend()
		keys = ["auto", "cuda", "cpu"]; key = keys[self.backend_combo.currentIndex()]
		_select_backend(key)
		kw = _get_model_kwargs("v6_small")
		return RapidOCR(**(kw or {}))

	def _release_engines(self) -> None:
		for e in ([self.ocr_engine] if self.ocr_engine else []) + self.ocr_engines:
			try: del e
			except Exception: pass
		self.ocr_engine = None; self.ocr_engines.clear()
		import gc; gc.collect()

	# ═══════════════════ 导出 ═══════════════════

	def _export_csv(self) -> None:
		if self.video_path is None or self.metadata is None:
			QMessageBox.warning(self, "未导入视频", "请先导入视频。"); return
		roi = self._get_roi()
		if roi is None:
			QMessageBox.warning(self, "识别范围不完整", "请先填写或拖拽选择识别范围。"); return

		out, _ = QFileDialog.getSaveFileName(self, "保存 CSV",
			str(self.video_path.parent / f"{self.video_path.stem}_log.csv"),
			"CSV 文件 (*.csv)")
		if not out: return

		try:
			ms = float(self.max_speed_edit.text()); ma = float(self.max_accel_edit.text())
			fd = int(self.div_edit.text()); th = float(self.target_h_edit.text())
			pp = float(self.pad_edit.text()); nw = int(self.workers_edit.text())
		except ValueError:
			QMessageBox.warning(self, "参数错误", "请检查数值参数。"); return

		self._export_btn.setEnabled(False); self._cancel_btn.setEnabled(True)
		self._export_thread = _ExportThread(self, Path(out), roi, ms, ma, fd, th, pp, nw)
		self._export_thread._progress.connect(self._on_progress)
		self._export_thread._finished.connect(self._on_done)
		self._export_thread._error.connect(self._on_error)
		self._export_thread._cancelled.connect(self._on_cancel)
		self._export_thread.start()

	def _cancel_export(self) -> None:
		if self._export_thread: self._export_thread._cancel_flag = True
		self._cancel_btn.setEnabled(False); self._status_label.setText("正在取消...")

	def _on_progress(self, msg: str, pct: float) -> None:
		self._status_label.setText(msg); self._progress_bar.setValue(int(pct))

	def _on_done(self, mode: str) -> None:
		self._finish_export()
		self._status_label.setText("自动锚点完成 — 结果已保存。" if mode == "auto"
			else "人工基准完成 — 结果已保存。")

	def _on_error(self, err: str) -> None:
		self._finish_export(); QMessageBox.critical(self, "导出失败", err)

	def _on_cancel(self) -> None:
		self._finish_export(); self._status_label.setText("已取消。")

	def _finish_export(self) -> None:
		self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
		self._export_thread = None; self._release_engines()

	def _on_tab(self, index: int) -> None:
		if index == 1:
			self._footer.hide()
		else:
			self._footer.show()

	def closeEvent(self, event) -> None:
		if self._preview_cap is not None: self._preview_cap.release()
		self._release_engines()
		super().closeEvent(event)
