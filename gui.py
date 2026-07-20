"""RaceVideoToLog PySide6 GUI — 主窗口 + 导出线程。

分离自 RaceVideoToLog.py，包含所有 GUI 相关逻辑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
	QMainWindow, QWidget, QStackedWidget,
	QDialog, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from widget_utils import make_static_card
from pipeline import ProcessingPipeline
from PySide6.QtGui import (
	QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QShortcut,
)

from ocr_engine import (
    RapidOCR, VideoMetadata,
    codec_from_fourcc, format_duration,
    _reset_backend, _select_backend, _get_model_kwargs,
    _CancelExport,
)
from gui_analysis import AnalysisTab
from gui_review import ReviewDialog
from theme_manager import ThemeManager

from qfluentwidgets import (setTheme, Theme, isDarkTheme,
	PushButton, PrimaryPushButton, LineEdit, ComboBox, CheckBox, RadioButton,
	BodyLabel, StrongBodyLabel, CaptionLabel, CardWidget, Slider, ProgressBar, CompactSpinBox, Pivot)


class _ExportThread(QThread):
	"""后台导出线程：在原生线程中运行 Pipeline，通过信号与 GUI 通信。

	避免 QThread 导致的 CUDA ONNX 推理性能损失（~4.6x）。
	"""

	_progress = Signal(str, float)
	_finished = Signal(str)
	_review_data = Signal(list, list, list, list)
	_error = Signal(str)
	_cancelled = Signal()

	def __init__(self, app: "RaceVideoToLogApp", output_path: Path,
			region: tuple, max_speed_kmh: float, max_accel_mps2: float,
			frame_div: int, target_h: float, pad_px: float, buffer_size: int,
			backend: str = "auto", parent: QWidget | None = None) -> None:
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
		self._backend = backend
		self._cancel_flag = False


	def run(self) -> None:
		"""Run Pipeline in a native threading.Thread, wait for completion."""
		import threading
		done = threading.Event()
		error_container: list[Exception] = []
		result_container: dict = {}

		def _worker() -> None:
			try:
				self._check_cancel()
				mode = self.app.correction_mode
				assert self.app.video_path is not None
				pipeline = ProcessingPipeline(
					video_path=self.app.video_path,
					roi=self._region,
					max_speed=self._max_speed_kmh,
					max_accel=self._max_accel_mps2,
					frame_div=self._frame_div,
					target_h=self._target_h,
					pad=self._pad_px,
					buffer_size=self._buffer_size,
					backend=self._backend,
					ocr_model=self.app.model_combo.currentText(),
					reocr_model=self.app._reocr_model(),
					speed_format=self.app.speed_format,
					frame_start=self.app.frame_start_edit.text(),
					frame_end=self.app.frame_end_edit.text(),
					progress_cb=self._emit_progress,
				)
				if mode == "auto":
					pipeline.run_auto(self._output_path)
					result_container["mode"] = "auto"
				else:
					result = pipeline.run_review_pass1(self._output_path)
					if result is None:
						# No problem segments, CSV already written
						result_container["mode"] = "auto"
					else:
						result_container["mode"] = "review"
						result_container["review_data"] = result
						self.app._pipeline = pipeline
						self.app._review_output_path = self._output_path
			except _CancelExport:
				result_container["cancelled"] = True
			except Exception as exc:
				import traceback
				traceback.print_exc()
				error_container.append(exc)
			finally:
				done.set()

		t = threading.Thread(target=_worker, daemon=False)
		t.start()
		done.wait()
		t.join()

		if error_container:
			self._error.emit(str(error_container[0]))
		elif result_container.get("cancelled"):
			self._cancelled.emit()
		elif result_container["mode"] == "review":
			rows, obs, raw_frames, conf, segs = result_container["review_data"]
			self._review_data.emit(rows, obs, conf, segs)
			self.app._review_rows = rows
			self.app._review_observations = obs
			self.app._review_raw_frames = raw_frames
			self.app._review_confidences = conf
			self.app._review_segments = segs
			self._finished.emit("review")
		else:
			self._finished.emit("auto")

	def _check_cancel(self) -> None:
		if self._cancel_flag:
			raise _CancelExport()

	def _emit_progress(self, msg: str, pct: float) -> None:
		self._progress.emit(msg, pct)

class RaceVideoToLogApp(QMainWindow):
	"""RaceVideoToLog PySide6 主窗口。"""

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Race Video To Log")
		self.resize(1500, 920)
		self.setMinimumSize(1100, 760)

		# ── 状态变量 ──
		self.video_path: Path | None = None
		self._pipeline: object | None = None
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

		# 人工审核暂存
		self._review_rows: list = []
		self._review_observations: list = []
		self._review_raw_frames: list = []
		self._review_ocr: RapidOCR | None = None
		self._review_anchor_indices: set = set()
		self._review_output_path: Path | None = None
		self._review_confidences: list[dict] = []
		self._review_segments: list[dict] = []
		self._review_confirmed: set = set()

		# 预览
		self._preview_pm: QPixmap | None = None
		self._drag_active: bool = False
		self._drag_start: tuple = (0, 0)
		self._preview_scale: float = 1.0
		self._preview_ox: float = 0.0
		self._preview_oy: float = 0.0
		self._redraw_timer: QTimer | None = None  # ROI 拖拽重绘节流

		self._build_ui()
		self._connect_signals()
		self._add_shortcuts()

	# ═══════════════════ 构建 UI ═══════════════════

	def _build_ui(self) -> None:
		# ── 中央内容 ──
		central = QWidget()
		self.setCentralWidget(central)
		self._register_theme_callbacks()
		ThemeManager.refresh()
		root = QVBoxLayout(central)
		root.setContentsMargins(12, 8, 12, 6)
		root.setSpacing(0)

		# ── 顶栏：主题按钮 ──
		top_bar = QWidget()
		tbl = QHBoxLayout(top_bar); tbl.setContentsMargins(0, 0, 0, 4)
		self._tab_pivot = Pivot(self)
		self._tab_pivot.setFixedWidth(160)
		tbl.addWidget(self._tab_pivot)
		tbl.addStretch()
		self._theme_btn = PushButton("☀" if not isDarkTheme() else "☾")
		self._theme_btn.setFixedSize(36, 28)
		self._theme_btn.setToolTip("切换亮色/暗色主题")
		self._theme_btn.clicked.connect(self._toggle_theme)
		tbl.addWidget(self._theme_btn)
		root.addWidget(top_bar)
		self._tab_stack = QStackedWidget()
		root.addWidget(self._tab_stack)

		# Tab 1: OCR 处理
		self._ocr_tab = QWidget()
		self._tab_stack.addWidget(self._ocr_tab)
		self._build_ocr_tab()
		self._tab_pivot.addItem('ocr', 'OCR 处理', lambda: self._tab_stack.setCurrentIndex(0))

		# Tab 2: 数据分析
		self._analysis_tab = AnalysisTab(self._tab_stack)
		ThemeManager.register(lambda dark: self._analysis_tab._sync_figure_theme())
		self._tab_pivot.addItem('analysis', '数据分析', lambda: self._tab_stack.setCurrentIndex(1))
		self._tab_pivot.setCurrentItem('ocr')
		self._tab_pivot.currentItemChanged.connect(self._on_pivot)

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
		ThemeManager.refresh()

	def _build_ocr_tab(self) -> None:
		layout = QVBoxLayout(self._ocr_tab)
		layout.setContentsMargins(0, 6, 0, 0); layout.setSpacing(8)

		# Header
		hdr = QHBoxLayout()
		self._import_video_btn = PushButton("导入视频")
		hdr.addWidget(self._import_video_btn)
		self._file_label = BodyLabel("未导入视频")
		self._file_label.setWordWrap(True)
		hdr.addWidget(self._file_label, 1)
		self._import_settings_btn = PushButton("导入设置")
		self._import_settings_btn.clicked.connect(self._import_settings)
		hdr.addWidget(self._import_settings_btn)
		self._export_btn = PrimaryPushButton("导出 CSV")
		hdr.addWidget(self._export_btn)
		self._cancel_btn = PushButton("取消")
		self._cancel_btn.setEnabled(False)
		hdr.addWidget(self._cancel_btn)
		layout.addLayout(hdr)

		# 视频信息 Card
		info = make_static_card()
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
		fmt_card = make_static_card()
		gl = QVBoxLayout(fmt_card)
		gl.addWidget(StrongBodyLabel("速度格式"))
		r = QHBoxLayout()
		self._fmt_ms = RadioButton("m/s"); self._fmt_kmh = RadioButton("km/h")
		self._fmt_kmh.setChecked(True); self._fmt_mph = RadioButton("mile/h")
		r.addWidget(self._fmt_ms); r.addWidget(self._fmt_kmh)
		r.addWidget(self._fmt_mph); r.addStretch()
		gl.addLayout(r)
		gl.addWidget(CaptionLabel("输出统一转换为 km/h。"))

		cg = make_static_card()
		cl = QGridLayout(cg)
		cl.addWidget(BodyLabel("最大速度 (km/h)"), 0, 0)
		self.max_speed_edit = LineEdit(); self.max_speed_edit.setText("400"); self.max_speed_edit.setFixedWidth(50)
		cl.addWidget(self.max_speed_edit, 0, 1)
		cl.addWidget(BodyLabel("最大加速度 (m/s²)"), 0, 2)
		self.max_accel_edit = LineEdit(); self.max_accel_edit.setText("50"); self.max_accel_edit.setFixedWidth(50)
		cl.addWidget(self.max_accel_edit, 0, 3)
		gl.addWidget(cg)
		ll.addWidget(fmt_card)

		# 性能 Card
		perf_card = make_static_card()
		pl = QGridLayout(perf_card)
		pl.addWidget(StrongBodyLabel("性能"), 0, 0, 1, 4)
		pl.addWidget(BodyLabel("采样率 1/"), 1, 0)
		self.div_spin = CompactSpinBox(); self.div_spin.setRange(1, 10); self.div_spin.setValue(2); self.div_spin.setFixedWidth(70)
		self._disable_spin_flyout(self.div_spin)
		pl.addWidget(self.div_spin, 1, 1)
		pl.addWidget(BodyLabel("并行线程数"), 1, 2)
		self.buffer_edit = LineEdit(); self.buffer_edit.setText("16"); self.buffer_edit.setFixedWidth(50)
		pl.addWidget(self.buffer_edit, 1, 3)
		pl.addWidget(BodyLabel("OCR 高度 (px)"), 2, 0)
		self.target_h_edit = LineEdit(); self.target_h_edit.setText("24"); self.target_h_edit.setFixedWidth(50)
		pl.addWidget(self.target_h_edit, 2, 1)
		pl.addWidget(BodyLabel("边缘填充 (px)"), 2, 2)
		self.pad_edit = LineEdit(); self.pad_edit.setText("0"); self.pad_edit.setFixedWidth(50)
		pl.addWidget(self.pad_edit, 2, 3)
		pl.addWidget(BodyLabel("OCR 后端"), 3, 0)
		self.backend_combo = ComboBox()
		self.backend_combo.addItems(["自动", "CUDA", "CPU"]); self.backend_combo.setCurrentIndex(0)
		pl.addWidget(self.backend_combo, 3, 1)
		self.debug_cb = CheckBox("调试日志"); pl.addWidget(self.debug_cb, 3, 2, 1, 2)
		pl.addWidget(BodyLabel("OCR 模型"), 4, 0)
		self.model_combo = ComboBox()
		self.model_combo.addItems(["v6_tiny", "v6_small"])
		self.model_combo.setCurrentIndex(0)  # default: tiny
		self.model_combo.setFixedWidth(95)
		pl.addWidget(self.model_combo, 4, 1)
		pl.addWidget(BodyLabel("重OCR"), 4, 2)
		self.reocr_model_combo = ComboBox()
		self.reocr_model_combo.addItems(["同主模型", "v6_tiny", "v6_small"])
		self.reocr_model_combo.setCurrentIndex(2)  # default: v6_small
		self.reocr_model_combo.setFixedWidth(120)
		pl.addWidget(self.reocr_model_combo, 4, 3)
		ll.addWidget(perf_card)

		# 纠错模式 Card
		mode_card = make_static_card()
		ml = QVBoxLayout(mode_card)
		ml.addWidget(StrongBodyLabel("纠错模式"))
		self.mode_auto = RadioButton("自动锚点纠错（全自动，推荐）")
		self.mode_auto.setChecked(True)
		self.mode_baseline = RadioButton("人工辅助纠错")
		ml.addWidget(self.mode_auto); ml.addWidget(self.mode_baseline)
		ll.addWidget(mode_card)

		# 时间轴范围 Card
		time_card = make_static_card()
		tl = QGridLayout(time_card)
		tl.addWidget(StrongBodyLabel("时间轴范围"), 0, 0, 1, 6)
		tl.addWidget(BodyLabel("起始帧"), 1, 0)
		self.frame_start_edit = LineEdit(); self.frame_start_edit.setFixedWidth(72)
		tl.addWidget(self.frame_start_edit, 1, 1)
		bfs = PushButton("设为当前"); bfs.setFixedWidth(90)
		bfs.clicked.connect(lambda: self.frame_start_edit.setText(str(self._slider.value())))
		tl.addWidget(bfs, 1, 2)
		tl.addWidget(BodyLabel("结束帧"), 1, 3)
		self.frame_end_edit = LineEdit(); self.frame_end_edit.setFixedWidth(72)
		tl.addWidget(self.frame_end_edit, 1, 4)
		bfe = PushButton("设为当前"); bfe.setFixedWidth(90)
		bfe.clicked.connect(lambda: self.frame_end_edit.setText(str(self._slider.value())))
		tl.addWidget(bfe, 1, 5)
		ll.addWidget(time_card)

		ll.addStretch()

	def _build_right_panel(self, rl: QVBoxLayout) -> None:
		# 识别范围 Card
		roi_card = make_static_card()
		rgl = QGridLayout(roi_card)
		rgl.addWidget(StrongBodyLabel("识别范围（像素）"), 0, 0, 1, 4)
		self.roi_x1 = CompactSpinBox(); self.roi_y1 = CompactSpinBox()
		self.roi_x2 = CompactSpinBox(); self.roi_y2 = CompactSpinBox()
		for s in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
			s.setRange(0, 9999); s.setFixedWidth(80)
			s.valueChanged.connect(lambda v, spin=s: self._on_roi_spin(spin))
			self._disable_spin_flyout(s)
		rgl.addWidget(CaptionLabel("左上 X"), 1, 0); rgl.addWidget(self.roi_x1, 2, 0)
		rgl.addWidget(CaptionLabel("左上 Y"), 1, 1); rgl.addWidget(self.roi_y1, 2, 1)
		rgl.addWidget(CaptionLabel("右下 X"), 1, 2); rgl.addWidget(self.roi_x2, 2, 2)
		rgl.addWidget(CaptionLabel("右下 Y"), 1, 3); rgl.addWidget(self.roi_y2, 2, 3)
		rgl.addWidget(CaptionLabel("← 在预览画面上拖拽选择识别范围"), 3, 0, 1, 4)
		rl.addWidget(roi_card)

		# 预览 Card
		pv = make_static_card()
		pvl = QVBoxLayout(pv)
		pvl.addWidget(StrongBodyLabel("识别范围预览"))
		self._preview_label = BodyLabel()
		self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self._preview_label.setMinimumSize(400, 300)
		self._preview_label.setStyleSheet("background-color: #111; border-radius: 6px;")
		self._preview_label.setMouseTracking(True)
		self._preview_label.setCursor(Qt.CursorShape.CrossCursor)
		self._preview_label.mousePressEvent = self._on_pv_press    # type: ignore[method-assign]
		self._preview_label.mouseMoveEvent = self._on_pv_move       # type: ignore[method-assign]
		self._preview_label.mouseReleaseEvent = self._on_pv_release # type: ignore[method-assign]
		pvl.addWidget(self._preview_label, 1)

		sr = QHBoxLayout()
		self._slider = Slider(Qt.Orientation.Horizontal)
		self._slider.setRange(0, 1); self._slider.setValue(0)
		self._slider.valueChanged.connect(self._on_slider)
		sr.addWidget(self._slider, 1)
		self._frame_label = CaptionLabel("#0"); self._frame_label.setFixedWidth(50)
		sr.addWidget(self._frame_label)
		pvl.addLayout(sr)
		rl.addWidget(pv, 1)

	# ═══════════════════ 信号连接 + 快捷键 ═══════════════════

	def _connect_signals(self) -> None:
		self._import_video_btn.clicked.connect(self._import_video)
		self._export_btn.clicked.connect(self._export_csv)
		self._cancel_btn.clicked.connect(self._cancel_export)
		self._fmt_ms.clicked.connect(lambda: self._on_fmt("m/s"))
		self._fmt_kmh.clicked.connect(lambda: self._on_fmt("km/h"))
		self._fmt_mph.clicked.connect(lambda: self._on_fmt("mile/h"))
		self.mode_auto.toggled.connect(lambda checked: checked and self._on_mode("auto"))
		self.mode_baseline.toggled.connect(lambda checked: checked and self._on_mode("baseline"))
		self.backend_combo.currentIndexChanged.connect(self._on_backend)

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

	def _disable_spin_flyout(self, spin) -> None:
		try:
			spin.compactSpinButton.clicked.disconnect()
		except Exception:
			pass
		spin._showFlyout = lambda: None


	def _register_theme_callbacks(self) -> None:
		# 主窗口背景色
		def _update_bg(dark: bool) -> None:
			bg = "#1f1f1f" if dark else "#f5f5f5"
			fg = "#f0f0f0" if dark else "#000000"
			from PySide6.QtGui import QPalette, QColor
			for w in (self, self.centralWidget(), getattr(self, "_tab_stack", None)):
				if w is None: continue
				p = w.palette()
				p.setColor(QPalette.ColorRole.Window, QColor(bg))
				p.setColor(QPalette.ColorRole.Base, QColor(bg))
				p.setColor(QPalette.ColorRole.WindowText, QColor(fg))
				p.setColor(QPalette.ColorRole.Text, QColor(fg))
				p.setColor(QPalette.ColorRole.ButtonText, QColor(fg))
				w.setPalette(p)
		ThemeManager.register(_update_bg)
		# Windows 标题栏
		def _update_titlebar(dark: bool) -> None:
			import sys, ctypes
			if sys.platform != "win32": return
			try:
				hwnd = int(self.winId())
				val = ctypes.c_int(1 if dark else 0)
				ctypes.windll.dwmapi.DwmSetWindowAttribute(
					hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
			except Exception: pass
		ThemeManager.register(_update_titlebar)
		# 主题图标
		def _update_icon(dark: bool) -> None:
			self._theme_btn.setText("☀" if not dark else "☾")
		ThemeManager.register(_update_icon)
		# 数据分析 matplotlib
		if hasattr(self, "_analysis_tab"):
			tab = self._analysis_tab
			ThemeManager.register(lambda dark: tab._sync_figure_theme())

	def _apply_theme(self) -> None:
		from qfluentwidgets import qconfig, Theme, isDarkTheme
		if qconfig.theme == Theme.DARK:
			setTheme(Theme.LIGHT)
		else:
			setTheme(Theme.DARK)
		ThemeManager.refresh()





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
		for s, m in [(self.roi_x1, w), (self.roi_y1, h), (self.roi_x2, w), (self.roi_y2, h)]:
			s.setMaximum(m - 1)

	# ═══════════════════ 预览 ═══════════════════

	def _on_pv_press(self, event) -> None:
		if not self.metadata or self.first_frame_qimg is None: return
		x, y = self._to_video(event.position().x(), event.position().y())
		self._drag_active = True; self._drag_start = (x, y)
		for s, v in [(self.roi_x1, x), (self.roi_y1, y), (self.roi_x2, x), (self.roi_y2, y)]:
			s.blockSignals(True); s.setValue(v); s.blockSignals(False)

	def _on_pv_move(self, event) -> None:
		if not self._drag_active or not self.metadata: return
		x, y = self._to_video(event.position().x(), event.position().y())
		x1 = min(self._drag_start[0], x); y1 = min(self._drag_start[1], y)
		x2 = max(self._drag_start[0], x); y2 = max(self._drag_start[1], y)
		for s, v in [(self.roi_x1, x1), (self.roi_y1, y1), (self.roi_x2, x2), (self.roi_y2, y2)]:
				s.blockSignals(True); s.setValue(v); s.blockSignals(False)
		self._schedule_redraw()

	def _on_roi_spin(self, spin) -> None:
		if spin is self.roi_x1 and self.roi_x1.value() > self.roi_x2.value() - 1:
			spin.blockSignals(True); spin.setValue(self.roi_x2.value() - 1); spin.blockSignals(False)
		elif spin is self.roi_x2 and self.roi_x2.value() < self.roi_x1.value() + 1:
			spin.blockSignals(True); spin.setValue(self.roi_x1.value() + 1); spin.blockSignals(False)
		elif spin is self.roi_y1 and self.roi_y1.value() > self.roi_y2.value() - 1:
			spin.blockSignals(True); spin.setValue(self.roi_y2.value() - 1); spin.blockSignals(False)
		elif spin is self.roi_y2 and self.roi_y2.value() < self.roi_y1.value() + 1:
			spin.blockSignals(True); spin.setValue(self.roi_y1.value() + 1); spin.blockSignals(False)
		self._schedule_redraw()

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

	def _schedule_redraw(self) -> None:
		"""节流重绘：16ms 单次定时器，避免拖拽时过度调用 _redraw。"""
		if self._redraw_timer is not None:
			return  # 定时器已在运行，跳过
		self._redraw_timer = QTimer(self)
		self._redraw_timer.setSingleShot(True)
		self._redraw_timer.timeout.connect(self._do_throttled_redraw)
		self._redraw_timer.start(16)  # ~60fps

	def _do_throttled_redraw(self) -> None:
		self._redraw_timer = None
		self._redraw()

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
			l = int(x1 * scale); t = int(y1 * scale)
			r = int(x2 * scale); b = int(y2 * scale)
			painter.drawRect(l, t, r - l, b - t)
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
			x1 = self.roi_x1.value(); y1 = self.roi_y1.value()
			x2 = self.roi_x2.value(); y2 = self.roi_y2.value()
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

	def _reocr_model(self) -> str | None:
		"""解析重 OCR 模型选择：'同主模型' → None，否则返回模型名。"""
		text = self.reocr_model_combo.currentText()
		return None if text == "同主模型" else text

	def _create_ocr(self) -> "RapidOCR":
		_reset_backend()
		keys = ["auto", "cuda", "cpu"]; key = keys[self.backend_combo.currentIndex()]
		_select_backend(key)
		kw = _get_model_kwargs(self.model_combo.currentText())
		return RapidOCR(**(kw or {}))

	def _release_engines(self) -> None:
		for e in ([self.ocr_engine] if self.ocr_engine else []) + self.ocr_engines:
			try: del e
			except Exception: pass
		self.ocr_engine = None; self.ocr_engines.clear()
		import gc; gc.collect()

	# ═══════════════════ 导出 ═══════════════════

	def _import_settings(self) -> None:
		path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "",
			"CSV 文件 (*.csv);;所有文件 (*.*)")
		if not path:
			return
		from ocr_engine import parse_csv_header
		settings = parse_csv_header(path)
		if not settings:
			QMessageBox.warning(self, "导入失败", "无法解析 CSV 文件头。")
			return
		if "roi" in settings:
			try:
				parts = [int(x.strip()) for x in settings["roi"].split(",")]
				if len(parts) == 4:
					# Block signals to prevent _on_roi_spin clamping during bulk set
					for s in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
						s.blockSignals(True)
					self.roi_x2.setValue(parts[2]); self.roi_y2.setValue(parts[3])
					self.roi_x1.setValue(parts[0]); self.roi_y1.setValue(parts[1])
					for s in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
						s.blockSignals(False)
					self._redraw()
			except ValueError:
				pass
		for key, widget, cast in [
			("max_speed", self.max_speed_edit, str),
			("max_accel", self.max_accel_edit, str),
			("div", self.div_spin, lambda v: int(float(v))),
			("target_h", self.target_h_edit, str),
			("pad", self.pad_edit, str),
			("buffer", self.buffer_edit, str),
			("frame_start", self.frame_start_edit, str),
			("frame_end", self.frame_end_edit, str),
		]:
			if key in settings:
				try:
					val = cast(settings[key])
					if hasattr(widget, "setValue"):
						widget.setValue(val)
					else:
						widget.setText(str(val))
				except Exception:
					pass
		if "backend" in settings:
			be = settings["backend"].lower()
			idx = {"auto": 0, "cuda": 1, "cpu": 2}.get(be, 0)
			self.backend_combo.setCurrentIndex(idx)
		if "model" in settings:
			model = settings["model"]
			idx = {"v6_tiny": 0, "v6_small": 1}.get(model, 1)
			self.model_combo.setCurrentIndex(idx)
		if "reocr_model" in settings:
			rmodel = settings["reocr_model"]
			idx = {"v6_tiny": 1, "v6_small": 2}.get(rmodel, 0)
			self.reocr_model_combo.setCurrentIndex(idx)
		if "format" in settings:
			fmt = settings["format"].lower()
			for rb, key in [(self._fmt_ms, "m/s"), (self._fmt_kmh, "km/h"),
			                (self._fmt_mph, "mile/h")]:
				if key == fmt:
					rb.setChecked(True); break
		if "manual_anchor" in settings:
			self.mode_auto.setChecked(False)
			self.mode_baseline.setChecked(True)
		elif "auto_anchor" in settings:
			self.mode_baseline.setChecked(False)
			self.mode_auto.setChecked(True)
		self._status_label.setText(f"已导入设置: {Path(path).name}")

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
			fd = self.div_spin.value(); th = float(self.target_h_edit.text())
			pp = float(self.pad_edit.text()); nw = int(self.buffer_edit.text())
			be = ["auto", "cuda", "cpu"][self.backend_combo.currentIndex()]
		except ValueError:
			QMessageBox.warning(self, "参数错误", "请检查数值参数。"); return

		self._export_btn.setEnabled(False); self._cancel_btn.setEnabled(True)
		self._export_thread = _ExportThread(self, Path(out), roi, ms, ma, fd, th, pp, nw, be)
		self._export_thread._progress.connect(self._on_progress)
		self._export_thread._finished.connect(self._on_done)
		self._export_thread._review_data.connect(self._on_review_needed)
		self._export_thread._error.connect(self._on_error)
		self._export_thread._cancelled.connect(self._on_cancel)
		self._export_thread.start()

	def _cancel_export(self) -> None:
		if self._export_thread: self._export_thread._cancel_flag = True
		self._cancel_btn.setEnabled(False); self._status_label.setText("正在取消...")

	def _on_progress(self, msg: str, pct: float) -> None:
		self._status_label.setText(msg); self._progress_bar.setValue(int(pct))

	def _on_review_needed(self, rows: list, observations: list,
			confidences: list[dict], segments: list[dict]) -> None:
		self._review_rows = rows
		self._review_observations = observations
		self._review_confidences = confidences
		self._review_segments = segments

	def _on_done(self, mode: str) -> None:
		if mode == "review":
			self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
			self._export_thread = None
			self._show_review_dialog()
		else:
			self._finish_export()
			self._status_label.setText("自动锚点完成 — 结果已保存。")

	def _on_error(self, err: str) -> None:
		self._finish_export(); QMessageBox.critical(self, "导出失败", err)

	def _on_cancel(self) -> None:
		self._finish_export(); self._status_label.setText("已取消。")

	def _show_review_dialog(self) -> None:
		try:
			ms = float(self.max_speed_edit.text())
		except ValueError:
			ms = 400.0
		dlg = ReviewDialog(self, self._review_rows, self._review_observations,
			self._review_raw_frames, self._review_confidences,
			self._review_segments, ms)
		if dlg.exec() == QDialog.DialogCode.Accepted:
			corrections = dlg.get_corrections()
			partial_corrections = dlg.get_partial_corrections()
			self._review_confirmed = dlg.get_confirmed()
			try:
				self._continue_with_manual_anchors(corrections, partial_corrections)
			except Exception as e:
				self._progress_bar.setValue(0)
				self._status_label.setText(f"审核失败: {e}")
				import traceback; traceback.print_exc()

	def _continue_with_manual_anchors(self, corrections: dict[int, float],
	                                   partial_corrections: dict[int, str] | None = None) -> None:
		"""非阻塞执行 pass2：使用 QThread 避免 GUI 冻结。"""
		pipeline = getattr(self, "_pipeline", None)
		if pipeline is None:
			self._status_label.setText("错误: 处理状态丢失"); return

		self._export_btn.setEnabled(False)
		self._status_label.setText("正在应用审核修正...")
		self._progress_bar.setValue(0)

		app = self  # 捕获 RaceVideoToLogApp 引用
		out_path = getattr(app, "_review_output_path",
			app.video_path.parent / f"{app.video_path.stem}_log.csv")
		confirmed = getattr(app, "_review_confirmed", set())

		class _Pass2Thread(QThread):
			_finished = Signal(bool, str)

			def run(self):
				try:
					pipeline.run_review_pass2(
						corrections=corrections,
						confirmed_segments=confirmed,
						output_path=out_path,
						partial_corrections=partial_corrections or None,
					)
					self._finished.emit(True,
						f"人工审核完成 — {len(corrections)} 帧已修正，结果已保存。")
				except Exception as exc:
					import traceback; traceback.print_exc()
					self._finished.emit(False, f"审核失败: {exc}")
		self._pass2_thread = _Pass2Thread(self)
		self._pass2_thread._finished.connect(self._on_pass2_done)
		self._pass2_thread.start()

	def _on_pass2_done(self, success: bool, message: str) -> None:
		self._export_btn.setEnabled(True)
		self._progress_bar.setValue(100 if success else 0)
		self._status_label.setText(message)
		self._pass2_thread = None

	def _finish_export(self) -> None:
		self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
		self._export_thread = None; self._release_engines()
	def _on_pivot(self, key: str) -> None:
		if key == "analysis":
			self._footer.hide()
		else:
			self._footer.show()


