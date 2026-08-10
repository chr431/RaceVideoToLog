"""RaceVideoToLog PySide6 GUI — 主窗口。

Import 自 gui_export、gui_settings 等子模块，保持 gui.py 作为对外入口。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ocr_native import OcrEngine
    from decord import VideoReader

import config

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QDialog, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer
from widget_utils import make_static_card, disable_spin_flyout, set_value_silent
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
)

from ocr_engine import (
    VideoMetadata,
    format_duration,
)
from gui_analysis import AnalysisTab
from gui_review import ReviewDialog
from gui_export import ExportThread
from gui_preview import PreviewWidget
from gui_settings import build_settings_panel
from theme_manager import ThemeManager

from qfluentwidgets import (setTheme, Theme, isDarkTheme,
    PushButton, PrimaryPushButton,
    BodyLabel, StrongBodyLabel, CaptionLabel, Slider, ProgressBar, CompactSpinBox, Pivot)

# ── qfluentwidgets watcher 保护 ──
# widget 销毁时 Paint/DynamicPropertyChange 事件在途会触发
# "Internal C++ object already deleted"（PySide6 已知问题，真实显示器上
# 关闭窗口/对话框时都可能出现）。捕获 RuntimeError 并忽略，避免 stderr 刷屏。
# 这是对第三方包内部事件过滤器的安全护栏，不影响正常样式应用。
import qfluentwidgets.common.style_sheet as _qfw_ss  # noqa: E402

for _watcher_cls in (_qfw_ss.CustomStyleSheetWatcher,
                     _qfw_ss.DirtyStyleSheetWatcher):
    _orig_event_filter = _watcher_cls.eventFilter

    def _safe_event_filter(self, obj, e, _orig=_orig_event_filter):
        try:
            return _orig(self, obj, e)
        except RuntimeError:
            # 底层 C++ 对象已删除（widget 销毁竞态）→ 不处理该事件
            return False

    _watcher_cls.eventFilter = _safe_event_filter


def _t(mark: str) -> None:
    """GUI 计时打点：写 %LOCALAPPDATA%/RaceVideoToLog/gui_timing.log（排查 EXE 卡顿）。"""
    from monitor import gui_mark
    gui_mark(mark)


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
        self.first_frame_qimg: QImage | None = None
        self._preview_vr: "VideoReader | None" = None  # decord VideoReader
        self._preview_frame_no: int = 0
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._show_throttled_frame)
        self.ocr_engine: "OcrEngine | None" = None

        self._export_thread: ExportThread | None = None
        self.speed_format: str = config.DEFAULT_SPEED_FORMAT

        self._review_output_path: Path | None = None

        # ── Settings widgets dict (populated by _build_ocr_tab) ──
        self._settings: dict = {}

        # 预览

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

        # 左侧面板 — 使用 gui_settings.build_settings_panel
        left = QWidget(); left.setFixedWidth(450)
        self._settings = build_settings_panel(left)
        main_w.addWidget(left)

        # 右侧 = 识别范围 + 预览
        right = QVBoxLayout(); right.setSpacing(8)
        self._build_right_panel(right)
        main_w.addLayout(right, 1)

        layout.addLayout(main_w, 1)

    def _build_right_panel(self, rl: QVBoxLayout) -> None:
        # 识别范围 Card
        roi_card = make_static_card()
        rgl = QGridLayout(roi_card)
        rgl.addWidget(StrongBodyLabel("识别范围（像素）"), 0, 0, 1, 4)
        self.roi_x1 = CompactSpinBox()
        self.roi_y1 = CompactSpinBox()
        self.roi_x2 = CompactSpinBox()
        self.roi_y2 = CompactSpinBox()
        for s in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
            s.setRange(0, 9999); s.setFixedWidth(80)
            s.valueChanged.connect(lambda v, spin=s: self._on_roi_spin(spin))
            disable_spin_flyout(s)
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
        self._preview_widget = PreviewWidget()
        self._preview_widget.roi_dragged.connect(self._on_preview_roi)
        pvl.addWidget(self._preview_widget, 1)

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
        s = self._settings
        s["format_ms"].clicked.connect(lambda: self._on_fmt("m/s"))
        s["format_kmh"].clicked.connect(lambda: self._on_fmt("km/h"))
        s["format_mph"].clicked.connect(lambda: self._on_fmt("mile/h"))
        # Timeline set-to-current buttons
        s["_set_start_btn"].clicked.connect(lambda: s["frame_start_edit"].setText(str(self._slider.value())))
        s["_set_end_btn"].clicked.connect(lambda: s["frame_end_edit"].setText(str(self._slider.value())))

    def _add_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self._step(10))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self._step(-10))

    def _on_fmt(self, fmt: str) -> None: self.speed_format = fmt

    # ═══════════════════ 主题切换 ═══════════════════

    def _register_theme_callbacks(self) -> None:
        # 主窗口背景色
        def _update_bg(dark: bool) -> None:
            bg = config.CANVAS_BG_DARK if dark else config.CANVAS_BG_LIGHT
            fg = config.CANVAS_FG_DARK if dark else config.CANVAS_FG_LIGHT
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
        # 保存句柄，closeEvent 时注销，避免窗口销毁后 ThemeManager 仍触发
        self._theme_callbacks = [_update_bg, _update_titlebar, _update_icon]

    def closeEvent(self, event) -> None:
        """关闭前清理：取消导出线程 + 注销主题回调。

        防止导出线程/主题回调在窗口销毁后访问已删除 widget（"Internal
        C++ object already deleted" 的常见来源）。全程防御式，避免 closeEvent
        自身异常放大事件过滤器级联错误。
        """
        try:
            self._cancel_export()
        except Exception:
            pass
        th = getattr(self, "_export_thread", None)
        if th is not None:
            try:
                th.wait(3000)  # 取消标志已设，worker 应在解码检查点退出
            except Exception:
                pass
            try:
                self._teardown_export_thread()
            except Exception:
                pass
        for cb in getattr(self, "_theme_callbacks", []):
            try:
                ThemeManager.unregister(cb)
            except Exception:
                pass
        super().closeEvent(event)

    def _toggle_theme(self) -> None:
        from qfluentwidgets import qconfig
        if qconfig.theme == Theme.DARK:
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.DARK)
        ThemeManager.refresh()

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
        from video_utils import open_decord_vr

        _t("load_video: start")
        vr, label = open_decord_vr(str(path))
        _t("load_video: decord open")
        # 编码信息直接来自 decord（自建版新增 get_codec），无子进程开销
        try:
            codec = vr.get_codec() or "?"
        except Exception:
            codec = "?"
        _t("load_video: codec")
        try:
            fc = len(vr)
            fps = vr.get_avg_fps()
            first = vr[0].asnumpy()  # decord returns RGB
            _t("load_video: first frame")
            h, w = first.shape[:2]
            dur = fc / fps if fps > 0 else 0.0
        except Exception:
            del vr
            raise RuntimeError("无法读取视频第一帧。")

        if self._preview_vr is not None:
            del self._preview_vr
        self._preview_vr = vr
        self._preview_frame_no = 0

        self.video_path = path
        self.metadata = VideoMetadata(path=path, duration_sec=dur, width=w, height=h,
            fps=fps, codec=codec, frame_count=fc)
        hh, ww, ch = first.shape
        self.first_frame_qimg = QImage(first.data, ww, hh, ch * ww,
            QImage.Format.Format_RGB888).copy()

        self._file_label.setText(str(path))
        self._dur_label.setText(format_duration(dur))
        self._res_label.setText(f"{w} x {h}")
        self._fps_label.setText(f"{fps:.3f}" if fps > 0 else "Unknown")
        self._codec_label.setText(codec)
        self._status_label.setText("视频已载入，请输入识别范围并预览。")
        self._slider.setRange(0, fc - 1); self._slider.setValue(0)
        self._frame_label.setText(f"#{0}/{fc}")
        self._preview_widget.set_video_size(w, h)
        self._preview_widget.set_roi(self.roi_x1.value(), self.roi_y1.value(),
                                     self.roi_x2.value(), self.roi_y2.value())
        self._show_frame(0)
        for s, m in [(self.roi_x1, w), (self.roi_y1, h), (self.roi_x2, w), (self.roi_y2, h)]:
            s.setMaximum(m - 1)

    # ═══════════════════ 预览 ═══════════════════

    def _on_preview_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """预览拖拽 ROI → 同步 spinbox（静默赋值，不触发联动校验）。"""
        for s, v in [(self.roi_x1, x1), (self.roi_y1, y1),
                     (self.roi_x2, x2), (self.roi_y2, y2)]:
            set_value_silent(s, v)


    def _show_frame(self, frame_no: int) -> None:
        pm = None
        vr = self._preview_vr
        if frame_no > 0 and vr is not None and frame_no < len(vr):
            try:
                frame = vr[frame_no].asnumpy()  # decord returns RGB
                h, w, ch = frame.shape
                qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                pm = QPixmap.fromImage(qimg)
            except Exception:
                pass
        if pm is None and self.first_frame_qimg is not None:
            pm = QPixmap.fromImage(self.first_frame_qimg)
        if pm is not None:
            self._preview_widget.set_frame(pm)

    def _on_slider(self, value: int) -> None:
        if self.metadata:
            self._frame_label.setText(f"#{value}/{self.metadata.frame_count}")
        if self._throttle_timer:
            self._throttle_timer.stop()
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(lambda: self._show_frame(value))
        self._throttle_timer.start(30)

    def _show_throttled_frame(self) -> None:
        self._show_frame(self._slider.value())

    def _step(self, delta: int) -> None:
        if not self.metadata: return
        v = max(0, min(self.metadata.frame_count - 1, self._slider.value() + delta))
        self._slider.setValue(v)

    def _on_roi_spin(self, spin) -> None:
        if spin is self.roi_x1 and self.roi_x1.value() > self.roi_x2.value() - 1:
            spin.blockSignals(True); spin.setValue(self.roi_x2.value() - 1); spin.blockSignals(False)
        elif spin is self.roi_x2 and self.roi_x2.value() < self.roi_x1.value() + 1:
            spin.blockSignals(True); spin.setValue(self.roi_x1.value() + 1); spin.blockSignals(False)
        elif spin is self.roi_y1 and self.roi_y1.value() > self.roi_y2.value() - 1:
            spin.blockSignals(True); spin.setValue(self.roi_y2.value() - 1); spin.blockSignals(False)
        elif spin is self.roi_y2 and self.roi_y2.value() < self.roi_y1.value() + 1:
            spin.blockSignals(True); spin.setValue(self.roi_y1.value() + 1); spin.blockSignals(False)
        self._preview_widget.set_roi(
            self.roi_x1.value(), self.roi_y1.value(),
            self.roi_x2.value(), self.roi_y2.value())

    # ═══════════════════ OCR 引擎 ═══════════════════

    def _release_engines(self) -> None:
        for e in ([self.ocr_engine] if self.ocr_engine else []):
            try: del e
            except Exception: pass
        self.ocr_engine = None
        import gc; gc.collect()

    # ═══════════════════ 导出 ═══════════════════

    def _import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "",
            "CSV 文件 (*.csv);;所有文件 (*.*)")
        if not path:
            return
        from ocr_engine import parse_csv_header, parse_csv_setting
        settings = parse_csv_header(path)
        if not settings:
            QMessageBox.warning(self, "导入失败", "无法解析 CSV 文件头。")
            return
        s = self._settings

        # ── ROI（特殊：4 个 spinbox）──
        if "roi" in settings:
            parts = parse_csv_setting("roi", settings["roi"])
            if isinstance(parts, (list, tuple)) and len(parts) == 4:
                for spin in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
                    spin.blockSignals(True)
                self.roi_x2.setValue(parts[2]); self.roi_y2.setValue(parts[3])
                self.roi_x1.setValue(parts[0]); self.roi_y1.setValue(parts[1])
                for spin in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
                    spin.blockSignals(False)
                self._preview_widget.set_roi(
                    self.roi_x1.value(), self.roi_y1.value(),
                    self.roi_x2.value(), self.roi_y2.value())

        # ── 数值字段（统一使用共享解析器）──
        _num_fields = {
            "max_speed": s["max_speed_edit"], "max_accel": s["max_accel_edit"],
            "frame_start": s["frame_start_edit"], "frame_end": s["frame_end_edit"],
        }
        for key, widget in _num_fields.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                widget.setText(str(val))

        _spin_fields = {
            "buffer": s["buffer_spin"], "target_h": s["target_h_spin"],
            "max_width": s["max_width_spin"],
            "pad": s["pad_spin"],
        }
        for key, widget in _spin_fields.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                widget.setValue(val)

        # ── 下拉框字段 ──
        _combo_map = {
            "backend": (s["backend_combo"], {
                "auto": 0, "cpu": 1, "nvdec": 2,
                "decord/cpu": 1, "decord/gpu": 2,
            }),
        }
        for key, (combo, mapping) in _combo_map.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                idx = mapping.get(str(val).lower(), 0)
                combo.setCurrentIndex(idx)
        if "format" in settings:
            fmt = settings["format"].lower()
            for rb, key in [(s["format_ms"], "m/s"), (s["format_kmh"], "km/h"),
                            (s["format_mph"], "mile/h")]:
                if key == fmt:
                    rb.setChecked(True); break
        self._status_label.setText(f"已导入设置: {Path(path).name}")

    def _export_csv(self) -> None:
        if self.video_path is None or self.metadata is None:
            QMessageBox.warning(self, "未导入视频", "请先导入视频。"); return
        roi = (self.roi_x1.value(), self.roi_y1.value(),
               self.roi_x2.value(), self.roi_y2.value())
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            QMessageBox.warning(self, "识别范围不完整", "请先填写或拖拽选择识别范围。"); return

        out, _ = QFileDialog.getSaveFileName(self, "保存 CSV",
            str(self.video_path.parent / f"{self.video_path.stem}_log.csv"),
            "CSV 文件 (*.csv)")
        if not out: return

        s = self._settings
        try:
            ms = float(s["max_speed_edit"].text())
            ma = float(s["max_accel_edit"].text())
            bu = s["buffer_spin"].value(); th = s["target_h_spin"].value()
            mw = s["max_width_spin"].value()
            pp = s["pad_spin"].value()
            monitor_enabled = s["monitor_checkbox"].isChecked()
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请检查数值参数。"); return

        # 断开旧线程信号，防止泄漏到新线程
        self._teardown_export_thread()

        self._export_btn.setEnabled(False); self._cancel_btn.setEnabled(True)

        # ── 检查可选依赖可用性 ──
        try:
            import decord  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "decord 加载失败",
                "视频解码需要自建 decord fork（PyPI 版不支持）。\n\n"
                "修复：运行 setup_venv.bat，或从 chr431/decord 获取发布产物到 _decord_build\\")
            self._finish_export()
            return
        self._export_thread = ExportThread(
            video_path=self.video_path,
            roi=roi,
            max_speed_kmh=ms, max_accel_mps2=ma,
            buffer_size=bu, decode_backend=config.DECODE_BACKEND_KEYS[
                s["backend_combo"].currentIndex()],
            ocr_backend=config.OCR_BACKEND_KEYS[
                s["ocr_backend_combo"].currentIndex()],
            target_h=th, pad_px=pp,
            max_width=mw,
            speed_format=self.speed_format,
            frame_start=s["frame_start_edit"].text(),
            frame_end=s["frame_end_edit"].text(),
            monitor_enabled=monitor_enabled,
            output_path=Path(out),
            parent=self,
        )
        self._export_thread.progress_updated.connect(self._on_progress)
        self._export_thread.finished.connect(self._on_done)
        self._export_thread.error_occurred.connect(self._on_error)
        self._export_thread.cancelled.connect(self._on_cancel)
        self._export_thread.pipeline_ready.connect(self._on_pipeline_ready)
        self._export_thread.start()

    def _on_pipeline_ready(self, pipeline, output_path) -> None:
        self._pipeline = pipeline
        self._review_output_path = output_path

    def _cancel_export(self) -> None:
        if self._export_thread: self._export_thread._cancel_flag = True
        self._cancel_btn.setEnabled(False); self._status_label.setText("正在取消...")

    def _on_progress(self, msg: str, pct: float) -> None:
        self._status_label.setText(msg); self._progress_bar.setValue(int(pct))

    def _on_done(self, mode: str) -> None:
        if self.sender() is not self._export_thread:
            return
        self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._export_thread = None
        self._show_final_check()

    def _show_final_check(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        out = getattr(self, "_review_output_path", None)
        if pipeline is None or out is None:
            return
        # 段级 review：段值 + 代表帧预览，用户可改段值（段内不混速度，改段=改全段）
        dlg = ReviewDialog(self, pipeline.segments,
                           pipeline._max_speed, pipeline._max_accel,
                           pipeline._fps or 1.0)
        _t("final_check: before exec")
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        corrections = dlg.get_corrections()
        _t("final_check: dialog closed")
        # finalize（写 CSV/统计）在 frozen 环境可能被安全扫描拖慢数秒 →
        # 后台执行，完成后回主线程收尾
        import threading as _th
        from PySide6.QtCore import QTimer as _QT
        def _finalize_bg():
            _t("final_check: finalize start")
            if accepted and corrections:
                vals = [seg["value"] for seg in pipeline.segments]
                for si, v in corrections.items():
                    if 0 <= si < len(vals):
                        vals[si] = v
                pipeline.finalize(out, vals)
            else:
                pipeline.finalize(out)
            _t("final_check: finalize done")
            _QT.singleShot(0, lambda: (self._finish_export(),
                                       self._status_label.setText("最终检查完成 — 结果已保存。"),
                                       _t("final_check: finish_export done")))
        _th.Thread(target=_finalize_bg, daemon=True).start()

    def _on_error(self, err: str) -> None:
        if self.sender() is not self._export_thread:
            return
        self._finish_export(); QMessageBox.critical(self, "导出失败", err)

    def _on_cancel(self) -> None:
        if self.sender() is not self._export_thread:
            return
        self._finish_export(); self._status_label.setText("已取消。")

    def _teardown_export_thread(self) -> None:
        """拆除导出线程：断开全部信号并释放引用（幂等）。"""
        if self._export_thread is not None:
            try:
                self._export_thread.progress_updated.disconnect()
                self._export_thread.finished.disconnect()
                self._export_thread.error_occurred.disconnect()
                self._export_thread.cancelled.disconnect()
                self._export_thread.pipeline_ready.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._export_thread = None

    def _finish_export(self) -> None:
        self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._teardown_export_thread()
        # Release pipeline memory (crops etc.) on cancel/error
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            import logging
            _log = logging.getLogger("RaceVideoToLog.gui")
            try:
                from video_utils import rss_mb, sum_nbytes
                _raw_mb = sum_nbytes(list(pipeline.crops.values())) / 1e6
                _log.info("[MEM] _finish_export PRE-clear: crops=%d(%.1fMB) rss=%.0fMB",
                    len(pipeline.crops), _raw_mb, rss_mb())
            except Exception:
                pass
            pipeline.crops.clear()
            if getattr(pipeline, '_diag', None):
                pipeline._diag.clear()
            import gc; gc.collect()
            try:
                from video_utils import rss_mb
                _log.info("[MEM] _finish_export POST-clear: rss=%.0fMB", rss_mb())
            except Exception:
                pass
            self._pipeline = None
        _t("finish_export: pipeline cleared")
        self._release_engines()
        _t("finish_export: engines released")

    def _on_pivot(self, key: str) -> None:
        if key == "analysis":
            self._footer.hide()
        else:
            self._footer.show()
