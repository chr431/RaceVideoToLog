"""RaceVideoToLog PySide6 GUI — 主窗口。

Import 自 gui_export、gui_settings 等子模块，保持 gui.py 作为对外入口。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decord import VideoReader

import config

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer
from widget_utils import make_static_card, disable_spin_flyout
from PySide6.QtGui import (
    QImage, QKeySequence, QShortcut,
)

# gui_analysis / gui_review 延迟导入（顶层 import pyqtgraph ~0.8s，
# AnalysisTab 实例化另 ~0.6s —— 移出启动路径，首次用到时再加载）
from export_controller import ExportControllerMixin
from gui_preview import PreviewWidget
from gui_settings import build_settings_panel
from gui_video import VideoLoadMixin
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


class RaceVideoToLogApp(VideoLoadMixin, ExportControllerMixin, QMainWindow):
    """RaceVideoToLog PySide6 主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Race Video To Log")
        self.resize(1500, 920)
        self.setMinimumSize(1100, 760)

        # ── 状态变量 ──
        self.video_path: Path | None = None
        self._pipeline: object | None = None
        self.metadata: object | None = None
        self.first_frame_qimg: QImage | None = None
        self._preview_vr: "VideoReader | None" = None  # decord VideoReader
        self._preview_frame_no: int = 0
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._show_throttled_frame)

        self._export_thread: object | None = None
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

        # Tab 2: 数据分析（延迟创建：pyqtgraph 加载 + AnalysisTab 实例化
        # ~1.4s 移出启动路径，首次切到该 tab 时才加载）
        self._analysis_tab = None

        def _ensure_analysis_tab():
            if self._analysis_tab is None:
                from gui_analysis import AnalysisTab
                self._analysis_tab = AnalysisTab(self._tab_stack)
            return self._analysis_tab

        self._ensure_analysis_tab = _ensure_analysis_tab
        ThemeManager.register(
            lambda dark: self._analysis_tab._sync_figure_theme()
            if self._analysis_tab else None)
        self._tab_pivot.addItem(
            'analysis', '数据分析',
            lambda: (self._ensure_analysis_tab(),
                     self._tab_stack.setCurrentWidget(self._analysis_tab)))
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
        s["log_level_combo"].currentIndexChanged.connect(self._on_log_level_changed)

    def _add_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self._step(10))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self._step(-10))

    def _on_fmt(self, fmt: str) -> None: self.speed_format = fmt

    def _on_log_level_changed(self, index: int) -> None:
        from logging_setup import configure_logging
        configure_logging(("normal", "detailed", "debug")[index])

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

    def _on_pivot(self, key: str) -> None:
        if key == "analysis":
            self._footer.hide()
        else:
            self._footer.show()
