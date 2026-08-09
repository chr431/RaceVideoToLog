"""最终检查对话框（段级）— 段值曲线 + 代表帧预览 + 段值修正。

分段已验证不混合速度（可用 truth 视频 0 混合段），段值对段内所有帧可靠；
段级修正即可保证全段正确。橙点 = 需审核段（管线已纠正 / OCR 未读出）。
"""
from __future__ import annotations

import numpy as np

import config

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor
from theme_manager import ThemeManager
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, isDarkTheme)
from widget_utils import (make_static_card, disable_spin_flyout,
    make_brush, ModPlotWidget)
from config import (COLOR_RED, COLOR_ORANGE, COLOR_BLUE,
    COLOR_LIGHT_GRAY, COLOR_LIGHTER_GRAY, chart_colors)
import pyqtgraph as pg


class ReviewDialog(QDialog):
    """段级最终检查 — 段值曲线 + 代表帧预览 + 段值修正。"""

    def __init__(self, parent: QWidget, segments: list[dict],
                 max_speed: float,
                 max_accel: float = config.DEFAULT_MAX_ACCEL,
                 fps: float = 1.0) -> None:
        super().__init__(parent)
        self.setWindowTitle("最终检查 — 点击图中任意段修正该段速度")
        self.resize(1200, 750)
        self.setMinimumSize(1000, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._segments = segments
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._fps = fps
        # 修正：段索引 → 新值（应用到整个段范围）
        self._corrections: dict[int, float] = {}
        self._current_seg: int = 0

        self._build_ui()
        self._register_theme_callbacks()

    # ═══════════════ UI 构建 ═══════════════

    def _register_theme_callbacks(self) -> None:
        def _update(dark: bool) -> None:
            bg = QColor(config.CANVAS_BG_DARK if dark else config.CANVAS_BG_LIGHT)
            fg = QColor(config.CANVAS_FG_DARK if dark else config.CANVAS_FG_LIGHT)
            btn_bg = QColor("#3a3a3a" if dark else "#e8e8e8")
            img_bg = config.PREVIEW_BG if dark else config.PREVIEW_BG_LIGHT
            p = self.palette()
            for role, color in [(QPalette.ColorRole.Window, bg), (QPalette.ColorRole.Base, btn_bg),
                                (QPalette.ColorRole.WindowText, fg), (QPalette.ColorRole.Text, fg),
                                (QPalette.ColorRole.Button, btn_bg), (QPalette.ColorRole.ButtonText, fg)]:
                p.setColor(role, color)
            self.setPalette(p)
            self._img_label.setStyleSheet(f"background-color: {img_bg}; border-radius: 4px;")
            if hasattr(self, '_figure'): self._redraw_chart()
        self._theme_cb = ThemeManager.register(_update)
        _update(isDarkTheme())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        n_review = self._needs_review_count()
        header.addWidget(StrongBodyLabel("最终检查"))
        header.addWidget(CaptionLabel(
            f"  — 点击散点选段修正，橙色=需审核({n_review}段)，完成后点击「确认保存」"))
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

        chart_card = make_static_card()
        cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
        cl.addWidget(CaptionLabel("段值曲线（点击数据点选段，蓝点=已修正，橙点=需审核，红圈=当前段）"))
        self._canvas = self._create_chart()
        cl.addWidget(self._canvas, 1)
        rl.addWidget(chart_card, 2)

        bottom_row = QHBoxLayout(); bottom_row.setSpacing(8)

        img_card = make_static_card()
        il = QVBoxLayout(img_card); il.setContentsMargins(8, 8, 8, 4)
        il.addWidget(CaptionLabel("当前段代表帧原始图像（ROI 裁剪区域）"))
        self._img_label = QLabel("选择段后显示")
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(120, 80)
        self._img_label.setStyleSheet(f"background-color: {config.PREVIEW_BG}; border-radius: 4px;")
        il.addWidget(self._img_label, 1)
        bottom_row.addWidget(img_card, 1)

        ctrl_card = make_static_card()
        ctrl = QVBoxLayout(ctrl_card); ctrl.setContentsMargins(8, 8, 8, 4)

        cf_row = QHBoxLayout()
        cf_row.addWidget(BodyLabel("当前段: "))
        self._seg_label = BodyLabel("—")
        cf_row.addWidget(self._seg_label)
        cf_row.addSpacing(12)
        self._speed_value_label = BodyLabel("")
        cf_row.addWidget(self._speed_value_label)
        cf_row.addStretch()
        ctrl.addLayout(cf_row)

        ctrl.addSpacing(4)
        cr = QHBoxLayout()
        cr.addWidget(BodyLabel("修正速度"))
        from qfluentwidgets import CompactSpinBox
        self._speed_edit = CompactSpinBox()
        self._speed_edit.setFixedWidth(110)
        self._speed_edit.setRange(0, int(self._max_speed))
        self._speed_edit.setSuffix(" km/h")
        self._speed_edit.setSpecialValueText("(无效)")
        disable_spin_flyout(self._speed_edit)
        self._speed_edit.valueChanged.connect(self._on_spinbox_changed)
        self._speed_edit.installEventFilter(self)
        cr.addWidget(self._speed_edit)
        cr.addStretch()
        ctrl.addLayout(cr)

        btn_row = QHBoxLayout()
        btn_add = PrimaryPushButton("添加修正")
        btn_add.setFixedWidth(100)
        btn_add.clicked.connect(self._add_correction)
        btn_row.addWidget(btn_add)
        self._btn_delete = PushButton("删除修正")
        self._btn_delete.setFixedWidth(100)
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_correction)
        btn_row.addWidget(self._btn_delete)
        btn_row.addStretch()
        ctrl.addLayout(btn_row)

        bottom_row.addWidget(ctrl_card, 2)
        rl.addLayout(bottom_row, 1)

        btn_finish = PrimaryPushButton("确认保存")
        btn_finish.setFixedWidth(150)
        finish_row = QHBoxLayout()
        finish_row.addStretch()
        finish_row.addWidget(btn_finish)
        btn_finish.clicked.connect(self._finish)
        rl.addLayout(finish_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self._shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._shortcut_left.activated.connect(self._on_left_key)
        self._shortcut_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._shortcut_right.activated.connect(self._on_right_key)
        self._shortcut_up = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        self._shortcut_up.activated.connect(self._on_up_key)
        self._shortcut_down = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        self._shortcut_down.activated.connect(self._on_down_key)
        self.setFocus()

    # ═══════════════ 段数据 ═══════════════

    def _seg_value(self, si: int) -> float | None:
        if si in self._corrections:
            return self._corrections[si]
        return self._segments[si].get("value")

    def _needs_review(self, si: int) -> bool:
        """需审核段：管线纠正过（value≠ocr_value）或 OCR 未读出。"""
        seg = self._segments[si]
        ov = seg.get("ocr_value")
        cv = seg.get("value")
        if cv is None:
            return True
        if ov is not None and ov != cv:
            return True
        return False

    def _needs_review_count(self) -> int:
        return sum(1 for si in range(len(self._segments)) if self._needs_review(si))

    # ═══════════════ 图表 ═══════════════

    def _create_chart(self):
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=False)
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        plot = ModPlotWidget()
        plot.setMinimumHeight(150)
        plot.setBackground(bg)
        plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        vb = plot_item.getViewBox()
        assert vb is not None
        vb.setBorder(pg.mkPen(fg))
        vb.setMouseEnabled(x=True, y=True)
        self._plot = plot
        self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
        self._chart_artists = {}
        self._saved_range = None
        self._user_zoomed = False
        vb.sigRangeChangedManually.connect(self._on_range_changed)
        self._setup_hover(plot)
        self._redraw_chart(plot)
        return plot

    def _on_range_changed(self, mask) -> None:
        self._user_zoomed = True
        plot_item = self._plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        self._saved_range = vb.viewRange()

    def _setup_hover(self, plot) -> None:
        import pyqtgraph as pg
        from PySide6.QtCore import Qt as _Qt
        self._hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(
            config.COLOR_GRAY, width=1, style=_Qt.PenStyle.DashLine))
        self._hover_line.setVisible(False)
        plot.addItem(self._hover_line, ignoreBounds=True)
        fg = chart_colors(isDarkTheme())[1]
        self._hover_text = pg.TextItem("", color=fg, anchor=(0, 0))
        self._hover_text.setVisible(False)
        plot.addItem(self._hover_text, ignoreBounds=True)
        if not hasattr(self, '_hover_connected'):
            plot.scene().sigMouseMoved.connect(self._on_hover_moved)
            plot.plotItem.vb.sigRangeChanged.connect(self._pin_hover_text)
            self._hover_connected = True
        self._pin_hover_text()

    def _pin_hover_text(self, *args) -> None:
        plot_item = self._plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        self._hover_text.setPos(xmin, ymax)

    def _on_hover_moved(self, pos) -> None:
        plot = self._plot
        plot_item = plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        if not plot.sceneBoundingRect().contains(pos):
            self._hover_line.setVisible(False)
            self._hover_text.setVisible(False)
            return
        pt = vb.mapSceneToView(pos)
        x = pt.x()
        cache = getattr(self, '_chart_cache', None)
        xs = (cache or {}).get('xs') or []
        sidx = (cache or {}).get('sidx') or []
        if not xs:
            return
        import bisect
        idx = bisect.bisect_left(xs, x)
        if idx >= len(xs):
            idx = len(xs) - 1
        elif idx > 0 and abs(xs[idx - 1] - x) < abs(xs[idx] - x):
            idx -= 1
        si = sidx[idx]
        seg = self._segments[si]
        v = self._seg_value(si)
        self._hover_line.setPos(x)
        self._hover_line.setVisible(True)
        text = (f"段#{si} [{seg['start']}-{seg['end']}]: {v:.0f} km/h"
                if v is not None and v >= 0 else f"段#{si}: (无效)")
        self._hover_text.setText(text)
        self._hover_text.setVisible(True)

    def _frame_data(self):
        """逐帧数据：所有采样帧的 (x=帧号, y=段值, 段索引, 连线数组)。

        y 取当前段值（含修正预览）；None 段 y=-1（无效标记）。连线数组
        connect[i]=同段相邻帧 → 段内连线、段间断开。
        """
        xs, ys, sidx = [], [], []
        for si, seg in enumerate(self._segments):
            v = self._seg_value(si)
            frames = seg.get('frames') or [seg['start']]
            xs.extend(frames)
            ys.extend([v if v is not None else -1] * len(frames))
            sidx.extend([si] * len(frames))
        n = len(xs)
        # connect 数组长度 = N（点数）：True 连接点 i → i+1（同段相连，段间断开）
        sidx_arr = np.asarray(sidx)
        conn = np.zeros(n, dtype=bool)
        if n > 1:
            conn[:-1] = sidx_arr[1:] == sidx_arr[:-1]
        return xs, ys, sidx, conn

    def _redraw_chart(self, plot=None) -> None:
        if plot is None:
            plot = self._plot
        plot_item = plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        saved_range = self._saved_range if getattr(self, '_user_zoomed', False) else None

        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        prev_dark = getattr(self, '_chart_params', {}).get('dark')
        prev_corr = getattr(self, '_chart_corrections_fs', None)
        cur_corr = frozenset(self._corrections.keys())
        self._chart_corrections_fs = cur_corr

        xs, ys, sidx, conn = self._frame_data()
        prev_data = self._chart_cache.get('data_hash', 0) if hasattr(self, '_chart_cache') else 0
        data_hash = hash((len(xs), xs[0] if xs else 0, xs[-1] if xs else 0,
                          sum(ys), len(self._corrections)))
        self._chart_cache = {'xs': xs, 'ys': ys, 'sidx': sidx,
                             'data_hash': data_hash}
        needs_rebuild = (prev_dark != dark or not hasattr(self, '_chart_cache')
                         or prev_corr != cur_corr or prev_data != data_hash)

        if needs_rebuild:
            plot.clear()
            self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
            plot.setBackground(bg)
            plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
            self._setup_hover(plot)
            self._chart_artists = {}

            gray_c = COLOR_LIGHT_GRAY if not dark else COLOR_LIGHTER_GRAY
            # 段连线（step 曲线：同段相邻帧连线，段间断开）—— 显示全部点
            line = pg.PlotDataItem(xs, ys, connect=conn,
                                   pen=pg.mkPen(gray_c, width=1))
            plot.addItem(line)
            self._chart_artists['seg_line'] = line

            # 散点分层：正常灰 / 需审核橙 / 已修正蓝
            gx, gy, gi, ox, oy, oi = [], [], [], [], [], []
            for k in range(len(xs)):
                si = sidx[k]
                if si in self._corrections:
                    continue
                if ys[k] < 0 or self._needs_review(si):
                    ox.append(xs[k]); oy.append(ys[k]); oi.append(si)
                else:
                    gx.append(xs[k]); gy.append(ys[k]); gi.append(si)
            gray = pg.ScatterPlotItem(size=4, brush=make_brush(gray_c, 100), pen=None)
            gray.setData(x=gx, y=gy, data=gi)
            gray.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(gray)
            self._chart_artists['bg_gray'] = gray
            orange = pg.ScatterPlotItem(size=5, brush=make_brush(COLOR_ORANGE, 180), pen=None)
            orange.setData(x=ox, y=oy, data=oi)
            orange.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(orange)
            self._chart_artists['bg_orange'] = orange

            corr = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(COLOR_BLUE),
                                      pen=pg.mkPen('w', width=1.0))
            plot.addItem(corr)
            self._chart_artists['corrections'] = corr
            cur = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(COLOR_RED),
                                     pen=pg.mkPen('w', width=1.5))
            plot.addItem(cur)
            self._chart_artists['cur_highlight'] = cur
            self._update_corr_and_cur()

            plot.setLabel('bottom', '帧', color=fg)
            plot.setLabel('left', '速度 (km/h)', color=fg)
            plot_item = plot.getPlotItem()
            assert plot_item is not None
            for ax_name in ('left', 'bottom'):
                ax_item = plot_item.getAxis(ax_name)
                ax_item.setTextPen(fg)
                ax_item.setPen(fg)
            vb.setBorder(pg.mkPen(fg))
        else:
            self._update_corr_and_cur()

        if saved_range is not None:
            xr, yr = saved_range
            vb.setXRange(xr[0], xr[1], padding=0)
            vb.setYRange(yr[0], yr[1], padding=0)
        vb.enableAutoRange(None, False)

    def _update_corr_and_cur(self) -> None:
        corr = self._chart_artists.get('corrections')
        cur = self._chart_artists.get('cur_highlight')
        if corr is None or cur is None:
            return
        cache = self._chart_cache
        xs = cache.get('xs') or []
        ys = cache.get('ys') or []
        sidx = cache.get('sidx') or []
        # 修正蓝点：已修正段的全部帧
        corr_set = set(self._corrections.keys())
        cx = [xs[k] for k in range(len(xs)) if sidx[k] in corr_set]
        cy = [ys[k] for k in range(len(xs)) if sidx[k] in corr_set]
        corr.setData(x=cx, y=cy)
        # 当前红点：选中段的全部帧（水平 run）
        si = self._current_seg
        sel = [k for k in range(len(xs)) if sidx[k] == si]
        cx2 = [xs[k] for k in sel]
        cy2 = [ys[k] for k in sel]
        cur.setData(x=cx2, y=cy2)
        cur.setBrush(make_brush(COLOR_BLUE if si in self._corrections else COLOR_RED))

    def _on_scatter_clicked(self, scatter, points) -> None:
        for pt in points:
            idx = pt.data()
            if idx is not None:
                self._navigate_to(int(idx))
                break

    # ═══════════════ 图像 + 导航 ═══════════════

    def _show_seg_image(self, si: int) -> None:
        if 0 <= si < len(self._segments):
            crop = self._segments[si].get("rep_crop")
            if crop is not None and crop.size > 0:
                rgb = crop
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pm = QPixmap.fromImage(qimg)
                lw = max(50, self._img_label.width() - 8); lh = max(50, self._img_label.height() - 8)
                scaled = pm.scaled(lw, lh,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
                self._img_label.setPixmap(scaled)
                return
        self._img_label.setText("(无图像)")

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                self.keyPressEvent(event)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            return
        super().keyPressEvent(event)

    def _on_left_key(self) -> None:
        if self._current_seg > 0:
            self._navigate_to(self._current_seg - 1)

    def _on_up_key(self) -> None:
        v = int(self._speed_edit.value()) + 1
        if v <= int(self._max_speed):
            self._speed_edit.setValue(v)

    def _on_down_key(self) -> None:
        v = int(self._speed_edit.value()) - 1
        if v >= 0:
            self._speed_edit.setValue(v)

    def _on_right_key(self) -> None:
        if self._current_seg < len(self._segments) - 1:
            self._navigate_to(self._current_seg + 1)

    def _speed_input_value(self, si: int) -> int:
        if si in self._corrections:
            return int(self._corrections[si])
        v = self._segments[si].get("value")
        return int(v) if v is not None and v >= 0 else 0

    def _navigate_to(self, si: int) -> None:
        self._current_seg = si
        seg = self._segments[si]
        self._seg_label.setText(f"#{si} [{seg['start']}-{seg['end']}]")
        self._show_seg_image(si)
        self._speed_edit.blockSignals(True)
        self._speed_edit.setValue(self._speed_input_value(si))
        self._speed_edit.blockSignals(False)
        v = self._seg_value(si)
        self._speed_value_label.setText(f"速度: (无效)" if v is None or v < 0
                                        else f"速度: {v:.0f} km/h")
        self._btn_delete.setEnabled(si in self._corrections)
        self._redraw_chart()

    def _on_spinbox_changed(self, value: int) -> None:
        si = self._current_seg
        if si < 0 or si >= len(self._segments):
            return
        # 段值预览：未提交前不写入 corrections
        self._speed_value_label.setText(f"速度: {value:.0f} km/h (预览)")
        self._redraw_chart()

    def closeEvent(self, event) -> None:
        ThemeManager.unregister(getattr(self, '_theme_cb', lambda dark: None))
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_current_seg'):
            self._show_seg_image(self._current_seg)

    # ═══════════════ 修正操作 ═══════════════

    def _check_accel(self, si: int, v: float) -> tuple[bool, str]:
        """段级加速度检查：新值相对相邻段是否物理合理。"""
        n = len(self._segments)
        neighbors: list[float] = []
        for j in (si - 1, si + 1):
            if 0 <= j < n:
                nv = self._seg_value(j)
                if nv is not None and nv >= 0:
                    neighbors.append(nv)
        if not neighbors:
            return True, ""
        # 相邻段时间间隔（帧 → 秒）
        seg = self._segments[si]
        dt_min = 1e-3
        for j in (si - 1, si + 1):
            if 0 <= j < n:
                s2 = self._segments[j]
                dt = (s2['start'] - seg['end']) / self._fps if s2['start'] >= seg['end'] \
                    else (seg['start'] - s2['end']) / self._fps
                dt = max(abs(dt), 1e-3)
                dt_min = min(dt_min, dt)
        max_dv = self._max_accel / 3.6 * dt_min  # km/h
        worst = max(abs(v - nv) for nv in neighbors)
        if worst > max_dv * 3.0:  # 3× 容差（段间可能跨真实跳变）
            msg = (f"段 #{si} 输入值 {v:.0f} km/h 与相邻段物理不一致\n\n"
                    f"相邻段值: {neighbors}\n"
                    f"在 {self._max_accel:.0f} m/s² 约束与 {dt_min:.2f}s 间隔下允许 "
                    f"{max_dv * 3.0:.0f} km/h。\n\n确定要使用此值吗？")
            return False, msg
        return True, ""

    def _add_correction(self) -> None:
        si = getattr(self, "_current_seg", 0)
        v = float(self._speed_edit.value())
        ok, warning = self._check_accel(si, v)
        if not ok:
            reply = QMessageBox.warning(self, "加速度异常", warning,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        self._corrections[si] = v
        self._redraw_chart()
        self._show_seg_image(si)
        self._btn_delete.setEnabled(True)

    def _delete_correction(self) -> None:
        si = getattr(self, "_current_seg", 0)
        if si not in self._corrections:
            return
        self._corrections.pop(si, None)
        self._speed_edit.setValue(self._speed_input_value(si))
        self._redraw_chart()
        self._btn_delete.setEnabled(False)
        self._show_seg_image(si)

    def _finish(self) -> None:
        self.accept()

    def get_corrections(self) -> dict[int, float]:
        return dict(self._corrections)
