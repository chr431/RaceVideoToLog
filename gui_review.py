"""最终检查对话框 — 全帧速度曲线 + 置信度着色 + 逐帧修正。"""
from __future__ import annotations

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
    COLOR_LIGHT_GRAY, COLOR_LIGHTER_GRAY, chart_colors,
    MANUAL_EDIT_ACCEL_WARNING)
import pyqtgraph as pg


class ReviewDialog(QDialog):
    """最终检查对话框 — 全帧速度曲线 + 图像预览 + 逐帧修正。"""

    def __init__(self, parent: QWidget, rows: list, observations: list,
                    raw_frames: list, confidences: list[dict],
                    max_speed: float,
                    max_accel: float = config.DEFAULT_MAX_ACCEL,
                    review_scope: str = "auto",
                        fps: float = 1.0) -> None:
        super().__init__(parent)
        self.setWindowTitle("最终检查 — 点击图中任意点修正单帧")
        self.resize(1200, 750)
        self.setMinimumSize(1000, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._rows = rows
        self._observations = observations
        self._raw_frames = raw_frames
        self._confidences = confidences
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._fps = fps
        self._review_scope = review_scope  # "auto"=仅剖面偏离, "full"=所有信号
        self._corrections: dict[int, float] = {}
        self._current_frame: int = 0

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
        # ThemeManager 回调 + closeEvent 显式注销：
        # PySide SignalInstance.connect 不支持 receiver 关键字，
        # 手动管理生命周期避免静态回调列表泄漏。
        self._theme_cb = ThemeManager.register(_update)
        _update(isDarkTheme())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        regions = self._low_confidence_regions()
        n_low = sum(e - s + 1 for s, e in regions)
        header.addWidget(StrongBodyLabel("最终检查"))
        header.addWidget(CaptionLabel(f"  — 点击散点图选帧修正，橙色=需审核({n_low}帧)，完成后点击「确认保存」"))
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

        chart_card = make_static_card()
        cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
        cl.addWidget(CaptionLabel("速度曲线（点击数据点选帧，蓝点=已修正，橙点=低置信度，红圈=当前帧）"))
        self._canvas = self._create_chart()
        cl.addWidget(self._canvas, 1)
        rl.addWidget(chart_card, 2)

        bottom_row = QHBoxLayout(); bottom_row.setSpacing(8)

        img_card = make_static_card()
        il = QVBoxLayout(img_card); il.setContentsMargins(8, 8, 8, 4)
        il.addWidget(CaptionLabel("当前帧原始图像（ROI 裁剪区域）"))
        self._img_label = QLabel("选择帧后显示")
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(120, 80)
        self._img_label.setStyleSheet(f"background-color: {config.PREVIEW_BG}; border-radius: 4px;")
        il.addWidget(self._img_label, 1)
        bottom_row.addWidget(img_card, 1)

        ctrl_card = make_static_card()
        ctrl = QVBoxLayout(ctrl_card); ctrl.setContentsMargins(8, 8, 8, 4)

        cf_row = QHBoxLayout()
        cf_row.addWidget(BodyLabel("当前帧: "))
        self._frame_label = BodyLabel("—")
        cf_row.addWidget(self._frame_label)
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

    # ═══════════════ 图表 ═══════════════

    def _create_chart(self):
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=False)  # 大数据散点关闭抗锯齿
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        plot = ModPlotWidget()
        plot.setMinimumHeight(150)
        plot.setBackground(bg)
        plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        # 上/右框线：ViewBox.setBorder（空轴零尺寸会被渲染裁剪不显示）
        plot.getPlotItem().getViewBox().setBorder(pg.mkPen(fg))
        vb = plot.getPlotItem().getViewBox()
        vb.setMouseEnabled(x=True, y=True)  # 滚轮缩放 + 左键拖拽平移（原生）
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
        """用户手动缩放/平移后记录视图范围。
        sigRangeChangedManually 发射的是轴启用掩码（list），取范围用 self._plot。"""
        self._user_zoomed = True
        self._saved_range = self._plot.plotItem.vb.viewRange()

    def _setup_hover(self, plot) -> None:
        """悬停竖线 + 左上角最近点速度（pyqtgraph 原生 InfiniteLine + TextItem）。"""
        import pyqtgraph as pg
        from PySide6.QtCore import Qt as _Qt
        self._hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(
            config.COLOR_GRAY, width=1, style=_Qt.PenStyle.DashLine))
        self._hover_line.setVisible(False)
        plot.addItem(self._hover_line)
        fg = chart_colors(isDarkTheme())[1]
        # anchor=(0, 0)：文字左上角贴住 pos（(0,1) 会把文字顶到视图外被裁掉）
        self._hover_text = pg.TextItem("", color=fg, anchor=(0, 0))
        self._hover_text.setVisible(False)
        plot.addItem(self._hover_text, ignoreBounds=True)
        if not hasattr(self, '_hover_connected'):
            plot.scene().sigMouseMoved.connect(self._on_hover_moved)
            plot.plotItem.vb.sigRangeChanged.connect(self._pin_hover_text)
            self._hover_connected = True

    def _pin_hover_text(self, *args) -> None:
        """悬停文字钉在视图左上角：缩放/平移时跟随，不拖后腿跳回。"""
        vb = self._plot.plotItem.vb
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        self._hover_text.setPos(xmin, ymax)

    def _on_hover_moved(self, pos) -> None:
        plot = self._plot
        vb = plot.plotItem.vb
        if not plot.sceneBoundingRect().contains(pos):
            self._hover_line.setVisible(False)
            self._hover_text.setVisible(False)
            return
        pt = vb.mapSceneToView(pos)
        x = pt.x()
        cache = getattr(self, '_chart_cache', None)
        times = (cache or {}).get('times') or []
        speeds = (cache or {}).get('speeds') or []
        if not times:
            return
        import bisect
        idx = bisect.bisect_left(times, x)
        if idx >= len(times):
            idx = len(times) - 1
        elif idx > 0 and abs(times[idx - 1] - x) < abs(times[idx] - x):
            idx -= 1
        v = speeds[idx]
        self._hover_line.setPos(x)
        self._hover_line.setVisible(True)
        text = (f"#{int(times[idx])}: {v:.0f} km/h"
                if v >= 0 else f"#{int(times[idx])}: 无效")
        self._hover_text.setText(text)
        self._hover_text.setVisible(True)

    def _redraw_chart(self, plot=None) -> None:
        if plot is None:
            plot = self._plot
        vb = plot.plotItem.vb
        saved_range = self._saved_range if getattr(self, '_user_zoomed', False) else None

        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        prev_dark = getattr(self, '_chart_params', {}).get('dark')
        prev_corr = getattr(self, '_chart_corrections_fs', None)
        cur_corr = frozenset(self._corrections.keys())
        self._chart_corrections_fs = cur_corr

        times = [r[0] for r in self._rows]  # 帧号
        speeds = [r[2] for r in self._rows]
        prev_data = self._chart_cache.get('data_hash', 0) if hasattr(self, '_chart_cache') else 0
        data_hash = hash((len(times), times[0], times[-1],
                          sum(speeds), len(self._corrections)))
        self._chart_cache = {'times': times, 'speeds': speeds, 'data_hash': data_hash}
        needs_rebuild = (prev_dark != dark or not hasattr(self, '_chart_cache')
                         or prev_corr != cur_corr or prev_data != data_hash)

        low_set = set()
        for s, e in self._low_confidence_regions():
            low_set.update(range(s, e + 1))

        if needs_rebuild:
            plot.clear()
            self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
            plot.setBackground(bg)
            plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
            self._setup_hover(plot)
            self._chart_artists = {}

            # 低置信度区间（不可移动的 LinearRegionItem）
            done = set()
            for s, e in self._low_confidence_regions():
                key = (s, e)
                if key not in done:
                    done.add(key)
                    # pen=pg.mkPen(None)（NoPen 透明）而非 None：
                    # pen=None 会让 InfiniteLine 回落到默认黄色 (200,200,100)
                    region = pg.LinearRegionItem(
                        values=(times[s], times[min(e, len(times) - 1)]),
                        orientation='vertical', movable=False,
                        brush=make_brush(COLOR_ORANGE, 20),
                        pen=pg.mkPen(None))
                    plot.addItem(region)

            # 灰色/橙色散点（大数据：ScatterPlotItem 原生高性能）
            gray_c = COLOR_LIGHT_GRAY if not dark else COLOR_LIGHTER_GRAY
            gx, gy, gi, ox, oy, oi = [], [], [], [], [], []
            for i in range(len(times)):
                if i in self._corrections:
                    continue
                if i in low_set:
                    ox.append(times[i]); oy.append(speeds[i]); oi.append(i)
                else:
                    gx.append(times[i]); gy.append(speeds[i]); gi.append(i)
            gray = pg.ScatterPlotItem(size=4, brush=make_brush(gray_c, 100),
                                      pen=None)
            gray.setData(x=gx, y=gy, data=gi)
            gray.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(gray)
            self._chart_artists['bg_gray'] = gray
            orange = pg.ScatterPlotItem(size=9, brush=make_brush(COLOR_ORANGE, 180),
                                        pen=None)
            orange.setData(x=ox, y=oy, data=oi)
            orange.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(orange)
            self._chart_artists['bg_orange'] = orange

            # 修正蓝点 + 当前帧红点（红点 ≈ 灰点 2 倍直径，白描边区分）
            corr = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(COLOR_BLUE),
                                      pen=pg.mkPen('w', width=1.0))
            plot.addItem(corr)
            self._chart_artists['corrections'] = corr
            cur = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(COLOR_RED),
                                     pen=pg.mkPen('w', width=1.5))
            plot.addItem(cur)
            self._chart_artists['cur_highlight'] = cur
            self._update_corr_and_cur(times)

            # 轴样式 + 上/右框线（ViewBox 边框）
            plot.setLabel('bottom', '帧', color=fg)
            plot.setLabel('left', '速度 (km/h)', color=fg)
            for ax_name in ('left', 'bottom'):
                ax_item = plot.getPlotItem().getAxis(ax_name)
                ax_item.setTextPen(fg)
                ax_item.setPen(fg)
            plot.getPlotItem().getViewBox().setBorder(pg.mkPen(fg))
        else:
            self._update_corr_and_cur(times)

        if saved_range is not None:
            xr, yr = saved_range
            vb.setXRange(xr[0], xr[1], padding=0)
            vb.setYRange(yr[0], yr[1], padding=0)

    def _update_corr_and_cur(self, times: list) -> None:
        """增量更新修正蓝点与当前帧红点（pyqtgraph setData 高效）。"""
        corr = self._chart_artists.get('corrections')
        cur = self._chart_artists.get('cur_highlight')
        if corr is None or cur is None:
            return
        pv = getattr(self, '_preview_vals', {})
        cx = [times[fi] for fi in self._corrections if fi < len(times)]
        cy = [pv.get(fi, self._corrections[fi])
              for fi in self._corrections if fi < len(times)]
        corr.setData(x=cx, y=cy)
        cur_fi = self._current_frame
        if cur_fi in pv:
            target = pv[cur_fi]
        elif cur_fi in self._corrections:
            target = self._corrections[cur_fi]
        else:
            target = self._rows[cur_fi][2] if 0 <= cur_fi < len(self._rows) else -1
        if 0 <= cur_fi < len(times) and target >= 0:
            cur.setData(x=[times[cur_fi]], y=[target])
            # 已确定点选中：蓝点 + 白描边；未确定点：红点 + 白描边
            cur.setBrush(make_brush(COLOR_BLUE if cur_fi in self._corrections
                                    else COLOR_RED))
        else:
            cur.setData(x=[], y=[])

    def _on_scatter_clicked(self, scatter, points) -> None:
        """点击散点导航到对应帧（点数据里存帧号）。"""
        for pt in points:
            idx = pt.data()
            if idx is not None:
                self._navigate_to(idx)
                break

    def _low_confidence_regions(self) -> list[tuple[int, int]]:
        """Use Phase 1 multi-signal confidence scores (inc. accel spikes)."""
        n = len(self._rows)
        if n < 5:
            return []
        regions = []
        i = 0
        while i < n:
            c = self._confidences[i] if i < len(self._confidences) else {}
            score = c.get("score", 100)
            corrected = c.get("is_corrected", False)
            is_suspect = score < 70 or corrected
            if is_suspect:
                start = i
                while i < n:
                    c2 = self._confidences[i] if i < len(self._confidences) else {}
                    if not (c2.get("score", 100) < 70 or c2.get("is_corrected", False)):
                        break
                    i += 1
                regions.append((start, i - 1))
            i += 1
        return regions

    def _get_correction_xy(self, times: list[float]
                            ) -> tuple[list[float], list[float]]:
        cx = [times[fi] for fi in self._corrections if fi < len(times)]
        cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
        return cx, cy

    # ═══════════════ 图像 + 导航 ═══════════════

    def _show_frame_image(self, frame_index: int) -> None:
        if 0 <= frame_index < len(self._raw_frames):
            _, crop = self._raw_frames[frame_index]
            if crop is not None and crop.size > 0:
                # decord returns RGB — no conversion needed
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
        cur = self._current_frame
        if cur > 0:
            self._navigate_to(cur - 1)

    def _on_up_key(self) -> None:
        v = int(self._speed_edit.value()) + 1
        if v <= int(self._max_speed):
            self._speed_edit.setValue(v)

    def _on_down_key(self) -> None:
        v = int(self._speed_edit.value()) - 1
        if v >= 0:
            self._speed_edit.setValue(v)

    def _on_right_key(self) -> None:
        cur = self._current_frame
        if cur < len(self._rows) - 1:
            self._navigate_to(cur + 1)

    # pick 已由 _on_scatter_clicked 处理（pyqtgraph sigClicked）


    @staticmethod
    def _speed_label(val: float) -> str:
        return "失败" if val < 0 else f"{val:.0f}km/h"

    def _speed_input_value(self, fi: int) -> int:
        if fi in self._corrections:
            return int(self._corrections[fi])
        v = self._rows[fi][2]
        return int(v) if v >= 0 else 0

    def _navigate_to(self, fi: int) -> None:
        prev = self._current_frame
        backup = getattr(self, '_preview_backup', {})
        if prev != fi:
            # 离开帧：丢弃未提交的预览（已确定点恢复蓝点原值）
            getattr(self, '_preview_vals', {}).pop(prev, None)
            if prev in backup and prev not in self._corrections:
                self._rows[prev][2] = backup.pop(prev)
        self._current_frame = fi
        self._frame_label.setText(f"#{fi}")
        self._show_frame_image(fi)
        self._speed_edit.blockSignals(True)
        self._speed_edit.setValue(self._speed_input_value(fi))
        self._speed_edit.blockSignals(False)
        v = self._rows[fi][2]
        self._speed_value_label.setText(f"速度: (无效)" if v < 0 else f"速度: {v:.0f} km/h")
        self._btn_delete.setEnabled(fi in self._corrections)
        self._redraw_chart()

    def _on_spinbox_changed(self, value: int) -> None:
        fi = self._current_frame
        if fi < 0 or fi >= len(self._rows):
            return
        if not hasattr(self, '_preview_backup'):
            self._preview_backup: dict[int, float] = {}
        if not hasattr(self, '_preview_vals'):
            self._preview_vals: dict[int, float] = {}
        if fi in self._corrections:
            # 已确定点：预览值单独记录（蓝点/选中点跟随），不破坏原数据
            self._preview_vals[fi] = int(value)
        else:
            if fi not in self._preview_backup:
                self._preview_backup[fi] = self._rows[fi][2]
            self._rows[fi][2] = int(value)
        self._redraw_chart()
        self._speed_value_label.setText(f"速度: {value:.0f} km/h (预览)")

    def closeEvent(self, event) -> None:
        """注销主题回调，防止对话框销毁后泄漏。"""
        ThemeManager.unregister(getattr(self, '_theme_cb', lambda dark: None))
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_current_frame'):
            self._show_frame_image(self._current_frame)

    # ═══════════════ 修正操作 ═══════════════

    def _check_accel(self, fi: int, v: float) -> tuple[bool, str]:
        from ocr_engine import _neighbor_consistency_score
        times = [r[0] / self._fps for r in self._rows]
        score = _neighbor_consistency_score(fi, v, self._rows, times,
                                               self._max_speed, self._max_accel)
        if score < MANUAL_EDIT_ACCEL_WARNING:
            msg = (f"帧 #{fi} 输入值 {v:.0f} km/h 与周围邻居物理不一致\n\n"
                    f"邻域一致性分数: {score:.2f} / 1.00\n"
                    f"该值在 {self._max_accel:.0f} m/s² 约束下与邻近帧矛盾。\n\n"
                    f"确定要使用此值吗？")
            return False, msg
        return True, ""

    def _add_correction(self) -> None:
        fi = getattr(self, "_current_frame", 0)
        v = float(self._speed_edit.value())
        ok, warning = self._check_accel(fi, v)
        if not ok:
            reply = QMessageBox.warning(self, "加速度异常", warning,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        self._corrections[fi] = v
        getattr(self, '_preview_backup', {}).pop(fi, None)
        getattr(self, '_preview_vals', {}).pop(fi, None)
        self._redraw_chart()
        self._show_frame_image(fi)
        self._btn_delete.setEnabled(True)

    def _delete_correction(self) -> None:
        fi = getattr(self, "_current_frame", 0)
        if fi not in self._corrections:
            return
        self._corrections.pop(fi, None)
        getattr(self, '_preview_vals', {}).pop(fi, None)
        self._speed_edit.setValue(self._speed_input_value(fi))
        self._redraw_chart()
        self._btn_delete.setEnabled(False)
        self._show_frame_image(fi)

    def _finish(self) -> None:
        self.accept()

    def get_corrections(self) -> dict[int, float]:
        return dict(self._corrections)
