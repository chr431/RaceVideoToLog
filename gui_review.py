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
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, isDarkTheme)
from theme_manager import ThemeManager
from widget_utils import make_static_card, setup_chart_zoom_pan
from config import (COLOR_RED, COLOR_ORANGE, COLOR_BLUE,
    COLOR_LIGHT_GRAY, COLOR_LIGHTER_GRAY, chart_colors,
    LCS_WARNING_THRESHOLD)

import cv2
import numpy as np


class ReviewDialog(QDialog):
    """最终检查对话框 — 全帧速度曲线 + 图像预览 + 逐帧修正。"""

    def __init__(self, parent: QWidget, rows: list, observations: list,
				 raw_frames: list, confidences: list[dict],
				 max_speed: float,
				 max_accel: float = config.DEFAULT_MAX_ACCEL,
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
        self._corrections: dict[int, float] = {}
        self._current_frame: int = 0

        self._build_ui()
        self._register_theme_callbacks()

    # ═══════════════ UI 构建 ═══════════════

    def _register_theme_callbacks(self) -> None:
        def _update(dark: bool) -> None:
            bg = QColor("#1f1f1f" if dark else "#f5f5f5")
            fg = QColor("#f0f0f0" if dark else "#000000")
            btn_bg = QColor("#3a3a3a" if dark else "#e8e8e8")
            img_bg = "#111" if dark else "#e0e0e0"
            p = self.palette()
            for role, color in [(QPalette.ColorRole.Window, bg), (QPalette.ColorRole.Base, btn_bg),
                                (QPalette.ColorRole.WindowText, fg), (QPalette.ColorRole.Text, fg),
                                (QPalette.ColorRole.Button, btn_bg), (QPalette.ColorRole.ButtonText, fg)]:
                p.setColor(role, color)
            self.setPalette(p)
            self._img_label.setStyleSheet(f"background-color: {img_bg}; border-radius: 4px;")
            if hasattr(self, '_figure'): self._redraw_chart()
        ThemeManager.register(_update)
        _update(isDarkTheme())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        n_low = sum(1 for c in self._confidences if c.get("score", 100) < 30)
        header.addWidget(StrongBodyLabel("最终检查"))
        header.addWidget(CaptionLabel(f"  — 点击散点图选帧修正，橙色=低置信度({n_low}帧)，完成后点击「确认保存」"))
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

        chart_card = make_static_card()
        cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
        cl.addWidget(CaptionLabel("速度曲线（点击数据点选帧，蓝点=已修正，橙点=低置信度，红圈=当前帧）"))
        self._figure, self._ax, self._canvas = self._create_chart()
        cl.addWidget(self._canvas, 1)
        rl.addWidget(chart_card, 2)

        bottom_row = QHBoxLayout(); bottom_row.setSpacing(8)

        img_card = make_static_card()
        il = QVBoxLayout(img_card); il.setContentsMargins(8, 8, 8, 4)
        il.addWidget(CaptionLabel("当前帧原始图像（ROI 裁剪区域）"))
        self._img_label = QLabel("选择帧后显示")
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(120, 80)
        self._img_label.setStyleSheet("background-color: #111; border-radius: 4px;")
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
        try:
            self._speed_edit.compactSpinButton.clicked.disconnect()
        except Exception:
            pass
        self._speed_edit._showFlyout = lambda: None
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
        self.setFocus()

    # ═══════════════ 图表 ═══════════════

    def _create_chart(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        fig = Figure(figsize=(8, 3.5), dpi=100, layout='none')
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
        ax = fig.add_subplot(111)
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)
        self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}

        from PySide6.QtWidgets import QSizePolicy
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(150)
        self._canvas = canvas

        canvas.mpl_connect('pick_event', self._on_pick)
        canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._setup_chart_zoom_pan(ax, canvas)
        self._redraw_chart(ax, fig)
        return fig, ax, canvas

    def _setup_chart_zoom_pan(self, ax, canvas) -> None:
        self._user_zoomed_ref, self._saved_limits = setup_chart_zoom_pan(
            ax, canvas, throttle_ms=40)

    def _redraw_chart(self, ax=None, fig=None) -> None:
        if ax is None:
            ax = self._ax
        if fig is None:
            fig = self._figure

        user_zoomed = (hasattr(self, '_user_zoomed_ref')
					   and self._user_zoomed_ref[0])
        if user_zoomed:
            saved_xlim = self._saved_limits["xlim"]
            saved_ylim = self._saved_limits["ylim"]

        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        prev_dark = getattr(self, '_chart_params', {}).get('dark')
        needs_rebuild = (prev_dark != dark or not hasattr(self, '_chart_cache'))

        times = [r[0] / self._fps for r in self._rows]
        speeds = [r[2] for r in self._rows]
        self._chart_cache = {'times': times, 'speeds': speeds}

        low_regions = self._low_confidence_regions()

        if needs_rebuild:
            ax.clear()
            self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
            self._chart_artists = {}

            self._chart_artists['low_spans'] = []
            for s, e in low_regions:
                span = ax.axvspan(times[s], times[min(e, len(times) - 1)],
				                  facecolor=COLOR_ORANGE, alpha=0.08, zorder=0)
                self._chart_artists['low_spans'].append(span)

            low_indices = set()
            for s, e in low_regions:
                low_indices.update(range(s, e + 1))

            gx, gy = [], []
            ox, oy = [], []
            for i in range(len(times)):
                if i in self._corrections:
                    continue
                if i in low_indices:
                    ox.append(times[i]); oy.append(speeds[i])
                else:
                    gx.append(times[i]); gy.append(speeds[i])

            gray_c = COLOR_LIGHT_GRAY if not dark else COLOR_LIGHTER_GRAY
            if gx:
                self._chart_artists['bg_gray'] = ax.plot(
                    gx, gy, ".", color=gray_c, markersize=2, alpha=0.4,
                    zorder=1, rasterized=True, picker=True, pickradius=5)[0]
            if ox:
                self._chart_artists['bg_orange'] = ax.plot(
                    ox, oy, ".", color=COLOR_ORANGE, markersize=3, alpha=0.7,
                    zorder=2, rasterized=True, picker=True, pickradius=5)[0]

            cur_fi = self._current_frame
            if 0 <= cur_fi < len(times):
                cur_v = self._rows[cur_fi][2]
                if cur_v >= 0:
                    self._chart_artists['cur_highlight'] = ax.scatter(
                        [times[cur_fi]], [cur_v], c=COLOR_RED, s=40,
                        zorder=6, edgecolors='white', linewidths=1.0,
                        marker='o', facecolors='none')
                else:
                    self._chart_artists['cur_highlight'] = None
            else:
                self._chart_artists['cur_highlight'] = None

            cx, cy = self._get_correction_xy(times)
            self._chart_artists['corrections'] = ax.scatter(
                cx, cy, c=COLOR_BLUE, s=16, zorder=5, marker='o',
                edgecolors='white', linewidths=0.5) if cx else None

            ax.set_facecolor(bg)
            fig.set_facecolor(bg)
            ax.set_xlabel("时间 (s)", color=fg)
            ax.set_ylabel("速度 (km/h)", color=fg)
            ax.tick_params(colors=fg, labelsize=8)
            ax.spines["bottom"].set_color(fg if dark else "#888")
            ax.spines["left"].set_color(fg if dark else "#888")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15 if dark else 0.25)
            ax.autoscale_view()
        else:
            cx, cy = self._get_correction_xy(times)
            corr_artist = self._chart_artists.get('corrections')
            if cx:
                if corr_artist is None:
                    self._chart_artists['corrections'] = ax.scatter(
                        cx, cy, c=COLOR_BLUE, s=16, zorder=5, marker='o',
                        edgecolors='white', linewidths=0.5)
                else:
                    corr_artist.set_offsets(np.column_stack([cx, cy]) if cx
					                        else np.empty((0, 2)))
            elif corr_artist is not None:
                corr_artist.set_offsets(np.empty((0, 2)))

            cur_hl = self._chart_artists.get('cur_highlight')
            cur_fi = self._current_frame
            if cur_hl is not None:
                if 0 <= cur_fi < len(times):
                    cur_v = self._rows[cur_fi][2]
                    if cur_v >= 0:
                        cur_hl.set_offsets([[times[cur_fi], cur_v]])
                    else:
                        cur_hl.set_offsets(np.empty((0, 2)))
                else:
                    cur_hl.set_offsets(np.empty((0, 2)))

        if user_zoomed:
            ax.set_xlim(saved_xlim)
            ax.set_ylim(saved_ylim)

        self._canvas.draw_idle()

    def _low_confidence_regions(self) -> list[tuple[int, int]]:
        regions = []
        n = len(self._confidences)
        i = 0
        while i < n:
            if self._confidences[i].get("score", 100) < 30:
                start = i
                while i < n and self._confidences[i].get("score", 100) < 30:
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
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
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
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                self.keyPressEvent(event)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            return
        super().keyPressEvent(event)

    def _on_left_key(self) -> None:
        cur = self._current_frame
        if cur > 0:
            self._navigate_to(cur - 1)

    def _on_right_key(self) -> None:
        cur = self._current_frame
        if cur < len(self._rows) - 1:
            self._navigate_to(cur + 1)

    def _on_pick(self, event) -> None:
        mouse_btn = getattr(getattr(event, 'mouseevent', None), 'button', None)
        if mouse_btn is not None and mouse_btn != 1:
            return
        ind = event.ind
        if ind is None or len(ind) == 0:
            return
        self._navigate_to(ind[0])

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
        if prev != fi and prev in backup and prev not in self._corrections:
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
        if fi not in self._preview_backup and fi not in self._corrections:
            self._preview_backup[fi] = self._rows[fi][2]
        self._rows[fi][2] = float(value)
        self._redraw_chart()
        self._speed_value_label.setText(f"速度: {value:.0f} km/h (预览)")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_current_frame'):
            self._show_frame_image(self._current_frame)

    # ═══════════════ 修正操作 ═══════════════

    def _check_accel(self, fi: int, v: float) -> tuple[bool, str]:
        from ocr_engine import _lcs_score_for_value
        times = [r[0] / self._fps for r in self._rows]
        score = _lcs_score_for_value(fi, v, self._rows, times,
		                              self._max_speed, self._max_accel)
        if score < LCS_WARNING_THRESHOLD:
            msg = (f"帧 #{fi} 输入值 {v:.0f} km/h 与周围邻居物理不一致\n\n"
			       f"LCS 一致性分数: {score:.2f} / 1.00\n"
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
        self._redraw_chart()
        self._show_frame_image(fi)
        self._btn_delete.setEnabled(True)

    def _delete_correction(self) -> None:
        fi = getattr(self, "_current_frame", 0)
        if fi not in self._corrections:
            return
        self._corrections.pop(fi, None)
        self._speed_edit.setValue(self._speed_input_value(fi))
        self._redraw_chart()
        self._btn_delete.setEnabled(False)
        self._show_frame_image(fi)

    def _finish(self) -> None:
        self.accept()

    def get_corrections(self) -> dict[int, float]:
        return dict(self._corrections)
