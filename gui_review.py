"""人工审核对话框 — 聚焦问题段审核，含速度曲线 + 原始图像预览。

在自动纠错后展示置信度低的问题段，人工审核关键帧后重新纠错。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QListWidget, QListWidgetItem, QSpinBox, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, CardWidget)

import cv2
import numpy as np


class ReviewDialog(QDialog):
    """人工审核对话框 — 左侧问题段列表，右侧速度曲线 + 图像 + 修正控件。"""

    def __init__(self, parent: QWidget, rows: list, observations: list,
                 raw_frames: list, confidences: list[dict],
                 segments: list[dict], max_speed: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("人工审核 — 聚焦问题段")
        self.resize(1200, 750)
        self.setMinimumSize(1000, 600)

        self._rows = rows
        self._observations = observations
        self._raw_frames = raw_frames  # [(timestamp, np.ndarray), ...]
        self._confidences = confidences
        self._segments = segments
        self._max_speed = max_speed
        self._corrections: dict[int, float] = {}
        self._confirmed: set[int] = set()

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("聚焦人工审核"))
        header.addStretch()
        total = sum(s['count'] for s in self._segments)
        header.addWidget(CaptionLabel(f"发现 {len(self._segments)} 个问题段，共 {total} 帧待审核"))
        root.addLayout(header)

        # ── 主内容：左右分栏 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：问题段列表
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 8, 0); ll.setSpacing(4)
        ll.addWidget(CaptionLabel("问题段"))
        self._list = QListWidget()
        self._list.setFixedWidth(290)
        self._list.currentRowChanged.connect(self._on_select)
        ll.addWidget(self._list, 1)
        for seg in self._segments:
            self._add_segment_item(seg)
        splitter.addWidget(left)

        # 右侧：图表 + 原始图像 + 控件
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

        # 速度曲线
        chart_card = CardWidget()
        cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
        cl.addWidget(CaptionLabel("速度曲线（当前段加粗高亮，红色=问题段，绿色=已确认，蓝点=已修正）"))
        self._figure, self._ax, self._canvas = self._create_chart()
        cl.addWidget(self._canvas, 1)
        rl.addWidget(chart_card, 2)

        # 原始图像 + 修正控件（水平排列）
        bottom_row = QHBoxLayout(); bottom_row.setSpacing(8)

        # 原始图像预览
        img_card = CardWidget()
        il = QVBoxLayout(img_card); il.setContentsMargins(8, 8, 8, 4)
        il.addWidget(CaptionLabel("当前帧原始图像（ROI 裁剪区域）"))
        self._img_label = QLabel("选择帧后显示")
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(250, 60)
        self._img_label.setStyleSheet("background-color: #111; border-radius: 4px;")
        il.addWidget(self._img_label, 1)
        bottom_row.addWidget(img_card, 1)

        # 修正控件
        ctrl_card = CardWidget()
        ctrl = QVBoxLayout(ctrl_card); ctrl.setContentsMargins(8, 8, 8, 4)

        self._suggested_widget = QWidget()
        sl = QHBoxLayout(self._suggested_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(CaptionLabel("建议审核帧: "))
        self._suggested_btns: list[PushButton] = []
        sl.addStretch()
        ctrl.addWidget(self._suggested_widget)
        self._suggested_widget.hide()

        ctrl.addSpacing(4)
        cr = QHBoxLayout()
        cr.addWidget(BodyLabel("帧 #"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, len(self._rows) - 1)
        self._frame_spin.setFixedWidth(90)
        cr.addWidget(self._frame_spin)
        cr.addWidget(BodyLabel("速度"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(0, int(self._max_speed))
        self._speed_spin.setFixedWidth(90)
        cr.addWidget(self._speed_spin)
        cr.addWidget(BodyLabel("km/h"))
        cr.addStretch()
        ctrl.addLayout(cr)

        btn_row = QHBoxLayout()
        btn_add = PrimaryPushButton("添加修正")
        btn_add.setFixedWidth(100)
        btn_add.clicked.connect(self._add_correction)
        btn_row.addWidget(btn_add)
        btn_confirm = PushButton("此段正确")
        btn_confirm.setFixedWidth(100)
        btn_confirm.clicked.connect(self._confirm_segment)
        btn_row.addWidget(btn_confirm)
        btn_row.addStretch()
        ctrl.addLayout(btn_row)

        self._corr_label = CaptionLabel("（尚无手动修正）")
        self._corr_label.setWordWrap(True)
        ctrl.addWidget(self._corr_label)

        bottom_row.addWidget(ctrl_card, 1)
        rl.addLayout(bottom_row, 1)

        # 底部完成按钮
        btn_finish = PrimaryPushButton("完成审核，重新纠错")
        btn_finish.setFixedWidth(200)
        finish_row = QHBoxLayout()
        finish_row.addStretch()
        finish_row.addWidget(btn_finish)
        btn_finish.clicked.connect(self._finish)
        rl.addLayout(finish_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        if self._segments:
            self._list.setCurrentRow(0)

    def _create_chart(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qfluentwidgets import isDarkTheme

        fig = Figure(figsize=(8, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        dark = isDarkTheme()
        bg = "#2a2a2a" if dark else "#ffffff"
        fg = "#e0e0e0" if dark else "#333333"
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)

        self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}

        canvas = FigureCanvasQTAgg(fig)
        canvas.setParent(self)
        self._canvas = canvas
        self._redraw_chart(ax, fig)
        fig.tight_layout()
        return fig, ax, canvas

    def _redraw_chart(self, ax=None, fig=None) -> None:
        if ax is None:
            ax = self._ax
        if fig is None:
            fig = self._figure
        ax.clear()
        p = getattr(self, '_chart_params', {})
        dark = p.get('dark', False)
        bg = p.get('bg', '#ffffff')
        fg = p.get('fg', '#333333')

        times = [r[0] for r in self._rows]
        speeds = [r[2] for r in self._rows]
        cur_row = self._list.currentRow()
        cur_seg = None
        if cur_row >= 0:
            cur_seg = self._list.item(cur_row).data(Qt.ItemDataRole.UserRole)

        # 全曲线（极淡灰）
        ax.plot(times, speeds, color="#cccccc", linewidth=0.4, alpha=0.5, zorder=0)

        # 各问题段着色
        for seg in self._segments:
            s, e = seg['start'], seg['end']
            if seg['start'] in self._confirmed:
                ax.plot(times[s:e+1], speeds[s:e+1], color="#4CAF50",
                        linewidth=1.0, alpha=0.6, zorder=1)
            elif seg is cur_seg:
                ax.plot(times[s:e+1], speeds[s:e+1], color="#FF9800",
                        linewidth=2.5, zorder=4)
            else:
                ax.plot(times[s:e+1], speeds[s:e+1], color="#F44336",
                        linewidth=1.0, alpha=0.7, zorder=2)

        # 已修正帧
        if self._corrections:
            cx = [times[fi] for fi in self._corrections if fi < len(times)]
            cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
            if cx:
                ax.scatter(cx, cy, c="#2196F3", s=40, zorder=5, marker='o',
                           edgecolors='white', linewidths=0.8)

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
        fig.tight_layout()
        self._canvas.draw_idle()

    def _show_frame_image(self, frame_index: int) -> None:
        """显示指定帧的原始 ROI 图像。"""
        if 0 <= frame_index < len(self._raw_frames):
            _, crop = self._raw_frames[frame_index]
            if crop is not None and crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                # Scale to fit
                pm = QPixmap.fromImage(qimg)
                scaled = pm.scaled(self._img_label.width() - 8, self._img_label.height() - 8,
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                self._img_label.setPixmap(scaled)
                return
        self._img_label.setText("(无图像)")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_frame_spin'):
            if hasattr(self, '_current_frame'): self._show_frame_image(self._current_frame)

    def _add_segment_item(self, seg: dict) -> None:
        text = (f"帧 {seg['start']}-{seg['end']} ({seg['count']}帧)  "
                f"置信度 {seg['avg_score']:.0f}")
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, seg)
        if seg['start'] in self._confirmed:
            item.setForeground(Qt.GlobalColor.gray)
            item.setText(item.text() + " ✓")
        self._list.addItem(item)

    def _on_select(self, row: int) -> None:
        if row < 0:
            return
        seg = self._list.item(row).data(Qt.ItemDataRole.UserRole)

        for b in self._suggested_btns:
            self._suggested_widget.layout().removeWidget(b)
            b.deleteLater()
        self._suggested_btns.clear()

        for fi in seg['suggested']:
            v = self._rows[fi][2]
            btn = PushButton(f"#{fi} ({v:.0f}km/h)")
            btn.setFixedWidth(110)
            btn.clicked.connect(lambda checked, f=fi, val=v: self._quick_correct(f, val))
            self._suggested_widget.layout().insertWidget(
                self._suggested_widget.layout().count() - 1, btn)
            self._suggested_btns.append(btn)
        self._suggested_widget.show()

        self._frame_spin.setValue(seg['start'])
        self._show_frame_image(seg['start'])
        seg_vals = [self._rows[i][2] for i in range(seg['start'], seg['end'] + 1)]
        avg_val = int(sum(seg_vals) / max(len(seg_vals), 1))
        self._speed_spin.setValue(avg_val)
        self._update_corr_label()
        self._redraw_chart()

    def _quick_correct(self, fi: int, val: float) -> None:
        self._frame_spin.setValue(fi)
        self._speed_spin.setValue(int(val))
        self._show_frame_image(fi)

    def _add_correction(self) -> None:
        fi = self._frame_spin.value()
        speed = self._speed_spin.value()
        self._corrections[fi] = float(speed)
        self._update_corr_label()
        self._redraw_chart()
        self._show_frame_image(fi)

    def _update_corr_label(self) -> None:
        if not self._corrections:
            self._corr_label.setText("（尚无手动修正）")
            return
        items = [f"#{f}={v:.0f}km/h" for f, v in sorted(self._corrections.items())]
        self._corr_label.setText("已修正: " + ", ".join(items))

    def _confirm_segment(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        seg = self._list.item(row).data(Qt.ItemDataRole.UserRole)
        self._confirmed.add(seg['start'])
        item = self._list.item(row)
        item.setForeground(Qt.GlobalColor.gray)
        item.setText(item.text() + " ✓")
        self._redraw_chart()
        for r in range(row + 1, self._list.count()):
            s = self._list.item(r).data(Qt.ItemDataRole.UserRole)
            if s['start'] not in self._confirmed:
                self._list.setCurrentRow(r)
                return

    def _finish(self) -> None:
        if not self._corrections and not self._confirmed:
            QMessageBox.information(self, "提示",
                "未添加任何修正且未确认任何段。\n将使用自动纠错结果。")
        self.accept()

    def get_corrections(self) -> dict[int, float]:
        return dict(self._corrections)

    def get_confirmed(self) -> set[int]:
        return set(self._confirmed)	        # 修正控件
