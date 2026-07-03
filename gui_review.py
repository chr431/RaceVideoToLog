"""人工审核对话框 — 聚焦问题段审核，含速度曲线预览。

在自动纠错后展示置信度低的问题段，人工审核关键帧后重新纠错。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QSpinBox, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, CardWidget)


class ReviewDialog(QDialog):
    """人工审核对话框 — 左侧问题段列表，右侧速度曲线 + 修正控件。"""

    def __init__(self, parent: QWidget, rows: list, observations: list,
                 confidences: list[dict], segments: list[dict],
                 max_speed: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("人工审核 — 聚焦问题段")
        self.resize(1100, 680)
        self.setMinimumSize(900, 550)

        self._rows = rows
        self._observations = observations
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
        info = CaptionLabel(
            f"发现 {len(self._segments)} 个问题段，共 "
            f"{sum(s['count'] for s in self._segments)} 帧待审核")
        header.addWidget(info)
        root.addLayout(header)

        # ── 主内容：左右分栏 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：问题段列表
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 8, 0); ll.setSpacing(4)
        ll.addWidget(CaptionLabel("问题段列表"))
        self._list = QListWidget()
        self._list.setFixedWidth(280)
        self._list.currentRowChanged.connect(self._on_select)
        ll.addWidget(self._list, 1)
        for seg in self._segments:
            self._add_segment_item(seg)
        splitter.addWidget(left)

        # 右侧：图表 + 修正控件
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(8)

        # 速度曲线
        chart_card = CardWidget()
        cl = QVBoxLayout(chart_card)
        cl.addWidget(CaptionLabel("速度曲线（红色=问题段，蓝色=已修正帧）"))
        self._figure, self._ax, self._canvas = self._create_chart()
        cl.addWidget(self._canvas, 1)
        rl.addWidget(chart_card, 1)

        # 修正控件
        ctrl_card = CardWidget()
        ctrl = QVBoxLayout(ctrl_card)

        # 建议审核帧行
        self._suggested_widget = QWidget()
        sl = QHBoxLayout(self._suggested_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(CaptionLabel("建议审核帧: "))
        self._suggested_btns: list[QPushButton] = []
        sl.addStretch()
        ctrl.addWidget(self._suggested_widget)
        self._suggested_widget.hide()

        # 手动修正行
        corr_row = QWidget()
        crl = QHBoxLayout(corr_row); crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(BodyLabel("帧 #"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, len(self._rows) - 1)
        self._frame_spin.setFixedWidth(80)
        crl.addWidget(self._frame_spin)
        crl.addWidget(BodyLabel("速度"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(0, int(self._max_speed))
        self._speed_spin.setFixedWidth(80)
        crl.addWidget(self._speed_spin)
        crl.addWidget(BodyLabel("km/h"))
        btn_add = PrimaryPushButton("添加修正")
        btn_add.clicked.connect(self._add_correction)
        crl.addWidget(btn_add)
        crl.addStretch()
        ctrl.addWidget(corr_row)

        # 已修正列表
        self._corr_label = CaptionLabel("（尚无手动修正）")
        self._corr_label.setWordWrap(True)
        ctrl.addWidget(self._corr_label)

        rl.addWidget(ctrl_card)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_confirm = PushButton("此段正确，跳过")
        btn_confirm.clicked.connect(self._confirm_segment)
        btn_row.addWidget(btn_confirm)
        btn_row.addStretch()
        btn_finish = PrimaryPushButton("完成审核，重新纠错")
        btn_finish.clicked.connect(self._finish)
        btn_row.addWidget(btn_finish)
        rl.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # 选中第一个问题段
        if self._segments:
            self._list.setCurrentRow(0)

    def _create_chart(self):
        """创建速度曲线图表。"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qfluentwidgets import isDarkTheme

        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        dark = isDarkTheme()
        bg = "#2a2a2a" if dark else "#ffffff"
        fg = "#e0e0e0" if dark else "#333333"
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)

        times = [r[0] for r in self._rows]
        speeds = [r[2] for r in self._rows]

        # 全曲线（灰色细线）
        ax.plot(times, speeds, color="#888888", linewidth=0.5, alpha=0.7, zorder=1)

        # 问题段高亮为红色
        for seg in self._segments:
            s, e = seg['start'], seg['end']
            ax.plot(times[s:e+1], speeds[s:e+1], color="#F44336",
                    linewidth=1.2, zorder=2)

        # 已修正帧显示为蓝点
        if self._corrections:
            cx = [times[fi] for fi in self._corrections if fi < len(times)]
            cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
            if cx:
                ax.scatter(cx, cy, c="#2196F3", s=30, zorder=3, marker='o',
                           edgecolors='white', linewidths=0.5)

        ax.set_xlabel("时间 (s)" if not dark else "时间 (s)", color=fg)
        ax.set_ylabel("速度 (km/h)", color=fg)
        ax.tick_params(colors=fg)
        ax.spines["bottom"].set_color(fg if dark else "#888")
        ax.spines["left"].set_color(fg if dark else "#888")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2 if dark else 0.3)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        canvas.setParent(self)
        return fig, ax, canvas

    def _redraw_chart(self) -> None:
        """刷新图表以反映当前修正状态。"""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        from qfluentwidgets import isDarkTheme
        dark = isDarkTheme()
        bg = "#2a2a2a" if dark else "#ffffff"
        fg = "#e0e0e0" if dark else "#333333"

        times = [r[0] for r in self._rows]
        speeds = [r[2] for r in self._rows]

        confirmed_starts = self._confirmed
        for seg in self._segments:
            s, e = seg['start'], seg['end']
            if s in confirmed_starts:
                ax.plot(times[s:e+1], speeds[s:e+1], color="#4CAF50",
                        linewidth=0.8, alpha=0.5, zorder=1)
            else:
                ax.plot(times[s:e+1], speeds[s:e+1], color="#F44336",
                        linewidth=1.2, zorder=2)

        ax.plot(times, speeds, color="#888888", linewidth=0.5, alpha=0.5, zorder=0)

        if self._corrections:
            cx = [times[fi] for fi in self._corrections if fi < len(times)]
            cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
            if cx:
                ax.scatter(cx, cy, c="#2196F3", s=30, zorder=3, marker='o',
                           edgecolors='white', linewidths=0.5)

        ax.set_facecolor(bg)
        self._figure.set_facecolor(bg)
        ax.set_xlabel("时间 (s)", color=fg)
        ax.set_ylabel("速度 (km/h)", color=fg)
        ax.tick_params(colors=fg)
        ax.spines["bottom"].set_color(fg if dark else "#888")
        ax.spines["left"].set_color(fg if dark else "#888")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2 if dark else 0.3)
        self._figure.tight_layout()
        self._canvas.draw_idle()

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

        # Update suggested frames
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
        seg_vals = [self._rows[i][2] for i in range(seg['start'], seg['end'] + 1)]
        avg_val = int(sum(seg_vals) / max(len(seg_vals), 1))
        self._speed_spin.setValue(avg_val)
        self._update_corr_label()

        # Highlight current segment on chart
        self._redraw_chart()

    def _quick_correct(self, fi: int, val: float) -> None:
        self._frame_spin.setValue(fi)
        self._speed_spin.setValue(int(val))

    def _add_correction(self) -> None:
        fi = self._frame_spin.value()
        speed = self._speed_spin.value()
        self._corrections[fi] = float(speed)
        self._update_corr_label()
        self._redraw_chart()

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
