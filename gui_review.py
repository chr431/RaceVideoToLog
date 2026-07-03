"""人工审核对话框 — 聚焦问题段审核。

在自动纠错后展示置信度低的问题段，人工审核关键帧后重新纠错。
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui import RaceVideoToLogApp

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QSpinBox, QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, CardWidget)


class ReviewDialog(QDialog):
    """人工审核对话框。"""
    _review_done = Signal(list)  # [(frame_index, corrected_speed), ...]

    def __init__(self, parent: QWidget, rows: list, observations: list,
                 confidences: list[dict], segments: list[dict],
                 max_speed: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("人工审核 — 聚焦问题段")
        self.resize(700, 600)
        self.setMinimumSize(600, 400)

        self._rows = rows
        self._observations = observations
        self._confidences = confidences
        self._segments = segments
        self._max_speed = max_speed
        self._corrections: dict[int, float] = {}  # frame_index -> corrected_speed
        self._confirmed: set[int] = set()         # segments confirmed as correct

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        # Header
        layout.addWidget(StrongBodyLabel("聚焦人工审核"))
        layout.addWidget(CaptionLabel(
            f"自动纠错发现 {len(self._segments)} 个问题段。"
            "审核关键帧后，算法将重新纠错。"))

        # Segment list
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        layout.addWidget(self._list, 1)

        for seg in self._segments:
            self._add_segment_item(seg)

        # Detail panel
        detail_card = CardWidget()
        dl = QVBoxLayout(detail_card)
        self._detail_label = BodyLabel("选择左侧问题段查看详情")
        self._detail_label.setWordWrap(True)
        dl.addWidget(self._detail_label)

        # Suggested frames row
        self._suggested_widget = QWidget()
        sl = QHBoxLayout(self._suggested_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(CaptionLabel("建议审核帧: "))
        self._suggested_btns: list[QPushButton] = []
        sl.addStretch()
        dl.addWidget(self._suggested_widget)
        self._suggested_widget.hide()

        # Frame correction row
        corr_row = QWidget()
        crl = QHBoxLayout(corr_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(BodyLabel("手动修正帧 #"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, len(self._rows) - 1)
        self._frame_spin.setFixedWidth(80)
        crl.addWidget(self._frame_spin)
        crl.addWidget(BodyLabel("为"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(0, int(self._max_speed))
        self._speed_spin.setFixedWidth(80)
        crl.addWidget(self._speed_spin)
        crl.addWidget(BodyLabel("km/h"))
        btn_add = PrimaryPushButton("添加修正")
        btn_add.clicked.connect(self._add_correction)
        crl.addWidget(btn_add)
        crl.addStretch()
        dl.addWidget(corr_row)

        # Current corrections
        self._corr_label = CaptionLabel("")
        self._corr_label.setWordWrap(True)
        dl.addWidget(self._corr_label)

        layout.addWidget(detail_card)

        # Buttons
        btn_row = QHBoxLayout()
        btn_confirm = PushButton("此段正确，跳过")
        btn_confirm.clicked.connect(self._confirm_segment)
        btn_row.addWidget(btn_confirm)
        btn_row.addStretch()
        btn_finish = PrimaryPushButton("完成审核，重新纠错")
        btn_finish.clicked.connect(self._finish)
        btn_row.addWidget(btn_finish)
        layout.addLayout(btn_row)

    def _add_segment_item(self, seg: dict) -> None:
        text = (f"帧 {seg['start']}-{seg['end']} ({seg['count']}帧)  "
                f"置信度 {seg['avg_score']:.0f}  — {seg['reason']}")
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
        detail = (
            f"帧范围: {seg['start']} – {seg['end']} ({seg['count']} 帧)\n"
            f"问题: {seg['reason']}\n"
            f"平均置信度: {seg['avg_score']:.0f} / 100\n"
            f"建议审核 {len(seg['suggested'])} 个关键帧，算法可从段内其余帧通过物理约束插值恢复。"
        )
        self._detail_label.setText(detail)

        # Suggested frame buttons
        for b in self._suggested_btns:
            self._suggested_widget.layout().removeWidget(b)
            b.deleteLater()
        self._suggested_btns.clear()

        for fi in seg['suggested']:
            v = self._rows[fi][2]
            conf = next((c['score'] for c in self._confidences if c['index'] == fi), 0)
            btn = PushButton(f"#{fi} ({v:.0f}km/h)")
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda checked, f=fi, val=v: self._quick_correct(f, val))
            self._suggested_widget.layout().insertWidget(
                self._suggested_widget.layout().count() - 1, btn)
            self._suggested_btns.append(btn)
        self._suggested_widget.show()

        self._frame_spin.setValue(seg['start'])
        seg_vals = [self._rows[i][2] for i in range(seg['start'], seg['end'] + 1)]
        avg_val = int(sum(seg_vals) / len(seg_vals))
        self._speed_spin.setValue(avg_val)

        self._update_corr_label()

    def _quick_correct(self, fi: int, val: float) -> None:
        self._frame_spin.setValue(fi)
        self._speed_spin.setValue(int(val))

    def _add_correction(self) -> None:
        fi = self._frame_spin.value()
        speed = self._speed_spin.value()
        self._corrections[fi] = float(speed)
        self._update_corr_label()

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
        # Move to next unconfirmed
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
