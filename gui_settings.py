"""RaceVideoToLog settings panel — builds the left-side parameter cards.

Separated from gui.py so the main window delegates parameter UI construction
while keeping widget references for export/import operations.
"""
from __future__ import annotations

from qfluentwidgets import (
    CardWidget, ComboBox, LineEdit, RadioButton, CheckBox,
    BodyLabel, StrongBodyLabel, CaptionLabel, PushButton,
)

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
from widget_utils import make_static_card, make_int_spinbox
import config


def build_settings_panel(parent) -> dict:
    """Build the left settings panel and return a dict of all setting widgets.

    Caller stores the returned dict and reads values from it during export
    and import. All widgets are parented to `parent`.

    Returns dict keys:
        format_ms, format_kmh, format_mph  -- RadioButton (speed format)
        max_speed_edit, max_accel_edit      -- LineEdit
        buffer_spin, target_h_spin, pad_spin, max_width_spin  -- CompactSpinBox
        model_combo                         -- ComboBox (主 OCR 模型)
        log_level_combo                     -- ComboBox
        monitor_checkbox                    -- CheckBox (资源监控开关)
        frame_start_edit, frame_end_edit    -- LineEdit
    """
    widgets: dict = {}

    # ── Speed format card ──
    fmt_card = make_static_card(parent)
    gl = QVBoxLayout(fmt_card)
    gl.addWidget(StrongBodyLabel("速度格式"))
    r = QHBoxLayout()
    widgets["format_ms"] = RadioButton("m/s")
    widgets["format_kmh"] = RadioButton("km/h")
    widgets["format_kmh"].setChecked(True)
    widgets["format_mph"] = RadioButton("mile/h")
    r.addWidget(widgets["format_ms"])
    r.addWidget(widgets["format_kmh"])
    r.addWidget(widgets["format_mph"])
    r.addStretch()
    gl.addLayout(r)
    gl.addWidget(CaptionLabel("输出统一转换为 km/h。"))

    # ── Speed limits sub-card ──
    cg = make_static_card(parent)
    cl = QGridLayout(cg)
    cl.addWidget(BodyLabel("最大速度 (km/h)"), 0, 0)
    widgets["max_speed_edit"] = LineEdit()
    widgets["max_speed_edit"].setText(str(int(config.DEFAULT_MAX_SPEED)))
    widgets["max_speed_edit"].setFixedWidth(50)
    cl.addWidget(widgets["max_speed_edit"], 0, 1)
    cl.addWidget(BodyLabel("最大加速度 (m/s²)"), 0, 2)
    widgets["max_accel_edit"] = LineEdit()
    widgets["max_accel_edit"].setText(str(int(config.DEFAULT_MAX_ACCEL)))
    widgets["max_accel_edit"].setFixedWidth(50)
    cl.addWidget(widgets["max_accel_edit"], 0, 3)
    gl.addWidget(cg)

    # ── Performance card ──
    perf_card = make_static_card(parent)
    pl = QGridLayout(perf_card)
    pl.addWidget(StrongBodyLabel("性能"), 0, 0, 1, 4)
    pl.addWidget(BodyLabel("队列缓冲"), 1, 0)
    widgets["buffer_spin"] = make_int_spinbox(4, 256, config.DEFAULT_BUFFER_SIZE, 70)
    pl.addWidget(widgets["buffer_spin"], 1, 1)
    pl.addWidget(BodyLabel("OCR 高度 (px)"), 1, 2)
    widgets["target_h_spin"] = make_int_spinbox(8, 256, config.DEFAULT_TARGET_H, 70)
    pl.addWidget(widgets["target_h_spin"], 1, 3)
    pl.addWidget(BodyLabel("最大宽度 (px)"), 2, 0)
    widgets["max_width_spin"] = make_int_spinbox(0, 512, config.DEFAULT_MAX_WIDTH, 70)
    pl.addWidget(widgets["max_width_spin"], 2, 1)
    pl.addWidget(BodyLabel("边缘填充 (px)"), 2, 2)
    widgets["pad_spin"] = make_int_spinbox(0, 64, config.DEFAULT_PAD, 70)
    pl.addWidget(widgets["pad_spin"], 2, 3)
    pl.addWidget(BodyLabel("OCR 模型"), 3, 0)
    widgets["model_combo"] = ComboBox()
    widgets["model_combo"].addItems(["v6_small"])
    widgets["model_combo"].setCurrentIndex(0)
    widgets["model_combo"].setFixedWidth(95)
    pl.addWidget(widgets["model_combo"], 3, 1)
    pl.addWidget(BodyLabel("日志级别"), 3, 2)
    widgets["log_level_combo"] = ComboBox()
    widgets["log_level_combo"].addItems(["正常", "详细", "调试"])
    widgets["log_level_combo"].setCurrentIndex(0)
    widgets["log_level_combo"].setFixedWidth(120)
    pl.addWidget(widgets["log_level_combo"], 3, 3)
    pl.addWidget(BodyLabel("资源监控"), 4, 0)
    widgets["monitor_checkbox"] = CheckBox("内存 / CPU / GPU 采样")
    widgets["monitor_checkbox"].setChecked(config.MONITOR_ENABLED)
    pl.addWidget(widgets["monitor_checkbox"], 4, 1, 1, 2)

    # ── Timeline range card ──
    time_card = make_static_card(parent)
    tl = QGridLayout(time_card)
    tl.addWidget(StrongBodyLabel("时间轴范围"), 0, 0, 1, 6)
    tl.addWidget(BodyLabel("起始帧"), 1, 0)
    widgets["frame_start_edit"] = LineEdit()
    widgets["frame_start_edit"].setFixedWidth(72)
    tl.addWidget(widgets["frame_start_edit"], 1, 1)
    bfs = PushButton("设为当前")
    bfs.setFixedWidth(90)
    widgets["_set_start_btn"] = bfs
    tl.addWidget(bfs, 1, 2)
    tl.addWidget(BodyLabel("结束帧"), 1, 3)
    widgets["frame_end_edit"] = LineEdit()
    widgets["frame_end_edit"].setFixedWidth(72)
    tl.addWidget(widgets["frame_end_edit"], 1, 4)
    bfe = PushButton("设为当前")
    bfe.setFixedWidth(90)
    widgets["_set_end_btn"] = bfe
    tl.addWidget(bfe, 1, 5)

    # ── Assemble into layout ──
    ll = QVBoxLayout(parent)
    ll.setContentsMargins(0, 0, 0, 0)
    ll.setSpacing(6)
    ll.addWidget(fmt_card)
    ll.addWidget(perf_card)
    ll.addWidget(time_card)
    ll.addStretch()

    return widgets
