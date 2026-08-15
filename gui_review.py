"""最终检查对话框（段级）— 段值曲线 + 代表帧预览 + 段值修正。

分段已验证不混合速度（可用 truth 视频 0 混合段），段值对段内所有帧可靠；
段级修正即可保证全段正确。橙点 = 需审核段（管线已纠正 / OCR 未读出）。
图表渲染在 review_chart.ReviewChartMixin 中。
"""
from __future__ import annotations

import threading

import numpy as np

import config

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor
from review_chart import ReviewChartMixin
from theme_manager import ThemeManager
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, isDarkTheme)
from widget_utils import (make_static_card, disable_spin_flyout)


class ReviewDialog(ReviewChartMixin, QDialog):
    """段级最终检查 — 段值曲线 + 代表帧预览 + 段值修正。"""

    def __init__(self, parent: QWidget, segments: list[dict],
                 max_speed: float,
                 max_accel: float = config.DEFAULT_MAX_ACCEL,
                 fps: float = 1.0,
                 preview_loader=None,
                 is_crop_cached=None,
                 preload_loader=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("最终检查 — 点击图中任意段修正该段速度")
        self.resize(1200, 750)
        self.setMinimumSize(1000, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._segments = segments
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._fps = fps
        # preview_loader(frame) -> RGB ROI ndarray；None 时回退到段内保存的
        # 灰度代表帧（生产管线 gray 输出不含色彩）
        self._preview_loader = preview_loader
        # is_crop_cached(frame) -> bool：帧的 RGB 是否已进 pipeline 缓存。
        # 未缓存时主线程不调用 preview_loader（其内部会 seek 解码卡 UI），
        # 先显示灰度占位，等后台预取完成后再重试换彩。
        self._is_crop_cached = is_crop_cached
        # preload_loader(frames, priority_frame=..., stop_event=...)：
        # 后台批量预取全部代表帧 RGB 的 pipeline 入口。
        self._preload_loader = preload_loader
        self._preload_thread: threading.Thread | None = None
        self._preload_stop = threading.Event()
        # 可变优先级 {"frame": int, "rev": int}：用户跳到未缓存段时更新，
        # 后台预取线程据此把该目标重排到剩余队列最前
        self._preload_priority: dict | None = None
        self._rgb_retry_pending: set[int] = set()
        # 预览源图缓存：切段/窗口缩放只重新缩放已解码的 QPixmap，
        # 不重复 seek/解码（修复初次显示裁剪与切段卡顿）
        self._preview_source: QPixmap | None = None
        self._preview_source_seg: int = -1
        self._preview_source_rgb: bool = False
        # 修正：段索引 → 新值（应用到整个段范围）
        self._corrections: dict[int, float] = {}
        self._current_seg: int = 0
        # 预览：spinbox 改变时的 (段索引, 预览值)，用于隐藏现有段并临时渲染
        self._preview: tuple[int, float] | None = None

        self._build_ui()
        self._register_theme_callbacks()
        # 等布局真正完成后再初始化当前段显示，避免用布局尚未稳定时的
        # label 尺寸缩放图像（旧 bug：初始图被放大裁边）
        QTimer.singleShot(0, lambda: self._navigate_to(self._current_seg))
        # 后台开始缓存全部代表帧 RGB：按距离当前段排序，附近先可用；
        # 长距离点击时若目标尚未缓存，先显示灰度占位，不阻塞 UI
        QTimer.singleShot(100, self._start_rgb_preload)

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

    # ═══════════════ 图像 + 导航 ═══════════════

    def _show_seg_image(self, si: int) -> None:
        """加载段 si 的代表帧（仅首次），随后按当前 label 尺寸缩放。

        源图缓存为 QPixmap；resizeEvent/切回已看段都只做缩放，不重新解码。
        彩色源图来自 pipeline 的 RGB 缓存：后台预取完成前先显示段内灰度
        占位并定时重试，主线程绝不 seek/解码（长距离跳转不卡 UI）。
        """
        if not (0 <= si < len(self._segments)):
            self._img_label.setText("(无图像)")
            return
        if si == self._preview_source_seg and self._preview_source is not None:
            # 已是彩色源图（或没有 loader 可升级）→ 直接缩放显示。
            # 灰度占位时继续往下走，等待 RGB 缓存就绪后升级。
            if self._preview_loader is None or self._preview_source_rgb:
                self._apply_image_scale()
                return
        seg = self._segments[si]
        crop = seg.get("rep_crop")
        rep_frame = int(seg.get("rep_frame", -1))
        color = None
        # 生产管线保存的是 gray 输出；原始 RGB 在 pipeline 缓存里按需取。
        # 仅在确认缓存就绪后调用 loader（未就绪的调用会 seek+解码阻塞 UI）
        if self._preview_loader is not None and rep_frame >= 0:
            cached = (self._is_crop_cached is None
                      or self._is_crop_cached(rep_frame))
            if cached:
                try:
                    color = self._preview_loader(rep_frame)
                except Exception:
                    color = None
            if color is None and self._is_crop_cached is not None:
                # 后台预取还没跑到这一帧：把目标提到剩余队列最前，
                # 先灰度显示，稍后自动升级为彩色
                self._prioritize_rgb_preload(rep_frame)
                self._schedule_rgb_retry(si)
        if color is not None and color.size > 0:
            crop = color
        if crop is None or crop.size <= 0:
            self._preview_source = None
            self._preview_source_seg = -1
            self._preview_source_rgb = False
            self._img_label.setText("(无图像)")
            return
        if crop.ndim == 2:
            rgb = np.repeat(crop[..., None], 3, axis=-1)
        elif crop.shape[-1] == 1:
            rgb = np.repeat(crop, 3, axis=-1)
        elif crop.shape[-1] >= 4:
            rgb = crop[..., :3]
        else:
            rgb = crop
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ch = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w,
                      QImage.Format.Format_RGB888).copy()
        self._preview_source = QPixmap.fromImage(qimg)
        self._preview_source_seg = si
        self._preview_source_rgb = (color is not None and color.size > 0)
        self._apply_image_scale()

    def _schedule_rgb_retry(self, si: int) -> None:
        """RGB 缓存未就绪时安排一次重试（同一段只挂一个定时器）。"""
        if si in self._rgb_retry_pending:
            return
        self._rgb_retry_pending.add(si)
        QTimer.singleShot(250, lambda: self._retry_rgb(si))

    def _retry_rgb(self, si: int) -> None:
        self._rgb_retry_pending.discard(si)
        if (self.isVisible() and si == self._current_seg
                and 0 <= si < len(self._segments)):
            self._show_seg_image(si)

    def _apply_image_scale(self) -> None:
        """把缓存的源图按当前 label 尺寸缩放显示（窗口变化时不重新解码）。"""
        if self._preview_source is None or self._preview_source.isNull():
            self._img_label.setText("(无图像)")
            return
        lw = max(50, self._img_label.width() - 8)
        lh = max(50, self._img_label.height() - 8)
        scaled = self._preview_source.scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._img_label.setPixmap(scaled)

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
        self._preview = None  # 离开段丢弃未提交预览
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

    def _start_rgb_preload(self) -> None:
        """后台缓存全部代表帧 RGB：先按当前段排序，点击新目标时重排。"""
        if self._preload_loader is None or not self._segments:
            return
        frames: list[int] = []
        for seg in self._segments:
            rf = int(seg.get("rep_frame", -1))
            if rf >= 0:
                frames.append(rf)
        if not frames:
            return
        if self._preload_priority is None:
            priority = None
            if 0 <= self._current_seg < len(self._segments):
                priority = int(self._segments[self._current_seg].get(
                    "rep_frame", -1))
            self._preload_priority = {"frame": priority or 0, "rev": 0}
        self._preload_stop.clear()

        def _worker() -> None:
            try:
                self._preload_loader(
                    list(frames),
                    priority_frame=self._preload_priority["frame"],
                    stop_event=self._preload_stop,
                    priority_ref=self._preload_priority)
            except Exception:
                # 预取失败不阻断：点击未缓存段时走灰度占位 + 重试
                pass

        self._preload_thread = threading.Thread(
            target=_worker, name="rgb-preload", daemon=True)
        self._preload_thread.start()

    def _prioritize_rgb_preload(self, rep_frame: int) -> None:
        """通知预取线程：用户刚跳到 rep_frame，把它提到剩余队列最前。"""
        if self._preload_priority is None:
            self._preload_priority = {"frame": int(rep_frame), "rev": 1}
        else:
            self._preload_priority["frame"] = int(rep_frame)
            self._preload_priority["rev"] = self._preload_priority.get(
                "rev", 0) + 1

    def stop_rgb_preload(self) -> None:
        """停止后台 RGB 预取（对话框关闭时调用，允许安全释放 reader）。"""
        self._preload_stop.set()
        thread = self._preload_thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            # 最多等一个批解码完成；超时后 daemon 线程也会在下一批
            # 开始前检查 stop_event 退出，pipeline 锁保证不会重开 reader
            thread.join(timeout=2.0)
        self._preload_thread = None

    def _on_spinbox_changed(self, value: int) -> None:
        si = self._current_seg
        if si < 0 or si >= len(self._segments):
            return
        # 段值预览：未提交前不写入 corrections；隐藏现有段并临时渲染预览段
        self._preview = (si, float(value))
        self._speed_value_label.setText(f"速度: {value:.0f} km/h (预览)")
        self._redraw_chart()

    def closeEvent(self, event) -> None:
        self.stop_rgb_preload()
        ThemeManager.unregister(getattr(self, '_theme_cb', lambda dark: None))
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_preview_source'):
            self._apply_image_scale()

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
        # 相邻段时间间隔（帧 → 秒）：取两侧最近的间隙
        seg = self._segments[si]
        dt_min = float("inf")
        for j in (si - 1, si + 1):
            if 0 <= j < n:
                s2 = self._segments[j]
                dt = (s2['start'] - seg['end']) / self._fps if s2['start'] >= seg['end'] \
                    else (seg['start'] - s2['end']) / self._fps
                dt_min = min(dt_min, max(abs(dt), 1e-3))
        if dt_min == float("inf"):
            return True, ""  # 无相邻段，无需检查
        # m/s² × dt = m/s；转 km/h 用 config.MPS_TO_KMH（×3.6）
        max_dv = self._max_accel * config.MPS_TO_KMH * dt_min  # km/h
        worst = max(abs(v - nv) for nv in neighbors)
        if worst > max_dv * config.REVIEW_ACCEL_TOLERANCE:  # 容差倍率（段间可能跨真实跳变）
            msg = (f"段 #{si} 输入值 {v:.0f} km/h 与相邻段物理不一致\n\n"
                    f"相邻段值: {neighbors}\n"
                    f"在 {self._max_accel:.0f} m/s² 约束与 {dt_min:.2f}s 间隔下允许 "
                    f"{max_dv * config.REVIEW_ACCEL_TOLERANCE:.0f} km/h。\n\n确定要使用此值吗？")
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
        self._preview = None  # 提交后清除预览
        self._redraw_chart()
        self._show_seg_image(si)
        self._btn_delete.setEnabled(True)

    def _delete_correction(self) -> None:
        si = getattr(self, "_current_seg", 0)
        if si not in self._corrections:
            return
        self._corrections.pop(si, None)
        self._preview = None
        self._speed_edit.setValue(self._speed_input_value(si))
        self._redraw_chart()
        self._btn_delete.setEnabled(False)
        self._show_seg_image(si)

    def _finish(self) -> None:
        self.accept()

    def get_corrections(self) -> dict[int, float]:
        return dict(self._corrections)
