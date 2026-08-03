"""共享 GUI 组件和工具函数。

供 gui.py, gui_review.py, gui_analysis.py 使用，避免代码重复。
"""
from __future__ import annotations

from qfluentwidgets import CardWidget, CompactSpinBox


def make_static_card(parent=None):
    """创建禁用 hover 高亮的 CardWidget。

    统一用于主窗口、审核对话框和数据分析 Tab，
    消除各文件中的重复定义。
    """
    w = CardWidget(parent)
    w.enterEvent = lambda e: None
    w.leaveEvent = lambda e: None
    return w


from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget
import pyqtgraph as pg


def make_int_spinbox(min_val: int, max_val: int, default: int, width: int = 70):
    """创建整数 CompactSpinBox：禁用浮点 flyout 面板。

    供 GUI 中所有像素/线程数等整型参数统一使用。
    """
    spin = CompactSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(default)
    spin.setFixedWidth(width)
    disable_spin_flyout(spin)
    return spin


def set_value_silent(spin, value) -> None:
    """设置 spinbox 值但不触发 valueChanged（ROI 联动赋值统一用法）。"""
    spin.blockSignals(True)
    spin.setValue(value)
    spin.blockSignals(False)


def disable_spin_flyout(spin) -> None:
    """禁用 CompactSpinBox 点击弹出的浮点输入面板（统一 flyout 禁用逻辑）。"""
    try:
        spin.compactSpinButton.clicked.disconnect()
    except Exception:
        pass
    spin._showFlyout = lambda: None


def setup_chart_zoom_pan(ax, canvas, throttle_ms: int = 40):
    """为 matplotlib 图表配置滚轮缩放 + 右键拖拽平移。

    集中管理缩放/平移逻辑，消除 gui_review.py 和 gui_analysis.py
    中的 ~40 行重复代码。

    Args:
        ax: matplotlib Axes 对象
        canvas: FigureCanvasQTAgg 实例
        throttle_ms: 平移时的节流间隔 (毫秒)，0 表示不节流
    Returns:
        (user_zoomed_flag, saved_limits_dict) — 可变容器，
        调用方可读取 user_zoomed[0] 和 saved_limits 来恢复状态。
    """
    import time as _time
    user_zoomed = [False]
    saved_limits: dict[str, tuple | None] = {"xlim": None, "ylim": None}
    _pan_start = [None, None]
    _last_draw = [0.0]

    def _throttled_draw() -> None:
        now = _time.time()
        if throttle_ms <= 0 or now - _last_draw[0] >= throttle_ms / 1000.0:
            canvas.draw_idle()
            _last_draw[0] = now

    def _on_scroll(event: object) -> None:
        # 修饰键：Ctrl+滚轮缩放纵轴，Shift+滚轮缩放横轴，无修饰双轴缩放
        from PySide6.QtCore import Qt
        mods = Qt.KeyboardModifier.NoModifier
        gui_ev = getattr(event, 'guiEvent', None)
        if gui_ev is not None:
            try:
                mods = gui_ev.modifiers()
            except Exception:
                pass
        s = 0.85 if getattr(event, 'button', '') == 'up' else 1.15
        if mods & Qt.KeyboardModifier.ControlModifier:
            yl = ax.get_ylim(); ym = (yl[0] + yl[1]) / 2
            ax.set_ylim(ym - (ym - yl[0]) * s, ym + (yl[1] - ym) * s)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            xl = ax.get_xlim(); xm = (xl[0] + xl[1]) / 2
            ax.set_xlim(xm - (xm - xl[0]) * s, xm + (xl[1] - xm) * s)
        else:
            xl = ax.get_xlim(); yl = ax.get_ylim()
            xm = (xl[0] + xl[1]) / 2; ym = (yl[0] + yl[1]) / 2
            ax.set_xlim(xm - (xm - xl[0]) * s, xm + (xl[1] - xm) * s)
            ax.set_ylim(ym - (ym - yl[0]) * s, ym + (yl[1] - ym) * s)
        user_zoomed[0] = True
        saved_limits["xlim"] = tuple(ax.get_xlim())
        saved_limits["ylim"] = tuple(ax.get_ylim())
        canvas.draw_idle()

    def _on_press(event: object) -> None:
        if getattr(event, 'button', 0) == 3:
            _pan_start[0] = getattr(event, 'xdata', None)
            _pan_start[1] = getattr(event, 'ydata', None)

    def _on_motion(event: object) -> None:
        if getattr(event, 'button', 0) == 3 and _pan_start[0] is not None:
            xd = getattr(event, 'xdata', None)
            if xd is not None:
                dx = _pan_start[0] - xd
                dy = (_pan_start[1] or 0) - (getattr(event, 'ydata', None) or 0)
                xl = ax.get_xlim(); yl = ax.get_ylim()
                ax.set_xlim(xl[0] + dx, xl[1] + dx)
                ax.set_ylim(yl[0] + dy, yl[1] + dy)
                user_zoomed[0] = True
                saved_limits["xlim"] = tuple(ax.get_xlim())
                saved_limits["ylim"] = tuple(ax.get_ylim())
                if throttle_ms > 0:
                    _throttled_draw()
                else:
                    canvas.draw_idle()

    canvas.mpl_connect("scroll_event", _on_scroll)
    canvas.mpl_connect("button_press_event", _on_press)
    canvas.mpl_connect("motion_notify_event", _on_motion)

    return user_zoomed, saved_limits


class HoverOverlay(QWidget):
    """matplotlib 画布上的 Qt 覆盖层：悬停竖线 + 左上文本。

    完全绕开 matplotlib 重绘（blit/全量重绘都有抖动或卡顿）：
    竖线和文本由 Qt 直接绘制，鼠标移动只触发本 widget 的局部重绘。
    """

    def __init__(self, canvas, fg_color: str = "#333333",
                 line_color: str = "#888888") -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._x_px: float | None = None
        self._text: str = ""
        self._fg = QColor(fg_color)
        self._line = QColor(line_color)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(canvas.rect())
        self.hide()

    def set_hover(self, x_px: float, text: str) -> None:
        """更新竖线像素位置与文本并局部重绘（Qt 层，无 matplotlib 开销）。"""
        self._x_px = x_px
        self._text = text
        self.setGeometry(self._canvas.rect())
        self.show()
        self.update()

    def clear(self) -> None:
        self.hide()

    def paintEvent(self, event) -> None:
        if self._x_px is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(self._line, 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(int(self._x_px), 0, int(self._x_px), self.height())
        if self._text:
            p.setPen(self._fg)
            p.drawText(8, 14, self._text)
        p.end()


class _RegionViewBox(pg.ViewBox):
    """ViewBox：右键拖拽 = 选择范围，右键点击（无移动）= 取消选择（左键平移保留）。"""

    sig_drag_range = Signal(float, float)  # (x0, x1) 数据坐标（拖拽）
    sig_drag_click = Signal(float)         # 单击位置数据坐标（点击）

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.MouseButton.RightButton:
            if ev.isStart():
                self._drag_x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
                self._drag_p0 = ev.buttonDownScenePos()
            elif ev.isFinish():
                p1 = ev.scenePos()
                if p1 is not None and (p1 - self._drag_p0).manhattanLength() < 5:
                    # 点击（无拖动）→ 取消选择信号
                    x = self.mapSceneToView(p1).x()
                    self.sig_drag_click.emit(x)
                else:
                    x1 = self.mapSceneToView(ev.scenePos()).x()
                    self.sig_drag_range.emit(self._drag_x0, x1)
            else:
                x1 = self.mapSceneToView(ev.scenePos()).x()
                self.sig_drag_range.emit(self._drag_x0, x1)
            ev.accept()
        else:
            super().mouseDragEvent(ev, axis)


class ModPlotWidget(pg.PlotWidget):
    """PlotWidget：Ctrl+滚轮缩放纵轴，Shift+滚轮缩放横轴，无修饰双轴；
    右键拖拽 = 选择范围（sig_drag_range 信号）。

    pyqtgraph 0.14 的 ViewBox.wheelEvent 使用 PyQt5 旧 API (ev.delta())，
    PySide6 下抛 AttributeError — 在此自实现缩放。
    """

    sig_drag_range = Signal(float, float)
    sig_drag_click = Signal(float)

    def __init__(self, *args, **kwargs):
        vb = _RegionViewBox()
        kwargs.setdefault('plotItem', pg.PlotItem(viewBox=vb))
        super().__init__(*args, **kwargs)
        self._region_vb = vb
        vb.sig_drag_range.connect(self.sig_drag_range)
        vb.sig_drag_click.connect(self.sig_drag_click)

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        s = 0.85 if delta > 0 else 1.15  # 上滚放大
        vb = self.getPlotItem().getViewBox()
        xr = vb.viewRange()[0]
        yr = vb.viewRange()[1]
        if mods & Qt.KeyboardModifier.ControlModifier:
            ym = (yr[0] + yr[1]) / 2
            vb.setYRange(ym - (ym - yr[0]) * s, ym + (yr[1] - ym) * s, padding=0)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            xm = (xr[0] + xr[1]) / 2
            vb.setXRange(xm - (xm - xr[0]) * s, xm + (xr[1] - xm) * s, padding=0)
        else:
            xm = (xr[0] + xr[1]) / 2
            vb.setXRange(xm - (xm - xr[0]) * s, xm + (xr[1] - xm) * s, padding=0)
            ym = (yr[0] + yr[1]) / 2
            vb.setYRange(ym - (ym - yr[0]) * s, ym + (yr[1] - ym) * s, padding=0)
        ev.accept()
