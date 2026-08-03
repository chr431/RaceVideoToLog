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
from PySide6.QtGui import QColor
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


def make_brush(color, alpha: int = 255):
    """pyqtgraph 画笔：mkBrush(color, alpha=...) 会静默丢弃 alpha 关键字，
    导致选区/散点完全透明化失效。这里显式构造带透明度的 QColor。"""
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


class _RegionViewBox(pg.ViewBox):
    """ViewBox：右键拖拽 = 选择范围，右键点击（无移动）= 取消选择（左键平移保留）。"""

    sig_drag_range = Signal(float, float)  # (x0, x1) 数据坐标（拖拽）
    sig_drag_click = Signal(float)         # 单击位置数据坐标（点击）

    def mouseClickEvent(self, ev):
        """纯点击（无拖动）→ pyqtgraph 路由到 mouseClickEvent 而非
        mouseDragEvent：在此补发取消选择信号。"""
        if ev.button() == Qt.MouseButton.RightButton:
            x = self.mapSceneToView(ev.scenePos()).x()
            self.sig_drag_click.emit(x)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

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
            mask = [False, True]
            ym = (yr[0] + yr[1]) / 2
            vb.setYRange(ym - (ym - yr[0]) * s, ym + (yr[1] - ym) * s, padding=0)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            mask = [True, False]
            xm = (xr[0] + xr[1]) / 2
            vb.setXRange(xm - (xm - xr[0]) * s, xm + (xr[1] - xm) * s, padding=0)
        else:
            mask = [True, True]
            xm = (xr[0] + xr[1]) / 2
            vb.setXRange(xm - (xm - xr[0]) * s, xm + (xr[1] - xm) * s, padding=0)
            ym = (yr[0] + yr[1]) / 2
            vb.setYRange(ym - (ym - yr[0]) * s, ym + (yr[1] - ym) * s, padding=0)
        # 自实现缩放绕过了 ViewBox.wheelEvent → 补发手动变更信号
        # （审核窗口依赖它记录视图范围，避免重绘时被 autoRange 重置）
        self._region_vb.sigRangeChangedManually.emit(mask)
        ev.accept()
