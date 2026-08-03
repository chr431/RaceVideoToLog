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
        s = 0.85 if getattr(event, 'button', '') == 'up' else 1.15
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
