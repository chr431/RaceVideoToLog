"""分析 Tab 的 pyqtgraph 绘制辅助（循环卷绕断线 + 诊断段着色）。"""
from __future__ import annotations

import config
from constants import Flag


def plot_wrapped_pg(plot, x, y, color, width=1.0):
    """绘制可能循环卷绕的数据：x 下降跳变处断线（分段绘制）。"""
    import pyqtgraph as pg
    x = list(x); y = list(y)
    item = pg.PlotDataItem(pen=pg.mkPen(color, width=width))
    plot.addItem(item)
    brk = None
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            brk = i
            break
    if brk is None:
        item.setData(x, y)
    else:
        # 两段拼接（中间 NaN 断线）
        xs = x[:brk] + [x[brk - 1], x[brk]] + x[brk:]
        ys = y[:brk] + [float('nan'), float('nan')] + y[brk:]
        item.setData(xs, ys)
    return item


def plot_segmented_pg(plot, x, y, flags, color, show_diagnostics, smooth_strength):
    """平滑 + 诊断段着色（pyqtgraph）：主曲线 + 红/绿段覆盖。

    show_diagnostics=True 时叠加自动纠错段（Flag.is_corrected，红）与
    高可信段（Flag.is_trusted，绿）。
    """
    from analysis import smooth_data
    import pyqtgraph as pg
    x = list(x); y = list(y); flags = list(flags)
    if smooth_strength > 0:
        x, y = smooth_data(x, y, smooth_strength)
    item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.0))
    item.setData(x, y)
    plot.addItem(item)
    if not show_diagnostics or not any(f >= 1 for f in flags):
        return item

    n_orig = len(flags)
    n_smooth = len(x)

    def _range_segments(predicate):
        segs = []
        i = 0
        while i < n_orig:
            if predicate(flags[i]):
                j = i
                while j < n_orig and predicate(flags[j]):
                    j += 1
                si = int(max(0, i - 0.5) * n_smooth / n_orig)
                ei = int(min(n_orig, j + 0.5) * n_smooth / n_orig)
                si = max(0, min(si, n_smooth - 2))
                ei = min(n_smooth, max(ei, si + 1))
                segs.append((si, ei))
                i = j + 1
            else:
                i += 1
        return segs

    for predicate, seg_color, width in [
            (Flag.is_corrected, config.COLOR_RED, 2.0),
            (Flag.is_trusted, config.COLOR_GREEN, 1.5)]:
        segs = _range_segments(predicate)
        if not segs:
            continue
        rx, ry = [], []
        for si, ei in segs:
            rx.extend(x[si:ei] + [float('nan')])
            ry.extend(y[si:ei] + [float('nan')])
        overlay = pg.PlotDataItem(pen=pg.mkPen(seg_color, width=width))
        overlay.setData(rx, ry)
        plot.addItem(overlay)
    return item
