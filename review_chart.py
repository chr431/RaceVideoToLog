"""段级 review 的 pyqtgraph 图表渲染（ReviewDialog 的 mixin）。"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

import config
from config import (COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE,
                    COLOR_LIGHT_GRAY, COLOR_LIGHTER_GRAY, chart_colors)
from PySide6.QtCore import Qt
from qfluentwidgets import isDarkTheme
from widget_utils import make_brush, ModPlotWidget


class ReviewChartMixin:
    """段值曲线：散点分层（灰/橙/蓝）+ 预览段 + 当前段高亮 + hover。

    依赖宿主（ReviewDialog）提供：_segments、_corrections、_preview、
    _current_seg、_seg_value、_needs_review、_navigate_to。
    """

    def _create_chart(self):
        pg.setConfigOptions(antialias=False)
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        plot = ModPlotWidget()
        plot.setMinimumHeight(150)
        plot.setBackground(bg)
        plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        vb = plot_item.getViewBox()
        assert vb is not None
        vb.setBorder(pg.mkPen(fg))
        vb.setMouseEnabled(x=True, y=True)
        self._plot = plot
        self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
        self._chart_artists = {}
        self._saved_range = None
        self._user_zoomed = False
        vb.sigRangeChangedManually.connect(self._on_range_changed)
        self._setup_hover(plot)
        self._redraw_chart(plot)
        return plot

    def _on_range_changed(self, mask) -> None:
        self._user_zoomed = True
        plot_item = self._plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        self._saved_range = vb.viewRange()

    def _setup_hover(self, plot) -> None:
        from PySide6.QtCore import Qt as _Qt
        self._hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(
            config.COLOR_GRAY, width=1, style=_Qt.PenStyle.DashLine))
        self._hover_line.setVisible(False)
        plot.addItem(self._hover_line, ignoreBounds=True)
        fg = chart_colors(isDarkTheme())[1]
        self._hover_text = pg.TextItem("", color=fg, anchor=(0, 0))
        self._hover_text.setVisible(False)
        plot.addItem(self._hover_text, ignoreBounds=True)
        if not hasattr(self, '_hover_connected'):
            plot.scene().sigMouseMoved.connect(self._on_hover_moved)
            plot.plotItem.vb.sigRangeChanged.connect(self._pin_hover_text)
            self._hover_connected = True
        self._pin_hover_text()

    def _pin_hover_text(self, *args) -> None:
        plot_item = self._plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        self._hover_text.setPos(xmin, ymax)

    def _on_hover_moved(self, pos) -> None:
        plot = self._plot
        plot_item = plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        if not plot.sceneBoundingRect().contains(pos):
            self._hover_line.setVisible(False)
            self._hover_text.setVisible(False)
            return
        pt = vb.mapSceneToView(pos)
        x = pt.x()
        cache = getattr(self, '_chart_cache', None)
        xs = (cache or {}).get('xs') or []
        sidx = (cache or {}).get('sidx') or []
        if not xs:
            return
        import bisect
        idx = bisect.bisect_left(xs, x)
        if idx >= len(xs):
            idx = len(xs) - 1
        elif idx > 0 and abs(xs[idx - 1] - x) < abs(xs[idx] - x):
            idx -= 1
        si = sidx[idx]
        seg = self._segments[si]
        v = self._seg_value(si)
        self._hover_line.setPos(x)
        self._hover_line.setVisible(True)
        text = (f"段#{si} [{seg['start']}-{seg['end']}]: {v:.0f} km/h"
                if v is not None and v >= 0 else f"段#{si}: (无效)")
        self._hover_text.setText(text)
        self._hover_text.setVisible(True)

    def _frame_data(self):
        """逐帧数据：所有采样帧的 (x=帧号, y=段值, 段索引, 连线数组)。

        y 取当前段值（含修正预览）；None 段 y=-1（无效标记）。连线数组
        connect[i]=同段相邻帧 → 段内连线、段间断开。同时缓存每段点集，
        供 _update_corr_and_cur 在导航时 O(段长) 更新，避免全量扫描。
        """
        xs, ys, sidx = [], [], []
        seg_xs: dict = {}
        seg_ys: dict = {}
        for si, seg in enumerate(self._segments):
            v = self._seg_value(si)
            frames = seg.get('frames') or [seg['start']]
            xs.extend(frames)
            ys.extend([v if v is not None else -1] * len(frames))
            sidx.extend([si] * len(frames))
            seg_xs[si] = list(frames)
            seg_ys[si] = [v if v is not None else -1] * len(frames)
        n = len(xs)
        # connect 数组长度 = N（点数）：True 连接点 i → i+1（同段相连，段间断开）
        sidx_arr = np.asarray(sidx)
        conn = np.zeros(n, dtype=bool)
        if n > 1:
            conn[:-1] = sidx_arr[1:] == sidx_arr[:-1]
        self._chart_seg_points = {"x": seg_xs, "y": seg_ys}
        return xs, ys, sidx, conn

    def _redraw_chart(self, plot=None) -> None:
        if plot is None:
            plot = self._plot
        plot_item = plot.plotItem
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        saved_range = self._saved_range if getattr(self, '_user_zoomed', False) else None

        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        prev_dark = getattr(self, '_chart_params', {}).get('dark')
        prev_corr = getattr(self, '_chart_corrections_fs', None)
        cur_corr = frozenset(self._corrections.keys())
        self._chart_corrections_fs = cur_corr

        pv_seg = self._preview[0] if self._preview else None
        pv_val = self._preview[1] if self._preview else None

        xs, ys, sidx, conn = self._frame_data()
        prev_data = self._chart_cache.get('data_hash', 0) if hasattr(self, '_chart_cache') else 0
        data_hash = hash((len(xs), xs[0] if xs else 0, xs[-1] if xs else 0,
                          sum(ys), len(self._corrections)))
        self._chart_cache = {'xs': xs, 'ys': ys, 'sidx': sidx,
                             'data_hash': data_hash}
        prev_preview = getattr(self, '_chart_preview', None)
        cur_preview = self._preview
        self._chart_preview = cur_preview
        needs_rebuild = (prev_dark != dark or prev_corr != cur_corr
                         or prev_preview != cur_preview or prev_data != data_hash)

        if needs_rebuild:
            plot.clear()
            self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
            plot.setBackground(bg)
            plot.showGrid(x=True, y=True, alpha=0.15 if dark else 0.25)
            self._setup_hover(plot)
            self._chart_artists = {}

            gray_c = COLOR_LIGHT_GRAY if not dark else COLOR_LIGHTER_GRAY
            # 段连线（step 曲线）：预览段 NaN 隐藏 + 断开其连接
            line_ys = list(ys)
            conn_line = conn.copy()
            if pv_seg is not None:
                for k in range(len(xs)):
                    if sidx[k] == pv_seg:
                        line_ys[k] = float('nan')
                for k in range(len(xs) - 1):
                    if sidx[k] == pv_seg or sidx[k + 1] == pv_seg:
                        conn_line[k] = False
            line = pg.PlotDataItem(xs, line_ys, connect=conn_line,
                                   pen=pg.mkPen(gray_c, width=1))
            plot.addItem(line)
            self._chart_artists['seg_line'] = line

            # 散点分层：正常灰 / 需审核橙 / 已修正蓝（跳过预览段）
            gx, gy, gi, ox, oy, oi = [], [], [], [], [], []
            for k in range(len(xs)):
                si = sidx[k]
                if si == pv_seg:
                    continue
                if si in self._corrections:
                    continue
                if ys[k] < 0 or self._needs_review(si):
                    ox.append(xs[k]); oy.append(ys[k]); oi.append(si)
                else:
                    gx.append(xs[k]); gy.append(ys[k]); gi.append(si)
            gray = pg.ScatterPlotItem(size=4, brush=make_brush(gray_c, 100), pen=None)
            gray.setData(x=gx, y=gy, data=gi)
            gray.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(gray)
            self._chart_artists['bg_gray'] = gray
            orange = pg.ScatterPlotItem(size=5, brush=make_brush(COLOR_ORANGE, 180), pen=None)
            orange.setData(x=ox, y=oy, data=oi)
            orange.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(orange)
            self._chart_artists['bg_orange'] = orange

            # 临时预览段（绿色虚线 + 实心点）：隐藏现有段后渲染新位置
            if pv_seg is not None:
                frames = self._segments[pv_seg]['frames']
                px = list(frames)
                py_ = [pv_val] * len(px)
                pline = pg.PlotDataItem(
                    px, py_, connect='all',
                    pen=pg.mkPen(COLOR_GREEN, width=1.5,
                                 style=Qt.PenStyle.DashLine))
                plot.addItem(pline)
                self._chart_artists['preview_line'] = pline
                ppts = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(COLOR_GREEN),
                                          pen=pg.mkPen('w', width=1.0))
                ppts.setData(x=px, y=py_)
                plot.addItem(ppts)
                self._chart_artists['preview_pts'] = ppts

            corr = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(COLOR_BLUE),
                                      pen=pg.mkPen('w', width=1.0))
            plot.addItem(corr)
            self._chart_artists['corrections'] = corr
            cur = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(COLOR_RED),
                                     pen=pg.mkPen('w', width=1.5))
            plot.addItem(cur)
            self._chart_artists['cur_highlight'] = cur
            self._update_corr_and_cur()

            plot.setLabel('bottom', '帧', color=fg)
            plot.setLabel('left', '速度 (km/h)', color=fg)
            plot_item = plot.getPlotItem()
            assert plot_item is not None
            for ax_name in ('left', 'bottom'):
                ax_item = plot_item.getAxis(ax_name)
                ax_item.setTextPen(fg)
                ax_item.setPen(fg)
            vb.setBorder(pg.mkPen(fg))
        else:
            self._update_corr_and_cur()

        if saved_range is not None:
            xr, yr = saved_range
            vb.setXRange(xr[0], xr[1], padding=0)
            vb.setYRange(yr[0], yr[1], padding=0)
        vb.enableAutoRange(None, False)

    def _update_corr_and_cur(self) -> None:
        corr = self._chart_artists.get('corrections')
        cur = self._chart_artists.get('cur_highlight')
        if corr is None or cur is None:
            return
        points = getattr(self, '_chart_seg_points', None)
        if points is None:  # 理论上不会发生：_redraw_chart 总会先 _frame_data
            return
        pv_seg = self._preview[0] if self._preview else None
        # 修正蓝点：只遍历已修正段（通常远少于总段数）
        corr_set = set(self._corrections.keys())
        if pv_seg is not None:
            corr_set.discard(pv_seg)
        cx, cy = [], []
        for si in corr_set:
            cx.extend(points["x"].get(si, ()))
            cy.extend(points["y"].get(si, ()))
        corr.setData(x=cx, y=cy)
        # 当前红点：预览时隐藏（绿色预览段替代）；否则高亮选中段全部帧
        si = self._current_seg
        if pv_seg is not None:
            cur.setData(x=[], y=[])
        else:
            cur.setData(x=points["x"].get(si, []), y=points["y"].get(si, []))
            cur.setBrush(make_brush(COLOR_BLUE if si in self._corrections else COLOR_RED))

    def _on_scatter_clicked(self, scatter, points) -> None:
        for pt in points:
            idx = pt.data()
            if idx is not None:
                self._navigate_to(int(idx))
                break
