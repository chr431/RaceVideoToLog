"""数据分析 Tab — PySide6 GUI。

嵌入主窗口 QTabWidget，提供 CSV 导入、多模式图表渲染、范围选择器等功能。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QStackedWidget,
)
from PySide6.QtCore import Qt, QTimer
from qfluentwidgets import (PushButton, PrimaryPushButton, CompactSpinBox, isDarkTheme,
    RadioButton, CheckBox, BodyLabel, Slider, CaptionLabel)
from widget_utils import (make_static_card, make_int_spinbox,
    make_brush, ModPlotWidget)

from analysis import parse_csv, smooth_data
from analysis_plot import plot_segmented_pg, plot_wrapped_pg
import config
from config import (COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, MPS_TO_KMH, chart_colors)


def _shift_csv(times, dists, speeds, flags, offset: int):
    """按帧偏移做循环卷绕（一整圈语义）：负偏移 = 前 |offset| 帧移到末尾。

    offset 单位为帧（CSV 行索引位移），越界部分卷绕到另一端。
    返回新顺序的 (times, dists, speeds, flags)，不修改原数据。
    """
    if not offset or not times:
        return times, dists, speeds, flags
    k = (-offset) % len(times)
    return (times[k:] + times[:k], dists[k:] + dists[:k],
            speeds[k:] + speeds[:k], flags[k:] + flags[:k])


class AnalysisTab:
    """数据分析 Tab — 嵌入 QStackedWidget，修改自动刷新。"""

    def __init__(self, stack: QStackedWidget) -> None:
        # pyqtgraph：无 Figure 类缓存

        self._stack = stack

        # 状态
        self._csvs: list[str | None] = [None, None, None]
        self._labels: list = []
        self._plot: "pg.PlotWidget | None" = None
        self._chart_mode: str = "v-x"
        self._show_corrected: bool = False
        self._saved_limits: dict[str, tuple | None] = {}
        self._last_mode: str | None = None
        self._smooth_str: int = 0
        self._span_selector = None
        self._offsets: list[int] = [0, 0, 0]  # 每行 CSV 的帧偏移（仅 GUI）
        self._offset_timer: QTimer | None = None

        self._build_tab()

    def _build_tab(self) -> None:
        tab = QWidget()
        self._stack.addWidget(tab)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 6)

        # ── 顶部控制栏 ──
        ctrl = QWidget()
        cl = QGridLayout(ctrl)
        cl.setContentsMargins(0, 0, 0, 0)

        for i in range(3):
            slot = make_static_card()
            sl = QHBoxLayout(slot)
            btn_import = PushButton("导入")
            btn_import.clicked.connect(lambda checked, idx=i: self._import(idx))
            sl.addWidget(btn_import)
            btn_clear = PushButton("清除")
            btn_clear.clicked.connect(lambda checked, idx=i: self._clear(idx))
            sl.addWidget(btn_clear)
            lbl = BodyLabel("未导入")
            self._labels.append(lbl)
            sl.addWidget(lbl)
            off_label = CaptionLabel("偏移")
            off_spin = make_int_spinbox(-99999, 99999, 0, 84)
            off_spin.setToolTip("帧偏移（循环）：正数右移、负数左移，越界部分卷绕到另一端。 不修改 CSV 内容。")
            off_spin.valueChanged.connect(
                lambda v, idx=i: self._schedule_offset_render(idx, v))
            sl.addWidget(off_label)
            sl.addWidget(off_spin)
            cl.addWidget(slot, 0, i)



        btn_export = PrimaryPushButton("导出 PNG")
        btn_export.setFixedWidth(96)
        btn_export.clicked.connect(self._export_png)
        cl.addWidget(btn_export, 0, 3, Qt.AlignmentFlag.AlignRight)

        # 第二行：模式 + 平滑 + 自动调整
        row2 = QHBoxLayout()
        self._cb_corrected = CheckBox("显示诊断信息")
        self._cb_corrected.toggled.connect(lambda v: (setattr(self, '_show_corrected', v), self._render()))
        row2.addWidget(self._cb_corrected)

        row2.addWidget(BodyLabel("平滑"))
        self._smooth_slider = Slider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(0, 100); self._smooth_slider.setValue(0)
        self._smooth_slider.setFixedWidth(100)
        self._smooth_slider.valueChanged.connect(lambda v: (setattr(self, '_smooth_str', v), self._render()))
        row2.addWidget(self._smooth_slider)

        self._smooth_spin = CompactSpinBox(); self._smooth_spin.setRange(0, 100); self._smooth_spin.setValue(0); self._smooth_spin.setFixedWidth(70)
        try:
            self._smooth_spin.compactSpinButton.clicked.disconnect()
        except Exception:
            pass
        self._smooth_spin._showFlyout = lambda: None
        self._smooth_spin.valueChanged.connect(self._smooth_slider.setValue)
        self._smooth_slider.valueChanged.connect(self._smooth_spin.setValue)
        row2.addWidget(self._smooth_spin)

        self._rb_vt = RadioButton("v-t"); self._rb_vx = RadioButton("v-x")
        self._rb_vx.setChecked(True); self._rb_dtx = RadioButton("Δt-x")
        for mode, rb in [("v-t", self._rb_vt), ("v-x", self._rb_vx), ("dt-x", self._rb_dtx)]:
            rb.toggled.connect(lambda checked, m=mode: (self._on_mode(m), self._render()) if checked else None)
            row2.addWidget(rb)

        btn_fit = PushButton("自动调整")
        btn_fit.clicked.connect(self._auto_fit)
        row2.addWidget(btn_fit)

        cl.addLayout(row2, 1, 0, 1, 4)
        layout.addWidget(ctrl)

        # ── pyqtgraph 画布 ──
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=False)
        plot = ModPlotWidget()
        plot.setBackground(chart_colors(isDarkTheme())[0])
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        vb = plot_item.getViewBox()
        assert vb is not None
        # 上/右框线：ViewBox.setBorder（空轴零尺寸会被渲染裁剪不显示）
        vb.setBorder(pg.mkPen(chart_colors(isDarkTheme())[1]))
        self._plot = plot
        layout.addWidget(plot, 1)
        self._ready = True

    # ═══════════════════ 事件 ═══════════════════

    def _sync_figure_theme(self) -> None:
        """同步图表背景/文字颜色到当前主题（pyqtgraph）。"""
        from qfluentwidgets import isDarkTheme
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        plot = self._plot
        if plot is None:
            return
        plot.setBackground(bg)
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        for ax_name in ('left', 'bottom'):
            ax_item = plot_item.getAxis(ax_name)
            ax_item.setTextPen(fg)
            ax_item.setPen(fg)
        vb = plot_item.getViewBox()
        assert vb is not None
        vb.setBorder(pg.mkPen(fg))
        # 悬停/统计文字同步主题前景色（初始用 COLOR_FG_LIGHT 的话深色下不可见）
        for t in (getattr(self, '_delta_text', None), getattr(self, '_hover_text', None)):
            if t is not None:
                t.setColor(fg)

    def _on_mode(self, mode: str) -> None:
        self._chart_mode = mode

    def _import(self, index: int) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._stack, f"选择 CSV {index + 1}", "",
            "CSV 文件 (*.csv);;所有文件 (*.*)")
        if path:
            self._csvs[index] = path
            self._labels[index].setText(Path(path).name)
            self._invalidate_and_render()

    def _clear(self, index: int) -> None:
        self._csvs[index] = None
        self._labels[index].setText("未导入")
        self._invalidate_and_render()

    def _invalidate_and_render(self) -> None:
        """清除缓存状态并重新渲染。"""
        self._saved_limits.clear()
        self._last_mode = None
        self._render()

    def _schedule_offset_render(self, idx: int, value: int) -> None:
        """偏移 spinbox 变化：节流 150ms 后重建（拖动时不卡顿）。"""
        from PySide6.QtCore import QTimer
        self._offsets[idx] = value
        timer = self._offset_timer
        if timer is None:
            timer = QTimer()  # 无 parent：AnalysisTab 非 QObject
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_offset_render)
            self._offset_timer = timer
        timer.start(150)

    def _flush_offset_render(self) -> None:
        self._offset_timer = None
        self._invalidate_and_render()

    # ═══════════════════ 渲染 ═══════════════════


    def _setup_chart_interactions(self, plot, all_x, all_y, is_dtx, is_vt,
                                 label):
        """交互：LinearRegionItem 拖选区间统计 + 悬停竖线 + 修饰键缩放。"""
        import pyqtgraph as pg
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None

        # ── 拖选区间统计（LinearRegionItem，原生高性能）──
        # anchor=(1, 0)：文字右上角贴住 pos（(1,1) 会把文字顶到视图外被裁掉）
        fg = chart_colors(isDarkTheme())[1]
        delta_text = pg.TextItem("", color=fg, anchor=(1, 0))
        plot.addItem(delta_text, ignoreBounds=True)
        self._delta_text = delta_text

        def _pin_delta_text(*args) -> None:
            """把统计文本钉在视图右上角（autoRange/缩放/平移后跟随）。"""
            xmin_v, xmax_v = vb.viewRange()[0]
            ymin_v, ymax_v = vb.viewRange()[1]
            delta_text.setPos(xmax_v, ymax_v)

        def _update_delta_text() -> None:
            region = self._region
            if region is None:
                return
            # pyqtgraph 的 getRegion 类型标注过宽（含 list）→ 显式 cast
            xmin, xmax = cast("tuple[float, float]", region.getRegion())
            results = []
            for i in range(3):
                xd = all_x[i]
                if not xd:
                    continue
                n = Path(self._csvs[i] or "").stem
                total = 0.0
                if is_dtx:
                    ys = ye = None
                    for j, x in enumerate(xd):
                        if ys is None and x >= xmin:
                            ys = all_y[i][j]
                        if x <= xmax:
                            ye = all_y[i][j]
                    if ys is not None and ye is not None:
                        total = ye - ys
                else:
                    for j, x in enumerate(xd):
                        if xmin <= x <= xmax and j > 0:
                            if is_vt:
                                dt_v = xd[j] - xd[j - 1]
                                avg = (all_y[i][j] + all_y[i][j - 1]) / 2 / MPS_TO_KMH
                                total += avg * dt_v
                            else:
                                dx_v = xd[j] - xd[j - 1]
                                avg = ((all_y[i][j] + all_y[i][j - 1]) / 2) / MPS_TO_KMH
                                total += dx_v / avg if avg > 0 and dx_v > 0 else 0
                if is_dtx:
                    sign = "+" if total >= 0 else ""
                    results.append(f"{label}: {sign}{total:.2f}s")
                elif total > 0:
                    unit = "m" if is_vt else "s"
                    results.append(f"{n}: {total:.2f}{unit}")
            delta_text.setText(chr(10).join(results) if results else "")
            delta_text.setVisible(True)  # 常驻右上（提示或统计）

        # 重建时先断开旧连接（否则每次 rebuild 都向 scene/vb/region 累积回调）
        scene_sig = plot.scene().sigMouseMoved
        for sig, h in ((scene_sig, getattr(self, '_h_mouse', None)),
                       (getattr(self, '_h_region_sig', None), getattr(self, '_h_region', None)),
                       (vb.sigRangeChanged, getattr(self, '_pin_delta_text', None))):
            if sig is not None and h is not None:
                try:
                    sig.disconnect(h)
                except (TypeError, RuntimeError):
                    pass
        self._pin_delta_text = _pin_delta_text
        vb.sigRangeChanged.connect(_pin_delta_text)
        _pin_delta_text()

        # ── 悬停竖线 + 最近点速度（仅 v-t / v-x）──
        hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(
            config.COLOR_GRAY, width=1, style=Qt.PenStyle.DashLine))
        hover_line.setVisible(False)
        # ignoreBounds：悬停辅助线不参与 ViewBox 自动范围计算
        plot.addItem(hover_line, ignoreBounds=True)
        # anchor=(0, 0)：文字左上角贴住 pos（(0,1) 同样会被顶到视图外）
        hover_text = pg.TextItem("", color=fg, anchor=(0, 0))
        hover_text.setVisible(False)
        plot.addItem(hover_text, ignoreBounds=True)
        self._hover_text = hover_text
        self._hover_text_visible = False
        # 钉在视图左上角：缩放/平移时跟随，不出现拖后腿跳回
        if getattr(self, '_pin_hover_text', None) is not None:
            try:
                vb.sigRangeChanged.disconnect(self._pin_hover_text)
            except (TypeError, RuntimeError):
                pass

        def _pin_hover_text(*args) -> None:
            xmin_v, xmax_v = vb.viewRange()[0]
            ymin_v, ymax_v = vb.viewRange()[1]
            hover_text.setPos(xmin_v, ymax_v)

        self._pin_hover_text = _pin_hover_text
        vb.sigRangeChanged.connect(_pin_hover_text)
        _pin_hover_text()  # 初始定位：重建后视图未变化时不触发 sigRangeChanged
        import bisect

        def _on_mouse_moved(pos) -> None:
            if is_dtx:
                return
            if not plot.sceneBoundingRect().contains(pos):
                hover_line.setVisible(False)
                hover_text.setVisible(False)
                self._hover_text_visible = False
                return
            pt = vb.mapSceneToView(pos)
            x = pt.x()
            lines = []
            for i in range(3):
                xd = all_x[i]
                if not xd:
                    continue
                idx = bisect.bisect_left(xd, x)
                if idx >= len(xd):
                    idx = len(xd) - 1
                elif idx > 0 and abs(xd[idx - 1] - x) < abs(xd[idx] - x):
                    idx -= 1
                n = Path(self._csvs[i] or "").stem
                v = all_y[i][idx]
                lines.append(f"{n}: {v:.0f} km/h" if v >= 0 else f"{n}: 无效")
            hover_line.setPos(x)
            hover_line.setVisible(True)
            hover_text.setText(chr(10).join(lines) if lines else "")
            hover_text.setVisible(bool(lines))
            self._hover_text_visible = bool(lines)

        self._h_mouse = _on_mouse_moved
        plot.scene().sigMouseMoved.connect(_on_mouse_moved)

        # 拖选 region：初始覆盖全数据范围（拖 handles 调整，统计实时更新）
        xs_all = [x for xd in all_x for x in xd]
        xmin0 = min(xs_all) if xs_all else 0.0
        xmax0 = max(xs_all) if xs_all else 1.0
        if xmax0 <= xmin0:
            xmax0 = xmin0 + 1.0
        # mkBrush 会丢弃 alpha → 用 make_brush 显式构造半透明填充；
        # 边界线用深蓝（默认黄色与主题不搭）
        region = pg.LinearRegionItem(values=(xmin0, xmax0), orientation='vertical',
                                     movable=True,
                                     brush=make_brush(config.COLOR_BLUE, 20),
                                     pen=pg.mkPen("#1565C0"))
        self._region = region
        self._h_region = lambda r: _update_delta_text()
        self._h_region_sig = region.sigRegionChanged
        region.sigRegionChanged.connect(self._h_region)
        region.setVisible(False)  # 重绘后默认不选择，仅右键拖拽才绘制
        plot.addItem(region)

        def _on_drag_range(x0: float, x1: float) -> None:
            """右键拖拽选择范围（模拟重构前 SpanSelector）。"""
            if x0 > x1:
                x0, x1 = x1, x0
            region.setRegion((x0, x1))
            region.setVisible(True)

        def _on_drag_click(x: float) -> None:
            """右键点击（无拖动）：若在选区外则取消选择，恢复提示文字。"""
            if not region.isVisible():
                return
            x0, x1 = cast("tuple[float, float]", region.getRegion())
            if x < x0 or x > x1:
                region.setVisible(False)
                delta_text.setText("← 右键拖拽选择范围，点击选区外取消")
                delta_text.setVisible(True)

        plot.sig_drag_range.connect(_on_drag_range)
        plot.sig_drag_click.connect(_on_drag_click)

        # 拖选统计文本放右上（悬停文本在左上，避免重叠）
        delta_text.setText("← 右键拖拽选择范围，点击选区外取消")
        delta_text.setVisible(True)  # 初始提示可见

    def _render(self) -> None:
        """高性能渲染：缓存 CSV 解析结果，smooth 变化时仅更新线数据。

        仅在 CSV 文件变更或模式切换时完全重建图表。
        """
        if not self._ready:
            return

        plot = self._plot
        if plot is None:
            return

        if not any(self._csvs):
            return

        mode = self._chart_mode
        show_cd = self._show_corrected
        smooth_str = self._smooth_str
        is_dtx = (mode == "dt-x")
        is_vt = (mode == "v-t")
        colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]

        # ── 检测是否需要完全重建 ──
        csv_key = (tuple(self._csvs), tuple(self._offsets))
        last_key = getattr(self, '_render_cache_key', None)
        needs_rebuild = (csv_key != last_key or mode != getattr(self, '_render_last_mode', '') or show_cd != getattr(self, '_render_last_cd', None))
        smooth_changed = (smooth_str != getattr(self, '_render_last_smooth', -1))

        if not needs_rebuild and not smooth_changed:
            return  # nothing changed

        if needs_rebuild:
            # ── 解析 + 缓存 CSV 数据 ──
            all_x: list[list[float]] = [[], [], []]
            all_y: list[list[float]] = [[], [], []]
            all_raw: list = [None, None, None]
            name1 = name2 = label = ""
            has_data = False

            if is_dtx:
                if not self._csvs[0] or not self._csvs[1]:
                    QMessageBox.warning(self._stack, "数据不足",
                        "Δt-x 需要 CSV 1 和 CSV 2 均已导入。")
                    return
                t1, d1, s1, _ = parse_csv(self._csvs[0])
                t2, d2, s2, _ = parse_csv(self._csvs[1])
                # 帧偏移（GUI only，循环卷绕）：时间轴循环平移对齐起跑线，
                # 越界部分卷绕到另一端（一整圈语义）；不修改 CSV 内容。
                t1, d1, s1, f1 = _shift_csv(t1, d1, s1, [0]*len(s1), self._offsets[0])
                t2, d2, s2, f2 = _shift_csv(t2, d2, s2, [0]*len(s2), self._offsets[1])
                t2_interp = np.interp(d1, d2, t2)
                dt = np.array(t1) - t2_interp
                all_x[0] = d1; all_y[0] = dt.tolist()
                all_raw[0] = (d1, dt.tolist(), None)
                name1 = Path(self._csvs[0]).stem
                name2 = Path(self._csvs[1]).stem
                label = f"{name1} - {name2}"
                has_data = True
            else:
                for i, csv_path in enumerate(self._csvs):
                    if not csv_path:
                        continue
                    try:
                        times, dists, speeds, flags = parse_csv(csv_path)
                        name = Path(csv_path).stem
                        # 循环卷绕偏移（帧 = 索引位移）；周期取模重标起点：
                        # 新 t=0 / x=0 起点，之后按圈周期卷绕（x/t 始终 >= 0，
                        # 起点前的圈尾路段出现在末端 — 周期拓展语义）
                        _lap_t = times[-1] - times[0] if len(times) > 1 else 0.0
                        _lap_d = dists[-1] - dists[0] if len(dists) > 1 else 0.0
                        times, dists, speeds, flags = _shift_csv(
                            times, dists, speeds, flags, self._offsets[i])
                        if self._offsets[i]:
                            base_t = times[0]
                            base_d = dists[0]
                            if is_vt:
                                x_data = [((x - base_t) % _lap_t) if _lap_t > 0 else (x - base_t)
                                          for x in times]
                            else:
                                x_data = [((d - base_d) % _lap_d) if _lap_d > 0 else (d - base_d)
                                          for d in dists]
                        else:
                            x_data = times if is_vt else dists
                        all_x[i] = x_data; all_y[i] = speeds
                        all_raw[i] = (x_data, speeds, flags)
                        has_data = True
                    except Exception as e:
                        QMessageBox.critical(self._stack, "解析失败",
                            f"{Path(csv_path).name}: {e}")
                        return

            if not has_data:
                return

            # ── 缓存原始数据 ──
            self._render_cache = {
                'all_x': all_x, 'all_y': all_y, 'all_raw': all_raw,
                'is_dtx': is_dtx, 'is_vt': is_vt,
                'name1': name1, 'name2': name2, 'label': label,
            }
            self._render_cache_key = csv_key
            self._render_last_mode = mode
            self._render_last_smooth = smooth_str
            self._render_last_cd = show_cd

            # ── 保存当前视图 ──
            if self._last_mode and self._last_mode != "dt-x":
                plot_item0 = plot.getPlotItem()
                assert plot_item0 is not None
                vb_cur = plot_item0.vb
                assert vb_cur is not None
                self._saved_limits[self._last_mode] = tuple(vb_cur.viewRange())

            # ── 完全重建（pyqtgraph）──
            plot.clear()
            from qfluentwidgets import isDarkTheme
            dark = isDarkTheme()
            bg, fg = chart_colors(dark)
            plot.setBackground(bg)

            # ── 绘制主曲线 ──
            if is_dtx:
                d1 = all_x[0]; dt_list = all_y[0]
                x_vals, y_vals = smooth_data(d1, dt_list, smooth_str) if smooth_str > 0 else (d1, dt_list)
                plot_wrapped_pg(plot, x_vals, y_vals, colors[0])
            else:
                for i, raw in enumerate(all_raw):
                    if raw is None:
                        continue
                    x_data, speeds, flags = raw
                    plot_segmented_pg(plot, x_data, speeds, flags,
                                      colors[i], show_cd, smooth_str)

            # ── 标签 ──
            if is_dtx:
                xlabel, ylabel = "距离 (m)", "Δt (s)"
                title = f"时间差-距离 ({name1} vs {name2})"
            elif is_vt:
                xlabel, ylabel = "时间 (s)", "速度 (km/h)"
                title = "速度-时间曲线"
            else:
                xlabel, ylabel = "距离 (m)", "速度 (km/h)"
                title = "速度-距离曲线"

            plot.setLabel('bottom', xlabel, color=fg)
            plot.setLabel('left', ylabel, color=fg)
            plot.setTitle(title, color=fg)
            plot_item = plot.getPlotItem()
            assert plot_item is not None
            for ax_name in ('left', 'bottom'):
                ax_item = plot_item.getAxis(ax_name)
                ax_item.setTextPen(fg)
                ax_item.setPen(fg)
            if is_dtx:
                zero = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(
                    config.COLOR_GRAY, width=1, style=Qt.PenStyle.DashLine))
                plot.addItem(zero)

            # ── 交互 ──
            self._setup_chart_interactions(plot, all_x, all_y,
                is_dtx, is_vt, label)

            vb2 = plot_item.vb
            assert vb2 is not None
            if is_dtx:
                # Δt-x：y=0 线纵向居中（对称范围）
                dt_vals = all_y[0]
                max_abs = max(abs(min(dt_vals)), abs(max(dt_vals))) if dt_vals else 1.0
                vb2.setYRange(-max_abs, max_abs, padding=0)
            elif saved := self._saved_limits.get(mode):
                xr, yr = saved
                vb2.setXRange(xr[0], xr[1], padding=0)
                vb2.setYRange(yr[0], yr[1], padding=0)
            else:
                vb2.autoRange()  # 新模式：自适应数据
            # 渲染后禁用自动范围检测：悬停/交互 item 的变化不再扩展视图
            # enableAutoRange(axis, enable)：False 作 axis 是 no-op，必须传 enable 关键字
            vb2.enableAutoRange(None, False)

            self._last_mode = mode
            self._sync_figure_theme()

        elif smooth_changed:
            # ── 增量更新或诊断切换：重建线条 ──
            self._render_last_smooth = smooth_str
            self._render_last_cd = show_cd
            cache = self._render_cache
            is_dtx = cache['is_dtx']; is_vt = cache['is_vt']
            all_raw = cache['all_raw']

            # 增量：清除旧曲线（保留 region/hover 等交互 item）
            for it in list(plot.listDataItems()):
                plot.removeItem(it)

            if is_dtx:
                d1 = cache['all_x'][0]; dt_list = cache['all_y'][0]
                x_vals, y_vals = smooth_data(d1, dt_list, smooth_str) if smooth_str > 0 else (d1, dt_list)
                plot_wrapped_pg(plot, x_vals, y_vals, colors[0])
            else:
                for i, raw in enumerate(all_raw):
                    if raw is None:
                        continue
                    x_data, speeds, flags = raw
                    plot_segmented_pg(plot, x_data, speeds, flags,
                                      colors[i], show_cd, smooth_str)

    # ═══════════════════ 其他 ═══════════════════

    def _auto_fit(self) -> None:
        """自动缩放视图（pyqtgraph ViewBox.autoRange）。"""
        plot = self._plot
        if plot is None:
            return
        self._saved_limits.pop(self._chart_mode, None)
        plot_item = plot.getPlotItem()
        assert plot_item is not None
        vb = plot_item.vb
        assert vb is not None
        vb.autoRange()

    def _export_png(self) -> None:
        """导出当前图表为 PNG（pyqtgraph ImageExporter）。"""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self._stack, "导出 PNG", "", "PNG 图片 (*.png)")
        if not path:
            return
        try:
            import pyqtgraph.exporters as _exp
            plot = self._plot
            if plot is None:
                return
            plot_item = plot.plotItem
            assert plot_item is not None
            exp = _exp.ImageExporter(plot_item)
            exp.export(path)
        except Exception as e:
            QMessageBox.critical(self._stack, "导出失败", str(e))
