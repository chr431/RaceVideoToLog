"""数据分析 Tab — PySide6 GUI。

嵌入主窗口 QTabWidget，提供 CSV 导入、多模式图表渲染、范围选择器等功能。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QStackedWidget, QSpinBox,
)
from PySide6.QtCore import Qt
from qfluentwidgets import (PushButton, PrimaryPushButton, CompactSpinBox,
    RadioButton, CheckBox, BodyLabel, Slider, CaptionLabel)
from widget_utils import make_static_card, setup_chart_zoom_pan, make_int_spinbox

from analysis import parse_csv, smooth_data, plot_segmented
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


def _plot_wrapped(ax, x, y, color, lw=0.8):
    """绘制可能循环卷绕的数据：x 下降跳变处断线（两段绘制，避免跨圈连线）。"""
    x = list(x); y = list(y)
    brk = None
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            brk = i
            break
    if brk is None:
        ax.plot(x, y, color=color, linewidth=lw)
    else:
        ax.plot(x[:brk], y[:brk], color=color, linewidth=lw)
        ax.plot(x[brk:], y[brk:], color=color, linewidth=lw)


def _read_fps(path: str) -> float:
    """读取 CSV 头的 fps（偏移秒数换算用）。"""
    from csv_io import parse_csv_header
    try:
        return float(parse_csv_header(path).get("fps", "0"))
    except (ValueError, TypeError):
        return 0.0


class AnalysisTab:
    """数据分析 Tab — 嵌入 QStackedWidget，修改自动刷新。"""

    def __init__(self, stack: QStackedWidget) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._Figure = Figure
        self._FigureCanvas = FigureCanvasQTAgg

        self._stack = stack

        # 状态
        self._csvs: list[str | None] = [None, None, None]
        self._labels: list = []
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._chart_mode: str = "v-x"
        self._show_corrected: bool = False
        self._saved_limits: dict[str, tuple | None] = {}
        self._last_mode: str | None = None
        self._smooth_str: int = 0
        self._span_selector = None
        self._offsets: list[int] = [0, 0, 0]  # 每行 CSV 的帧偏移（仅 GUI）
        self._fps: list[float] = [0.0, 0.0, 0.0]  # 每行 CSV 的 fps（偏移换算用）
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

        # ── Matplotlib 画布 ──
        self._figure = self._Figure(figsize=(8, 5), dpi=100)
        self._canvas = self._FigureCanvas(self._figure)
        self._canvas.setParent(tab)
        layout.addWidget(self._canvas, 1)
        self._sync_figure_theme()
        self._ready = True

    # ═══════════════════ 事件 ═══════════════════

    def _sync_figure_theme(self) -> None:
        """根据应用当前主题同步 matplotlib 画布背景色和文字颜色。"""
        from PySide6.QtGui import QPalette, QColor
        from qfluentwidgets import isDarkTheme
        dark = isDarkTheme()
        bg, fg = chart_colors(dark)
        if self._figure:
            self._figure.set_facecolor(bg)
            if self._figure.axes:
                for ax in self._figure.axes:
                    ax.set_facecolor(bg)
                    ax.tick_params(colors=fg)
                    ax.xaxis.label.set_color(fg)
                    ax.yaxis.label.set_color(fg)
                    ax.title.set_color(fg)
                    ax.spines["bottom"].set_color(fg if dark else "#888")
                    ax.spines["left"].set_color(fg if dark else "#888")
                    ax.spines["top"].set_color(fg if dark else "#888")
                    ax.spines["right"].set_color(fg if dark else "#888")
                    ax.grid(True, alpha=0.2 if dark else 0.3)
        if self._canvas:
            # 直接设置 canvas widget 的背景色（覆盖 QSS）
            c = QColor(bg)
            p = self._canvas.palette()
            p.setColor(QPalette.ColorRole.Window, c)
            p.setColor(QPalette.ColorRole.Base, c)
            self._canvas.setPalette(p)
            self._canvas.setAutoFillBackground(True)
            self._canvas.draw_idle()

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
        if self._offset_timer is None:
            self._offset_timer = QTimer(self)
            self._offset_timer.setSingleShot(True)
            self._offset_timer.timeout.connect(self._flush_offset_render)
        self._offset_timer.start(150)

    def _flush_offset_render(self) -> None:
        self._offset_timer = None
        self._invalidate_and_render()

    # ═══════════════════ 渲染 ═══════════════════


    def _setup_chart_interactions(self, ax, canvas, all_x, all_y, is_dtx, is_vt,
                                    delta_label, label):
        """配置图表交互：SpanSelector 范围选择 + 缩放/平移。"""
        delta_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
            va="top", fontsize=9, color=config.COLOR_FG_LIGHT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        def _on_select(xmin: float, xmax: float) -> None:
            if xmin > xmax:
                xmin, xmax = xmax, xmin
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
            delta_text.set_text("\n".join(results) if results else "")

        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
        from matplotlib.widgets import SpanSelector
        self._span_selector = SpanSelector(ax, _on_select, "horizontal",
            props=dict(facecolor=COLOR_BLUE, alpha=0.15),
            interactive=True, drag_from_anywhere=True,
            button=1)
        delta_text.set_text(f"← 拖拽选择范围查看{delta_label}")

        # ── 悬停竖线 + 最近数据点速度（仅 v-t / v-x，Δt-x 保持原样）──
        # blit 增量重绘：只重画竖线和文本，避免整图重绘（数据点多时跟手）
        import bisect
        hover_line = ax.axvline(0, color=config.COLOR_GRAY, linewidth=0.8,
                                linestyle="--", alpha=0.7, visible=False)
        hover_bg = [None]

        def _save_hover_bg(event: object) -> None:
            if event.canvas is canvas:
                try:
                    hover_bg[0] = canvas.copy_from_bbox(ax.bbox)
                except Exception:
                    hover_bg[0] = None

        _hover_last = [0.0]
        import time as _time

        def _hover_redraw() -> None:
            now = _time.time()
            if now - _hover_last[0] < 0.016:
                return  # 节流：~60fps 上限
            _hover_last[0] = now
            # 抑制 stale：防止 matplotlib 空闲循环自动整图重绘（导致抖动）
            hover_line.stale = False
            delta_text.stale = False
            if hover_bg[0] is not None:
                try:
                    canvas.restore_region(hover_bg[0])
                    ax.draw_artist(hover_line)
                    ax.draw_artist(delta_text)
                    canvas.blit(ax.bbox)
                    return
                except Exception:
                    pass
            canvas.draw_idle()

        def _on_motion(event: object) -> None:
            if is_dtx:
                return
            selected = getattr(self._span_selector, 'visible', False)
            if event.xdata is None:
                hover_line.set_visible(False)
                if not selected:
                    delta_text.set_text(f"← 拖拽选择范围查看{delta_label}")
                _hover_redraw()
                return
            if selected:
                hover_line.set_visible(False)  # 拖选后保持区间信息
                return
            hover_line.set_visible(True)
            hover_line.set_xdata([event.xdata, event.xdata])
            lines = []
            for i in range(3):
                xd = all_x[i]
                if not xd:
                    continue
                pos = bisect.bisect_left(xd, event.xdata)
                if pos >= len(xd):
                    pos = len(xd) - 1
                elif pos > 0 and abs(xd[pos - 1] - event.xdata) < abs(xd[pos] - event.xdata):
                    pos -= 1
                n = Path(self._csvs[i] or "").stem
                v = all_y[i][pos]
                lines.append(f"{n}: {v:.0f} km/h" if v >= 0 else f"{n}: 无效")
            delta_text.set_text(chr(10).join(lines) if lines else "")
            _hover_redraw()

        canvas.mpl_connect("motion_notify_event", _on_motion)
        canvas.mpl_connect("draw_event", _save_hover_bg)

        # 滚轮缩放 + 右键平移
        setup_chart_zoom_pan(ax, canvas, throttle_ms=0)

    def _render(self) -> None:
        """高性能渲染：缓存 CSV 解析结果，smooth 变化时仅更新线数据。

        仅在 CSV 文件变更或模式切换时完全重建图表。
        """
        if not self._ready:
            return

        fig = self._figure
        canvas = self._canvas
        if fig is None or canvas is None:
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
            all_flags: list[list[int]] = [[], [], []]
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
                t1, d1, s1, _ = _shift_csv(t1, d1, s1, self._offsets[0])
                t2, d2, s2, _ = _shift_csv(t2, d2, s2, self._offsets[1])
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
                        self._fps[i] = _read_fps(csv_path)
                        name = Path(csv_path).stem
                        # 循环卷绕偏移（帧 = 索引位移），v-x 时新 t=0 点为 x=0
                        times, dists, speeds, flags = _shift_csv(
                            times, dists, speeds, flags, self._offsets[i])
                        x_data = times if is_vt else dists
                        if not is_vt and self._offsets[i]:
                            base = dists[0]
                            x_data = [d - base for d in dists]
                        all_x[i] = x_data; all_y[i] = speeds
                        all_flags[i] = flags
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
                'all_x': all_x, 'all_y': all_y, 'all_flags': all_flags,
                'all_raw': all_raw, 'is_dtx': is_dtx, 'is_vt': is_vt,
                'name1': name1, 'name2': name2, 'label': label,
            }
            self._render_cache_key = csv_key
            self._render_last_mode = mode
            self._render_last_smooth = smooth_str
            self._render_last_cd = show_cd

            # ── 保存当前视图 ──
            if fig.axes and self._last_mode and self._last_mode != "dt-x":
                self._saved_limits[self._last_mode] = (
                    fig.axes[0].get_xlim(), fig.axes[0].get_ylim())

            # ── 完全重建 ──
            fig.clear()
            from qfluentwidgets import isDarkTheme
            dark = isDarkTheme()
            fig.set_facecolor(chart_colors(dark)[0])
            ax = fig.add_subplot(111)

            # ── 绘制主曲线 ──
            self._chart_lines = []  # 缓存线条引用用于增量更新
            if is_dtx:
                d1 = all_x[0]; dt_list = all_y[0]
                x_vals, y_vals = smooth_data(d1, dt_list, smooth_str) if smooth_str > 0 else (d1, dt_list)
                _plot_wrapped(ax, x_vals, y_vals, colors[0])
                self._chart_lines.append((line, 0, d1, dt_list, None))
                ax.plot([], [], color=colors[0], linewidth=0.8, label=label)
            else:
                for i, raw in enumerate(all_raw):
                    if raw is None:
                        continue
                    x_data, speeds, flags = raw
                    ln_refs = plot_segmented(ax, x_data, speeds, flags,
                                                colors[i], show_cd, smooth_str)
                    # Cache for smooth updates
                    self._chart_lines.append((ax.lines[-2] if len(ax.lines) >= 2 else ax.lines[-1],
                                                i, x_data, speeds, flags))
                    ax.plot([], [], color=colors[i], linewidth=0.8,
                            label=Path(self._csvs[i] or "").stem)

            # ── 标签 ──
            if is_dtx:
                xlabel, ylabel = "距离 (m)", "Δt (s)"
                title = f"时间差-距离 ({name1} vs {name2})"
                delta_label = "Δ(Δt)"
            elif is_vt:
                xlabel, ylabel = "时间 (s)", "速度 (km/h)"
                title = "速度-时间曲线"; delta_label = "行驶距离"
            else:
                xlabel, ylabel = "距离 (m)", "速度 (km/h)"
                title = "速度-距离曲线"; delta_label = "用时"

            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
            if is_dtx:
                ax.axhline(y=0, color=config.COLOR_GRAY, linewidth=1.2, linestyle="--", alpha=0.7)

            # ── 交互 ──
            self._setup_chart_interactions(ax, canvas, all_x, all_y,
                is_dtx, is_vt, delta_label, label)

            fig.tight_layout()
            if not is_dtx:
                saved = self._saved_limits.get(mode)
                if saved is not None:
                    ax.set_xlim(saved[0]); ax.set_ylim(saved[1])

            canvas.draw()
            self._last_mode = mode
            self._sync_figure_theme()

        elif smooth_changed:
            # ── 增量更新或诊断切换：重建线条 ──
            self._render_last_smooth = smooth_str
            self._render_last_cd = show_cd
            ax = fig.axes[0]
            cache = self._render_cache
            is_dtx = cache['is_dtx']; is_vt = cache['is_vt']
            all_raw = cache['all_raw']

            if ax.lines and not is_dtx:
                # 清除旧诊断段（红色/绿色）— 重建非平滑基线
                pass  # plot_segmented handles this internally

            # 重建：清除所有 lines，保留非-line artists
            for line in ax.lines[:]:
                line.remove()
            # 重建图例占位
            self._chart_lines = []

            if is_dtx:
                d1 = cache['all_x'][0]; dt_list = cache['all_y'][0]
                x_vals, y_vals = smooth_data(d1, dt_list, smooth_str) if smooth_str > 0 else (d1, dt_list)
                _plot_wrapped(ax, x_vals, y_vals, colors[0])
                self._chart_lines.append((line, 0, d1, dt_list, None))
            else:
                for i, raw in enumerate(all_raw):
                    if raw is None:
                        continue
                    x_data, speeds, flags = raw
                    _ = plot_segmented(ax, x_data, speeds, flags,
                                        colors[i], show_cd, smooth_str)
                    self._chart_lines.append((ax.lines[-2] if len(ax.lines) >= 2 else ax.lines[-1],
                                                i, x_data, speeds, flags))
                # 恢复图例
                for i, raw in enumerate(all_raw):
                    if raw is not None:
                        ax.plot([], [], color=colors[i], linewidth=0.8,
                                label=Path(self._csvs[i] or "").stem)

            canvas.draw_idle()

    # ═══════════════════ 其他 ═══════════════════

    def _auto_fit(self) -> None:
        fig = self._figure; canvas = self._canvas
        if fig is None or canvas is None or not fig.axes:
            return
        self._saved_limits.pop(self._chart_mode, None)
        ax = fig.axes[0]
        ax.autoscale(enable=True, axis="both")
        ax.relim(); ax.autoscale_view()
        canvas.draw_idle()

    def _export_png(self) -> None:
        fig = self._figure
        if fig is None or not fig.axes:
            QMessageBox.warning(self._stack, "无数据", "请先渲染曲线。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self._stack, "导出 PNG", "", "PNG 图片 (*.png)")
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            QMessageBox.information(self._stack, "导出完成", f"已保存: {path}")
