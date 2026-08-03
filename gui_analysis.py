"""数据分析 Tab — PySide6 GUI。

嵌入主窗口 QTabWidget，提供 CSV 导入、多模式图表渲染、范围选择器等功能。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QStackedWidget,
)
from PySide6.QtCore import Qt
from qfluentwidgets import (PushButton, PrimaryPushButton, CompactSpinBox,
    RadioButton, CheckBox, BodyLabel, Slider)
from widget_utils import make_static_card, setup_chart_zoom_pan

from analysis import parse_csv, smooth_data, plot_segmented
from config import (COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, MPS_TO_KMH, chart_colors)


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
        csv_key = tuple(self._csvs)
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
                        x_data = times if is_vt else dists
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
                line, = ax.plot(x_vals, y_vals, color=colors[0], linewidth=0.8)
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
                is_dtx, is_vt, delta_label, label, name1, name2)

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
                line, = ax.plot(x_vals, y_vals, color=colors[0], linewidth=0.8)
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
