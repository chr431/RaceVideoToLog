"""数据分析 Tab — PySide6 GUI。

嵌入主窗口 QTabWidget，提供 CSV 导入、多模式图表渲染、范围选择器等功能。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
	QWidget, QPushButton, QLabel, QRadioButton, QCheckBox, QGroupBox,
	QSlider, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
	QStackedWidget, QLineEdit,
)
from PySide6.QtCore import Qt
from qfluentwidgets import PushButton, PrimaryPushButton, CompactSpinBox

from analysis import parse_csv, smooth_data, plot_segmented


class AnalysisTab:
	"""数据分析 Tab — 嵌入 QStackedWidget。"""

	def __init__(self, stack: QStackedWidget) -> None:
		from matplotlib.figure import Figure
		from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
		self._Figure = Figure
		self._FigureCanvas = FigureCanvasQTAgg

		self._stack = stack

		# 状态
		self._csvs: list[str | None] = [None, None, None]
		self._labels: list[QLabel] = []
		self._figure: Figure | None = None
		self._canvas: FigureCanvasQTAgg | None = None
		self._chart_mode: str = "v-x"
		self._show_corrected: bool = False
		self._saved_limits: dict[str, tuple | None] = {}
		self._last_mode: str | None = None
		self._smooth_str: int = 25
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
			slot = QGroupBox(f"CSV {i+1}")
			sl = QHBoxLayout(slot)
			btn_import = PushButton("导入")
			btn_import.clicked.connect(lambda checked, idx=i: self._import(idx))
			sl.addWidget(btn_import)
			btn_clear = PushButton("清除")
			btn_clear.clicked.connect(lambda checked, idx=i: self._clear(idx))
			sl.addWidget(btn_clear)
			lbl = QLabel("未导入")
			self._labels.append(lbl)
			sl.addWidget(lbl)
			cl.addWidget(slot, 0, i)

		btn_render = PrimaryPushButton("渲染曲线")
		btn_render.clicked.connect(self._render)
		cl.addWidget(btn_render, 0, 3)

		btn_export = PushButton("导出 PNG")
		btn_export.clicked.connect(self._export_png)
		cl.addWidget(btn_export, 0, 4)

		# 第二行：模式 + 平滑 + 自动调整
		row2 = QHBoxLayout()
		self._cb_corrected = QCheckBox("显示诊断信息")
		self._cb_corrected.toggled.connect(lambda v: setattr(self, '_show_corrected', v))
		row2.addWidget(self._cb_corrected)

		row2.addWidget(QLabel("平滑"))
		self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
		self._smooth_slider.setRange(0, 100); self._smooth_slider.setValue(25)
		self._smooth_slider.setFixedWidth(100)
		self._smooth_slider.valueChanged.connect(lambda v: setattr(self, '_smooth_str', v))
		row2.addWidget(self._smooth_slider)

		self._smooth_spin = CompactSpinBox(); self._smooth_spin.setRange(0, 100); self._smooth_spin.setValue(25); self._smooth_spin.setFixedWidth(60)
		try:
			self._smooth_spin.compactSpinButton.clicked.disconnect()
		except Exception:
			pass
		self._smooth_spin._showFlyout = lambda: None
		self._smooth_spin.valueChanged.connect(self._smooth_slider.setValue)
		self._smooth_slider.valueChanged.connect(self._smooth_spin.setValue)
		row2.addWidget(self._smooth_spin)

		self._rb_vt = QRadioButton("v-t"); self._rb_vx = QRadioButton("v-x")
		self._rb_vx.setChecked(True); self._rb_dtx = QRadioButton("Δt-x")
		for mode, rb in [("v-t", self._rb_vt), ("v-x", self._rb_vx), ("dt-x", self._rb_dtx)]:
			rb.toggled.connect(lambda checked, m=mode: self._on_mode(m) if checked else None)
			row2.addWidget(rb)

		btn_fit = PushButton("自动调整")
		btn_fit.clicked.connect(self._auto_fit)
		row2.addWidget(btn_fit)

		cl.addLayout(row2, 1, 0, 1, 5)
		layout.addWidget(ctrl)

		# ── Matplotlib 画布 ──
		self._figure = self._Figure(figsize=(8, 5), dpi=100)
		self._canvas = self._FigureCanvas(self._figure)
		self._canvas.setParent(tab)
		layout.addWidget(self._canvas, 1)
		self._sync_figure_theme()

	# ═══════════════════ 事件 ═══════════════════

	def _sync_figure_theme(self) -> None:
		"""根据应用当前主题同步 matplotlib 画布背景色和文字颜色。"""
		from PySide6.QtGui import QPalette, QColor
		from qfluentwidgets import isDarkTheme
		dark = isDarkTheme()
		bg = "#2a2a2a" if dark else "#ffffff"
		fg = "#e0e0e0" if dark else "#333333"
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
			self._saved_limits.clear()

	def _clear(self, index: int) -> None:
		self._csvs[index] = None
		self._labels[index].setText("未导入")
		self._saved_limits.clear()

	# ═══════════════════ 渲染 ═══════════════════

	def _render(self) -> None:
		from matplotlib.widgets import SpanSelector

		fig = self._figure
		canvas = self._canvas
		if fig is None or canvas is None:
			return

		if fig.axes and self._last_mode and self._last_mode != "dt-x":
			self._saved_limits[self._last_mode] = (
				fig.axes[0].get_xlim(), fig.axes[0].get_ylim())

		fig.clear()
		# 新建 axes 前设置背景色（后续 _sync_figure_theme 会同步完整主题）
		from PySide6.QtWidgets import QApplication
		app = QApplication.instance()
		dark = bool(app.property("dark_mode")) if app else False
		fig.set_facecolor("#2a2a2a" if dark else "#ffffff")
		ax = fig.add_subplot(111)
		colors = ["#2196F3", "#FF5722", "#4CAF50"]
		mode = self._chart_mode
		show_cd = self._show_corrected
		smooth_str = self._smooth_str

		all_x: list[list[float]] = [[], [], []]
		all_y: list[list[float]] = [[], [], []]
		all_flags: list[list[int]] = [[], [], []]
		is_vt = (mode == "v-t")
		is_dtx = (mode == "dt-x")
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
			name1 = Path(self._csvs[0]).stem
			name2 = Path(self._csvs[1]).stem
			label = f"{name1} - {name2}"
			if smooth_str > 0:
				sx, sy = smooth_data(d1, dt.tolist(), smooth_str)
				ax.plot(sx, sy, color=colors[0], linewidth=0.8)
			else:
				ax.plot(d1, dt.tolist(), color=colors[0], linewidth=0.8)
			ax.plot([], [], color=colors[0], linewidth=0.8, label=label)
			has_data = True
		else:
			for i, csv_path in enumerate(self._csvs):
				if not csv_path:
					continue
				try:
					times, dists, speeds, flags = parse_csv(csv_path)
					name = Path(csv_path).stem
					x_data = times if is_vt else dists
					all_x[i] = x_data; all_y[i] = speeds; all_flags[i] = flags
					if show_cd or smooth_str > 0:
						plot_segmented(ax, x_data, speeds, flags,
							colors[i], show_cd, smooth_str)
					else:
						ax.plot(x_data, speeds, color=colors[i], linewidth=0.8)
					ax.plot([], [], color=colors[i], linewidth=0.8, label=name)
					has_data = True
				except Exception as e:
					QMessageBox.critical(self._stack, "解析失败",
						f"{Path(csv_path).name}: {e}")
					return

		if not has_data:
			return

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
			ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)

		delta_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
			va="top", fontsize=9, color="#333333",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

		# SpanSelector 回调
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
								avg = (all_y[i][j] + all_y[i][j - 1]) / 2 / 3.6
								total += avg * dt_v
							else:
								dx_v = xd[j] - xd[j - 1]
								avg = ((all_y[i][j] + all_y[i][j - 1]) / 2) / 3.6
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
		self._span_selector = SpanSelector(ax, _on_select, "horizontal",
			props=dict(facecolor="#2196F3", alpha=0.15),
			interactive=True, drag_from_anywhere=True,
			button=1)  # type: ignore[arg-type]
		delta_text.set_text(f"← 拖拽选择范围查看{delta_label}")

		# 滚轮缩放 + 右键平移
		_px = [None, None]

		def _on_scroll(event: object) -> None:
			s = 0.85 if getattr(event, 'button', '') == 'up' else 1.15
			xl = ax.get_xlim(); yl = ax.get_ylim()
			xm = (xl[0] + xl[1]) / 2; ym = (yl[0] + yl[1]) / 2
			ax.set_xlim(xm - (xm - xl[0]) * s, xm + (xl[1] - xm) * s)
			ax.set_ylim(ym - (ym - yl[0]) * s, ym + (yl[1] - ym) * s)
			canvas.draw_idle()

		def _on_press(event: object) -> None:
			if getattr(event, 'button', 0) == 3:
				_px[0], _px[1] = getattr(event, 'xdata', None), getattr(event, 'ydata', None)

		def _on_motion(event: object) -> None:
			if getattr(event, 'button', 0) == 3 and _px[0] is not None:
				xd = getattr(event, 'xdata', None)
				if xd is not None:
					dx = _px[0] - xd
					dy = (_px[1] or 0) - (getattr(event, 'ydata', None) or 0)
					xl = ax.get_xlim(); yl = ax.get_ylim()
					ax.set_xlim(xl[0] + dx, xl[1] + dx)
					ax.set_ylim(yl[0] + dy, yl[1] + dy)
					canvas.draw_idle()

		fig.canvas.mpl_connect("scroll_event", _on_scroll)
		fig.canvas.mpl_connect("button_press_event", _on_press)
		fig.canvas.mpl_connect("motion_notify_event", _on_motion)

		fig.tight_layout()
		if not is_dtx:
			saved = self._saved_limits.get(mode)
			if saved is not None:
				ax.set_xlim(saved[0]); ax.set_ylim(saved[1])

		canvas.draw()
		self._last_mode = mode
		self._sync_figure_theme()

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
