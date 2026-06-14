"""RaceVideoToLog — 数据分析模块。

支持 GUI 交互式分析和 CLI 无头导出：
- GUI: AnalysisTab 类，嵌入主窗口的 Notebook
- CLI: run_analysis_headless()，从两个 CSV 导出 3 张 PNG

用法:
  # GUI 模式
  from analysis import AnalysisTab
  tab = AnalysisTab(notebook, footer, status_var, progress_var)

  # CLI 模式
  python RaceVideoToLog.py --analysis csv1.csv csv2.csv [--analysis-out PREFIX]
"""
from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ocr_engine import _savgol_filter_np


# ═══════════════════════════════════════════════════════════════
# 模块级工具函数
# ═══════════════════════════════════════════════════════════════

def parse_csv(path: str | Path) -> tuple[list[float], list[float], list[float], list[int]]:
	"""解析 CSV 文件，返回 (times, dists, speeds, flags)。

	每行格式: timestamp,distance,speed_kmh,flag
	- 跳过以 # 开头的注释行和空行
	- try/except 保护浮点转换
	- 裁剪起始零速帧，距离和时间归零
	"""
	times, dists, speeds, flags = [], [], [], []
	with open(str(path), "r", encoding="utf-8-sig") as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or not line:
				continue
			parts = line.split(",")
			if len(parts) >= 3:
				try:
					times.append(float(parts[0]))
					dists.append(float(parts[1]))
					speeds.append(float(parts[2]))
					flags.append(int(parts[3]) if len(parts) > 3 else 0)
				except ValueError:
					continue
	# 裁剪起始零速帧
	start = 0
	for i, s in enumerate(speeds):
		if s > 0:
			start = i
			break
	if start > 0:
		times = times[start:]
		speeds = speeds[start:]
		base_dist = dists[start]
		dists = [d - base_dist for d in dists[start:]]
		flags = flags[start:]
		base_time = times[0]
		times = [t - base_time for t in times]
	return times, dists, speeds, flags


def smooth_data(xv, yv, strength: int) -> tuple[np.ndarray, np.ndarray]:
	"""Savitzky-Golay 滤波（纯 numpy 实现）：多项式滑动窗口拟合，保留峰谷形状。

	Args:
		xv: x 轴数据
		yv: y 轴数据
		strength: 平滑强度 (0-100)，0 表示不平滑
	Returns:
		(xv_array, smoothed_yv) — xv 保持不变，yv 平滑后
	"""
	if strength <= 0 or len(xv) < 5:
		return np.array(xv, dtype=float), np.array(yv, dtype=float)
	win = int(len(xv) * strength / 100.0 * 0.0175)
	win = max(5, min(win, len(xv) - 2))
	if win % 2 == 0:
		win += 1
	polyorder = min(3, win - 1)
	sy = _savgol_filter_np(np.array(yv, dtype=float), win, polyorder)
	return np.array(xv, dtype=float), sy


def plot_segmented(ax, x, y, flags, normal_color: str, show_red: bool,
                   smooth_strength: int) -> None:
	"""平滑 + 纠错段着色。

	- 红色 (#F44336): 自动纠错 (flag=1)
	- 绿色 (#81C784): 人工纠错 (flag>=2)
	"""
	red = "#F44336"
	green = "#81C784"

	if smooth_strength > 0:
		x, y = smooth_data(x, y, smooth_strength)

	ax.plot(x, y, color=normal_color, linewidth=0.8)

	if not show_red or not flags or not any(f >= 1 for f in flags):
		return

	n_orig = len(flags)
	n_smooth = len(x)
	_x = x.tolist() if hasattr(x, 'tolist') else list(x)
	_y = y.tolist() if hasattr(y, 'tolist') else list(y)

	# 红色段（flag=1 自动纠错）
	rx, ry = [], []
	for i in range(n_orig - 1):
		if flags[i] == 1 and flags[i + 1] == 1:
			si = int(i * n_smooth / n_orig)
			ei = int((i + 2) * n_smooth / n_orig)
			si = min(si, n_smooth - 2)
			ei = min(ei, n_smooth)
			if ei > si:
				rx.extend(_x[si:ei] + [float('nan')])
				ry.extend(_y[si:ei] + [float('nan')])
	if rx:
		ax.plot(rx, ry, color=red, linewidth=1.2)

	# 绿色段（flag>=2 人工纠错）
	gx, gy = [], []
	for i in range(n_orig - 1):
		if flags[i] >= 2 or flags[i + 1] >= 2:
			si = int(i * n_smooth / n_orig)
			ei = int((i + 2) * n_smooth / n_orig)
			si = min(si, n_smooth - 2)
			ei = min(ei, n_smooth)
			if ei > si:
				gx.extend(_x[si:ei] + [float('nan')])
				gy.extend(_y[si:ei] + [float('nan')])
	if gx:
		ax.plot(gx, gy, color=green, linewidth=1.5, alpha=0.8)


# ═══════════════════════════════════════════════════════════════
# GUI: 数据分析 Tab
# ═══════════════════════════════════════════════════════════════

class AnalysisTab:
	"""数据分析 Tab — 嵌入主窗口的 Notebook。

	提供 CSV 导入、多模式图表渲染、范围选择器、平滑控制等功能。
	"""

	def __init__(self, notebook: ttk.Notebook, footer: ttk.Frame,
		status_var: tk.StringVar, progress_var: tk.DoubleVar) -> None:

		from matplotlib.figure import Figure
		from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
		self._Figure = Figure
		self._FigureCanvasTkAgg = FigureCanvasTkAgg

		self._notebook = notebook
		self._footer = footer
		self.status_var = status_var
		self.progress_var = progress_var

		# 状态变量
		self._analysis_csvs: list[str | None] = [None, None, None]
		self._analysis_labels: list[tk.StringVar] = []
		self._analysis_figure: Figure | None = None
		self._analysis_canvas: FigureCanvasTkAgg | None = None
		self._chart_mode = tk.StringVar(value="v-x")
		self._show_corrected = tk.BooleanVar(value=False)
		self._saved_limits: dict[str, tuple | None] = {}
		self._last_rendered_mode: str | None = None
		self._smooth_strength = tk.IntVar(value=25)
		self._smooth_entry_var = tk.StringVar(value="25")
		self._span_selector = None

		self._build_tab()

	def _build_tab(self) -> None:
		"""构建数据分析标签页 UI。"""
		tab = ttk.Frame(self._notebook)
		self._notebook.add(tab, text="数据分析")
		tab.columnconfigure(0, weight=1)
		tab.rowconfigure(1, weight=1)

		# 顶部控制栏
		ctrl = ttk.Frame(tab, padding=(12, 10, 12, 6))
		ctrl.grid(row=0, column=0, sticky="ew")
		for i in range(3):
			ctrl.columnconfigure(i, weight=1)
			var = tk.StringVar(value="未导入")
			self._analysis_labels.append(var)
			slot = ttk.LabelFrame(ctrl, text=f"CSV {i+1}", padding=(8, 6, 8, 6))
			slot.grid(row=0, column=i, sticky="ew", padx=(0, 6) if i < 2 else (0, 0))
			ttk.Button(slot, text="导入", command=lambda idx=i: self._import_csv(idx)).grid(row=0, column=0, sticky="w")
			ttk.Button(slot, text="清除", command=lambda idx=i: self._clear_csv(idx)).grid(row=0, column=1, sticky="w", padx=(4, 0))
			ttk.Label(slot, textvariable=var, foreground="#555555").grid(row=0, column=2, sticky="w", padx=(6, 0))

		ttk.Button(ctrl, text="渲染曲线", command=self._render_curves).grid(row=0, column=3, sticky="e", padx=(12, 6))
		ttk.Button(ctrl, text="导出 PNG", command=self._export_png).grid(row=0, column=4, sticky="e")
		ttk.Radiobutton(ctrl, text="v-t", variable=self._chart_mode, value="v-t").grid(row=1, column=3, sticky="e", padx=(12, 0))
		ttk.Radiobutton(ctrl, text="v-x", variable=self._chart_mode, value="v-x").grid(row=1, column=4, sticky="w")
		ttk.Radiobutton(ctrl, text="Δt-x", variable=self._chart_mode, value="dt-x").grid(row=1, column=5, sticky="w", padx=(6, 0))
		ttk.Button(ctrl, text="自动调整", command=self._auto_fit).grid(row=1, column=6, sticky="e", padx=(6, 0))
		ttk.Checkbutton(ctrl, text="标记纠错点", variable=self._show_corrected).grid(row=1, column=0, sticky="w", padx=(0, 6))
		ttk.Label(ctrl, text="平滑").grid(row=1, column=1, sticky="e", padx=(0, 2))
		ttk.Scale(ctrl, from_=0, to=100, variable=self._smooth_strength,
			orient="horizontal", length=80).grid(row=1, column=2, sticky="w")
		smooth_entry = ttk.Entry(ctrl, textvariable=self._smooth_entry_var, width=4, justify="center")
		smooth_entry.grid(row=1, column=2, sticky="e", padx=(0, 4))

		# 滑块 ↔ 输入框双向同步
		def _slider_to_entry(*_):
			self._smooth_entry_var.set(str(self._smooth_strength.get()))

		def _entry_to_slider(*_):
			try:
				v = int(self._smooth_entry_var.get())
				self._smooth_strength.set(max(0, min(100, v)))
			except ValueError:
				pass

		self._smooth_strength.trace_add("write", _slider_to_entry)
		self._smooth_entry_var.trace_add("write", _entry_to_slider)

		# 切换到数据分析 tab 时隐藏底部进度条/状态
		def _on_tab_change(event):
			cur = self._notebook.index(self._notebook.select())
			if cur == 1:  # 数据分析 tab
				self.status_var.set("")
				self.progress_var.set(0.0)
			self._update_footer_visibility()

		self._notebook.bind("<<NotebookTabChanged>>", _on_tab_change)

		# Matplotlib 画布
		self._analysis_figure = self._Figure(figsize=(8, 5), dpi=100)
		self._analysis_canvas = self._FigureCanvasTkAgg(self._analysis_figure, master=tab)
		self._analysis_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))

	def _update_footer_visibility(self) -> None:
		"""OCR 处理 tab 显示底部状态栏，数据分析 tab 隐藏。"""
		cur = self._notebook.index(self._notebook.select())
		if cur == 1:
			self._footer.grid_remove()
		else:
			self._footer.grid()

	def _import_csv(self, index: int) -> None:
		path = filedialog.askopenfilename(
			title=f"选择 CSV {index + 1}",
			filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
		)
		if path:
			self._analysis_csvs[index] = path
			self._analysis_labels[index].set(Path(path).name)
			self._saved_limits.clear()

	def _clear_csv(self, index: int) -> None:
		self._analysis_csvs[index] = None
		self._analysis_labels[index].set("未导入")
		self._saved_limits.clear()

	def _render_curves(self) -> None:
		from matplotlib.widgets import SpanSelector

		fig = self._analysis_figure

		# 保存当前视图范围
		if fig.axes and self._last_rendered_mode and self._last_rendered_mode != "dt-x":
			self._saved_limits[self._last_rendered_mode] = (
				fig.axes[0].get_xlim(), fig.axes[0].get_ylim()
			)

		fig.clear()
		ax = fig.add_subplot(111)
		colors = ["#2196F3", "#FF5722", "#4CAF50"]
		mode = self._chart_mode.get()
		show_corrected = self._show_corrected.get()
		smooth_str = self._smooth_strength.get()
		has_data = False

		all_x_data: list[list[float]] = [[], [], []]
		all_y_data: list[list[float]] = [[], [], []]
		all_times: list[list[float]] = [[], [], []]
		all_dists: list[list[float]] = [[], [], []]
		all_flags: list[list[int]] = [[], [], []]
		is_vt = (mode == "v-t")
		is_dtx = (mode == "dt-x")

		if is_dtx:
			if not self._analysis_csvs[0] or not self._analysis_csvs[1]:
				messagebox.showwarning("数据不足", "Δt-x 需要 CSV 1 和 CSV 2 均已导入。")
				return
			times1, dists1, speeds1, _ = parse_csv(self._analysis_csvs[0])
			times2, dists2, speeds2, _ = parse_csv(self._analysis_csvs[1])
			t2_interp = np.interp(dists1, dists2, times2)
			dt = np.array(times1) - t2_interp
			all_x_data[0] = dists1
			all_y_data[0] = dt.tolist()
			x_data = dists1
			y_data = dt.tolist()
			name1 = Path(self._analysis_csvs[0]).stem
			name2 = Path(self._analysis_csvs[1]).stem
			label = f"{name1} - {name2}"
			if smooth_str > 0:
				sx, sy = smooth_data(x_data, y_data, smooth_str)
				ax.plot(sx, sy, color=colors[0], linewidth=0.8)
			else:
				ax.plot(x_data, y_data, color=colors[0], linewidth=0.8)
			ax.plot([], [], color=colors[0], linewidth=0.8, label=label)
			has_data = True
		else:
			for i, csv_path in enumerate(self._analysis_csvs):
				if not csv_path:
					continue
				try:
					times, dists, speeds, flags = parse_csv(csv_path)
					name = Path(csv_path).stem
					all_times[i] = times
					all_dists[i] = dists
					if is_vt:
						x_data = times
						y_data = speeds
					else:
						x_data = dists
						y_data = speeds
					all_x_data[i] = x_data
					all_y_data[i] = y_data
					all_flags[i] = flags

					if show_corrected or smooth_str > 0:
						plot_segmented(ax, x_data, speeds, flags, colors[i], show_corrected, smooth_str)
					else:
						ax.plot(x_data, speeds, color=colors[i], linewidth=0.8)
					ax.plot([], [], color=colors[i], linewidth=0.8, label=name)
					has_data = True
				except Exception as e:
					messagebox.showerror("解析失败", f"{Path(csv_path).name}: {e}")
					return

		if not has_data:
			return

		if is_dtx:
			xlabel = "距离 (m)"
			ylabel = "Δt (s)"
			title = f"时间差-距离 ({name1} vs {name2})"
			delta_label_text = "Δ(Δt)"
		elif is_vt:
			xlabel = "时间 (s)"
			ylabel = "速度 (km/h)"
			title = "速度-时间曲线"
			delta_label_text = "行驶距离"
		else:
			xlabel = "距离 (m)"
			ylabel = "速度 (km/h)"
			title = "速度-距离曲线"
			delta_label_text = "用时"

		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		ax.legend(loc="upper right")
		ax.grid(True, alpha=0.3)

		if is_dtx:
			ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)

		delta_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
			va="top", fontsize=9, color="#333333",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

		def _on_select(xmin, xmax):
			if xmin > xmax:
				xmin, xmax = xmax, xmin
			results = []
			for i in range(3):
				xd = all_x_data[i]
				if not xd:
					continue
				name = Path(self._analysis_csvs[i]).stem if self._analysis_csvs[i] else ""
				total = 0.0
				if is_dtx:
					y_start = y_end = None
					for j, x in enumerate(xd):
						if y_start is None and x >= xmin:
							y_start = all_y_data[i][j]
						if x <= xmax:
							y_end = all_y_data[i][j]
					if y_start is not None and y_end is not None:
						total = y_end - y_start
				else:
					for j, x in enumerate(xd):
						if xmin <= x <= xmax:
							if is_vt:
								if j > 0:
									dt = xd[j] - xd[j - 1]
									avg_spd = (all_y_data[i][j] + all_y_data[i][j - 1]) / 2 / 3.6
									total += avg_spd * dt
							else:
								if j > 0:
									dx = xd[j] - xd[j - 1]
									avg_spd_mps = ((all_y_data[i][j] + all_y_data[i][j - 1]) / 2) / 3.6 if dx > 0 else 999
									total += dx / avg_spd_mps if avg_spd_mps > 0 else 0
				if is_dtx:
					sign = "+" if total >= 0 else ""
					results.append(f"{label}: {sign}{total:.2f}s")
				elif total > 0:
					unit = "m" if is_vt else "s"
					results.append(f"{name}: {total:.2f}{unit}")
			delta_text.set_text("\n".join(results) if results else "")

		if self._span_selector is not None:
			try:
				self._span_selector.disconnect_events()
			except Exception:
				pass
		self._span_selector = SpanSelector(ax, _on_select, "horizontal",
			props=dict(facecolor="#2196F3", alpha=0.15),
			interactive=True, drag_from_anywhere=True,
			button=1)
		delta_text.set_text(f"← 拖拽选择范围查看{delta_label_text}")

		# ── 滚轮缩放 + 右键拖动平移 ──
		_press_xy = [None, None]

		def _on_scroll(event):
			scale = 0.85 if event.button == "up" else 1.15
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			xmid = (xlim[0] + xlim[1]) / 2
			ymid = (ylim[0] + ylim[1]) / 2
			ax.set_xlim(xmid - (xmid - xlim[0]) * scale, xmid + (xlim[1] - xmid) * scale)
			ax.set_ylim(ymid - (ymid - ylim[0]) * scale, ymid + (ylim[1] - ymid) * scale)
			self._analysis_canvas.draw_idle()

		def _on_press(event):
			if event.button == 3:
				_press_xy[0], _press_xy[1] = event.xdata, event.ydata

		def _on_motion(event):
			if event.button == 3 and _press_xy[0] is not None and event.xdata is not None:
				dx = _press_xy[0] - event.xdata
				dy = _press_xy[1] - event.ydata
				xlim = ax.get_xlim()
				ylim = ax.get_ylim()
				ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
				ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
				self._analysis_canvas.draw_idle()

		fig.canvas.mpl_connect("scroll_event", _on_scroll)
		fig.canvas.mpl_connect("button_press_event", _on_press)
		fig.canvas.mpl_connect("motion_notify_event", _on_motion)

		fig.tight_layout()

		if not is_dtx:
			saved = self._saved_limits.get(mode)
			if saved is not None:
				ax.set_xlim(saved[0])
				ax.set_ylim(saved[1])

		self._analysis_canvas.draw()
		self._last_rendered_mode = mode

	def _auto_fit(self) -> None:
		fig = self._analysis_figure
		if not fig.axes:
			return
		mode = self._chart_mode.get()
		self._saved_limits.pop(mode, None)
		ax = fig.axes[0]
		ax.autoscale(enable=True, axis="both")
		ax.relim()
		ax.autoscale_view()
		self._analysis_canvas.draw_idle()

	def _export_png(self) -> None:
		if self._analysis_figure is None or not self._analysis_figure.axes:
			messagebox.showwarning("无数据", "请先渲染曲线。")
			return
		path = filedialog.asksaveasfilename(
			title="导出 PNG",
			defaultextension=".png",
			filetypes=[("PNG 图片", "*.png")],
		)
		if path:
			self._analysis_figure.savefig(path, dpi=150, bbox_inches="tight")
			messagebox.showinfo("导出完成", f"已保存: {path}")


# ═══════════════════════════════════════════════════════════════
# CLI: 无头分析导出
# ═══════════════════════════════════════════════════════════════

def run_analysis_headless(args) -> None:
	"""无头数据分析：从两个 CSV 导出 v-t、v-x、Δt-x 三张 PNG。

	Args:
		args: argparse.Namespace，需包含:
			- analysis: [csv1, csv2]
			- analysis_out: 输出前缀（可选）
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	csv1, csv2 = Path(args.analysis[0]), Path(args.analysis[1])
	if not csv1.exists() or not csv2.exists():
		print("错误: CSV 文件不存在")
		import sys
		sys.exit(1)

	out_prefix = Path(args.analysis_out) if args.analysis_out else csv1.parent / "分析"
	out_prefix.parent.mkdir(parents=True, exist_ok=True)

	t1, d1, s1, f1 = parse_csv(csv1)
	t2, d2, s2, f2 = parse_csv(csv2)
	name1, name2 = csv1.stem, csv2.stem

	# ── v-t ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, times, name, c in [(s1, t1, name1, "#2196F3"), (s2, t2, name2, "#FF5722")]:
		_, sy = smooth_data(times, data, 25)
		ax.plot(times, sy, color=c, linewidth=0.8, label=name)
	ax.set_xlabel("时间 (s)"); ax.set_ylabel("速度 (km/h)")
	ax.set_title("速度-时间曲线"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_v-t.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"v-t: {out_prefix}_v-t.png")

	# ── v-x ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, dists, name, c in [(s1, d1, name1, "#2196F3"), (s2, d2, name2, "#FF5722")]:
		_, sy = smooth_data(dists, data, 25)
		ax.plot(dists, sy, color=c, linewidth=0.8, label=name)
	ax.set_xlabel("距离 (m)"); ax.set_ylabel("速度 (km/h)")
	ax.set_title("速度-距离曲线"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_v-x.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"v-x: {out_prefix}_v-x.png")

	# ── Δt-x ──
	fig, ax = plt.subplots(figsize=(10, 6))
	t2_interp = np.interp(d1, d2, t2)
	dt = np.array(t1) - t2_interp
	_, sdt = smooth_data(d1, dt, 25)
	ax.plot(d1, sdt, color="#2196F3", linewidth=0.8, label=f"{name1} - {name2}")
	ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)
	ax.set_xlabel("距离 (m)"); ax.set_ylabel("Δt (s)")
	ax.set_title(f"时间差-距离 ({name1} vs {name2})"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_Δt-x.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Δt-x: {out_prefix}_Δt-x.png")

	print("分析完成。")
