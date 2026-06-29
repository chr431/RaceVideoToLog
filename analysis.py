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

from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
	# 裁剪起始零速帧，并将时间轴和距离轴归零
	start = 0
	for i, s in enumerate(speeds):
		if s > 0:
			start = i
			break
	if start > 0:
		times = times[start:]
		speeds = speeds[start:]
		dists = dists[start:]
		flags = flags[start:]
	# 始终归零时间轴和距离轴（不受 --frame-start 参数影响）
	if times:
		base_time = times[0]
		times = [t - base_time for t in times]
	if dists:
		base_dist = dists[0]
		dists = [d - base_dist for d in dists]
	return times, dists, speeds, flags


def smooth_data(xv: "np.ndarray | list[float]", yv: "np.ndarray | list[float]", strength: int) -> tuple[np.ndarray, np.ndarray]:
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


def plot_segmented(ax: "Axes", x: "np.ndarray | list[float]", y: "np.ndarray | list[float]", flags: "list[int]", normal_color: str, show_red: bool,
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
	x_arr = np.asarray(x); y_arr = np.asarray(y)
	_x = x_arr.tolist()
	_y = y_arr.tolist()

	# 红色段（flag=1 自动纠错）
	rx, ry = [], []
	i = 0
	while i < n_orig:
		if flags[i] == 1:
			# 找到连续 flag=1 段
			j = i
			while j < n_orig and flags[j] == 1:
				j += 1
			run_len = j - i
			# 映射到平滑后的索引：覆盖 run_len+1 个数据点（段头尾各延半帧）
			si = int(max(0, i - 0.5) * n_smooth / n_orig)
			ei = int(min(n_orig, j + 0.5) * n_smooth / n_orig)
			si = max(0, min(si, n_smooth - 2))
			ei = min(n_smooth, max(ei, si + 1))
			rx.extend(_x[si:ei] + [float('nan')])
			ry.extend(_y[si:ei] + [float('nan')])
			i = j + 1
		else:
			i += 1
	if rx:
		ax.plot(rx, ry, color=red, linewidth=2.0)

	# 绿色段（flag>=2 人工纠错）
	gx, gy = [], []
	i = 0
	while i < n_orig:
		if flags[i] >= 2:
			j = i
			while j < n_orig and flags[j] >= 2:
				j += 1
			si = int(max(0, i - 0.5) * n_smooth / n_orig)
			ei = int(min(n_orig, j + 0.5) * n_smooth / n_orig)
			si = max(0, min(si, n_smooth - 2))
			ei = min(n_smooth, max(ei, si + 1))
			gx.extend(_x[si:ei] + [float('nan')])
			gy.extend(_y[si:ei] + [float('nan')])
			i = j + 1
		else:
			i += 1
	if gx:
		ax.plot(gx, gy, color=green, linewidth=1.5, alpha=0.8)



# ═══════════════════════════════════════════════════════════════
# GUI: 数据分析 Tab
# ═══════════════════════════════════════════════════════════════
# 已迁移至 gui_analysis.py (PySide6)
# 从 gui_analysis import AnalysisTab

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
