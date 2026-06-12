"""RaceVideoToLog — 赛车视频速度 OCR 提取工具。

从车载视频中实时 OCR 识别速度数字，支持 GPU (CUDA) / CPU 两种后端，
输出时间-速度-距离 CSV 文件。

用法:
  python RaceVideoToLog.py                          # GUI 模式
  python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2  # CLI 模式
"""
from __future__ import annotations

import argparse
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from queue import Queue
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ocr_engine
from ocr_engine import *

class RaceVideoToLogApp:
	def __init__(self) -> None:
		self.root = tk.Tk()
		self.root.title("Race Video To Log")
		self.root.geometry("1180x860")
		self.root.minsize(980, 720)

		self.video_path: Path | None = None
		self.metadata: VideoMetadata | None = None
		self.first_frame_bgr: np.ndarray | None = None
		self.preview_photo: ImageTk.PhotoImage | None = None
		self.preview_after_id: str | None = None
		self.ocr_engine: RapidOCR | None = None
		self.ocr_engines: list[RapidOCR] = []

		self.file_var = tk.StringVar(value="未导入视频")
		self.duration_var = tk.StringVar(value="-")
		self.resolution_var = tk.StringVar(value="-")
		self.fps_var = tk.StringVar(value="-")
		self.codec_var = tk.StringVar(value="-")
		self.status_var = tk.StringVar(value="请选择视频并设置识别范围。")

		self.left_x_var = tk.StringVar()
		self.left_y_var = tk.StringVar()
		self.right_x_var = tk.StringVar()
		self.right_y_var = tk.StringVar()
		self.speed_format_var = tk.StringVar(value="km/h")
		self.max_speed_var = tk.StringVar(value="400")
		self.max_accel_var = tk.StringVar(value="50")
		self.frame_div_var = tk.StringVar(value="2")
		self.target_height_var = tk.StringVar(value="24")
		self.pad_var = tk.StringVar(value="0")
		self.num_workers_var = tk.StringVar(value="4")
		self.backend_var = tk.StringVar(value="auto")
		self._ocr_model_var = tk.StringVar(value="v5 首选(推荐)")  # OCR 模型版本

		# 时间轴范围
		self._frame_start_var = tk.StringVar(value="")
		self._frame_end_var = tk.StringVar(value="")
		self._manual_correction_var = tk.BooleanVar(value=False)  # 人工纠错

		self.is_exporting = False
		self._cancel_flag = False
		self.progress_var = tk.DoubleVar(value=0.0)

		self.first_frame_pil: Image.Image | None = None
		self._preview_scale = 1.0
		self._preview_offset_x = 0.0
		self._preview_offset_y = 0.0
		self._last_canvas_w = 0
		self._last_canvas_h = 0

		self._drag_start_x: int | None = None
		self._drag_start_y: int | None = None
		self._preview_frame_pos = tk.DoubleVar(value=0)  # 预览帧位置

		# 数据分析 tab
		self._analysis_csvs: list[str | None] = [None, None, None]  # 最多 3 个 CSV
		self._analysis_labels: list[tk.StringVar] = []
		self._analysis_figure: Figure | None = None
		self._analysis_canvas: FigureCanvasTkAgg | None = None
		self._chart_mode = tk.StringVar(value="v-x")
		self._show_corrected = tk.BooleanVar(value=False)
		self._saved_limits: dict[str, tuple | None] = {}  # 按模式保存视图范围
		self._last_rendered_mode: str | None = None  # 上次实际渲染的模式
		self._smooth_strength = tk.IntVar(value=25)
		self._smooth_entry_var = tk.StringVar(value="25")  # 平滑输入框同步
		self._span_selector = None  # matplotlib SpanSelector

		self._build_ui()
		self._bind_preview_updates()

	def _build_ui(self) -> None:
		self.root.columnconfigure(0, weight=1)
		self.root.rowconfigure(0, weight=1)

		# Notebook 占满主区域
		self._notebook = ttk.Notebook(self.root)
		self._notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 10))

		# ── Tab 1: OCR 处理 ──
		tab_ocr = ttk.Frame(self._notebook)
		self._notebook.add(tab_ocr, text="OCR 处理")
		tab_ocr.columnconfigure(0, weight=1)
		tab_ocr.rowconfigure(2, weight=1)  # 主内容区可拉伸

		# OCR Header
		header = ttk.Frame(tab_ocr, padding=(12, 6, 12, 6))
		header.grid(row=0, column=0, sticky="ew")
		header.columnconfigure(1, weight=1)

		ttk.Button(header, text="导入视频", command=self.import_video).grid(row=0, column=0, sticky="w")
		self.export_btn = ttk.Button(header, text="导出 CSV", command=self.export_csv)
		self.export_btn.grid(row=0, column=1, sticky="e")
		self.cancel_btn = ttk.Button(header, text="取消", command=self._cancel_export, state="disabled")
		self.cancel_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))
		ttk.Label(header, textvariable=self.file_var).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

		# OCR 视频信息
		info = ttk.LabelFrame(tab_ocr, text="视频信息", padding=(12, 10, 12, 12))
		info.grid(row=1, column=0, sticky="ew", pady=(0, 10))
		for index in range(4):
			info.columnconfigure(index, weight=1)
		self._add_info_row(info, 0, "时长", self.duration_var)
		self._add_info_row(info, 1, "分辨率", self.resolution_var)
		self._add_info_row(info, 2, "帧率", self.fps_var)
		self._add_info_row(info, 3, "编码", self.codec_var)

		# OCR 主内容
		ocr_main = ttk.Frame(tab_ocr)
		ocr_main.grid(row=2, column=0, sticky="nsew")
		ocr_main.columnconfigure(1, weight=3)
		ocr_main.columnconfigure(0, weight=1)
		ocr_main.rowconfigure(0, weight=1)

		config_col = ttk.Frame(ocr_main, padding=(0, 0, 6, 0))
		config_col.grid(row=0, column=0, sticky="nsew")
		config_col.columnconfigure(0, weight=1)

		range_box = ttk.LabelFrame(config_col, text="识别范围（像素）", padding=(12, 10, 12, 12))
		range_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
		for index in range(4): range_box.columnconfigure(index, weight=1)
		self._add_range_entry(range_box, 0, 0, "左上 X", self.left_x_var)
		self._add_range_entry(range_box, 0, 2, "右下 X", self.right_x_var)
		self._add_range_entry(range_box, 1, 0, "左上 Y", self.left_y_var)
		self._add_range_entry(range_box, 1, 2, "右下 Y", self.right_y_var)

		format_box = ttk.LabelFrame(config_col, text="速度格式", padding=(12, 10, 12, 12))
		format_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
		ttk.Radiobutton(format_box, text="m/s", value="m/s", variable=self.speed_format_var).grid(row=0, column=0, sticky="w")
		ttk.Radiobutton(format_box, text="km/h", value="km/h", variable=self.speed_format_var).grid(row=0, column=1, sticky="w", padx=(20, 0))
		ttk.Radiobutton(format_box, text="mile/h", value="mile/h", variable=self.speed_format_var).grid(row=0, column=2, sticky="w", padx=(20, 0))
		ttk.Label(format_box, text="输出统一转换为 km/h。", foreground="#555555").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

		constraint_box = ttk.LabelFrame(format_box, text="物理约束纠错", padding=(10, 8, 10, 10))
		constraint_box.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
		constraint_box.columnconfigure(1, weight=1); constraint_box.columnconfigure(3, weight=1)
		ttk.Label(constraint_box, text="最大速度 (km/h)").grid(row=0, column=0, sticky="w")
		ttk.Entry(constraint_box, textvariable=self.max_speed_var, width=10).grid(row=0, column=1, sticky="ew", padx=(6, 14))
		ttk.Label(constraint_box, text="最大加速度 (m/s²)").grid(row=0, column=2, sticky="w")
		ttk.Entry(constraint_box, textvariable=self.max_accel_var, width=10).grid(row=0, column=3, sticky="ew", padx=(6, 0))
		ttk.Label(constraint_box, text="设为 0 则不限制。用于自动修正丢位、多位和跳变异常。", foreground="#555555").grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
		ttk.Checkbutton(constraint_box, text="人工纠错（识别结束后手动修正）", variable=self._manual_correction_var).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

		perf_box = ttk.LabelFrame(config_col, text="性能", padding=(12, 10, 12, 12))
		perf_box.grid(row=2, column=0, sticky="ew")
		perf_box.columnconfigure(1, weight=1); perf_box.columnconfigure(3, weight=1); perf_box.columnconfigure(5, weight=1)

		ttk.Label(perf_box, text="采样间隔").grid(row=0, column=0, sticky="w")
		self.frame_div_combo = ttk.Combobox(perf_box, textvariable=self.frame_div_var, values=["1","2","3","4","5"], width=6, state="readonly")
		self.frame_div_combo.grid(row=0, column=1, sticky="ew", padx=(6, 2))
		ttk.Label(perf_box, text="1/N 采集", foreground="#555555").grid(row=0, column=2, sticky="w")

		ttk.Label(perf_box, text="OCR 后端").grid(row=0, column=3, sticky="w", padx=(20,0))
		_BL = {"auto": "自动", "cuda": "CUDA", "cpu": "CPU"}
		self.backend_combo = ttk.Combobox(perf_box, textvariable=self.backend_var, values=[_BL[k] for k in ["auto","cuda","cpu"]], width=10, state="readonly")
		self.backend_combo.grid(row=0, column=4, sticky="ew", padx=(6, 2))
		self.backend_combo.bind("<<ComboboxSelected>>", self._on_backend_changed)

		ttk.Label(perf_box, text="模型").grid(row=0, column=5, sticky="w", padx=(12,0))
		_MODELS = {"v3": "v3 备选", "v5_mobile": "v5 首选(推荐)"}
		self._model_combo = ttk.Combobox(perf_box, textvariable=self._ocr_model_var, values=[_MODELS[k] for k in ["v3","v5_mobile"]], width=11, state="readonly")
		self._model_combo.grid(row=0, column=6, sticky="ew", padx=(6, 2))

		ttk.Label(perf_box, text="OCR 高度 (px)").grid(row=1, column=0, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.target_height_var, width=8).grid(row=1, column=1, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text="边缘填充 (px)").grid(row=1, column=2, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.pad_var, width=8).grid(row=1, column=3, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text="并行线程数").grid(row=1, column=4, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.num_workers_var, width=8).grid(row=1, column=5, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text=">1 时启用并行推理。", foreground="#555555").grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

		# 时间轴范围
		time_box = ttk.LabelFrame(config_col, text="时间轴范围", padding=(12, 10, 12, 12))
		time_box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
		time_box.columnconfigure(1, weight=1); time_box.columnconfigure(4, weight=1)
		ttk.Label(time_box, text="起始帧").grid(row=0, column=0, sticky="w")
		ttk.Entry(time_box, textvariable=self._frame_start_var, width=8).grid(row=0, column=1, sticky="ew", padx=(4, 4))
		ttk.Button(time_box, text="设为当前", command=lambda: self._frame_start_var.set(str(int(self._preview_slider.get())))).grid(row=0, column=2, padx=(0, 8))
		ttk.Label(time_box, text="结束帧").grid(row=0, column=3, sticky="w")
		ttk.Entry(time_box, textvariable=self._frame_end_var, width=8).grid(row=0, column=4, sticky="ew", padx=(4, 4))
		ttk.Button(time_box, text="设为当前", command=lambda: self._frame_end_var.set(str(int(self._preview_slider.get())))).grid(row=0, column=5)
		ttk.Label(time_box, text="留空=全部。仅处理 [起始, 结束) 之间的帧。", foreground="#555555").grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

		# 右侧预览
		preview_box = ttk.LabelFrame(ocr_main, text="识别范围预览", padding=(6, 6, 6, 6))
		preview_box.grid(row=0, column=1, sticky="nsew")
		preview_box.columnconfigure(0, weight=1); preview_box.rowconfigure(0, weight=1)

		self.preview_canvas = tk.Canvas(preview_box, background="#151515", highlightthickness=0, cursor="crosshair")
		self.preview_canvas.grid(row=0, column=0, sticky="nsew")
		self.preview_canvas.bind("<Configure>", lambda event: self.schedule_preview_refresh())
		self.preview_canvas.bind("<ButtonPress-1>", self._on_drag_start)
		self.preview_canvas.bind("<B1-Motion>", self._on_drag_motion)
		self.preview_canvas.bind("<ButtonRelease-1>", self._on_drag_end)
		self.preview_canvas.bind("<ButtonPress-3>", lambda e: None)  # 右键保留

		# 视频帧位置滑动条 + 刷新按钮
		slider_row = ttk.Frame(preview_box)
		slider_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
		slider_row.columnconfigure(0, weight=1)
		self._preview_slider = ttk.Scale(slider_row, from_=0, to=1, variable=self._preview_frame_pos,
			orient="horizontal")
		self._preview_slider.grid(row=0, column=0, sticky="ew")
		ttk.Button(slider_row, text="刷新预览", command=self.refresh_preview).grid(row=0, column=1, padx=(8, 0))

		# 预览画布右键：重置视图
		# ── Tab 2: 数据分析 ──
		self._build_analysis_tab()

		# Row 1: 底部状态栏（OCR 处理 tab 使用，数据分析 tab 隐藏）
		self._footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
		self._footer.grid(row=1, column=0, sticky="ew")
		self._footer.columnconfigure(0, weight=1)
		ttk.Label(self._footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
		self.progress_bar = ttk.Progressbar(self._footer, variable=self.progress_var, maximum=100.0)
		self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

	def _add_info_row(self, parent: ttk.LabelFrame, column: int, title: str, variable: tk.StringVar) -> None:
		cell = ttk.Frame(parent)
		cell.grid(row=0, column=column, sticky="ew", padx=6)
		ttk.Label(cell, text=title).grid(row=0, column=0, sticky="w")
		ttk.Label(cell, textvariable=variable).grid(row=1, column=0, sticky="w", pady=(4, 0))

	def _add_range_entry(self, parent: ttk.LabelFrame, row: int, column: int, label: str, variable: tk.StringVar) -> None:
		cell = ttk.Frame(parent)
		cell.grid(row=row, column=column, columnspan=2, sticky="ew", padx=4, pady=4)
		ttk.Label(cell, text=label).grid(row=0, column=0, sticky="w")
		ttk.Entry(cell, textvariable=variable, width=10).grid(row=1, column=0, sticky="ew", pady=(4, 0))

	# ═══════════════════ 数据分析 Tab ═══════════════════
	def _build_analysis_tab(self) -> None:
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
			btn = ttk.Button(slot, text="导入", command=lambda idx=i: self._import_csv(idx))
			btn.grid(row=0, column=0, sticky="w")
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
		self._analysis_figure = Figure(figsize=(8, 5), dpi=100)
		self._analysis_canvas = FigureCanvasTkAgg(self._analysis_figure, master=tab)
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
			self._saved_limits.clear()  # 数据已变，清空视图缓存

	def _clear_csv(self, index: int) -> None:
		"""清除已导入的 CSV。"""
		self._analysis_csvs[index] = None
		self._analysis_labels[index].set("未导入")
		self._saved_limits.clear()

	def _render_curves(self) -> None:
		from matplotlib.widgets import SpanSelector
		fig = self._analysis_figure

		# 保存当前视图范围（按上次实际渲染的模式记忆；Δt-x 不缓存）
		if fig.axes and self._last_rendered_mode and self._last_rendered_mode != "dt-x":
			self._saved_limits[self._last_rendered_mode] = (
				fig.axes[0].get_xlim(), fig.axes[0].get_ylim()
			)

		fig.clear()
		ax = fig.add_subplot(111)
		colors = ["#2196F3", "#FF5722", "#4CAF50"]
		mode = self._chart_mode.get()
		show_corrected = self._show_corrected.get()
		has_data = False

		# 存储所有数据用于范围选择计算
		all_x_data: list[list[float]] = [[], [], []]
		all_y_data: list[list[float]] = [[], [], []]
		all_times: list[list[float]] = [[], [], []]
		all_dists: list[list[float]] = [[], [], []]
		all_flags: list[list[int]] = [[], [], []]
		is_vt = (mode == "v-t")
		is_dtx = (mode == "dt-x")

		if is_dtx:
			# ── Δt-x 模式：仅用 CSV1/CSV2，无视 CSV3 ──
			if not self._analysis_csvs[0] or not self._analysis_csvs[1]:
				messagebox.showwarning("数据不足", "Δt-x 需要 CSV 1 和 CSV 2 均已导入。")
				return
			times1, dists1, speeds1, _ = self._parse_csv(self._analysis_csvs[0])
			times2, dists2, speeds2, _ = self._parse_csv(self._analysis_csvs[1])
			# 以 CSV1 距离为基准，插值 CSV2 时间
			t2_interp = np.interp(dists1, dists2, times2)
			dt = np.array(times1) - t2_interp  # 正数=CSV1更慢
			all_x_data[0] = dists1
			all_y_data[0] = dt.tolist()
			x_data = dists1
			y_data = dt.tolist()
			name1 = Path(self._analysis_csvs[0]).stem
			name2 = Path(self._analysis_csvs[1]).stem
			label = f"{name1} - {name2}"
			if self._smooth_strength.get() > 0:
				sx, sy = self._smooth_data(x_data, y_data, self._smooth_strength.get())
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
					times, dists, speeds, flags = self._parse_csv(csv_path)
					name = Path(csv_path).stem
					all_times[i] = times
					all_dists[i] = dists
					if is_vt:
						x_data = times
						y_data = speeds
					else:  # v-x
						x_data = dists
						y_data = speeds
					all_x_data[i] = x_data
					all_y_data[i] = y_data
					all_flags[i] = flags

					if show_corrected or self._smooth_strength.get() > 0:
						self._plot_segmented(ax, x_data, speeds, flags, colors[i], show_corrected)
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

		# Δt-x 模式：绘制 y=0 参考线
		if is_dtx:
			ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)

		# 范围选择器统计文本
		delta_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
			va="top", fontsize=9, color="#333333",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

		# 跨 CSV 的范围选择器
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
					# Δt-x: 显示 Δt 在该范围内的变化量
					y_start = y_end = None
					for j, x in enumerate(xd):
						if y_start is None and x >= xmin:
							y_start = all_y_data[i][j]
						if x <= xmax:
							y_end = all_y_data[i][j]
					if y_start is not None and y_end is not None:
						total = y_end - y_start  # Δ(Δt)
				else:
					for j, x in enumerate(xd):
						if xmin <= x <= xmax:
							if is_vt:
								# v-t: 累计行驶距离 = Σ(速度 × 时间间隔)
								if j > 0:
									dt = xd[j] - xd[j - 1]
									avg_spd = (all_y_data[i][j] + all_y_data[i][j - 1]) / 2 / 3.6
									total += avg_spd * dt
							else:
								# v-x: 累计用时
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

		# 移除旧的选择器
		if self._span_selector is not None:
			try:
				self._span_selector.disconnect_events()
			except Exception:
				pass
		self._span_selector = SpanSelector(ax, _on_select, "horizontal",
			props=dict(facecolor="#2196F3", alpha=0.15),
			interactive=True, drag_from_anywhere=True,
			button=1)  # 仅左键触发
		delta_text.set_text(f"← 拖拽选择范围查看{delta_label_text}")

		# ── 滚轮缩放 + 右键拖动平移 ──
		_press_xy = [None, None]  # 右键拖动起始点

		def _on_scroll(event):
			scale = 0.85 if event.button == "up" else 1.15
			xlim = ax.get_xlim(); ylim = ax.get_ylim()
			xmid = (xlim[0] + xlim[1]) / 2; ymid = (ylim[0] + ylim[1]) / 2
			dx = (xlim[1] - xlim[0]) * (1 - scale) / 2
			dy = (ylim[1] - ylim[0]) * (1 - scale) / 2
			ax.set_xlim(xmid - (xmid - xlim[0]) * scale, xmid + (xlim[1] - xmid) * scale)
			ax.set_ylim(ymid - (ymid - ylim[0]) * scale, ymid + (ylim[1] - ymid) * scale)
			self._analysis_canvas.draw_idle()

		def _on_press(event):
			if event.button == 3:  # 右键
				_press_xy[0], _press_xy[1] = event.xdata, event.ydata

		def _on_motion(event):
			if event.button == 3 and _press_xy[0] is not None and event.xdata is not None:
				dx = _press_xy[0] - event.xdata; dy = _press_xy[1] - event.ydata
				xlim = ax.get_xlim(); ylim = ax.get_ylim()
				ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
				ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
				self._analysis_canvas.draw_idle()

		fig.canvas.mpl_connect("scroll_event", _on_scroll)
		fig.canvas.mpl_connect("button_press_event", _on_press)
		fig.canvas.mpl_connect("motion_notify_event", _on_motion)

		fig.tight_layout()

		# 恢复当前模式的上次视图范围（Δt-x 不缓存）
		if not is_dtx:
			saved = self._saved_limits.get(mode)
			if saved is not None:
				ax.set_xlim(saved[0])
				ax.set_ylim(saved[1])

		self._analysis_canvas.draw()
		self._last_rendered_mode = mode

	def _auto_fit(self) -> None:
		"""重置图表缩放和位置到默认状态。"""
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

	def _smooth_data(self, xv, yv, strength):
		"""Savitzky-Golay 滤波（纯 numpy 实现）：多项式滑动窗口拟合，保留峰谷形状。"""
		if strength <= 0 or len(xv) < 5:
			return np.array(xv), np.array(yv, dtype=float)
		win = int(len(xv) * strength / 100.0 * 0.0175)
		win = max(5, min(win, len(xv) - 2))
		if win % 2 == 0:
			win += 1
		polyorder = min(3, win - 1)
		sy = _savgol_filter_np(np.array(yv, dtype=float), win, polyorder)
		return np.array(xv, dtype=float), sy

	def _plot_segmented(self, ax, x, y, flags, normal_color, show_red):
		"""平滑 + 纠错段着色。show_red 控制是否绘制红色段。"""
		red = "#F44336"
		strength = self._smooth_strength.get()

		if strength > 0:
			x, y = self._smooth_data(x, y, strength)

		ax.plot(x, y, color=normal_color, linewidth=0.8)

		if not show_red or not flags or not any(f >= 1 for f in flags):
			return

		n_orig = len(flags)
		n_smooth = len(x)
		# 统一为 list（平滑后是 ndarray，不滑时是 list）
		_x = x.tolist() if hasattr(x, 'tolist') else list(x)
		_y = y.tolist() if hasattr(y, 'tolist') else list(y)
		rx, ry = [], []
		for i in range(n_orig - 1):
			if flags[i] >= 1 and flags[i + 1] >= 1:
				si = int(i * n_smooth / n_orig)
				ei = int((i + 2) * n_smooth / n_orig)
				si = min(si, n_smooth - 2)
				ei = min(ei, n_smooth)
				if ei > si:
					rx.extend(_x[si:ei] + [float('nan')])
					ry.extend(_y[si:ei] + [float('nan')])
		if rx:
			ax.plot(rx, ry, color=red, linewidth=1.2)

	def _parse_csv(self, path: str) -> tuple[list[float], list[float], list[float], list[int]]:
		times, dists, speeds, flags = [], [], [], []
		with open(path, "r", encoding="utf-8-sig") as f:
			for line in f:
				parts = line.strip().split(",")
				if len(parts) >= 3:
					times.append(float(parts[0]))
					dists.append(float(parts[1]))
					speeds.append(float(parts[2]))
					flags.append(int(parts[3]) if len(parts) > 3 else 0)
		# 去除开头的静止帧（speed=0 且 distance=0），从第一个有效速度开始
		start = 0
		for i, s in enumerate(speeds):
			if s > 0:
				start = i
				break
		if start > 0:
			times = times[start:]
			speeds = speeds[start:]
			# 距离归一化：从第一帧有效速度处开始重新计算
			base_dist = dists[start]
			dists = [d - base_dist for d in dists[start:]]
			flags = flags[start:]
			# 时间也归零
			base_time = times[0]
			times = [t - base_time for t in times]
		return times, dists, speeds, flags

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

	def _parse_positive_float(self, value: str, field_name: str, allow_zero: bool = False) -> float:
		parsed = safe_float(value)
		if parsed is None:
			raise ValueError(f"{field_name} 不是有效数字。")
		if parsed < 0 or (not allow_zero and parsed == 0):
			raise ValueError(f"{field_name} 必须{'≥' if allow_zero else '>'} 0。")
		return parsed

	def _bind_preview_updates(self) -> None:
		for variable in (self.left_x_var, self.left_y_var, self.right_x_var, self.right_y_var):
			variable.trace_add("write", lambda *args: self._update_roi_rect())

	def _canvas_to_video_coords(self, cx: float, cy: float) -> tuple[int, int]:
		if not self.metadata or self._preview_scale <= 0:
			return 0, 0
		x = (cx - self._preview_offset_x) / self._preview_scale
		y = (cy - self._preview_offset_y) / self._preview_scale
		x = max(0, min(self.metadata.width - 1, int(x)))
		y = max(0, min(self.metadata.height - 1, int(y)))
		return x, y

	def _on_drag_start(self, event: tk.Event) -> None:
		if not self.metadata:
			return
		x, y = self._canvas_to_video_coords(event.x, event.y)
		self._drag_start_x = x
		self._drag_start_y = y
		self.left_x_var.set(str(x))
		self.left_y_var.set(str(y))
		self.right_x_var.set(str(x))
		self.right_y_var.set(str(y))

	def _on_drag_motion(self, event: tk.Event) -> None:
		if not self.metadata or self._drag_start_x is None or self._drag_start_y is None:
			return
		x, y = self._canvas_to_video_coords(event.x, event.y)
		x1 = min(self._drag_start_x, x)
		y1 = min(self._drag_start_y, y)
		x2 = max(self._drag_start_x, x)
		y2 = max(self._drag_start_y, y)
		self.left_x_var.set(str(x1))
		self.left_y_var.set(str(y1))
		self.right_x_var.set(str(x2))
		self.right_y_var.set(str(y2))

	def _on_drag_end(self, event: tk.Event) -> None:
		self._drag_start_x = None
		self._drag_start_y = None

	def _get_preview_frame(self) -> np.ndarray | None:
		"""根据滑动条位置读取对应视频帧。"""
		if self.video_path is None or self.metadata is None:
			return None
		pos = int(self._preview_slider.get())
		fi = int(pos)
		cap = cv2.VideoCapture(str(self.video_path))
		cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
		ok, frame = cap.read()
		cap.release()
		return frame if ok else None

	def _on_preview_slider(self, *args) -> None:
		"""滑动条变化回调（由刷新按钮触发预览，此处不做操作以保持拖拽流畅）。"""
		pass

	def schedule_preview_refresh(self) -> None:
		if self.preview_after_id is not None:
			self.root.after_cancel(self.preview_after_id)
		self.preview_after_id = self.root.after(200, self.refresh_preview)

	def load_video(self, path: Path) -> None:
		capture = cv2.VideoCapture(str(path))
		if not capture.isOpened():
			raise RuntimeError("无法打开视频文件。")

		frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		fourcc = capture.get(cv2.CAP_PROP_FOURCC) or 0.0
		duration_sec = (frame_count / fps) if fps > 0 else 0.0

		ok, frame = capture.read()
		capture.release()
		if not ok or frame is None:
			raise RuntimeError("无法读取视频第一帧。")

		self.video_path = path
		self.metadata = VideoMetadata(
			path=path,
			duration_sec=duration_sec,
			width=width,
			height=height,
			fps=fps,
			codec=codec_from_fourcc(fourcc),
			frame_count=frame_count,
		)
		self.first_frame_bgr = frame
		frame_rgb = cv2.cvtColor(self.first_frame_bgr, cv2.COLOR_BGR2RGB)
		self.first_frame_pil = Image.fromarray(frame_rgb)
		self._last_canvas_w = 0
		self._last_canvas_h = 0

		self.file_var.set(str(path))
		self.duration_var.set(format_duration(duration_sec))
		self.resolution_var.set(f"{width} x {height}")
		self.fps_var.set(f"{fps:.3f}" if fps > 0 else "Unknown")
		self.codec_var.set(self.metadata.codec)
		self.status_var.set("视频已载入，请输入识别范围并预览。")
		self._preview_slider.configure(to=frame_count - 1)
		self._preview_frame_pos.set(0)
		self.schedule_preview_refresh()

	def import_video(self) -> None:
		file_path = filedialog.askopenfilename(
			title="选择需要处理的视频",
			filetypes=[
				("视频文件", "*.mp4 *.mkv *.avi *.mov *.m4v *.wmv *.flv *.webm"),
				("所有文件", "*.*"),
			],
		)
		if not file_path:
			return

		try:
			self.load_video(Path(file_path))
		except Exception as exc:
			messagebox.showerror("导入失败", str(exc))
			self.status_var.set("导入失败，请检查视频文件是否可读。")

	def get_region(self) -> tuple[int, int, int, int] | None:
		if not self.metadata:
			return None

		x1 = safe_int(self.left_x_var.get())
		y1 = safe_int(self.left_y_var.get())
		x2 = safe_int(self.right_x_var.get())
		y2 = safe_int(self.right_y_var.get())
		if None in (x1, y1, x2, y2):
			return None
		return clamp_region(x1, y1, x2, y2, self.metadata.width, self.metadata.height)

	def refresh_preview(self) -> None:
		self.preview_after_id = None
		if self.first_frame_pil is None:
			self.preview_canvas.delete("all")
			return

		# 从滑动条直接读取位置
		pos = int(self._preview_slider.get())
		if pos > 0 and self.video_path is not None:
			cap = cv2.VideoCapture(str(self.video_path))
			cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
			ok, frame = cap.read()
			cap.release()
			if ok and frame is not None:
				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				self._draw_preview_image(Image.fromarray(frame_rgb))
				self.status_var.set(f"预览帧 #{pos}")
			else:
				self._draw_preview_image(self.first_frame_pil)
				self.status_var.set(f"无法读取帧 #{pos}")
		else:
			self._draw_preview_image(self.first_frame_pil)
			self.status_var.set("预览帧 #0（首帧）")
		self._update_roi_rect()

	def _update_roi_rect(self) -> None:
		self.preview_canvas.delete("roi_rect")
		region = self.get_region()
		if region is not None and getattr(self, "_preview_scale", 0.0) > 0:
			x1, y1, x2, y2 = region
			cx1 = x1 * self._preview_scale + self._preview_offset_x
			cy1 = y1 * self._preview_scale + self._preview_offset_y
			cx2 = x2 * self._preview_scale + self._preview_offset_x
			cy2 = y2 * self._preview_scale + self._preview_offset_y
			self.preview_canvas.create_rectangle(
				cx1, cy1, cx2, cy2, 
				outline="#ff5050", width=max(2, int(self._preview_scale * 2)), tag="roi_rect"
			)

	def _draw_preview_image(self, image: Image.Image) -> None:
		canvas_width = max(1, self.preview_canvas.winfo_width())
		canvas_height = max(1, self.preview_canvas.winfo_height())

		scale = min(canvas_width / image.width, canvas_height / image.height)
		if scale <= 0:
			scale = 1.0

		display_width = max(1, int(image.width * scale))
		display_height = max(1, int(image.height * scale))

		self._last_canvas_w = canvas_width
		self._last_canvas_h = canvas_height
		self._preview_scale = scale
		self._preview_offset_x = (canvas_width - display_width) / 2.0
		self._preview_offset_y = (canvas_height - display_height) / 2.0

		display_size = (display_width, display_height)
		resized = image.resize(display_size, Image.Resampling.LANCZOS)
		self.preview_photo = ImageTk.PhotoImage(resized)

		self.preview_canvas.delete("video_frame")
		self.preview_canvas.create_image(canvas_width / 2, canvas_height / 2, image=self.preview_photo, tag="video_frame")
		self.preview_canvas.tag_lower("video_frame")

	def _on_backend_changed(self, event: tk.Event | None = None) -> None:
		"""用户切换 OCR 后端时，重置引擎缓存并在状态栏提示。"""
		_reset_backend()
		self._release_ocr_engines()
		BACKEND_LABELS_REV = {"自动": "auto", "CUDA": "cuda", "CPU": "cpu"}
		selected_label = self.backend_var.get()
		selected_key = BACKEND_LABELS_REV.get(selected_label, "auto")
		actual = _select_backend(selected_key)
		status_map = {"CUDA": "CUDA (GPU)", "CPU": "CPU"}

		if selected_key == "cuda" and actual != "CUDA":
			_hint = ("请确认已安装 CUDA Toolkit 12.x 和 cuDNN 9.x，\n"
			         "并位于默认路径：\n"
			         "  C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin\n"
			         "  C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\...")
			self.root.after(100, lambda h=_hint: messagebox.showwarning(
				"后端不可用",
				f"{selected_label} 不可用。\n已自动回退为 {status_map.get(actual, actual)}。\n\n{h}"
			))

		self.status_var.set(f"OCR 后端: {status_map.get(actual, actual)}（选择: {selected_label}）")

	def _create_ocr_engine(self) -> RapidOCR:
		self._check_cancel()
		_reset_backend()
		BACKEND_LABELS_REV = {"自动": "auto", "CUDA": "cuda", "CPU": "cpu"}
		selected_label = self.backend_var.get()
		selected_key = BACKEND_LABELS_REV.get(selected_label, "auto")
		actual = _select_backend(selected_key)
		MODEL_REV = {"v3 备选": "v3", "v5 首选(推荐)": "v5_mobile"}
		model_key = MODEL_REV.get(self._ocr_model_var.get(), "v5_mobile")
		print(f"[OCR] 后端: {actual}, 模型: {model_key}", flush=True)
		kwargs = _get_model_kwargs(model_key)
		if kwargs is None and model_key != "v3":
			print(f"[OCR] 警告: {model_key} 模型文件不存在，回退到默认 v3")
		self._check_cancel()
		return RapidOCR(**(kwargs or {}))

	def get_ocr_engines(self, count: int) -> list[RapidOCR]:
		"""预创建 N 个 OCR 引擎用于 CUDA 并行推理。"""
		while len(self.ocr_engines) < count:
			self.ocr_engines.append(self._create_ocr_engine())
		return self.ocr_engines[:count]

	def get_ocr_engine(self) -> RapidOCR:
		if self.ocr_engine is None:
			self.ocr_engine = self._create_ocr_engine()
		return self.ocr_engine

	def preprocess_crop(self, crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		"""预处理：灰度化 + 缩放到 target_h（PP-OCR 内置归一化，无需 CLAHE/OTSU）。"""
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		h, w = gray.shape[:2]
		target_h = max(8.0, float(target_h))
		pad_px = max(0.0, float(pad_px))

		scale = target_h / float(h) if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)

		pad_int = int(pad_px)
		if pad_int > 0:
			gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)

		return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

	def _preprocess_fallback(self, crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		"""备选预处理：OTSU 二值化 + 缩放。"""
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		h, w = gray.shape[:2]
		th = max(8.0, float(target_h))
		scale = th / float(h) if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)), interpolation=cv2.INTER_LINEAR)
		pad_int = int(pad_px)
		if pad_int > 0:
			gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
		return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

	def _ocr_sequential(
		self,
		raw_frames: list[tuple[float, np.ndarray]],
		ocr: RapidOCR,
		target_h: float,
		pad_px: float,
		total_frames: int,
	) -> list[SpeedObservation]:
		observations: list[SpeedObservation] = []
		for idx, (timestamp, crop) in enumerate(raw_frames):
			proc = self.preprocess_crop(crop, target_h, pad_px)
			ocr_result, _ = ocr(proc)
			speed_value, raw_text = extract_speed_value(ocr_result)
			if speed_value is None:
				proc_fb = self._preprocess_fallback(crop, target_h, pad_px)
				ocr_result, _ = ocr(proc_fb)
				speed_value, raw_text = extract_speed_value(ocr_result)
			if speed_value is not None and raw_text is not None:
				observations.append(
					SpeedObservation(
						timestamp=timestamp,
						raw_speed_kmh=convert_speed_to_kmh(speed_value, self.speed_format_var.get()),
						raw_text=raw_text,
					)
				)
			if len(observations) % 10 == 0:
				pct = ((idx + 1) / total_frames * 90.0) + 5.0
				self.root.after(0, self._update_progress,
					f"[{ocr_engine._gpu_backend}] 正在处理... {len(observations)} 条 ({pct:.1f}%)", pct)
		return observations

	def _ocr_pipeline(
		self,
		raw_frames: list[tuple[float, np.ndarray]],
		ocr: RapidOCR,
		target_h: float,
		pad_px: float,
		total_frames: int,
		num_workers: int,
	) -> list[SpeedObservation]:
		queue_size = num_workers * 2
		q: Queue = Queue(maxsize=queue_size)
		errors: list[Exception] = []

		def producer() -> None:
			try:
				for timestamp, crop in raw_frames:
					proc = self.preprocess_crop(crop, target_h, pad_px)
					q.put((timestamp, proc))
				q.put(None)
			except Exception as exc:
				errors.append(exc)
				q.put(None)

		t = threading.Thread(target=producer, daemon=True)
		t.start()

		observations: list[SpeedObservation] = []
		done = 0
		while True:
			item = q.get()
			if item is None:
				break
			timestamp, proc_img = item
			ocr_result, _ = ocr(proc_img)
			speed_value, raw_text = extract_speed_value(ocr_result)
			if speed_value is not None and raw_text is not None:
				observations.append(
					SpeedObservation(
						timestamp=timestamp,
						raw_speed_kmh=convert_speed_to_kmh(speed_value, self.speed_format_var.get()),
						raw_text=raw_text,
					)
				)
			done += 1
			if done % 10 == 0:
				pct = (done / total_frames * 90.0) + 5.0
				self.root.after(0, self._update_progress,
					f"[{ocr_engine._gpu_backend}] 正在处理... {len(observations)} 条 ({pct:.1f}%)", pct)
		t.join()
		if errors:
			raise errors[0]
		return observations

	def export_csv(self) -> None:
		if getattr(self, "is_exporting", False):
			return
		if self.video_path is None or self.metadata is None or self.first_frame_bgr is None:
			messagebox.showwarning("未导入视频", "请先导入视频。")
			return

		try:
			max_speed_kmh = self._parse_positive_float(self.max_speed_var.get(), "最大速度上限", allow_zero=True)
			max_accel_mps2 = self._parse_positive_float(self.max_accel_var.get(), "最大加速度上限", allow_zero=True)
			frame_div = int(self._parse_positive_float(self.frame_div_var.get(), "采样间隔"))
			target_h = self._parse_positive_float(self.target_height_var.get(), "OCR 目标高度")
			pad_px = self._parse_positive_float(self.pad_var.get(), "边缘填充", allow_zero=True)
			num_workers = int(self._parse_positive_float(self.num_workers_var.get(), "并行线程数"))
		except ValueError as exc:
			messagebox.showwarning("参数错误", str(exc))
			return

		region = self.get_region()
		if region is None:
			messagebox.showwarning("识别范围不完整", "请先填写左上和右下坐标。")
			return

		output_path = filedialog.asksaveasfilename(
			title="保存 CSV",
			defaultextension=".csv",
			initialdir=str(self.video_path.parent),
			initialfile=f"{self.video_path.stem}_log.csv",
			filetypes=[("CSV 文件", "*.csv")],
		)
		if not output_path:
			return

		self.is_exporting = True
		self._cancel_flag = False
		self.export_btn.config(state="disabled")
		self.cancel_btn.config(state="normal")
		threading.Thread(
			target=self._run_export_thread,
			args=(Path(output_path), region, max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers),
			daemon=True,
		).start()

	def _run_export_thread(
		self,
		output_path: Path,
		region: tuple[int, int, int, int],
		max_speed_kmh: float,
		max_accel_mps2: float,
		frame_div: int,
		target_h: float,
		pad_px: float,
		num_workers: int,
	) -> None:
		try:
			self._run_export(output_path, region, max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers)
		except _CancelExport:
			self.root.after(0, self._on_export_cancelled)
		except Exception as exc:
			self.root.after(0, self._on_export_error, str(exc))

	def _release_ocr_engines(self) -> None:
		"""释放所有 OCR 引擎，回收 GPU 显存。"""
		engines_to_free = [self.ocr_engine] if self.ocr_engine else []
		engines_to_free.extend(self.ocr_engines)
		self.ocr_engine = None
		self.ocr_engines.clear()
		for engine in engines_to_free:
			try: del engine
			except: pass
		import gc; gc.collect()

	def _check_cancel(self) -> None:
		if self._cancel_flag:
			raise _CancelExport()

	def _cancel_export(self) -> None:
		self._cancel_flag = True
		self.cancel_btn.config(state="disabled")
		self.status_var.set("正在取消...")
		self.root.update()  # 立即刷新 GUI 显示取消状态

	def _on_export_cancelled(self) -> None:
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self.progress_var.set(0.0)
		self._release_ocr_engines()
		self.status_var.set("已取消。")

	def _on_export_error(self, err: str) -> None:
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self.progress_var.set(0.0)
		self._release_ocr_engines()
		messagebox.showerror("导出失败", err)
		self.status_var.set("导出失败。")

	def _on_export_success(self, output_path: Path, rows_len: int, elapsed: float,
	                       total_frames: int, accuracy: float, backend: str) -> None:
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self._release_ocr_engines()
		self.status_var.set(f"导出完成：{output_path}")
		self.progress_var.set(100.0)
		fps_val = total_frames / elapsed if elapsed > 0 else 0.0
		msg = (
			f"已导出 {rows_len} 条记录。\n"
			f"引擎: {backend}  |  "
			f"用时: {elapsed:.1f}s  |  "
			f"速度: {fps_val:.1f} fps  |  "
			f"准确率: {accuracy:.1f}%\n\n"
			f"{output_path}"
		)
		messagebox.showinfo("导出完成", msg)

	def _update_progress(self, msg: str, pct: float) -> None:
		self.status_var.set(msg)
		self.progress_var.set(pct)

	def _ocr_cuda_parallel(
		self,
		raw_frames: list[tuple[float, np.ndarray]],
		target_h: float,
		pad_px: float,
		total_frames: int,
		num_workers: int,
	) -> list[SpeedObservation]:
		"""并行推理：单引擎 + 多线程预处理，OCR 调用由 onnxruntime 内部并行。"""
		from concurrent.futures import ThreadPoolExecutor, as_completed
		engines = self.get_ocr_engines(1)
		engine = engines[0]
		self._check_cancel()

		with ThreadPoolExecutor(max_workers=num_workers) as pool:
			preprocessed = list(pool.map(
				lambda item: (item[0], self.preprocess_crop(item[1], target_h, pad_px)),
				raw_frames,
			))
		self._check_cancel()

		observations: list[SpeedObservation | None] = [None] * len(raw_frames)

		def _ocr_one(idx: int, ts: float, proc: np.ndarray, crop_bgr: np.ndarray) -> tuple[int, SpeedObservation | None]:
			ocr_result, _ = engine(proc)
			sv, rt = extract_speed_value(ocr_result)
			if sv is None:
				# 备选预处理：CLAHE 增强对比度
				proc_fb = self._preprocess_fallback(crop_bgr, target_h, pad_px)
				ocr_result, _ = engine(proc_fb)
				sv, rt = extract_speed_value(ocr_result)
			if sv is not None and rt is not None:
				return idx, SpeedObservation(
					timestamp=ts,
					raw_speed_kmh=convert_speed_to_kmh(sv, self.speed_format_var.get()),
					raw_text=rt,
				)
			return idx, None

		pool = ThreadPoolExecutor(max_workers=num_workers)
		try:
			futures = [pool.submit(_ocr_one, i, ts, proc, raw_frames[i][1]) for i, (ts, proc) in enumerate(preprocessed)]
			done = 0
			for f in as_completed(futures):
				if done % 10 == 0:
					self._check_cancel()
				idx, obs = f.result()
				observations[idx] = obs
				done += 1
				if done % 10 == 0:
					pct = (done / total_frames * 90.0) + 5.0
					self.root.after(0, self._update_progress,
						f"[{ocr_engine._gpu_backend}×{num_workers}] 正在处理... {done}/{total_frames} ({pct:.1f}%)", pct)
		finally:
			pool.shutdown(wait=False, cancel_futures=True)

		return [o for o in observations if o is not None]

	def _run_export(
		self,
		output_path: Path,
		region: tuple[int, int, int, int],
		max_speed_kmh: float,
		max_accel_mps2: float,
		frame_div: int,
		target_h: float,
		pad_px: float,
		num_workers: int,
	) -> None:
		import time as _time
		_t_start = _time.perf_counter()

		assert self.video_path is not None
		assert self.metadata is not None

		num_workers = max(1, min(num_workers, 8))

		self.root.after(0, self._update_progress, "正在初始化 OCR 引擎...", 0.0)
		self._check_cancel()
		self.root.update_idletasks()  # 确保进度消息显示
		ocr = self.get_ocr_engine()
		self.root.update_idletasks()  # 引擎就绪后刷新

		capture = cv2.VideoCapture(str(self.video_path))
		if not capture.isOpened():
			raise RuntimeError("无法重新打开视频文件。")

		x1, y1, x2, y2 = region
		frame_step = max(1, frame_div)

		raw_frames: list[tuple[float, np.ndarray]] = []
		frame_index = 0

		# 解析时间轴范围
		f_start = _parse_int_or_none(self._frame_start_var.get())
		f_end = _parse_int_or_none(self._frame_end_var.get())

		while True:
			ok, frame = capture.read()
			if not ok or frame is None:
				break
			if f_end is not None and frame_index >= f_end:
				break
			if f_start is not None and frame_index < f_start:
				frame_index += 1
				continue
			if frame_index % frame_step != 0:
				frame_index += 1
				continue
			if frame_index % (frame_step * 100) == 0:
				self._check_cancel()
			timestamp = frame_index / self.metadata.fps if self.metadata.fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
			crop = frame[y1 : y2 + 1, x1 : x2 + 1].copy()
			if crop.size == 0:
				capture.release()
				raise RuntimeError("识别范围超出视频画面。")
			raw_frames.append((timestamp, crop))
			frame_index += 1
		capture.release()

		total_frames = len(raw_frames)
		if total_frames == 0:
			raise RuntimeError("未从视频中读取到任何帧，请检查采样率设置。")

		self.root.after(0, self._update_progress,
			f"开始处理 {total_frames} 帧 (workers={num_workers})...", 5.0)
		self._check_cancel()

		# 仅 CUDA 支持并行推理
		if num_workers > 1 and ocr_engine._gpu_backend == "CUDA":
			observations = self._ocr_cuda_parallel(raw_frames, target_h, pad_px, total_frames, num_workers)
		elif num_workers > 1:
			observations = self._ocr_pipeline(raw_frames, ocr, target_h, pad_px, total_frames, num_workers)
		else:
			observations = self._ocr_sequential(raw_frames, ocr, target_h, pad_px, total_frames)

		if not observations:
			raise RuntimeError("未识别到任何速度数据，请检查识别范围与速度格式。")

		# 阶段4：物理约束纠错 + 积分（CPU）
		self.root.after(0, self._update_progress, "正在进行物理约束纠错...", 96.0)
		corrected_speeds = correct_speed_series(observations, max_speed_kmh, max_accel_mps2)

		# 计算统计信息
		_t_elapsed = _time.perf_counter() - _t_start
		_corrected_count = sum(
			1 for o, c in zip(observations, corrected_speeds)
			if abs(o.raw_speed_kmh - c) > 0.01
		)
		_accuracy = (1 - _corrected_count / len(observations)) * 100 if observations else 100.0

		rows: list[tuple[float, float, float, int]] = []
		distance_m = 0.0
		previous_sample_time: float | None = None
		previous_speed_ms: float | None = None
		prev_corrected_kmh: float | None = None  # 用于物理跳变检测
		for observation, corrected_speed_kmh in zip(observations, corrected_speeds):
			current_speed_ms = corrected_speed_kmh / 3.6
			if previous_sample_time is not None and previous_speed_ms is not None:
				delta_t = observation.timestamp - previous_sample_time
				if delta_t > 0:
					distance_m += (previous_speed_ms + current_speed_ms) * 0.5 * delta_t
			previous_sample_time = observation.timestamp
			previous_speed_ms = current_speed_ms
			corrected_flag = 1 if abs(observation.raw_speed_kmh - corrected_speed_kmh) > 0.01 else 0
			# 纠正后仍超物理极限的也标记（如 224→22 因无候选值而无法纠正的情况）
			if not corrected_flag and prev_corrected_kmh is not None:
				delta_t = observation.timestamp - (rows[-1][0] if rows else 0)
				if delta_t > 0 and abs(corrected_speed_kmh - prev_corrected_kmh) / (delta_t * 3.6) > max_accel_mps2:
					corrected_flag = 1
			prev_corrected_kmh = corrected_speed_kmh
			rows.append((observation.timestamp, distance_m, corrected_speed_kmh, corrected_flag))

		self._run_manual_correction(observations, raw_frames, rows)
		# 重算准确率（含人工纠正的 flag=2）
		_corrected_count = sum(1 for r in rows if r[3] >= 1)
		_accuracy = (1 - _corrected_count / len(rows)) * 100 if rows else 100.0
		self._write_csv_with_retry(output_path, rows, _t_elapsed, total_frames, _accuracy, ocr_engine._gpu_backend)

	def _run_manual_correction(self, observations, raw_frames, rows):
		"""人工纠错：弹出窗口依次展示标记帧，用户手动输入正确速度。"""
		if not self._manual_correction_var.get():
			return
		trust = _estimate_raw_trust(observations)
		# 收集所有 flag=1 的帧，按可信度升序（最不可信排最前）
		flagged = []
		for i, (t, d, s, f) in enumerate(rows):
			if f == 1:
				flagged.append((i, trust[i], observations[i]))
		if not flagged:
			return
		flagged.sort(key=lambda x: x[1])

		# 创建人工纠错窗口
		win = tk.Toplevel(self.root)
		win.title(f"人工纠错 ({len(flagged)} 帧)")
		win.geometry("500x440")
		win.transient(self.root)
		win.grab_set()
		win.resizable(False, False)

		idx_iter = iter(flagged)
		current = [None]  # mutable container

		# 预览图
		img_label = ttk.Label(win)
		img_label.grid(row=0, column=0, columnspan=2, pady=(12, 8))

		info_var = tk.StringVar()
		ttk.Label(win, textvariable=info_var, font=("", 10)).grid(row=1, column=0, columnspan=2)

		speed_var = tk.StringVar()
		entry_frame = ttk.Frame(win)
		entry_frame.grid(row=2, column=0, columnspan=2, pady=(12, 4))
		ttk.Label(entry_frame, text="正确速度 (km/h):").grid(row=0, column=0)
		ttk.Entry(entry_frame, textvariable=speed_var, width=10, font=("", 12), justify="center").grid(row=0, column=1, padx=(8, 0))

		progress_var = tk.StringVar()
		ttk.Label(win, textvariable=progress_var, foreground="#888888").grid(row=3, column=0, columnspan=2)

		done_flag = [False]
		total_flagged = len(flagged)

		def _show_next():
			nonlocal total_flagged
			try:
				ri, score, obs = next(idx_iter)
			except StopIteration:
				done_flag[0] = True
				win.destroy()
				return
			current[0] = (ri, obs, score)
			remaining = total_flagged - (rows[ri][3] if ri >= 0 else 0)  # 粗略估计
			# 实际剩余：统计还未处理的 flag=1 数量
			remaining = sum(1 for r in rows if r[3] == 1)
			progress_var.set(f"帧 #{ri+1}/{len(rows)}  |  可信度 {score:.2f}  |  剩余 {remaining} 帧")
			info_var.set(f"t={obs.timestamp:.2f}s  纠正值={rows[ri][2]:.1f} km/h  原始={obs.raw_speed_kmh:.1f}")
			speed_var.set("")
			# 显示 crop 预览
			crop = raw_frames[ri][1]
			h, w = crop.shape[:2]
			sc = min(200.0 / h, 350.0 / w, 1.0)
			disp = cv2.resize(crop, (int(w*sc), int(h*sc)))
			disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
			img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
			img_label.configure(image=img)
			img_label.image = img

		def _confirm():
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			try:
				val = float(speed_var.get().strip())
				t, d, s, f = rows[ri]
				rows[ri] = (t, d, val, 2)
			except ValueError:
				pass
			_show_next()

		def _use_raw():
			"""应用原始 OCR 值（flag=2 人工确认）。"""
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			t, d, s, f = rows[ri]
			rows[ri] = (t, d, obs.raw_speed_kmh, 2)
			_show_next()

		def _use_corrected():
			"""应用自动纠正值（flag=2 人工确认）。"""
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			t, d, s, f = rows[ri]
			rows[ri] = (t, d, s, 2)  # 保留原纠正值，标记 flag=2
			_show_next()

		def _skip_all():
			done_flag[0] = True
			win.destroy()

		btn_frame = ttk.Frame(win)
		btn_frame.grid(row=4, column=0, columnspan=2, pady=(12, 12))
		ttk.Button(btn_frame, text="确认 (Enter)", command=_confirm).grid(row=0, column=0, padx=(0, 8))
		ttk.Button(btn_frame, text="原值", command=_use_raw).grid(row=0, column=1, padx=(0, 8))
		ttk.Button(btn_frame, text="纠正值", command=_use_corrected).grid(row=0, column=2, padx=(0, 8))
		ttk.Button(btn_frame, text="全部跳过", command=_skip_all).grid(row=0, column=3)
		win.bind("<Return>", lambda e: _confirm())

		_show_next()
		self.root.wait_window(win)

	def _write_csv_with_retry(self, initial_path: Path, rows: list[tuple[float, float, float, int]],
	                          elapsed: float = 0.0, total_frames: int = 0,
	                          accuracy: float = 100.0, backend: str = "CPU") -> None:
		"""写入 CSV，若文件被占用则弹窗让用户另选路径，避免分析结果丢失。"""
		output_path = initial_path
		while True:
			try:
				with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
					writer = csv.writer(fh)
					for timestamp, distance, speed_kmh, flag in rows:
						writer.writerow([f"{timestamp:.2f}", f"{distance:.2f}", f"{speed_kmh:.2f}", str(flag)])
				self.root.after(0, self._on_export_success, output_path, len(rows),
				                elapsed, total_frames, accuracy, backend)
				return
			except (OSError, PermissionError):
				picked = threading.Event()
				result: list[str | None] = [None]

				def _ask() -> None:
					result[0] = filedialog.asksaveasfilename(
						title="无法写入，请选择其他位置",
						defaultextension=".csv",
						initialdir=str(output_path.parent),
						initialfile=f"{output_path.stem}_new.csv",
						filetypes=[("CSV 文件", "*.csv")],
					)
					picked.set()

				self.root.after(0, _ask)
				picked.wait()
				if not result[0]:
					self.root.after(0, self._on_export_error, "用户取消了保存。")
					return
				output_path = Path(result[0])

	def run(self) -> None:
		self.root.mainloop()


def main() -> None:
	import argparse, sys
	parser = argparse.ArgumentParser(description="RaceVideoToLog - 视频速度提取工具")
	parser.add_argument("video", nargs="?", help="视频文件路径")
	parser.add_argument("--roi", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="识别范围")
	parser.add_argument("--format", choices=["m/s","km/h","mile/h"], default="km/h")
	parser.add_argument("--div", type=int, default=2, choices=[1,2,3,4,5])
	parser.add_argument("--max-speed", type=float, default=400)
	parser.add_argument("--max-accel", type=float, default=50)
	parser.add_argument("--target-h", type=int, default=24)
	parser.add_argument("--pad", type=int, default=0)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--backend", choices=["auto","cuda","cpu"], default="auto")
	parser.add_argument("--ocr-model", choices=["v3","v5_mobile"], default="v5_mobile")
	parser.add_argument("-o", "--output", type=str)
	parser.add_argument("--analysis", nargs=2, metavar=("CSV1","CSV2"))
	parser.add_argument("--analysis-out", type=str)
	parser.add_argument("--frame-start", type=int, metavar="N")
	parser.add_argument("--frame-end", type=int, metavar="N")
	parser.add_argument("--multi-box", action="store_true")
	args = parser.parse_args()

	if args.video:
		from headless import run_headless
		run_headless(args)
	elif args.analysis:
		from headless import run_analysis_headless
		run_analysis_headless(args)
	else:
		if sys.platform == "win32":
			import ctypes
			ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
		app = RaceVideoToLogApp()
		app.run()


if __name__ == "__main__":
	main()