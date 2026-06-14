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
from analysis import AnalysisTab

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
		self._human_baseline_var = tk.BooleanVar(value=False)  # 人工基准模式
		self._baseline_freq_var = tk.StringVar(value="10")    # 人工基准抽样频率
		self._debug_log_var = tk.BooleanVar(value=False)       # 调试日志
		self._auto_anchor_var = tk.BooleanVar(value=False)    # 自动锚点纠错

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

		perf_box = ttk.LabelFrame(config_col, text="性能", padding=(12, 10, 12, 12))
		perf_box.grid(row=2, column=0, sticky="ew")
		perf_box.columnconfigure(1, weight=1); perf_box.columnconfigure(3, weight=1); perf_box.columnconfigure(5, weight=1)

		ttk.Label(perf_box, text="采样间隔").grid(row=0, column=0, sticky="w")
		self.frame_div_spinbox = ttk.Spinbox(perf_box, textvariable=self.frame_div_var, from_=1, to=10, width=5)
		self.frame_div_spinbox.grid(row=0, column=1, sticky="ew", padx=(6, 2))
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
		ttk.Label(perf_box, text=">1 时启用并行推理。", foreground="#555555").grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))
		baseline_frame = ttk.Frame(perf_box)
		baseline_frame.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(8, 0))
		ttk.Checkbutton(baseline_frame, text="人工基准",
			variable=self._human_baseline_var).grid(row=0, column=0, sticky="w")
		ttk.Label(baseline_frame, text="抽样频率 1/").grid(row=0, column=1, sticky="w", padx=(12, 0))
		self._baseline_spinbox = ttk.Spinbox(baseline_frame, textvariable=self._baseline_freq_var,
			from_=1, to=50, width=4)
		self._baseline_spinbox.grid(row=0, column=2, sticky="w")
		ttk.Label(baseline_frame, text="(1=全部人工)", foreground="#888888").grid(row=0, column=3, sticky="w", padx=(4, 0))
		ttk.Checkbutton(baseline_frame, text="调试日志", variable=self._debug_log_var).grid(row=0, column=4, sticky="w", padx=(12, 0))
		ttk.Checkbutton(baseline_frame, text="自动锚点纠错", variable=self._auto_anchor_var).grid(row=0, column=5, sticky="w", padx=(12, 0))

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
		# Row 1: 底部状态栏（OCR 处理 tab 使用，数据分析 tab 隐藏）
		self._footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
		self._footer.grid(row=1, column=0, sticky="ew")
		self._footer.columnconfigure(0, weight=1)
		ttk.Label(self._footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
		self.progress_bar = ttk.Progressbar(self._footer, variable=self.progress_var, maximum=100.0)
		self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

		# ── Tab 2: 数据分析 ──
		self._analysis_tab = AnalysisTab(self._notebook, self._footer, self.status_var, self.progress_var)

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
		"""预处理：灰度化 + 缩放到 target_h（PP-OCR 内置归一化，无需额外处理）。"""
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
		max_speed_kmh: float = 400,
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
			if speed_value is None:
				# 数字仪表后备链：use_det=False → EasyOCR
				speed_value, raw_text = ocr_digital_fallback(ocr, crop, max_speed_kmh)
			# 始终为每一帧生成 observation（OCR 失败用 -1.0，保证索引对齐）
			if speed_value is not None and raw_text is not None:
				observations.append(SpeedObservation(
					timestamp=timestamp,
					raw_speed_kmh=convert_speed_to_kmh(speed_value, self.speed_format_var.get()),
					raw_text=raw_text,
				))
			else:
				observations.append(SpeedObservation(
					timestamp=timestamp, raw_speed_kmh=-1.0, raw_text=""))
			if (idx + 1) % 10 == 0:
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
		max_speed_kmh: float = 400,
	) -> list[SpeedObservation]:
		queue_size = num_workers * 2
		q: Queue = Queue(maxsize=queue_size)
		errors: list[Exception] = []

		def producer() -> None:
			try:
				for timestamp, crop in raw_frames:
					proc = self.preprocess_crop(crop, target_h, pad_px)
					q.put((timestamp, proc, crop))
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
			timestamp, proc_img, crop = item
			ocr_result, _ = ocr(proc_img)
			speed_value, raw_text = extract_speed_value(ocr_result)
			if speed_value is None:
				proc_fb = self._preprocess_fallback(crop, target_h, pad_px)
				ocr_result, _ = ocr(proc_fb)
				speed_value, raw_text = extract_speed_value(ocr_result)
			if speed_value is None:
				# 数字仪表后备链：use_det=False → EasyOCR
				speed_value, raw_text = ocr_digital_fallback(ocr, crop, max_speed_kmh)
			# 始终为每一帧生成 observation
			if speed_value is not None and raw_text is not None:
				observations.append(SpeedObservation(
					timestamp=timestamp,
					raw_speed_kmh=convert_speed_to_kmh(speed_value, self.speed_format_var.get()),
					raw_text=raw_text,
				))
			else:
				observations.append(SpeedObservation(
					timestamp=timestamp, raw_speed_kmh=-1.0, raw_text=""))
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

	def _log(self, msg: str) -> None:
		"""调试日志：勾选"调试日志"时输出到终端。"""
		if self._debug_log_var.get():
			print(f"[DEBUG] {msg}", flush=True)

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

	def _finish_export_state(self) -> None:
		"""重置导出状态（不弹窗，用于已自行处理结果输出的流程）。"""
		print("[Baseline] _finish_export_state called", flush=True)
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self.progress_var.set(100.0)
		self.status_var.set("人工基准完成 — 结果已保存。")

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
		max_speed_kmh: float = 400,
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
			if sv is None:
				# 数字仪表后备链：use_det=False → EasyOCR
				sv, rt = ocr_digital_fallback(engine, crop_bgr, max_speed_kmh)
			if sv is not None and rt is not None:
				return idx, SpeedObservation(
					timestamp=ts,
					raw_speed_kmh=convert_speed_to_kmh(sv, self.speed_format_var.get()),
					raw_text=rt,
				)
			return idx, SpeedObservation(timestamp=ts, raw_speed_kmh=-1.0, raw_text="")

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

		return observations  # 所有帧均已填充（失败帧用 -1.0 标记）

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

		num_workers = max(1, min(num_workers, 32))

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

		# ── 自动锚点模式 ──
		if self._auto_anchor_var.get():
			try:
				self._run_auto_anchor_mode(raw_frames, total_frames, output_path, region,
					max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers,
					_t_start, ocr)
			except _CancelExport:
				self.root.after(0, self._on_export_cancelled)
			except Exception:
				import traceback
				traceback.print_exc()
				self.root.after(0, lambda: messagebox.showerror(
					"自动锚点错误", traceback.format_exc()))
				self.root.after(0, self._finish_export_state)
			else:
				self.root.after(0, self._finish_export_state)
			return

		# ── 人工基准模式 ──
		if self._human_baseline_var.get():
			try:
				self._run_baseline_mode(raw_frames, total_frames, output_path, region,
					max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers,
					_t_start, ocr)
			except _CancelExport:
				self.root.after(0, self._on_export_cancelled)
			except Exception:
				import traceback
				traceback.print_exc()
				self.root.after(0, lambda: messagebox.showerror(
					"人工基准错误", traceback.format_exc()))
				print("[Export] Scheduling _finish_export_state via root.after...", flush=True)
				self.root.after(0, self._finish_export_state)
			else:
				self.root.after(0, self._finish_export_state)
			return

		self.root.after(0, self._update_progress,
			f"开始处理 {total_frames} 帧 (workers={num_workers})...", 5.0)
		self._check_cancel()

		# 仅 CUDA 支持并行推理
		if num_workers > 1 and ocr_engine._gpu_backend == "CUDA":
			observations = self._ocr_cuda_parallel(raw_frames, target_h, pad_px, total_frames, num_workers, max_speed_kmh)
		elif num_workers > 1:
			observations = self._ocr_pipeline(raw_frames, ocr, target_h, pad_px, total_frames, num_workers, max_speed_kmh)
		else:
			observations = self._ocr_sequential(raw_frames, ocr, target_h, pad_px, total_frames, max_speed_kmh)

		if not observations:
			raise RuntimeError("未识别到任何速度数据，请检查识别范围与速度格式。")

		# 阶段4：物理约束纠错
		self.root.after(0, self._update_progress, "正在进行物理约束纠错...", 96.0)
		corrected_speeds = correct_speed_series(observations, max_speed_kmh, max_accel_mps2)

		# 构建初始 rows
		rows = self._build_rows(observations, corrected_speeds, max_accel_mps2)

		# 计算统计信息
		_t_elapsed = _time.perf_counter() - _t_start
		_corrected_count = sum(1 for r in rows if r[3] >= 1)
		_accuracy = (1 - _corrected_count / len(rows)) * 100 if rows else 100.0

		# 弹出结果对话框
		action = self._show_result_dialog(len(rows), _corrected_count, _accuracy)
		if action == "discard":
			raise _CancelExport()
		elif action == "correct":
			# 人工纠错（迭代，持久窗口）
			self._run_manual_correction_iterative(
				observations, raw_frames, rows, max_speed_kmh, max_accel_mps2)
			# 重新计算统计
			_corrected_count = sum(1 for r in rows if r[3] >= 1)
			_accuracy = (1 - _corrected_count / len(rows)) * 100 if rows else 100.0

		self._write_csv_with_retry(output_path, rows, _t_elapsed, total_frames, _accuracy, ocr_engine._gpu_backend)

	def _show_result_dialog(self, total_rows: int, corrected_count: int, accuracy: float) -> str:
		"""识别完成对话框。返回 "save" | "correct" | "discard"。"""
		result: list[str] = []

		win = tk.Toplevel(self.root)
		win.title("识别完成")
		win.geometry("420x260")
		win.transient(self.root)
		win.grab_set()
		win.resizable(False, False)
		# 居中于主窗口
		win.update_idletasks()
		rx = self.root.winfo_rootx() + (self.root.winfo_width() - 420) // 2
		ry = self.root.winfo_rooty() + (self.root.winfo_height() - 260) // 2
		win.geometry(f"+{rx}+{ry}")

		frame = ttk.Frame(win, padding=(20, 16, 20, 16))
		frame.pack(fill="both", expand=True)

		ttk.Label(frame, text="识别完成", font=("", 14, "bold")).pack(anchor="center")
		stats = (
			f"共 {total_rows} 条记录\n"
			f"自动纠错 {corrected_count} 条  |  准确率 {accuracy:.1f}%"
		)
		ttk.Label(frame, text=stats, font=("", 10), justify="center").pack(pady=(12, 4))

		hint_text = "自动纠错可能仍有误判，建议人工复核。"
		ttk.Label(frame, text=hint_text, foreground="#888888", font=("", 9)).pack(pady=(0, 16))

		btn_frame = ttk.Frame(frame)
		btn_frame.pack()
		ttk.Button(btn_frame, text="确认保存", command=lambda: _choose("save"), width=12).pack(side="left", padx=4)
		ttk.Button(btn_frame, text="人工纠错", command=lambda: _choose("correct"), width=12).pack(side="left", padx=4)
		ttk.Button(btn_frame, text="舍弃结果", command=lambda: _choose("discard"), width=12).pack(side="left", padx=4)

		def _choose(action: str) -> None:
			result.append(action)
			win.destroy()

		self.root.wait_window(win)
		return result[0] if result else "discard"

	def _build_rows(self, observations, corrected_speeds, max_accel_mps2):
		"""从 observations 和 corrected_speeds 构建 rows 列表。"""
		rows: list[tuple[float, float, float, int]] = []
		distance_m = 0.0
		previous_sample_time: float | None = None
		previous_speed_ms: float | None = None
		prev_corrected_kmh: float | None = None
		for observation, corrected_speed_kmh in zip(observations, corrected_speeds):
			current_speed_ms = corrected_speed_kmh / 3.6
			if previous_sample_time is not None and previous_speed_ms is not None:
				delta_t = observation.timestamp - previous_sample_time
				if delta_t > 0:
					distance_m += (previous_speed_ms + current_speed_ms) * 0.5 * delta_t
			previous_sample_time = observation.timestamp
			previous_speed_ms = current_speed_ms
			corrected_flag = 1 if abs(observation.raw_speed_kmh - corrected_speed_kmh) > 0.01 else 0
			if not corrected_flag and prev_corrected_kmh is not None:
				delta_t = observation.timestamp - (rows[-1][0] if rows else 0)
				if delta_t > 0 and abs(corrected_speed_kmh - prev_corrected_kmh) / (delta_t * 3.6) > max_accel_mps2:
					corrected_flag = 1
			prev_corrected_kmh = corrected_speed_kmh
			rows.append((observation.timestamp, distance_m, corrected_speed_kmh, corrected_flag))
		return rows

	def _correct_with_anchors(self, rows, observations, raw_frames, ocr, max_speed_kmh, max_accel_mps2, anchor_indices):
		"""纠错程序 B v2：5 阶段流水线。

		1. 错误检测 — 物理一致性检查标记异常帧
		2. 重 OCR — 多种预处理变体获取备选值
		3. 最优选择 — 评分备选值，选最物理合理的
		4. 残留检测 — 多轮迭代收敛
		5. 最终填充 — 插值 + 加速度裁剪 + 轻量平滑

		锚点帧 (anchor_indices) 的值固定不变。
		"""
		if len(anchor_indices) < 2:
			return rows

		n = len(rows)
		anchors = anchor_indices
		times = [r[0] for r in rows]

		self._log(f"Correction B v2: {n} rows, {len(anchors)} anchors")

		# ── 阶段 1：错误检测 ──
		error_set = self._detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		self._log(f"  Stage 1: detected {len(error_set)} errors")

		if not error_set:
			return rows

		# ── 阶段 2+3：重 OCR + 最优选择（首轮）──
		fixed = self._fix_errors(rows, observations, raw_frames, ocr, error_set, anchors, times, max_speed_kmh, max_accel_mps2)
		self._log(f"  Stage 2+3: fixed {fixed} frames in round 1")

		# ── 阶段 4：多轮迭代 ──
		max_rounds = 3
		prev_error_count = len(error_set)
		for rnd in range(2, max_rounds + 1):
			error_set = self._detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
			if not error_set or len(error_set) >= prev_error_count:
				break
			prev_error_count = len(error_set)
			fixed = self._fix_errors(rows, observations, raw_frames, ocr, error_set, anchors, times, max_speed_kmh, max_accel_mps2)
			self._log(f"  Stage 4 round {rnd}: {len(error_set)} errors, fixed {fixed}")

		# ── 阶段 5：最终填充不可恢复帧 + 轻量平滑 ──
		error_set = self._detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		if error_set:
			self._fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2)
			self._log(f"  Stage 5: filled {len(error_set)} unrecoverable frames")

		self._apply_final_smooth(rows, anchors, max_speed_kmh, max_accel_mps2)
		self._log("  Final smooth applied")

		return rows

	def _detect_errors(self, rows, anchors, times, max_speed_kmh, max_accel_mps2):
		"""阶段 1：错误检测。三种检测器并行标记异常帧。

		A. 邻帧跳变 — 与前后邻帧的加速度超限
		B. 锚点趋势偏离 — 偏离锚点间线性插值过多
		C. 孤立离群 (spike) — 与两边都冲突但邻居彼此一致
		"""
		n = len(rows)
		raw_vals = [r[2] for r in rows]
		error_set = set()

		for i in range(n):
			if i in anchors:
				continue
			v = raw_vals[i]
			if v < 0 or v > max_speed_kmh:
				error_set.add(i)
				continue

			# ── A. 邻帧跳变检测 ──
			fwd_fail = False
			bwd_fail = False

			if i > 0:
				prev_v = raw_vals[i - 1]
				if prev_v >= 0 and prev_v <= max_speed_kmh:
					dt = max(times[i] - times[i - 1], 0.001)
					max_dv = max_accel_mps2 * dt * 3.6 * 1.2
					if abs(v - prev_v) > max_dv:
						# 跳过显示保持帧（同值且 dt < 0.15s）
						if not (i + 1 < n and v == raw_vals[i + 1] and times[i + 1] - times[i] < 0.15):
							fwd_fail = True

			if i + 1 < n:
				next_v = raw_vals[i + 1]
				if next_v >= 0 and next_v <= max_speed_kmh:
					dt = max(times[i + 1] - times[i], 0.001)
					max_dv = max_accel_mps2 * dt * 3.6 * 1.2
					if abs(next_v - v) > max_dv:
						if not (i > 0 and v == raw_vals[i - 1] and times[i] - times[i - 1] < 0.15):
							bwd_fail = True

			if fwd_fail and bwd_fail:
				error_set.add(i)
				continue

			# ── B. 锚点趋势偏离 ──
			la = None; ra = None
			for j in range(i - 1, -1, -1):
				if j in anchors:
					la = j; break
			for j in range(i + 1, n):
				if j in anchors:
					ra = j; break
			if la is not None and ra is not None:
				lv = rows[la][2]; rv = rows[ra][2]
				lt = rows[la][0]; rt = rows[ra][0]
				total_dt = max(rt - lt, 0.001)
				frac = (times[i] - lt) / total_dt
				interp = lv + (rv - lv) * frac
				seg_dt = times[i] - lt
				threshold = max(5.0, 3.0 * max_accel_mps2 * max(seg_dt, 0.1) * 3.6)
				if abs(v - interp) > threshold:
					error_set.add(i)
					continue

			# ── C. 孤立离群 (spike) ──
			if i >= 2 and i + 2 < n:
				left_v = raw_vals[i - 1] if raw_vals[i - 1] >= 0 else (raw_vals[i - 2] if raw_vals[i - 2] >= 0 else None)
				right_v = raw_vals[i + 1] if raw_vals[i + 1] >= 0 else (raw_vals[i + 2] if raw_vals[i + 2] >= 0 else None)
				if left_v is not None and right_v is not None:
					dt_cross = max(times[i + 2] - times[i - 2], 0.01)
					max_dv_cross = max_accel_mps2 * dt_cross * 3.6 * 1.5
					if abs(right_v - left_v) <= max_dv_cross:
						dt_left = max(times[i] - times[i - 1], 0.001)
						dt_right = max(times[i + 1] - times[i], 0.001)
						max_dv_l = max_accel_mps2 * dt_left * 3.6 * 1.5
						max_dv_r = max_accel_mps2 * dt_right * 3.6 * 1.5
						if abs(v - left_v) > max_dv_l and abs(right_v - v) > max_dv_r:
							error_set.add(i)

			# ── D. 局部趋势偏离 ──
			# 5 帧滑动窗口中值作为局部期望值。
			# 若本帧偏离期望 > 3 km/h 且两侧邻帧与期望一致，则为短暂闪变（OCR 误读特征）。
			if i >= 2 and i + 2 < n:
				window = []
				for j in range(max(0, i - 2), min(n, i + 3)):
					if j != i and raw_vals[j] >= 0 and raw_vals[j] <= max_speed_kmh:
						window.append(raw_vals[j])
				if len(window) >= 3:
					window.sort()
					local_median = window[len(window) // 2]
					dev = abs(v - local_median)
					if dev > 3.0:
						# 验证：两侧邻帧与中位数一致（偏差 < 2 km/h）
						left_ok = (i >= 1 and raw_vals[i - 1] >= 0 and abs(raw_vals[i - 1] - local_median) < 2.0)
						right_ok = (i + 1 < n and raw_vals[i + 1] >= 0 and abs(raw_vals[i + 1] - local_median) < 2.0)
						if left_ok and right_ok:
							error_set.add(i)

		return error_set

	def _fix_errors(self, rows, observations, raw_frames, ocr, error_set, anchors, times, max_speed_kmh, max_accel_mps2):
		"""阶段 2+3：对每个 error 帧重 OCR 获取备选，选最优值填入。"""
		fixed = 0
		for i in error_set:
			if i in anchors:
				continue
			candidates = list(self._re_ocr_frame(raw_frames[i][1], ocr, max_speed_kmh))
			# 加入锚点插值候选
			interp_cand = self._interp_candidate(i, rows, anchors, times, max_speed_kmh)
			if interp_cand is not None:
				candidates.append(interp_cand)
			# 加入混淆表候选
			oid = min(i, len(observations) - 1)
			confusion_cands = build_speed_candidates(observations[oid].raw_text, max_speed_kmh)
			candidates.extend(c for c in confusion_cands if c not in candidates)

			if not candidates:
				continue

			best_val = None
			best_score = -1.0
			for cand in set(candidates):
				if not (0 <= cand <= max_speed_kmh):
					continue
				score = self._score_candidate(cand, i, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2)
				if score > best_score:
					best_score = score
					best_val = cand

			if best_val is not None and abs(rows[i][2] - best_val) > 0.5:
				rows[i][2] = best_val
				if rows[i][3] == 0:
					rows[i][3] = 1
				fixed += 1
		return fixed

	def _re_ocr_frame(self, crop_bgr, ocr, max_speed_kmh):
		"""阶段 2：对单帧尝试 6 种预处理变体重 OCR，返回所有有效备选值集合。"""
		candidates = set()
		if crop_bgr is None or crop_bgr.size == 0:
			return candidates

		gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
		h, w = gray.shape[:2]
		if h <= 0 or w <= 0:
			return candidates

		def _do_ocr(img_bgr):
			res, _ = ocr(img_bgr)
			sv, rt = extract_speed_value(res)
			if sv is not None and sv <= max_speed_kmh:
				candidates.add(float(sv))

		# 变体 1: 标准灰度 (h=24)
		scale = 24.0 / h if h > 0 else 1.0
		proc = cv2.resize(gray, (max(1, int(w * scale)), 24))
		_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

		# 变体 2: CLAHE + OTSU (h=32)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		_, otsu = cv2.threshold(clahe.apply(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		scale32 = 32.0 / h if h > 0 else 1.0
		proc = cv2.resize(otsu, (max(1, int(w * scale32)), 32))
		_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

		# 变体 3: OTSU 二值化 (h=32)
		_, otsu2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		proc = cv2.resize(otsu2, (max(1, int(w * scale32)), 32))
		_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

		# 变体 4: 反相灰度 (h=32)
		proc = cv2.resize(cv2.bitwise_not(gray), (max(1, int(w * scale32)), 32))
		_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

		# 变体 5: OTSU 反相 (h=32)
		_, otsu3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		proc = cv2.resize(otsu3, (max(1, int(w * scale32)), 32))
		_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

		# 变体 6: EasyOCR 后备
		try:
			sv, _rt = ocr_digital_fallback(ocr, crop_bgr, max_speed_kmh)
			if sv is not None:
				candidates.add(float(sv))
		except Exception:
			pass

		return candidates

	def _interp_candidate(self, i, rows, anchors, times, max_speed_kmh):
		"""计算帧 i 在左右锚点间的线性插值估计。"""
		n = len(rows)
		la = None; ra = None
		for j in range(i - 1, -1, -1):
			if j in anchors:
				la = j; break
		for j in range(i + 1, n):
			if j in anchors:
				ra = j; break
		if la is not None and ra is not None:
			lv = rows[la][2]; rv = rows[ra][2]
			lt = rows[la][0]; rt = rows[ra][0]
			total_dt = max(rt - lt, 0.001)
			frac = (times[i] - lt) / total_dt
			val = lv + (rv - lv) * frac
			if 0 <= val <= max_speed_kmh:
				return val
		return None

	def _score_candidate(self, val, i, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2):
		"""阶段 3：对候选值评分。

		score = neighbor_score * 0.4 + anchor_score * 0.35 + smoothness_score * 0.25
		"""
		n = len(rows)

		# 1. neighbor_score: 与前后可信邻帧的加速度一致性
		neighbor_score = 0.0
		count = 0
		for j in range(i - 1, max(i - 4, -1), -1):
			if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
				continue
			dt = max(times[i] - times[j], 0.001)
			max_dv = max_accel_mps2 * dt * 3.6
			dv = abs(val - rows[j][2])
			neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
			count += 1
			break
		for j in range(i + 1, min(i + 5, n)):
			if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
				continue
			dt = max(times[j] - times[i], 0.001)
			max_dv = max_accel_mps2 * dt * 3.6
			dv = abs(rows[j][2] - val)
			neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
			count += 1
			break
		neighbor_score = neighbor_score / max(count, 1)

		# 2. anchor_score: 与锚点插值的接近度
		anchor_score = 0.0
		interp = self._interp_candidate(i, rows, anchors, times, max_speed_kmh)
		if interp is not None:
			dev = abs(val - interp)
			threshold = max(5.0, max_accel_mps2 * 3.6)
			anchor_score = max(0.0, 1.0 - dev / threshold)

		# 3. smoothness_score: 检查是否产生平滑的加速度剖面
		smoothness_score = 0.5
		if i >= 1 and i + 1 < n:
			prev_v = None
			for j in range(i - 1, max(i - 3, -1), -1):
				if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
					prev_v = rows[j][2]; break
			next_v = None
			for j in range(i + 1, min(i + 4, n)):
				if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
					next_v = rows[j][2]; break
			if prev_v is not None and next_v is not None:
				expected = (prev_v + next_v) / 2.0
				dev2 = abs(val - expected)
				smoothness_score = max(0.0, 1.0 - dev2 / max(10.0, max_accel_mps2 * 1.8 * 3.6))

		return neighbor_score * 0.4 + anchor_score * 0.35 + smoothness_score * 0.25

	def _fill_unrecoverable(self, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2):
		"""阶段 5：对无法通过重 OCR 修复的帧，直接计算物理合理值。"""
		n = len(rows)
		for i in error_set:
			if i in anchors:
				continue
			la = None; ra = None
			for j in range(i - 1, -1, -1):
				if j in anchors or j not in error_set:
					if 0 <= rows[j][2] <= max_speed_kmh:
						la = j; break
			for j in range(i + 1, n):
				if j in anchors or j not in error_set:
					if 0 <= rows[j][2] <= max_speed_kmh:
						ra = j; break
			if la is not None and ra is not None:
				lv = rows[la][2]; rv = rows[ra][2]
				lt = rows[la][0]; rt = rows[ra][0]
				total_dt = max(rt - lt, 0.001)
				frac = (times[i] - lt) / total_dt
				val = lv + (rv - lv) * frac
			elif la is not None:
				val = rows[la][2]
			elif ra is not None:
				val = rows[ra][2]
			else:
				continue

			# 加速度裁剪
			if la is not None:
				dt = max(times[i] - rows[la][0], 0.001)
				max_dv = max_accel_mps2 * dt * 3.6
				val = max(rows[la][2] - max_dv, min(rows[la][2] + max_dv, val))

			val = max(0.0, min(max_speed_kmh, val))
			rows[i][2] = val
			if rows[i][3] == 0:
				rows[i][3] = 1

	def _apply_final_smooth(self, rows, anchors, max_speed_kmh, max_accel_mps2):
		"""阶段 5 末尾：轻量 Savitzky-Golay 平滑。只触动非锚点帧且变化 < 3 km/h。"""
		n = len(rows)
		if n < 7:
			return
		vals = [r[2] for r in rows]
		win = min(n, 7) | 1  # 奇数窗口
		try:
			smoothed = _savgol_filter_np(np.array(vals, dtype=float), win, 2)
			for i in range(n):
				if i in anchors:
					continue
				diff = abs(smoothed[i] - vals[i])
				if diff < 3.0:
					rows[i][2] = max(0.0, min(max_speed_kmh, float(smoothed[i])))
		except Exception:
			pass



	def _run_auto_anchor_mode(self, raw_frames, total_frames, output_path, region,
			max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers,
			_t_start, ocr):
		"""Auto anchor mode: select reliable OCR frames, run Correction B."""
		print(f'[AutoAnchor] START: {total_frames} frames', flush=True)
		import time as _time
		print('[AutoAnchor] Starting OCR...', flush=True)
		self.root.after(0, self._update_progress, "正在 OCR 自动识别...", 25.0)
		self._check_cancel()
		observations = self._ocr_sequential(raw_frames, ocr, target_h, pad_px, total_frames, max_speed_kmh)
		self._check_cancel()
		n_obs = len(observations)
		print(f'[AutoAnchor] OCR done: {n_obs} frames', flush=True)
		if n_obs == 0:
			raise RuntimeError("未识别到任何速度数据。")

		# Auto-select anchors
		self.root.after(0, self._update_progress, "正在自动识别可靠锚点...", 40.0)
		anchor_indices = auto_select_anchors(observations, max_speed_kmh, window=7, max_dev=5.0)
		print(f'[AutoAnchor] Selected {len(anchor_indices)} anchors ({100*len(anchor_indices)/n_obs:.1f}% of frames)', flush=True)
		if len(anchor_indices) < 3:
			raise RuntimeError("自动锚点选择失败：未找到足够的可靠帧。")

		# Build rows with anchors
		rows = []
		for i, obs in enumerate(observations):
			if i in anchor_indices:
				rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 2])
			else:
				rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 0])

		# Run Correction B
		self.root.after(0, self._update_progress, "正在以自动锚点进行物理约束纠错...", 60.0)
		self._check_cancel()
		print(f'[AutoAnchor] Running Correction B...', flush=True)
		rows = self._correct_with_anchors(rows, observations, raw_frames, ocr, max_speed_kmh, max_accel_mps2, anchor_indices)
		print(f'[AutoAnchor] Correction B done, integrating distance...', flush=True)

		# Distance integration
		dist = 0.0; prev_t, prev_v = None, None
		for r in rows:
			v = r[2] / 3.6
			if prev_t is not None and prev_v is not None:
				dt = r[0] - prev_t
				if dt > 0: dist += (prev_v + v) * 0.5 * dt
			prev_t, prev_v = r[0], v; r[1] = dist

		print(f'[AutoAnchor] Distance integrated, writing CSV...', flush=True)
		_t_elapsed = _time.perf_counter() - _t_start
		_corrected_count = sum(1 for r in rows if r[3] >= 1)
		_accuracy = (1 - _corrected_count / len(rows)) * 100 if rows else 100.0

		# Write CSV
		vhash = compute_video_hash(self.video_path)
		with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write(f"# RaceVideoToLog\n")
			fh.write(f"# video_hash={vhash}, video={self.video_path.name}\n")
			fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format={self.speed_format_var.get()}\n")
			fh.write(f"# max_speed={max_speed_kmh}, max_accel={max_accel_mps2}, div={frame_div}, target_h={target_h}, pad={pad_px}, backend={ocr_engine._gpu_backend}, model={self._ocr_model_var.get()}, workers={num_workers}, frame_start={self._frame_start_var.get() or ''}, frame_end={self._frame_end_var.get() or ''}, auto_anchor=1\n")
			w = csv.writer(fh)
			for r in rows:
				w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

		print(f'[AutoAnchor] CSV written: {output_path}', flush=True)

	def _run_baseline_mode(self, raw_frames, total_frames, output_path, region,
			max_speed_kmh, max_accel_mps2, frame_div, target_h, pad_px, num_workers,
			_t_start, ocr):
		"""人工基准模式完整流程。"""
		print(f'[Baseline] START: {total_frames} frames, freq={self._baseline_freq_var.get()}', flush=True)
		import time as _time
		baseline_freq = max(1, int(_parse_int_or_none(self._baseline_freq_var.get()) or 10))
		print('[Baseline] Starting OCR...', flush=True)
		self.root.after(0, self._update_progress, "正在 OCR 自动识别...", 25.0)
		self._check_cancel()
		# 基准模式使用串行 OCR（避免后台线程中并行引擎的潜在死锁）
		observations = self._ocr_sequential(raw_frames, ocr, target_h, pad_px, total_frames, max_speed_kmh)
		self._check_cancel()
		print(f'[Baseline] OCR done: {len(observations)} frames', flush=True); n_obs = len(observations)
		if n_obs == 0:
			raise RuntimeError("未识别到任何速度数据。")
		baseline_indices = set(range(0, n_obs, baseline_freq))
		n_baseline = len(baseline_indices)
		self.root.after(0, self._update_progress,
			f"人工基准模式：{n_obs} 帧中 {n_baseline} 帧需人工标注 (1/{baseline_freq})...", 20.0)
		self._check_cancel()
		rows = []
		for i, obs in enumerate(observations):
			if i in baseline_indices:
				rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 1])
			else:
				rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 0])
		if n_baseline > 0:
			for i in range(n_obs):
				if i not in baseline_indices:
					rows[i][3] = 3
			# 标注窗口必须在主线程创建（后台线程中 Tkinter 窗口无法渲染）
			import threading as _th
			_ann_done = _th.Event()
			_ann_error = []
			def _do_annotation():
				try:
					self._run_manual_correction_iterative(
						observations, raw_frames, rows, max_speed_kmh, max_accel_mps2,
						baseline_mode=True)
				except Exception as e:
					_ann_error.append(e)
				finally:
					_ann_done.set()
			self.root.after(0, _do_annotation)
			if not _ann_done.wait(timeout=3600):  # 最多等 1 小时
				raise RuntimeError("标注窗口超时未响应")
			if _ann_error:
				raise _ann_error[0]
			for i in range(n_obs):
				if rows[i][3] == 3:
					rows[i][3] = 0
		self.root.after(0, self._update_progress,
			"正在以人工基准为锚点进行物理约束纠错...", 85.0)
		self._check_cancel()
		self._log(f"Correction B: {n_obs} rows, anchors={sum(1 for i in range(n_obs) if rows[i][3] >= 2)}")
		print(f'[Baseline] Annotation done, running correction B...', flush=True); rows = self._correct_with_anchors(rows, observations, raw_frames, ocr, max_speed_kmh, max_accel_mps2,
			{i for i in range(n_obs) if rows[i][3] >= 2})
		print(f'[Baseline] Correction B done, integrating distance...', flush=True); dist = 0.0; prev_t, prev_v = None, None
		for r in rows:
			v = r[2] / 3.6
			if prev_t is not None and prev_v is not None:
				dt = r[0] - prev_t
				if dt > 0: dist += (prev_v + v) * 0.5 * dt
			prev_t, prev_v = r[0], v; r[1] = dist
		print(f"[Baseline] Distance integration done, computing hash...", flush=True)
		_t_elapsed = _time.perf_counter() - _t_start
		_corrected_count = sum(1 for r in rows if r[3] >= 1)
		_accuracy = (1 - _corrected_count / len(rows)) * 100 if rows else 100.0
		# 写出 CSV（含参数头）
		vhash = compute_video_hash(self.video_path)
		print(f"[Baseline] Hash computed, opening CSV...", flush=True)
		with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write(f"# RaceVideoToLog\n")
			fh.write(f"# video_hash={vhash}, video={self.video_path.name}\n")
			fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format={self.speed_format_var.get()}\n")
			fh.write(f"# max_speed={max_speed_kmh}, max_accel={max_accel_mps2}, div={frame_div}, target_h={target_h}, pad={pad_px}, backend={ocr_engine._gpu_backend}, model={self._ocr_model_var.get()}, workers={num_workers}, frame_start={self._frame_start_var.get() or ''}, frame_end={self._frame_end_var.get() or ''}, baseline_freq={baseline_freq}\n")
			w = csv.writer(fh)
			for r in rows:
				w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

		print(f"[Baseline] CSV written: {output_path}", flush=True)

	def _run_manual_correction_iterative(self, observations, raw_frames, rows, max_speed_kmh, max_accel_mps2,
			baseline_mode: bool = False):
		"""人工纠错窗口。baseline_mode=True 时单轮不迭代，支持跳过。"""
		trust = _estimate_raw_trust(observations)
		total_corrected = 0
		iteration = 0

		# ── 创建持久窗口 ──
		win = tk.Toplevel(self.root)
		win.title("人工纠错")
		win.geometry("500x480")
		win.transient(self.root)
		win.resizable(False, False)
		win.update_idletasks()
		rx = self.root.winfo_rootx() + (self.root.winfo_width() - 500) // 2
		ry = self.root.winfo_rooty() + (self.root.winfo_height() - 480) // 2
		win.geometry(f"+{rx}+{ry}")

		# ── 界面元素（持久）──
		img_label = ttk.Label(win)
		img_label.grid(row=0, column=0, columnspan=2, pady=(12, 8))
		info_var = tk.StringVar()
		ttk.Label(win, textvariable=info_var, font=("", 10)).grid(row=1, column=0, columnspan=2)
		speed_var = tk.StringVar()
		entry_frame = ttk.Frame(win)
		entry_frame.grid(row=2, column=0, columnspan=2, pady=(12, 4))
		ttk.Label(entry_frame, text="正确速度 (km/h):").grid(row=0, column=0)
		speed_entry = ttk.Entry(entry_frame, textvariable=speed_var, width=10, font=("", 12), justify="center")
		speed_entry.grid(row=0, column=1, padx=(8, 0))
		progress_var = tk.StringVar()
		ttk.Label(win, textvariable=progress_var, foreground="#888888").grid(row=3, column=0, columnspan=2)
		bottom_var = tk.StringVar()
		bottom_frame = ttk.Frame(win)
		bottom_frame.grid(row=5, column=0, columnspan=2, pady=(4, 12), sticky="ew")
		bottom_frame.columnconfigure(0, weight=1)
		ttk.Label(bottom_frame, textvariable=bottom_var, foreground="#555555", font=("", 9)).grid(row=0, column=0)

		btn_frame = ttk.Frame(win)
		btn_frame.grid(row=4, column=0, columnspan=2, pady=(12, 8))

		# ── 可变状态 ──
		current_flagged: list[tuple[int, float, SpeedObservation]] = []
		idx_iter = iter([])
		current = [None]
		done_flag = [False]

		def _rebuild_flagged():
			"""收集当前 flag=1 帧，更新迭代。"""
			nonlocal idx_iter, current_flagged
			current_flagged = []
			for i, r in enumerate(rows):
				if r[3] == 1:
					current_flagged.append((i, trust[i], observations[i]))
			if not current_flagged:
				return False
			if not baseline_mode:
					current_flagged.sort(key=lambda x: x[1])
			idx_iter = iter(current_flagged)
			return True

		def _refresh_window():
			"""更新标题和底部统计。"""
			nonlocal total_corrected
			total_corrected = sum(1 for r in rows if r[3] >= 2)
			remaining = sum(1 for r in rows if r[3] == 1)
			if baseline_mode:
				win.title("人工基准标注")
			else:
				win.title(f"人工纠错 — 第 {iteration} 轮")
			if baseline_mode:
				bottom_var.set(f"已标注 {total_corrected} 帧  |  剩余 {remaining} 帧  |  跳过=留空")
			else:
				bottom_var.set(f"已纠正 {total_corrected} 帧  |  当前第 {iteration} 轮  |  剩余 {remaining} 帧")

		def _show_next():
			try:
				ri, score, obs = next(idx_iter)
			except StopIteration:
				done_flag[0] = True
				if baseline_mode:
					win.destroy()
				else:
					_next_round()
				return
			current[0] = (ri, obs, score)
			remaining = sum(1 for r in rows if r[3] == 1)
			if baseline_mode:
				progress_var.set(f"帧 #{ri+1}/{len(rows)}  |  剩余 {remaining} 帧")
			else:
				progress_var.set(f"帧 #{ri+1}/{len(rows)}  |  可信度 {score:.2f}  |  剩余 {remaining} 帧")
			if baseline_mode:
				info_var.set(f"Frame #{ri}  t={obs.timestamp:.2f}s  输入正确速度后按确认")
			else:
				info_var.set(f"t={obs.timestamp:.2f}s  纠正值={rows[ri][2]:.1f} km/h  原始={obs.raw_speed_kmh:.1f}")
			speed_var.set("")
			speed_entry.focus_set()
			crop = raw_frames[ri][1]
			h, w = crop.shape[:2]
			sc = min(200.0 / h, 350.0 / w, 1.0)
			disp = cv2.resize(crop, (int(w*sc), int(h*sc)))
			disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
			img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
			img_label.configure(image=img)
			img_label.image = img
			_refresh_window()

		# ── 按钮动作 ──
		def _confirm():
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			try:
				val = float(speed_var.get().strip())
				if val >= 0:
					t, d, s, f = rows[ri]
					rows[ri] = [t, d, val, 2]
			except ValueError:
				pass
			_show_next()

		def _use_raw():
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			t, d, s, f = rows[ri]
			rows[ri] = (t, d, obs.raw_speed_kmh, 2)
			_show_next()

		def _use_corrected():
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			t, d, s, f = rows[ri]
			rows[ri] = (t, d, s, 2)
			_show_next()

		def _skip():
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			rows[ri][3] = 0
			_show_next()

		def _close():
			done_flag[0] = True
			win.destroy()

		def _next_round():
			"""本轮结束：重新纠错并展示下一轮。"""
			nonlocal iteration
			# 用 flag=2 帧重建 observations 再纠错
			flag2 = set(i for i, r in enumerate(rows) if r[3] == 2)
			if flag2:
				new_obs = list(observations)
				for i in flag2:
					t, d, s, f = rows[i]
					new_obs[i] = SpeedObservation(
						timestamp=observations[i].timestamp,
						raw_speed_kmh=s, raw_text=str(int(s)))
				new_speeds = correct_speed_series(new_obs, max_speed_kmh, max_accel_mps2)
				new_rows = self._build_rows(new_obs, new_speeds, max_accel_mps2)
				for i in flag2:
					new_rows[i] = rows[i]
				rows.clear()
				rows.extend(new_rows)

			iteration += 1
			if _rebuild_flagged():
				done_flag[0] = False
				current[0] = None
				_refresh_window()
				win.deiconify()
				win.lift()
				win.grab_set()
				win.after(10, _show_next)  # 异步调用，避免递归
			else:
				win.destroy()

		# ── 构建按钮 ──
		for widget in btn_frame.winfo_children():
			widget.destroy()
		if baseline_mode:
			ttk.Button(btn_frame, text="确认 (Enter)", command=_confirm).grid(row=0, column=0, padx=(0, 6))
			ttk.Button(btn_frame, text="跳过", command=_skip).grid(row=0, column=1, padx=(0, 6))
			ttk.Button(btn_frame, text="跳过剩余", command=_close).grid(row=0, column=2)
		else:
			ttk.Button(btn_frame, text="确认 (Enter)", command=_confirm).grid(row=0, column=0, padx=(0, 6))
			ttk.Button(btn_frame, text="原值", command=_use_raw).grid(row=0, column=1, padx=(0, 6))
			ttk.Button(btn_frame, text="纠正值", command=_use_corrected).grid(row=0, column=2, padx=(0, 6))
			ttk.Button(btn_frame, text="跳过剩余", command=_close).grid(row=0, column=3)
		win.bind("<Return>", lambda e: _confirm() if not done_flag[0] else None)

		# ── 开始第一轮 ──
		if _rebuild_flagged():
			_refresh_window()
			win.grab_set()
			_show_next()
			self.root.wait_window(win)
		else:
			win.destroy()

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
	parser.add_argument("--div", type=int, default=2, choices=list(range(1, 11)))
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
	parser.add_argument("--baseline-freq", type=int, default=0, metavar="N",
		help="人工基准抽样频率 1/N (1=全部人工)")
	parser.add_argument("--multi-box", action="store_true")
	args = parser.parse_args()

	if args.video:
		from headless import run_headless
		run_headless(args)
	elif args.analysis:
		from analysis import run_analysis_headless
		run_analysis_headless(args)
	else:
		if sys.platform == "win32":
			import ctypes
			ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
		app = RaceVideoToLogApp()
		app.run()


if __name__ == "__main__":
	main()