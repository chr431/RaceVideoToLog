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
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

import ocr_engine
from ocr_engine import *  # noqa: F403, F405  # pyright: ignore[reportWildcardImportFromLibrary]
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
		self._preview_cap: cv2.VideoCapture | None = None  # 持久化的预览用 VideoCapture
		self._preview_frame_no: int = 0  # 当前预览帧号
		self._preview_throttle_id: str | None = None  # 拖动节流
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

		# 时间轴范围
		self._frame_start_var = tk.StringVar(value="")
		self._frame_end_var = tk.StringVar(value="")
		self._correction_mode_var = tk.StringVar(value="auto")  # 纠错模式: auto/baseline
		self._baseline_freq_var = tk.StringVar(value="10")    # 人工基准抽样频率
		self._debug_log_var = tk.BooleanVar(value=False)       # 调试日志

		self.is_exporting = False
		self._cancel_flag = False
		self.progress_var = tk.DoubleVar(value=0.0)

		self.first_frame_pil: Image.Image | None = None
		self._preview_scale = 1.0
		self._preview_offset_x = 0.0
		self._preview_offset_y = 0.0

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

		# 右侧：识别范围 + 预览画面
		right_col = ttk.Frame(ocr_main)
		right_col.grid(row=0, column=1, sticky="nsew")
		right_col.columnconfigure(0, weight=1)
		right_col.rowconfigure(1, weight=1)  # 预览区可拉伸

		range_box = ttk.LabelFrame(right_col, text="识别范围（像素）", padding=(12, 10, 12, 12))
		range_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
		for index in range(4): range_box.columnconfigure(index, weight=1)
		self._add_range_entry(range_box, 0, 0, "左上 X", self.left_x_var)
		self._add_range_entry(range_box, 0, 1, "左上 Y", self.left_y_var)
		self._add_range_entry(range_box, 0, 2, "右下 X", self.right_x_var)
		self._add_range_entry(range_box, 0, 3, "右下 Y", self.right_y_var)

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


		ttk.Label(perf_box, text="OCR 高度 (px)").grid(row=1, column=0, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.target_height_var, width=8).grid(row=1, column=1, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text="边缘填充 (px)").grid(row=1, column=2, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.pad_var, width=8).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=(8,0))
		# 并行线程数 + 调试日志 单独一行
		ttk.Label(perf_box, text="并行线程数").grid(row=2, column=0, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.num_workers_var, width=8).grid(row=2, column=1, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Checkbutton(perf_box, text="调试日志", variable=self._debug_log_var).grid(row=2, column=2, columnspan=2, sticky="w", pady=(8,0))
		# 纠错模式选择
		mode_frame = ttk.LabelFrame(config_col, text="纠错模式", padding=(12, 10, 12, 12))
		mode_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
		mode_frame.columnconfigure(0, weight=1)
		ttk.Radiobutton(mode_frame, text="自动锚点纠错（全自动，推荐）", variable=self._correction_mode_var, value="auto").grid(row=0, column=0, sticky="w")
		ttk.Radiobutton(mode_frame, text="人工基准标注", variable=self._correction_mode_var, value="baseline").grid(row=1, column=0, sticky="w", pady=(4, 0))
		baseline_frame = ttk.Frame(mode_frame)
		baseline_frame.grid(row=1, column=0, sticky="w", padx=(140, 0), pady=(4, 0))
		ttk.Label(baseline_frame, text="抽样频率 1/").grid(row=0, column=0, sticky="w")
		self._baseline_spinbox = ttk.Spinbox(baseline_frame, textvariable=self._baseline_freq_var, from_=1, to=50, width=4)
		self._baseline_spinbox.grid(row=0, column=1, sticky="w")
		ttk.Label(baseline_frame, text="(1=全部人工)", foreground="#888888").grid(row=0, column=2, sticky="w", padx=(4, 0))

		# 时间轴范围
		time_box = ttk.LabelFrame(config_col, text="时间轴范围", padding=(12, 10, 12, 12))
		time_box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
		time_box.columnconfigure(1, weight=1); time_box.columnconfigure(4, weight=1)
		ttk.Label(time_box, text="起始帧").grid(row=0, column=0, sticky="w")
		ttk.Entry(time_box, textvariable=self._frame_start_var, width=8).grid(row=0, column=1, sticky="ew", padx=(4, 4))
		ttk.Button(time_box, text="设为当前", command=lambda: self._frame_start_var.set(str(int(self._preview_slider.get())))).grid(row=0, column=2, padx=(0, 8))
		ttk.Label(time_box, text="结束帧").grid(row=0, column=3, sticky="w")
		ttk.Entry(time_box, textvariable=self._frame_end_var, width=8).grid(row=0, column=4, sticky="ew", padx=(4, 4))
		ttk.Button(time_box, text="设为当前", command=lambda: self._frame_end_var.set(str(int(self._preview_slider.get())))).grid(row=0, column=5)
		ttk.Label(time_box, text="留空=全部。仅处理 [起始, 结束) 之间的帧。", foreground="#555555").grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

		# 右侧预览
		preview_box = ttk.LabelFrame(right_col, text="识别范围预览", padding=(6, 6, 6, 6))
		preview_box.grid(row=1, column=0, sticky="nsew")
		preview_box.columnconfigure(0, weight=1); preview_box.rowconfigure(0, weight=1)

		self.preview_canvas = tk.Canvas(preview_box, background="#151515", highlightthickness=0, cursor="crosshair")
		self.preview_canvas.grid(row=0, column=0, sticky="nsew")
		self.preview_canvas.bind("<Configure>", lambda event: self.schedule_preview_refresh())
		self.preview_canvas.bind("<ButtonPress-1>", self._on_drag_start)
		self.preview_canvas.bind("<B1-Motion>", self._on_drag_motion)
		self.preview_canvas.bind("<ButtonRelease-1>", self._on_drag_end)
		self.preview_canvas.bind("<ButtonPress-3>", lambda e: None)  # 右键保留
		# 方向键精确帧导航
		self.root.bind("<Left>", lambda e: self._step_preview_frame(-1))
		self.root.bind("<Right>", lambda e: self._step_preview_frame(1))
		self.root.bind("<Up>", lambda e: self._step_preview_frame(10))
		self.root.bind("<Down>", lambda e: self._step_preview_frame(-10))

		# 视频帧位置滑动条 + 帧号标签
		slider_row = ttk.Frame(preview_box)
		slider_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
		slider_row.columnconfigure(0, weight=1)
		self._preview_slider = ttk.Scale(slider_row, from_=0, to=1, variable=self._preview_frame_pos,
			orient="horizontal", command=self._on_slider_drag)
		self._preview_slider.grid(row=0, column=0, sticky="ew")
		self._preview_frame_label = ttk.Label(slider_row, text="#0", width=8, anchor="e")
		self._preview_frame_label.grid(row=0, column=1, padx=(6, 2))

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

	
	
	def schedule_preview_refresh(self) -> None:
		if self.preview_after_id is not None:
			self.root.after_cancel(self.preview_after_id)
		self.preview_after_id = self.root.after(200, self.refresh_preview)

	def _step_preview_frame(self, delta: int) -> None:
		"""以 delta 帧为单位移动预览位置。"""
		if not self.metadata:
			return
		pos = int(self._preview_slider.get())
		new_pos = max(0, min(self.metadata.frame_count - 1, pos + delta))
		self._preview_slider.set(new_pos)
		self._preview_frame_pos.set(new_pos)
		self._throttle_preview()

	def _on_slider_drag(self, _value: str) -> None:
		"""滑块拖动时实时更新预览（节流）。"""
		self._throttle_preview()

	def _throttle_preview(self) -> None:
		"""节流预览刷新：30ms 内只触发一次。"""
		if self._preview_throttle_id is not None:
			self.root.after_cancel(self._preview_throttle_id)
		self._preview_throttle_id = self.root.after(30, self._do_refresh_preview)

	def _do_refresh_preview(self) -> None:
		"""实际执行预览帧刷新（由节流器调用）。"""
		self._preview_throttle_id = None
		self.refresh_preview()

	def _seek_preview_frame(self, target: int) -> bool:
		"""高效定位 VideoCapture 到目标帧。小跳用 grab()，大跳用 set()。
		返回 True 表示成功。"""
		cap = self._preview_cap
		if cap is None:
			return False
		current = self._preview_frame_no
		diff = target - current
		# 小范围前跳：连续 grab() 比 seek 更快（避免关键帧搜索）
		if 0 < diff <= 30:
			for _ in range(diff):
				if not cap.grab():
					return False
			self._preview_frame_no = target
			return True
		# 其他情况：使用 set() 定位
		cap.set(cv2.CAP_PROP_POS_FRAMES, target)
		self._preview_frame_no = target
		return True

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
		if not ok or frame is None:
			capture.release()
			raise RuntimeError("无法读取视频第一帧。")

		# 保持 capture 打开用于高效预览（关闭旧的如果有）
		if self._preview_cap is not None:
			self._preview_cap.release()
		self._preview_cap = capture
		self._preview_frame_no = 0

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
		assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
		return clamp_region(x1, y1, x2, y2, self.metadata.width, self.metadata.height)

	def refresh_preview(self) -> None:
		self.preview_after_id = None
		if self.first_frame_pil is None:
			self.preview_canvas.delete("all")
			return

		pos = int(self._preview_slider.get())
		cap = self._preview_cap

		if pos > 0 and cap is not None and self.video_path is not None:
			if self._seek_preview_frame(pos):
				ok, frame = cap.retrieve()
				if ok and frame is not None:
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					self._draw_preview_image(Image.fromarray(frame_rgb))
					total = self.metadata.frame_count if self.metadata else 0
					self.status_var.set(f"预览帧 #{pos}/{total}")
					self._preview_frame_label.configure(text=f"#{pos}")
					self._update_roi_rect()
					return
			# 定位/解码失败，回退到首帧
			self._draw_preview_image(self.first_frame_pil)
			self.status_var.set(f"无法读取帧 #{pos}")
		else:
			self._draw_preview_image(self.first_frame_pil)
			self.status_var.set("预览帧 #0（首帧）")
			self._preview_frame_label.configure(text="#0")
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
				outline="#ff5050", width=max(2.0, self._preview_scale * 2), tags="roi_rect"
			)

	def _draw_preview_image(self, image: Image.Image) -> None:
		canvas_width = max(1, self.preview_canvas.winfo_width())
		canvas_height = max(1, self.preview_canvas.winfo_height())

		scale = min(canvas_width / image.width, canvas_height / image.height)
		if scale <= 0:
			scale = 1.0

		display_width = max(1, int(image.width * scale))
		display_height = max(1, int(image.height * scale))



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
		print(f"[OCR] 后端: {actual}", flush=True)
		kwargs = _get_model_kwargs("v6_small")
		if kwargs is None:
			print(f"[OCR] 警告: v6_small 模型文件不存在")
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
			except Exception: pass
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

	def _finish_export_state(self, mode: str = "") -> None:
		"""重置导出状态（不弹窗，用于已自行处理结果输出的流程）。"""
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self.progress_var.set(100.0)
		if mode == "auto":
			self.status_var.set("自动锚点完成 — 结果已保存。")
		elif mode == "baseline":
			self.status_var.set("人工基准完成 — 结果已保存。")
		else:
			self.status_var.set("导出完成。")

	def _on_export_error(self, err: str) -> None:
		self.is_exporting = False
		self._cancel_flag = False
		self.export_btn.config(state="normal")
		self.cancel_btn.config(state="disabled")
		self.progress_var.set(0.0)
		self._release_ocr_engines()
		messagebox.showerror("导出失败", err)
		self.status_var.set("导出失败。")

	def _update_progress(self, msg: str, pct: float) -> None:
		self.status_var.set(msg)
		self.progress_var.set(pct)

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

		self.root.after(0, self._update_progress, "加载 OCR 引擎...", 2.0)
		self._check_cancel()
		self.root.update_idletasks()
		ocr = self.get_ocr_engine()
		self.root.after(0, self._update_progress, "OCR 引擎就绪, 解码视频帧...", 5.0)
		self.root.update_idletasks()

		capture = cv2.VideoCapture(str(self.video_path))
		if not capture.isOpened():
			raise RuntimeError("无法重新打开视频文件。")

		x1, y1, x2, y2 = region
		frame_step = max(1, frame_div)
		total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

		raw_frames: list[tuple[float, np.ndarray]] = []
		frame_index = 0

		# 解析时间轴范围
		f_start = _parse_int_or_none(self._frame_start_var.get())
		f_end = _parse_int_or_none(self._frame_end_var.get())
		_end_limit = f_end if f_end is not None else total_video_frames

		while frame_index < total_video_frames:
			if frame_index >= _end_limit:
				break
			if f_start is not None and frame_index < f_start:
				capture.grab()  # 跳过: 只抓取不解码
				frame_index += 1
				continue
			if frame_index % frame_step != 0:
				capture.grab()  # 跳过: 只抓取不解码 (div>1 时大幅加速)
				frame_index += 1
				continue

			if not capture.grab():  # 抓取原始帧
				break
			ok, frame = capture.retrieve()  # 仅对需要的帧解码
			if not ok or frame is None:
				break

			if frame_index % max(1, frame_step * 200) == 0:
				self._check_cancel()
				pct = 5.0 + 15.0 * (frame_index / max(_end_limit, 1))
				self.root.after(0, self._update_progress,
					f"解码视频: {frame_index}/{_end_limit} 帧", pct)

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

		# ── 纠错模式分发 ──
		mode = self._correction_mode_var.get()
		if mode == "auto":
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
				self.root.after(0, lambda: self._finish_export_state("auto"))
			else:
				self.root.after(0, lambda: self._finish_export_state("auto"))
		elif mode == "baseline":
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
				self.root.after(0, lambda: self._finish_export_state("baseline"))
			else:
				self.root.after(0, lambda: self._finish_export_state("baseline"))
		else:
			messagebox.showwarning("未选择模式", "请选择纠错模式（自动锚点或人工基准）。")
			raise _CancelExport()
		return



	def _correct_with_anchors(self, rows: list, observations: list, raw_frames: list, ocr: "RapidOCR", max_speed_kmh: float, max_accel_mps2: float, anchor_indices: set) -> list:
		"""Correction pipeline. Delegates to correction.correct_with_anchors
		with a per-frame progress callback for precise x/y updates."""
		from correction import correct_with_anchors
		# Accumulate total across all correction stages so progress bar
		# advances monotonically 60→90% instead of resetting each round.
		_acc = [0, 0]  # [done_so_far, total_so_far]
		_initial_total: list[int] = [0]
		def _prog(done: int, total: int) -> None:
			if _initial_total[0] == 0:
				_initial_total[0] = total  # first round sets baseline
			_acc[0] = _initial_total[0] + done - total if total > 0 else _acc[0] + done
			pct = min(1.0, _acc[0] / max(_initial_total[0], 1))
			overall = 60.0 + pct * 30.0
			self.root.after(0, self._update_progress,
				f"物理纠错: {_acc[0]} 帧已处理", overall)
		return correct_with_anchors(rows, observations, raw_frames, ocr,
			max_speed_kmh, max_accel_mps2, anchor_indices,
			log_fn=self._log, progress_fn=_prog)


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
		anchor_indices = auto_select_anchors(observations, max_speed_kmh, max_accel_mps2=max_accel_mps2)
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

		# Run Correction B (progress updated via per-frame callback)
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
		assert self.video_path is not None
		vhash = compute_video_hash(self.video_path)
		with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write(f"# RaceVideoToLog\n")
			fh.write(f"# video_hash={vhash}, video={self.video_path.name}\n")
			fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format={self.speed_format_var.get()}\n")
			fh.write(f"# max_speed={max_speed_kmh}, max_accel={max_accel_mps2}, div={frame_div}, target_h={target_h}, pad={pad_px}, backend={ocr_engine._gpu_backend}, model=v6_small, workers={num_workers}, frame_start={self._frame_start_var.get() or ''}, frame_end={self._frame_end_var.get() or ''}, auto_anchor=1\n")
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
			def _do_annotation() -> None:
				try:
					self._run_baseline_annotation(
						observations, raw_frames, rows, max_speed_kmh, max_accel_mps2)
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
		# 自动锚点补充：在人工锚点之外自动选择可靠帧作为额外锚点
		self.root.after(0, self._update_progress,
			"正在自动识别补充锚点...", 80.0)
		auto_anchors = auto_select_anchors(observations, max_speed_kmh, max_accel_mps2=max_accel_mps2)
		manual_anchors = {i for i in range(n_obs) if rows[i][3] >= 2}
		merged_anchors = manual_anchors | (auto_anchors - manual_anchors)
		for i in auto_anchors:
			if i not in manual_anchors and rows[i][3] == 0:
				rows[i][3] = 2
		print(f'[Baseline] Anchors: {len(manual_anchors)} manual + {len(auto_anchors - manual_anchors)} auto = {len(merged_anchors)} total', flush=True)
		self._check_cancel()
		self._log(f"Correction B: {n_obs} rows, anchors={len(merged_anchors)} ({len(manual_anchors)} manual + {len(auto_anchors - manual_anchors)} auto)")
		print(f'[Baseline] Annotation done, running correction B...', flush=True); rows = self._correct_with_anchors(rows, observations, raw_frames, ocr, max_speed_kmh, max_accel_mps2,
			merged_anchors)
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
		assert self.video_path is not None
		vhash = compute_video_hash(self.video_path)
		print(f"[Baseline] Hash computed, opening CSV...", flush=True)
		with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
			fh.write(f"# RaceVideoToLog\n")
			fh.write(f"# video_hash={vhash}, video={self.video_path.name}\n")
			fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format={self.speed_format_var.get()}\n")
			fh.write(f"# max_speed={max_speed_kmh}, max_accel={max_accel_mps2}, div={frame_div}, target_h={target_h}, pad={pad_px}, backend={ocr_engine._gpu_backend}, model=v6_small, workers={num_workers}, frame_start={self._frame_start_var.get() or ''}, frame_end={self._frame_end_var.get() or ''}, baseline_freq={baseline_freq}\n")
			w = csv.writer(fh)
			for r in rows:
				w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

		print(f"[Baseline] CSV written: {output_path}", flush=True)

	def _run_baseline_annotation(self, observations: list, raw_frames: list, rows: list, max_speed_kmh: float, max_accel_mps2: float) -> None:
		"""人工基准标注窗口。用户逐帧检查并输入正确速度值。"""
		trust = _estimate_raw_trust(observations)
		total_labeled = 0

		# ── 创建窗口 ──
		win = tk.Toplevel(self.root)
		win.title("人工基准标注")
		win.geometry("500x480")
		win.transient(self.root)
		win.resizable(False, False)
		win.update_idletasks()
		rx = self.root.winfo_rootx() + (self.root.winfo_width() - 500) // 2
		ry = self.root.winfo_rooty() + (self.root.winfo_height() - 480) // 2
		win.geometry(f"+{rx}+{ry}")

		# ── UI 元素 ──
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

		# ── 状态 ──
		current_flagged: list[tuple[int, float, SpeedObservation]] = []
		idx_iter = iter([])
		current: list[tuple[int, SpeedObservation, float] | None] = [None]
		done_flag = [False]

		def _rebuild_flagged() -> bool:
			nonlocal idx_iter, current_flagged
			current_flagged = [(i, trust[i], observations[i]) for i, r in enumerate(rows) if r[3] == 1]
			if not current_flagged:
				return False
			idx_iter = iter(current_flagged)
			return True

		def _refresh_window() -> None:
			nonlocal total_labeled
			total_labeled = sum(1 for r in rows if r[3] >= 2)
			remaining = sum(1 for r in rows if r[3] == 1)
			bottom_var.set(f"已标注 {total_labeled} 帧  |  剩余 {remaining} 帧  |  跳过=留空并确认")

		def _show_next() -> None:
			try:
				ri, score, obs = next(idx_iter)
			except StopIteration:
				done_flag[0] = True
				win.destroy()
				return
			current[0] = (ri, obs, score)
			remaining = sum(1 for r in rows if r[3] == 1)
			progress_var.set(f"帧 #{ri+1}/{len(rows)}  |  剩余 {remaining} 帧")
			info_var.set(f"Frame #{ri}  t={obs.timestamp:.2f}s  输入正确速度后按确认")
			speed_var.set("")
			speed_entry.focus_set()
			crop = raw_frames[ri][1]
			h, w = crop.shape[:2]
			sc = min(200.0 / h, 350.0 / w, 1.0)
			disp = cv2.resize(crop, (int(w*sc), int(h*sc)))
			disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
			img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
			img_label.configure(image=img)
			setattr(img_label, 'image', img)  # keep reference to prevent GC
			_refresh_window()

		# ── 按钮 ──
		def _confirm() -> None:
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			try:
				val = float(speed_var.get().strip())
				if val >= 0:
					t, d, s, f = rows[ri]
					rows[ri] = [t, d, val, 2]
			except ValueError:
				# 留空=跳过当前帧
				rows[ri][3] = 0
			_show_next()

		def _skip() -> None:
			if current[0] is None or done_flag[0]:
				return
			ri, obs, _ = current[0]
			rows[ri][3] = 0
			_show_next()

		def _close() -> None:
			done_flag[0] = True
			win.destroy()

		# ── 构建按钮 ──
		for widget in btn_frame.winfo_children():
			widget.destroy()
		ttk.Button(btn_frame, text="确认 (Enter)", command=_confirm).grid(row=0, column=0, padx=(0, 6))
		ttk.Button(btn_frame, text="跳过", command=_skip).grid(row=0, column=1, padx=(0, 6))
		ttk.Button(btn_frame, text="跳过剩余", command=_close).grid(row=0, column=2)
		win.bind("<Return>", lambda e: _confirm() if not done_flag[0] else None)

		# ── 开始 ──
		if _rebuild_flagged():
			_refresh_window()
			win.grab_set()
			_show_next()
			self.root.wait_window(win)
		else:
			win.destroy()

	def _release_preview_cap(self) -> None:
		"""释放预览用的 VideoCapture。"""
		if self._preview_cap is not None:
			self._preview_cap.release()
			self._preview_cap = None
		self._preview_frame_no = 0

	def run(self) -> None:
		self.root.protocol("WM_DELETE_WINDOW", self._on_close)
		self.root.mainloop()

	def _on_close(self) -> None:
		"""关闭窗口时清理资源。"""
		self._release_preview_cap()
		self.root.destroy()


def main() -> None:
	import argparse
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
	parser.add_argument("--ocr-model", choices=["v6_small"], default="v6_small")
	parser.add_argument("-o", "--output", type=str)
	parser.add_argument("--analysis", nargs=2, metavar=("CSV1","CSV2"))
	parser.add_argument("--analysis-out", type=str)
	parser.add_argument("--frame-start", type=int, metavar="N")
	parser.add_argument("--frame-end", type=int, metavar="N")
	args = parser.parse_args()

	if args.video:
		from headless import run_headless
		run_headless(args)
	elif args.analysis:
		from analysis import run_analysis_headless
		run_analysis_headless(args)
	else:
		app = RaceVideoToLogApp()
		app.run()


if __name__ == "__main__":
	main()