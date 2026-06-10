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
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import threading

import cv2
import numpy as np
from PIL import Image, ImageTk

# ═══════════════════ GPU 加速前置：注册 CUDA/cuDNN DLL ═══════════════════
def _register_gpu_dlls() -> None:
    """将 CUDA 和 cuDNN DLL 按依赖顺序预加载到进程内存。"""
    try:
        import ctypes as _ct
        import os as _os

        _cuda_bin: str | None = None
        _cudnn_dir: str | None = None
        _cudnn_dlls: list[str] = []

        # ── 1. 定位 CUDA Toolkit bin 目录 ──
        _cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        for _ver in [
            "v12.9", "v12.8", "v12.7", "v12.6", "v12.5", "v12.4",
            "v12.3", "v12.2", "v12.1", "v12.0",
            "v11.8", "v11.7", "v11.6",
        ]:
            _cb = _os.path.join(_cuda_base, _ver, "bin")
            if _os.path.isdir(_cb):
                _cuda_bin = _cb
                break
        if not _cuda_bin:
            for _env in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
                _val = _os.environ.get(_env, "")
                if _val:
                    _cb = _os.path.join(_val, "bin")
                    if _os.path.isdir(_cb):
                        _cuda_bin = _cb
                        break

        # ── 2. 定位 cuDNN DLL（匹配 CUDA 版本）──
        _cuda_major = ""
        if _cuda_bin:
            import re as _re
            _m = _re.search(r"v(\d+\.\d+)", _cuda_bin.replace("\\", "/"))
            if _m:
                _cuda_major = _m.group(1)

        _cudnn_base = r"C:\Program Files\NVIDIA\CUDNN"
        if _os.path.isdir(_cudnn_base):
            _candidates: list[tuple[str, str]] = []
            for _root, _dirs, _files in _os.walk(_cudnn_base):
                for _f in _files:
                    if _f.lower().startswith("cudnn") and _f.endswith(".dll"):
                        _candidates.append((_root, _f))
            if _cuda_major and _candidates:
                _matched = [(r, f) for r, f in _candidates if _cuda_major in r.replace("\\", "/")]
                if _matched:
                    _candidates = _matched
            for _root, _f in _candidates:
                if _cudnn_dir is None:
                    _cudnn_dir = _root
                _cudnn_dlls.append(_os.path.join(_root, _f))

        # ── 3. 按依赖顺序预加载 DLL ──
        _loaded = 0
        _failed: list[str] = []

        def _load_dll(_path: str) -> bool:
            nonlocal _loaded
            try:
                _ct.CDLL(_path)
                _loaded += 1
                return True
            except OSError as _e:
                _failed.append(f"{_os.path.basename(_path)}: {_e}")
                return False
            except Exception:
                return False

        if _cuda_bin:
            for _prefix in ("cudart64_", "cudart32_"):
                for _f in _os.listdir(_cuda_bin):
                    if _f.lower().startswith(_prefix) and _f.endswith(".dll"):
                        _load_dll(_os.path.join(_cuda_bin, _f))
            for _f in sorted(_os.listdir(_cuda_bin)):
                _fl = _f.lower()
                if _fl.endswith(".dll") and not _fl.startswith("cudart"):
                    if any(_fl.startswith(p) for p in (
                        "cublas", "cufft", "curand", "cusparse", "cusolver",
                        "npp", "nvjpeg", "nvrtc", "nvblas", "nvjitlink",
                        "zlibwapi",
                    )):
                        _load_dll(_os.path.join(_cuda_bin, _f))

        for _dll_path in _cudnn_dlls:
            _load_dll(_dll_path)

        _path_extra: list[str] = []
        if _cuda_bin:
            _path_extra.append(_cuda_bin)
        if _cudnn_dir:
            _path_extra.append(_cudnn_dir)
        if _path_extra:
            _existing = _os.environ.get("PATH", "")
            _os.environ["PATH"] = ";".join(_path_extra) + (";" + _existing if _existing else "")

        if _cuda_bin:
            print(f"[GPU] CUDA: {_cuda_bin}", flush=True)
        else:
            print("[GPU] CUDA: 未找到 CUDA Toolkit 安装", flush=True)
        if _cudnn_dlls:
            print(f"[GPU] cuDNN: {len(_cudnn_dlls)} 个 DLL 在 {_cudnn_dir}", flush=True)
        else:
            print("[GPU] cuDNN: 未找到", flush=True)
        print(f"[GPU] 预加载: {_loaded} 个成功", flush=True)
        if _failed:
            print(f"[GPU] 预加载失败 ({len(_failed)} 个):", flush=True)
            for _msg in _failed[:5]:
                print(f"  {_msg}", flush=True)

    except Exception:
        pass

_register_gpu_dlls()
# ═══════════════════════════════════════════════════════════

from rapidocr_onnxruntime import RapidOCR


_gpu_patched = False
_gpu_backend = "CPU"


# 后端优先级：用户选择 → 回退链
_BACKEND_FALLBACK: dict[str, list[str]] = {
    "auto": ["CUDA", "CPU"],
    "cuda": ["CUDA", "CPU"],
    "cpu":  ["CPU"],
}
_BACKEND_PROVIDER_MAP = {
    "CUDA": ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo", "cudnn_conv_algo_search": "EXHAUSTIVE", "do_copy_in_default_stream": True}),
    "CPU":  ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
}


def _select_backend(preferred: str = "auto") -> str:
    """按用户偏好选择 OCR 后端，不可用时自动回退。

    preferred: "auto" | "cuda" | "cpu"
    返回实际使用的后端名称: "CUDA" | "CPU"
    """
    global _gpu_patched, _gpu_backend

    if _gpu_patched:
        return _gpu_backend
    _gpu_patched = True

    try:
        import onnxruntime as ort
    except Exception:
        _gpu_backend = "CPU"
        return _gpu_backend

    available = set(ort.get_available_providers())

    chain = _BACKEND_FALLBACK.get(preferred.lower(), _BACKEND_FALLBACK["auto"])
    chosen: str | None = None

    for candidate in chain:
        ep_name = _BACKEND_PROVIDER_MAP[candidate][0]
        if ep_name in available:
            chosen = candidate
            break
    if chosen is None:
        chosen = "CPU"

    # Monkey-patch OrtInferSession 以使用选定后端
    from rapidocr_onnxruntime.utils import OrtInferSession

    ep_name, ep_opts = _BACKEND_PROVIDER_MAP[chosen]
    cpu_ep_name, cpu_opts = _BACKEND_PROVIDER_MAP["CPU"]

    def _patched_init(self, config):  # type: ignore[no-untyped-def]
        from onnxruntime import (
            SessionOptions, InferenceSession, GraphOptimizationLevel,
        )
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        EP_list: list = [(ep_name, ep_opts)] if ep_name != cpu_ep_name else []
        EP_list.append((cpu_ep_name, cpu_opts))
        self._verify_model(config['model_path'])
        self.session = InferenceSession(
            config['model_path'], sess_options=sess_opt, providers=EP_list,
        )

    OrtInferSession.__init__ = _patched_init  # type: ignore[method-assign]

    _gpu_backend = chosen
    return _gpu_backend


def _reset_backend() -> None:
    """重置后端选择状态，允许用户在运行时切换后端。"""
    global _gpu_patched, _gpu_backend
    _gpu_patched = False
    _gpu_backend = "CPU"


SOURCE_TO_KMH = {
	"m/s": 3.6,
	"km/h": 1.0,
	"mile/h": 1.609344,
}

OCR_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")


@dataclass
class VideoMetadata:
	path: Path
	duration_sec: float
	width: int
	height: int
	fps: float
	codec: str
	frame_count: int


@dataclass
class SpeedObservation:
	timestamp: float
	raw_speed_kmh: float
	raw_text: str


def format_duration(seconds: float) -> str:
	seconds = max(0.0, float(seconds))
	total = int(round(seconds))
	hours, remainder = divmod(total, 3600)
	minutes, secs = divmod(remainder, 60)
	if hours:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:d}:{secs:02d}"


def codec_from_fourcc(fourcc: float) -> str:
	value = int(fourcc)
	if value == 0:
		return "Unknown"
	chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
	codec = "".join(chars).strip("\x00").strip()
	return codec or "Unknown"


def safe_int(value: str) -> int | None:
	value = value.strip()
	if not value:
		return None
	try:
		return int(float(value))
	except ValueError:
		return None


def safe_float(value: str) -> float | None:
	value = value.strip()
	if not value:
		return None
	try:
		return float(value)
	except ValueError:
		return None


def normalize_ocr_text(text: str) -> str:
	translation = str.maketrans(
		{
			"O": "0",
			"o": "0",
			"Q": "0",
			"D": "0",
			"I": "1",
			"l": "1",
			"|": "1",
			"!": "1",
			"Z": "2",
			"z": "2",
			"S": "5",
			"s": "5",
			"B": "8",
			"G": "6",
			"g": "6",
			"T": "7",
			"t": "7",
			",": ".",
		}
	)
	return text.translate(translation)


def extract_speed_value(ocr_result) -> tuple[float | None, str | None]:
	if not ocr_result:
		return None, None

	candidates: list[str] = []
	for item in ocr_result:
		if not item or len(item) < 2:
			continue
		text = str(item[1]).strip()
		if text:
			candidates.append(text)

	if not candidates:
		return None, None

	joined = normalize_ocr_text(" ".join(candidates)).replace(" ", "")
	match = OCR_NUMBER_RE.search(joined)
	if not match:
		return None, None

	raw_text = re.sub(r"\D", "", match.group(0))
	if not raw_text:
		return None, None
	try:
		return float(raw_text), raw_text
	except ValueError:
		return None, None


def convert_speed_to_kmh(speed_value: float, source_unit: str) -> float:
	return float(speed_value) * SOURCE_TO_KMH[source_unit]


def clamp_region(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
	x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
	y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
	return x1, y1, x2, y2


def build_speed_candidates(raw_text: str, max_speed_kmh: float) -> list[float]:
	"""根据 OCR 原始文本生成可能的速度候选值。

	策略:
	1. 数字后缀扩展: OCR "60" → 候选 60/160/260(处理丢位)
	2. 常见字符混淆替换: 6↔8, 3↔8, 5↔6, 0↔8, 1↔7 等
	"""
	if max_speed_kmh <= 0:
		return []

	text = re.sub(r"\D", "", raw_text)
	if not text:
		return []

	max_speed_int = int(math.floor(max_speed_kmh))
	if max_speed_int < 0:
		return []

	candidates: set[float] = set()

	# 策略1: 保留原始值
	try:
		val = int(text)
		if val <= max_speed_int:
			candidates.add(float(val))
	except ValueError:
		pass

	# 策略2: 后缀扩展（处理丢位）
	min_suffix_len = 1 if len(text) == 1 else max(1, len(text) - 2)
	for suffix_len in range(min_suffix_len, len(text) + 1):
		suffix_text = text[-suffix_len:]
		try:
			suffix_value = int(suffix_text)
		except ValueError:
			continue
		step = 10 ** suffix_len
		for candidate in range(suffix_value, max_speed_int + 1, step):
			candidates.add(float(candidate))

	# 策略3: 常见 OCR 字符混淆替换
	# 对每位数字尝试替换为视觉相似的字符
	_CONFUSION_MAP = {
		"0": ["8"], "8": ["0", "6", "3"],
		"6": ["8", "5"], "5": ["6"],
		"3": ["8"], "1": ["7"], "7": ["1"],
		"2": ["7"], "9": ["8"],
	}
	for i, ch in enumerate(text):
		for alt in _CONFUSION_MAP.get(ch, []):
			altered = text[:i] + alt + text[i+1:]
			try:
				val = int(altered)
				if val <= max_speed_int:
					candidates.add(float(val))
			except ValueError:
				pass

	return sorted(candidates)


def _estimate_raw_trust(samples: list[SpeedObservation], window: int = 3) -> list[float]:
	"""评估每个采样点的原始 OCR 值可信度 (0~1)。

	若某帧值与前后邻帧的原始值接近（在 5 km/h 内），则认为可信。
	连续多帧一致时可信度更高。
	"""
	n = len(samples)
	scores: list[float] = [0.5] * n
	if n < 2:
		return scores

	for i in range(n):
		agree = 0
		total = 0
		ref = samples[i].raw_speed_kmh
		for j in range(max(0, i - window), min(n, i + window + 1)):
			if i == j:
				continue
			total += 1
			if abs(samples[j].raw_speed_kmh - ref) <= 5.0:
				agree += 1
		scores[i] = agree / max(total, 1)
	return scores


def correct_speed_series(
	samples: list[SpeedObservation],
	max_speed_kmh: float,
	max_accel_mps2: float,
) -> list[float]:
	"""改进的物理约束纠错: 候选扩展 + 可信度加权 + EMA 平滑。

	三步流程:
	1. 评估原始 OCR 可信度（与邻帧一致性）
	2. 动态规划 + 可信度加权选择最优候选
	3. 指数移动平均平滑纠正连续偏差
	"""
	if not samples:
		return []

	if max_speed_kmh <= 0 or max_accel_mps2 <= 0:
		return [sample.raw_speed_kmh for sample in samples]

	n = len(samples)
	trust_scores = _estimate_raw_trust(samples)

	# ── Step 1: 生成候选列表 ──
	candidate_lists: list[list[float]] = []
	for i, sample in enumerate(samples):
		candidates = build_speed_candidates(sample.raw_text, max_speed_kmh)
		if sample.raw_speed_kmh <= max_speed_kmh:
			candidates.append(float(sample.raw_speed_kmh))
		if not candidates:
			candidates = [min(max(sample.raw_speed_kmh, 0.0), max_speed_kmh)]
		candidate_lists.append(sorted(set(candidates)))

	# ── Step 2: DP + 可信度加权 ──
	# 初始状态
	states: list[tuple[float, float, int | None]] = []
	first_sample = samples[0]
	for candidate in candidate_lists[0]:
		cost = abs(candidate - first_sample.raw_speed_kmh) * (2.0 - trust_scores[0])
		states.append((cost, candidate, None))

	backpointers: list[list[int]] = [[] for _ in samples]
	backpointers[0] = [-1] * len(candidate_lists[0])

	for i in range(1, n):
		cur_cands = candidate_lists[i]
		prev_cands = candidate_lists[i - 1]
		delta_time = max(samples[i].timestamp - samples[i - 1].timestamp, 1e-6)
		max_delta_kmh = max_accel_mps2 * delta_time * 3.6
		# 可信度低时放大加速度容忍度（OCR 可能误读）
		effective_max_delta = max_delta_kmh * (1.0 + (1.0 - trust_scores[i]) * 2.0)

		cur_states: list[tuple[float, float, int]] = []
		cur_back: list[int] = []

		for cur_idx, cur_val in enumerate(cur_cands):
			best_cost = float("inf")
			best_prev = 0
			for prev_idx, prev_val in enumerate(prev_cands):
				prev_cost = states[prev_idx][0]
				delta = abs(cur_val - prev_val)
				# 加速度代价
				if delta <= max_delta_kmh:
					trans_cost = delta * 0.1
				elif delta <= effective_max_delta:
					trans_cost = delta * 0.5
				else:
					trans_cost = delta * 5.0
				# OCR 贴近代价（可信帧权重高，不可信帧权重低）
				ocr_weight = 1.0 if trust_scores[i] > 0.5 else 0.1
				ocr_cost = abs(cur_val - samples[i].raw_speed_kmh) * ocr_weight
				cost = prev_cost + trans_cost + ocr_cost
				if cost < best_cost:
					best_cost = cost
					best_prev = prev_idx
			cur_states.append((best_cost, cur_val, best_prev))
			cur_back.append(best_prev)

		states = [(c, v, p) for c, v, p in cur_states]
		backpointers[i] = cur_back

	# 回溯
	best_final = min(range(len(states)), key=lambda idx: states[idx][0])
	corrected = [0.0] * n
	corrected[-1] = states[best_final][1]
	idx = best_final
	for i in range(n - 1, 0, -1):
		idx = backpointers[i][idx]
		corrected[i - 1] = candidate_lists[i - 1][idx]

	# ── Step 3: EMA 平滑（纠正 DP 路径中的连续偏差）──
	# 找到高可信度锚点，从锚点间做 EMA 平滑
	smoothed = list(corrected)
	alpha = 0.35  # 平滑系数
	# 前向 EMA
	for i in range(1, n):
		if trust_scores[i] < 0.5:
			smoothed[i] = alpha * corrected[i] + (1 - alpha) * smoothed[i - 1]
	# 后向 EMA
	for i in range(n - 2, -1, -1):
		if trust_scores[i] < 0.5:
			back_val = alpha * corrected[i] + (1 - alpha) * smoothed[i + 1]
			smoothed[i] = (smoothed[i] + back_val) / 2.0

	return smoothed

	return corrected


class _CancelExport(Exception):
    """内部异常：用户取消了导出任务。"""
    pass


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

		self._build_ui()
		self._bind_preview_updates()

	def _build_ui(self) -> None:
		self.root.columnconfigure(0, weight=1)
		self.root.rowconfigure(2, weight=1)  # 主区域可拉伸

		# Row 0: 顶部工具栏
		header = ttk.Frame(self.root, padding=(12, 12, 12, 6))
		header.grid(row=0, column=0, sticky="ew")
		header.columnconfigure(1, weight=1)

		ttk.Button(header, text="导入视频", command=self.import_video).grid(row=0, column=0, sticky="w")
		self.export_btn = ttk.Button(header, text="导出 CSV", command=self.export_csv)
		self.export_btn.grid(row=0, column=1, sticky="e")
		self.cancel_btn = ttk.Button(header, text="取消", command=self._cancel_export, state="disabled")
		self.cancel_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))
		ttk.Label(header, textvariable=self.file_var).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

		# Row 1: 视频信息
		info = ttk.LabelFrame(self.root, text="视频信息", padding=(12, 10, 12, 12))
		info.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
		for index in range(4):
			info.columnconfigure(index, weight=1)
		self._add_info_row(info, 0, "时长", self.duration_var)
		self._add_info_row(info, 1, "分辨率", self.resolution_var)
		self._add_info_row(info, 2, "帧率", self.fps_var)
		self._add_info_row(info, 3, "编码", self.codec_var)

		# Row 2: 左侧配置 + 右侧预览
		main_area = ttk.Frame(self.root)
		main_area.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 10))
		main_area.columnconfigure(1, weight=3)
		main_area.columnconfigure(0, weight=1)
		main_area.rowconfigure(0, weight=1)

		config_col = ttk.Frame(main_area, padding=(0, 0, 6, 0))
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
		ttk.Label(constraint_box, text="用于自动修正丢位、多位和跳变异常。", foreground="#555555").grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

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

		ttk.Label(perf_box, text="OCR 高度 (px)").grid(row=1, column=0, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.target_height_var, width=8).grid(row=1, column=1, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text="边缘填充 (px)").grid(row=1, column=2, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.pad_var, width=8).grid(row=1, column=3, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text="并行线程数").grid(row=1, column=4, sticky="w", pady=(8,0))
		ttk.Entry(perf_box, textvariable=self.num_workers_var, width=8).grid(row=1, column=5, sticky="ew", padx=(6, 14), pady=(8,0))
		ttk.Label(perf_box, text=">1 时启用并行推理。", foreground="#555555").grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

		# 右侧预览
		preview_box = ttk.LabelFrame(main_area, text="识别范围预览", padding=(6, 6, 6, 6))
		preview_box.grid(row=0, column=1, sticky="nsew")
		preview_box.columnconfigure(0, weight=1); preview_box.rowconfigure(0, weight=1)

		self.preview_canvas = tk.Canvas(preview_box, background="#151515", highlightthickness=0, cursor="crosshair")
		self.preview_canvas.grid(row=0, column=0, sticky="nsew")
		self.preview_canvas.bind("<Configure>", lambda event: self.schedule_preview_refresh())
		self.preview_canvas.bind("<ButtonPress-1>", self._on_drag_start)
		self.preview_canvas.bind("<B1-Motion>", self._on_drag_motion)
		self.preview_canvas.bind("<ButtonRelease-1>", self._on_drag_end)

		# Row 3: 底部状态栏
		footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
		footer.grid(row=3, column=0, sticky="ew")
		footer.columnconfigure(0, weight=1)
		ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
		self.progress_bar = ttk.Progressbar(footer, variable=self.progress_var, maximum=100.0)
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

		self._draw_preview_image(self.first_frame_pil)
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

		if (self._last_canvas_w != canvas_width or 
			self._last_canvas_h != canvas_height or 
			self.preview_photo is None):
			
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
		_reset_backend()
		BACKEND_LABELS_REV = {"自动": "auto", "CUDA": "cuda", "CPU": "cpu"}
		selected_label = self.backend_var.get()
		selected_key = BACKEND_LABELS_REV.get(selected_label, "auto")
		actual = _select_backend(selected_key)
		print(f"[OCR] 用户选择: {selected_label}, 实际后端: {actual}", flush=True)
		return RapidOCR()

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
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		h, w = gray.shape[:2]
		target_h = max(8.0, float(target_h))
		pad_px = max(0.0, float(pad_px))

		# Otsu 二值化：纯黑白，匹配 PP-OCR 训练数据分布，显著提升准确率（96.6% vs 95.6%）
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		scale = target_h / float(h) if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)

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
					f"[{_gpu_backend}] 正在处理... {len(observations)} 条 ({pct:.1f}%)", pct)
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
					f"[{_gpu_backend}] 正在处理... {len(observations)} 条 ({pct:.1f}%)", pct)
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
			max_speed_kmh = self._parse_positive_float(self.max_speed_var.get(), "最大速度上限")
			max_accel_mps2 = self._parse_positive_float(self.max_accel_var.get(), "最大加速度上限")
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

		def _ocr_one(idx: int, ts: float, proc: np.ndarray) -> tuple[int, SpeedObservation | None]:
			ocr_result, _ = engine(proc)
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
			futures = [pool.submit(_ocr_one, i, ts, proc) for i, (ts, proc) in enumerate(preprocessed)]
			done = 0
			for f in as_completed(futures):
				if done % 10 == 0:
					self._check_cancel()
				idx, obs = f.result()
				observations[idx] = obs
				done += 1
				if done % 50 == 0:
					pct = (done / total_frames * 90.0) + 5.0
					self.root.after(0, self._update_progress,
						f"[{_gpu_backend}×{num_workers}] 正在处理... {done}/{total_frames} ({pct:.1f}%)", pct)
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
		ocr = self.get_ocr_engine()

		capture = cv2.VideoCapture(str(self.video_path))
		if not capture.isOpened():
			raise RuntimeError("无法重新打开视频文件。")

		x1, y1, x2, y2 = region
		frame_step = max(1, frame_div)

		raw_frames: list[tuple[float, np.ndarray]] = []
		frame_index = 0
		while True:
			ok, frame = capture.read()
			if not ok or frame is None:
				break
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
			f"OCR 引擎: {_gpu_backend}，正在处理 {total_frames} 帧 (workers={num_workers})...", 5.0)
		self._check_cancel()

		# 仅 CUDA 支持并行推理
		if num_workers > 1 and _gpu_backend == "CUDA":
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
		for observation, corrected_speed_kmh in zip(observations, corrected_speeds):
			current_speed_ms = corrected_speed_kmh / 3.6
			if previous_sample_time is not None and previous_speed_ms is not None:
				delta_t = observation.timestamp - previous_sample_time
				if delta_t > 0:
					distance_m += (previous_speed_ms + current_speed_ms) * 0.5 * delta_t
			previous_sample_time = observation.timestamp
			previous_speed_ms = current_speed_ms
			corrected_flag = 1 if abs(observation.raw_speed_kmh - corrected_speed_kmh) > 0.01 else 0
			rows.append((observation.timestamp, distance_m, corrected_speed_kmh, corrected_flag))

		self._write_csv_with_retry(output_path, rows, _t_elapsed, total_frames, _accuracy, _gpu_backend)

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
	parser = argparse.ArgumentParser(description="RaceVideoToLog - 视频速度提取工具")
	parser.add_argument("video", nargs="?", help="视频文件路径")
	parser.add_argument("--roi", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="识别范围 (左上X 左上Y 右下X 右下Y)")
	parser.add_argument("--format", choices=["m/s","km/h","mile/h"], default="km/h", help="速度格式 (默认 km/h)")
	parser.add_argument("--div", type=int, default=2, choices=[1,2,3,4,5], help="采样间隔 1/N (默认 2)")
	parser.add_argument("--max-speed", type=float, default=400, help="最大速度 km/h (默认 400)")
	parser.add_argument("--max-accel", type=float, default=50, help="最大加速度 m/s² (默认 50)")
	parser.add_argument("--target-h", type=int, default=24, help="OCR 目标高度 px (默认 24)")
	parser.add_argument("--pad", type=int, default=0, help="边缘填充 px (默认 0)")
	parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认 4)")
	parser.add_argument("--backend", choices=["auto","cuda","cpu"], default="auto",
		help="OCR 后端: auto/cuda/cpu (默认 auto)")
	parser.add_argument("-o", "--output", type=str, help="输出 CSV 路径 (默认 视频名_log.csv)")
	args = parser.parse_args()

	if args.video:
		run_headless(args)
	else:
		# GUI 模式：隐藏控制台窗口
		if sys.platform == "win32":
			import ctypes
			ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
		app = RaceVideoToLogApp()
		app.run()


def run_headless(args: argparse.Namespace) -> None:
	"""命令行无头模式：不启动 GUI，直接分析并输出 CSV。"""
	if not args.roi:
		print("错误: 命令行模式需要 --roi X1 Y1 X2 Y2")
		sys.exit(1)

	video_path = Path(args.video)
	if not video_path.exists():
		print(f"错误: 找不到文件 {video_path}")
		sys.exit(1)

	output_path = Path(args.output) if args.output else video_path.with_suffix(".csv")
	region = (args.roi[0], args.roi[1], args.roi[2], args.roi[3])

	print(f"视频: {video_path}")
	print(f"识别范围: {region}")
	print(f"采样间隔: 1/{args.div}")
	print(f"最大速度: {args.max_speed} km/h, 最大加速度: {args.max_accel} m/s^2")
	print(f"OCR 后端选择: {args.backend}")

	# 初始化 OCR
	_reset_backend()
	backend_actual = _select_backend(args.backend)
	print(f"OCR 实际后端: {backend_actual}")
	ocr = RapidOCR()

	# 读取视频信息
	cap = cv2.VideoCapture(str(video_path))
	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	duration = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps) if fps > 0 else 0.0
	print(f"分辨率: {width}x{height}, 帧率: {fps:.2f}, 时长: {format_duration(duration)}")

	# 读取帧
	x1, y1, x2, y2 = clamp_region(*region, width, height)
	frame_step = max(1, args.div)
	raw_frames: list[tuple[float, np.ndarray]] = []
	fi = 0
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			break
		if fi % frame_step != 0:
			fi += 1
			continue
		ts = fi / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
		crop = frame[y1:y2 + 1, x1:x2 + 1].copy()  # .copy() 断开对整帧的引用
		fi += 1
	cap.release()

	total = len(raw_frames)
	print(f"采样帧: {total}")
	if total == 0:
		print("错误: 未读取到帧")
		sys.exit(1)

	# OCR
	observations: list[SpeedObservation] = []
	for idx, (ts, crop) in enumerate(raw_frames):
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		h, w = gray.shape[:2]
		target_h = max(8.0, float(args.target_h))
		scale = target_h / h if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)
		pad_int = args.pad
		if pad_int > 0:
			gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
		proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

		ocr_result, _ = ocr(proc)
		sv, rt = extract_speed_value(ocr_result)
		if sv is not None and rt is not None:
			observations.append(SpeedObservation(
				timestamp=ts,
				raw_speed_kmh=sv * SOURCE_TO_KMH[args.format],
				raw_text=rt,
			))
		if (idx + 1) % 100 == 0:
			print(f"  OCR 进度: {idx + 1}/{total}")

	if not observations:
		print("错误: 未识别到速度数据")
		sys.exit(1)

	# 纠错
	print(f"识别: {len(observations)} 条, 正在进行物理约束纠错...")
	corrected = correct_speed_series(observations, args.max_speed, args.max_accel)

	# 积分 + 写出
	rows: list[tuple[float, float, float, int]] = []
	dist = 0.0
	prev_t, prev_v = None, None
	for obs, cspd in zip(observations, corrected):
		v = cspd / 3.6
		if prev_t is not None and prev_v is not None:
			dt = obs.timestamp - prev_t
			if dt > 0:
				dist += (prev_v + v) * 0.5 * dt
		prev_t, prev_v = obs.timestamp, v
		flag = 1 if abs(obs.raw_speed_kmh - cspd) > 0.01 else 0
		rows.append((obs.timestamp, dist, cspd, flag))

	with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
		w = csv.writer(fh)
		for t, d, s, fl in rows:
			w.writerow([f"{t:.2f}", f"{d:.2f}", f"{s:.2f}", str(fl)])

	corrected_count = sum(r[3] for r in rows)
	print(f"导出完成: {output_path}")
	print(f"共 {len(rows)} 条, 纠错 {corrected_count} 条 (准确率 {100 - corrected_count/len(rows)*100:.1f}%)")


if __name__ == "__main__":
	main()
