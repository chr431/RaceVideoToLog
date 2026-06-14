"""OCR engine for RaceVideoToLog.

SpeedObservation, preprocessing, correction algorithms,
model configuration, and supporting utilities.
"""
from __future__ import annotations
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

try:
	import matplotlib
	matplotlib.use("TkAgg")
	matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
	matplotlib.rcParams["axes.unicode_minus"] = False
	from matplotlib.figure import Figure
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
	pass

# ── 导出列表：包含 _ 前缀的私有符号供 RaceVideoToLog.py / headless.py 使用 ──
__all__ = [
	"SpeedObservation", "VideoMetadata", "RapidOCR",
	"extract_speed_value", "convert_speed_to_kmh", "clamp_region",
	"correct_speed_series", "correct_speed_series_v2", "build_speed_candidates",
	"normalize_ocr_text", "format_duration", "codec_from_fourcc",
	"safe_int", "safe_float", "SOURCE_TO_KMH", "OCR_NUMBER_RE",
	"ocr_digital_fallback", "compute_video_hash",
	"_reset_backend", "_select_backend", "_get_model_kwargs",
	"_gpu_backend", "_gpu_patched", "_CancelExport",
	"_parse_int_or_none", "_estimate_raw_trust", "_savgol_filter_np",
	"_set_rec_keys_path",
]

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


def _parse_int_or_none(s: str) -> int | None:
	"""解析字符串为 int，空字符串返回 None。"""
	s = s.strip()
	if not s:
		return None
	try:
		return int(s)
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


def ocr_digital_fallback(
	ocr, crop_bgr, max_speed_kmh=400
) -> tuple[float | None, str | None]:
	"""数字仪表 OCR 后备链：CLAHE+OTSU → 常规检测 → 无检测模式。

	用于 PP-OCR 标准预处理未命中时的后备策略（如赛车 HUD 仪表字体）。
	返回 (speed_value, raw_text) 或 (None, None)。
	"""
	gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

	# ── 策略1: CLAHE + OTSU + 常规检测 ──
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(gray)
	_, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	h, w = enhanced.shape[:2]
	for th in (28, 32, 48):
		scale = th / h
		resized = cv2.resize(enhanced, (max(1, int(w * scale)), th))
		bgr_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
		try:
			result, _ = ocr(bgr_input)
			sv, rt = extract_speed_value(result)
			if sv is not None and sv <= max_speed_kmh:
				return sv, rt
		except Exception:
			pass

	# ── 策略2: use_det=False（跳过检测，多预处理变体）──
	variants = [
		("clahe_otsu", enhanced),
		("inv", cv2.bitwise_not(gray)),
		("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
		("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
	]
	for _label, img in variants:
		for th in (32, 48):
			scale = th / h
			resized = cv2.resize(img, (max(1, int(w * scale)), th))
			bgr_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
			try:
				result, _ = ocr(bgr_input, use_det=False)
				sv, rt = extract_speed_value(result)
				if sv is not None and sv <= max_speed_kmh:
					return sv, rt
			except Exception:
				pass

	return None, None


def convert_speed_to_kmh(speed_value: float, source_unit: str) -> float:
	return float(speed_value) * SOURCE_TO_KMH[source_unit]


def clamp_region(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
	x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
	y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
	return x1, y1, x2, y2


def _savgol_filter_np(y, window_length, polyorder):
	"""纯 numpy Savitzky-Golay 滤波，等价于 scipy.signal.savgol_filter。"""
	if window_length % 2 == 0 or window_length < 1:
		raise ValueError("window_length must be odd")
	if window_length <= polyorder:
		raise ValueError("window_length must be > polyorder")
	half = window_length // 2
	y = np.asarray(y, dtype=float)
	n = len(y)
	if n < window_length:
		return y.copy()
	x_full = np.arange(-half, half + 1, dtype=float)
	result = np.zeros(n)
	for i in range(n):
		lo = max(0, i - half)
		hi = min(n, i + half + 1)
		y_seg = y[lo:hi]
		if len(y_seg) < window_length:
			result[i] = y[i]
		else:
			x_seg = x_full[lo - (i - half):hi - (i - half)]
			A = np.vander(x_seg, polyorder + 1, increasing=True)
			coeffs = np.linalg.lstsq(A, y_seg, rcond=None)[0]
			result[i] = np.polyval(coeffs[::-1], 0)
	return result


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

	# 策略3: 常见 OCR 字符混淆替换（对称映射）
	_CONFUSION_MAP = {
		"0": ["8", "6", "9"],
		"1": ["7", "2"],
		"2": ["7", "1", "3"],
		"3": ["8", "9", "2", "5"],
		"4": ["7", "9"],
		"5": ["6", "3", "8", "9"],
		"6": ["8", "5", "0", "2"],
		"7": ["1", "2", "4"],
		"8": ["0", "6", "3", "5", "9"],
		"9": ["8", "3", "5", "0", "4"],
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


def _get_model_kwargs(variant: str, models_dir: str | None = None) -> dict | None:
	"""根据 OCR 模型变体返回 RapidOCR 的 kwargs。模型文件不存在时返回 None。"""
	import rapidocr_onnxruntime as rr
	if models_dir is None:
		models_dir = str(Path(rr.__file__).parent / "models")
	# 模型路径（仅 onnx 文件）
	_PATH_KEYS = {"det_model_path", "rec_model_path"}
	model_paths: dict[str, dict] = {
		"v3": {},
		"v3_server": {"det_model_path": f"{models_dir}/ch_PP-OCRv3_det_server_infer.onnx",
			"rec_model_path": f"{models_dir}/ch_PP-OCRv3_rec_server_infer.onnx"},

		"v4_server": {"det_model_path": f"{models_dir}/ch_PP-OCRv4_det_server_infer.onnx",
			"rec_model_path": f"{models_dir}/ch_PP-OCRv4_rec_server_infer.onnx"},

		"v5_mobile": {"det_model_path": f"{models_dir}/ch_PP-OCRv5_mobile_det_infer.onnx",
			"rec_model_path": f"{models_dir}/ch_PP-OCRv5_mobile_rec_infer.onnx",

			"text_score": 0.6, "use_angle_cls": False, "rec_batch_num": 12},

	}
	cfg = model_paths.get(variant)
	if cfg is None:
		return None
	if not cfg:  # v3 默认
		return None
	for key in _PATH_KEYS:
		if key in cfg and not Path(cfg[key]).exists():
			return None
	if variant.startswith("v5"):
		_set_rec_keys_path(str(Path(rr.__file__).parent / "config.yaml"),
			f"{models_dir}/ppocr_keys_v1.txt")

	return cfg


def _set_rec_keys_path(config_path: str, keys_path: str) -> None:
	"""临时修改 rapidocr config.yaml 的 Rec.keys_path。"""
	from rapidocr_onnxruntime.utils import read_yaml
	config = read_yaml(config_path)
	if config.get("Rec", {}).get("keys_path") == keys_path:
		return  # 已设置
	config.setdefault("Rec", {})["keys_path"] = keys_path
	import yaml
	with open(config_path, "w") as f:
		yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def compute_video_hash(video_path: str | Path, chunk_size: int = 1_048_576) -> str:
	"""计算视频文件的快速哈希（头尾各 1MB + 文件大小）。

	使用 SHA-256，足以唯一标识视频文件，同时避免读取整个大文件。
	"""
	import hashlib
	video_path = Path(video_path)
	if not video_path.exists():
		return "N/A"
	file_size = video_path.stat().st_size
	h = hashlib.sha256()
	h.update(str(file_size).encode())
	with open(video_path, "rb") as f:
		h.update(f.read(chunk_size))
		if file_size > chunk_size * 2:
			f.seek(-chunk_size, 2)
			h.update(f.read(chunk_size))
	return h.hexdigest()[:16]  # 前 16 字符足够区分




def auto_select_anchors(observations, max_speed_kmh=400.0, window=7, max_dev=5.0):
	"""Select reliable OCR frames as Correction B anchors.
	Uses local median filter: for each frame, compute median in a sliding
	window. If frame value deviates <= max_dev from median, it is reliable.
	Returns set of trusted frame indices."""
	n = len(observations)
	raw_vals = [o.raw_speed_kmh for o in observations]
	anchors = set()
	half = window // 2

	for i in range(half, n - half):
		if raw_vals[i] <= 0:
			continue
		local = []
		for j in range(i - half, i + half + 1):
			if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh:
				local.append(raw_vals[j])
		if len(local) < 3:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	# Head boundary frames
	for i in range(0, half):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(0, min(window, n))
		         if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	# Tail boundary frames
	for i in range(n - half, n):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(max(0, n - window), n)
		         if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	return anchors

def correct_speed_series(
	samples: list[SpeedObservation],
	max_speed_kmh: float,
	max_accel_mps2: float,
) -> list[float]:
	"""物理约束纠错: 仅纠正物理上不可能的帧。

	策略（保守）：
	1. 扫描找出物理上不一致的帧（与前后邻帧的加速度超限）
	2. 将连续不一致帧分组为"可疑段"
	3. 对每个可疑段运行 DP，段两端的可信帧作为锚点固定
	4. 非可疑帧保持原始 OCR 值不变
	"""
	if not samples:
		return []

	if max_speed_kmh <= 0 or max_accel_mps2 <= 0:
		return [sample.raw_speed_kmh for sample in samples]

	n = len(samples)
	if n < 2:
		return [s.raw_speed_kmh for s in samples]

	# ── Step 1: 多帧可达性扫描 ──
	# 从首帧出发，若存在任意可信前驱帧使得加速度在限制内，则本帧可达
	# 从尾帧出发同理。只有双向都不可达的帧才标记可疑。
	raw_vals = [s.raw_speed_kmh for s in samples]
	times = [s.timestamp for s in samples]

	can_reach_fwd = [False] * n
	can_reach_bwd = [False] * n
	can_reach_fwd[0] = True
	can_reach_bwd[-1] = True

	# 前向多帧可达（最多 10 帧，中间帧必须一致）
	REACH_WINDOW = 10
	for i in range(1, n):
		if raw_vals[i] > max_speed_kmh:
			continue
		for j in range(i - 1, max(-1, i - REACH_WINDOW - 1), -1):
			if can_reach_fwd[j]:
				dt = times[i] - times[j]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6
				# 显示更新：当前帧与下一帧同值 → 放宽容限
				if i + 1 < n and raw_vals[i] == raw_vals[i + 1]:
					max_dv = max_dv * 2 + 10.0
				if j < i - 1:
					# 跳帧查找：中间帧必须能从 j 到达
					mid_ok = True
					for k in range(j + 1, i):
						if raw_vals[k] > max_speed_kmh:
							continue
						# 中间帧也必须能从 j 到达（或显示更新）
						dt_k = times[k] - times[j]
						if dt_k > 0:
							mid_max = max_accel_mps2 * dt_k * 3.6
							if k + 1 < n and raw_vals[k] == raw_vals[k + 1]:
								mid_max = mid_max * 2 + 10.0
							if abs(raw_vals[k] - raw_vals[j]) > mid_max:
								mid_ok = False
								break
					if not mid_ok:
						continue
				if abs(raw_vals[i] - raw_vals[j]) <= max_dv:
					can_reach_fwd[i] = True
					break

	# 后向多帧可达（最多 10 帧，中间帧必须一致）
	for i in range(n - 2, -1, -1):
		if raw_vals[i] > max_speed_kmh:
			continue
		for j in range(i + 1, min(n, i + REACH_WINDOW + 1)):
			if can_reach_bwd[j]:
				dt = times[j] - times[i]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6
				# 显示更新检测
				if i + 1 < n and raw_vals[i] == raw_vals[i + 1]:
					max_dv = max_dv * 2 + 10.0
				if j > i + 1:
					mid_ok = True
					for k in range(i + 1, j):
						if raw_vals[k] > max_speed_kmh:
							continue
						dt_k = times[j] - times[k]
						if dt_k > 0:
							mid_max = max_accel_mps2 * dt_k * 3.6
							if k + 1 < n and raw_vals[k] == raw_vals[k + 1]:
								mid_max = mid_max * 2 + 10.0
							if abs(raw_vals[j] - raw_vals[k]) > mid_max:
								mid_ok = False
								break
					if not mid_ok:
						continue
				if abs(raw_vals[j] - raw_vals[i]) <= max_dv:
					can_reach_bwd[i] = True
					break

	# 双向都不可达 → 可疑。但显示稳定帧（与邻居同值）永不标记。
	suspect = [False] * n
	for i in range(n):
		reach_fail = not can_reach_fwd[i] and not can_reach_bwd[i]
		if not reach_fail:
			continue
		stable = (i > 0 and raw_vals[i] == raw_vals[i - 1]) or (i + 1 < n and raw_vals[i] == raw_vals[i + 1])
		if not stable:
			suspect[i] = True

	# ── 紧邻跳变检测：即使双向可达，若紧邻跳变超限且跳变源为孤立帧 → 标记 ──
	for i in range(1, n):
		dt = times[i] - times[i - 1]
		if dt <= 0:
			continue
		max_dv = max_accel_mps2 * dt * 3.6
		dv = abs(raw_vals[i] - raw_vals[i - 1])
		if dv <= max_dv:
			continue
		# 跳变超限。判断哪一侧是孤立帧（与邻居不一致）
		i_left_isolated = (i - 2 >= 0 and raw_vals[i - 1] != raw_vals[i - 2])
		i_right_isolated = (i + 1 < n and raw_vals[i] != raw_vals[i + 1])
		if i_left_isolated and not i_right_isolated:
			suspect[i - 1] = True
		elif i_right_isolated and not i_left_isolated:
			suspect[i] = True

	# ── Step 2: 分组连续可疑帧，扩展边界 ──
	# 每个段的边界向外扩展 1 帧作为 DP 锚点
	segments: list[tuple[int, int]] = []  # (start, end) inclusive in original indices
	i = 0
	while i < n:
		if suspect[i]:
			j = i
			while j < n and suspect[j]:
				j += 1
			# 扩展边界：左边界向外 1 帧（若存在且非可疑）
			seg_start = max(0, i - 1)
			seg_end = min(n - 1, j)  # j 是第一个非可疑帧，包含它作为锚点
			if j < n and not suspect[j]:
				seg_end = j  # 把右侧第一个可信帧纳入锚点
			segments.append((seg_start, seg_end))
			i = j + 1
		else:
			i += 1

	if not segments:
		return [sample.raw_speed_kmh for sample in samples]

	# ── Step 3: 对每个可疑段运行 DP ──
	result = [sample.raw_speed_kmh for sample in samples]

	for seg_start, seg_end in segments:
		# 为段内每帧生成候选
		seg_candidates: list[list[float]] = []
		for idx in range(seg_start, seg_end + 1):
			sample = samples[idx]
			cands = build_speed_candidates(sample.raw_text, max_speed_kmh)
			if sample.raw_speed_kmh <= max_speed_kmh:
				cands.append(float(sample.raw_speed_kmh))
			if not cands:
				cands = [min(max(sample.raw_speed_kmh, 0.0), max_speed_kmh)]
			seg_candidates.append(sorted(set(cands)))

		seg_n = seg_end - seg_start + 1

		# 第一帧：若为锚点（非可疑），只保留其原始值
		if not suspect[seg_start] and seg_candidates[0]:
			raw_val = float(samples[seg_start].raw_speed_kmh)
			if raw_val in seg_candidates[0]:
				seg_candidates[0] = [raw_val]
			else:
				seg_candidates[0] = [raw_val]

		# 最后一帧：若为锚点（非可疑），只保留其原始值
		if not suspect[seg_end] and seg_candidates[-1]:
			raw_val = float(samples[seg_end].raw_speed_kmh)
			if raw_val in seg_candidates[-1]:
				seg_candidates[-1] = [raw_val]
			else:
				seg_candidates[-1] = [raw_val]

		# ── 插值候选：基于锚点的线性插值 + 邻近可信帧引导 ──
		anchor_left = seg_candidates[0][0]
		anchor_right = seg_candidates[-1][0]
		seg_duration = times[seg_end] - times[seg_start]

		# 找段外最近的可信帧（非可疑）作为额外锚点参考
		ref_before = None
		for k in range(seg_start - 1, -1, -1):
			if not suspect[k] and raw_vals[k] <= max_speed_kmh:
				ref_before = raw_vals[k]
				break
		ref_after = None
		for k in range(seg_end + 1, n):
			if not suspect[k] and raw_vals[k] <= max_speed_kmh:
				ref_after = raw_vals[k]
				break

		for idx in range(seg_start, seg_end + 1):
			if idx == seg_start or idx == seg_end:
				continue
			local_i = idx - seg_start
			# 线性插值候选
			if seg_duration > 0:
				frac = (times[idx] - times[seg_start]) / seg_duration
				interp_val = anchor_left + (anchor_right - anchor_left) * frac
				lo = max(0.0, interp_val - 15.0)
				hi = min(max_speed_kmh, interp_val + 15.0)
				for v in range(int(lo), int(hi) + 1, 2):
					if v <= max_speed_kmh:
						seg_candidates[local_i].append(float(v))
			# 邻近可信帧候选（段外的可信邻居值）
			for ref_val in (ref_before, ref_after):
				if ref_val is not None:
					lo = max(0.0, ref_val - 10.0)
					hi = min(max_speed_kmh, ref_val + 10.0)
					for v in range(int(lo), int(hi) + 1, 2):
						if v <= max_speed_kmh:
							seg_candidates[local_i].append(float(v))
			# 去重排序
			seg_candidates[local_i] = sorted(set(seg_candidates[local_i]))
		# DP 初始化
		states: list[tuple[float, float, int | None]] = []
		first_cands = seg_candidates[0]
		for c in first_cands:
			cost = 0.0 if not suspect[seg_start] else abs(c - samples[seg_start].raw_speed_kmh)
			states.append((cost, c, None))

		bp: list[list[int]] = [[] for _ in range(seg_n)]
		bp[0] = [-1] * len(first_cands)

		for t in range(1, seg_n):
			abs_idx = seg_start + t
			cur_cands = seg_candidates[t]
			prev_cands = seg_candidates[t - 1]
			dt = max(samples[abs_idx].timestamp - samples[abs_idx - 1].timestamp, 1e-6)
			max_dv = max_accel_mps2 * dt * 3.6

			cur_states: list[tuple[float, float, int]] = []
			cur_bp: list[int] = []

			for ci, cv in enumerate(cur_cands):
				best_cost = float("inf")
				best_pi = 0
				for pi, pv in enumerate(prev_cands):
					delta = abs(cv - pv)
					if delta > max_dv:
						continue
					# OCR 贴近代价：仅可疑帧有代价
					ocr_cost = 0.0
					if suspect[abs_idx]:
						ocr_cost = abs(cv - samples[abs_idx].raw_speed_kmh)
					cost = states[pi][0] + delta * 0.3 + ocr_cost
					if cost < best_cost:
						best_cost = cost
						best_pi = pi
				if best_cost == float("inf"):
					for pi, pv in enumerate(prev_cands):
						delta = abs(cv - pv)
						cost = states[pi][0] + delta * 5.0
						if cost < best_cost:
							best_cost = cost
							best_pi = pi
				cur_states.append((best_cost, cv, best_pi))
				cur_bp.append(best_pi)

			states = [(c, v, p) for c, v, p in cur_states]
			bp[t] = cur_bp

		# 回溯
		best_final = min(range(len(states)), key=lambda idx: states[idx][0])
		seg_result = [0.0] * seg_n
		seg_result[-1] = states[best_final][1]
		trace = best_final
		for t in range(seg_n - 1, 0, -1):
			trace = bp[t][trace]
			seg_result[t - 1] = seg_candidates[t - 1][trace]

		# 写回结果（跳过锚点帧，用 DP 结果）
		for t in range(seg_n):
			abs_idx = seg_start + t
			if suspect[abs_idx]:
				result[abs_idx] = seg_result[t]

	# ── 后处理：中值离群值检测 ──
	# DP 可能因候选不足而无法修正某些帧。用滑动中值检测残差。
	POST_WINDOW = 5
	POST_THRESH = 30.0  # km/h，超过此偏差视为离群
	for i in range(n):
		lo = max(0, i - POST_WINDOW)
		hi = min(n, i + POST_WINDOW + 1)
		neighbors = [result[j] for j in range(lo, hi) if j != i]
		if len(neighbors) >= 3:
			neighbors.sort()
			median = neighbors[len(neighbors) // 2]
			if abs(result[i] - median) > POST_THRESH:
				# 离群：用最近可信邻居线性插值替换
				# 找前后最近的非离群帧
				left_idx = i - 1
				while left_idx >= 0:
					nb = [result[j] for j in range(max(0, left_idx - POST_WINDOW), min(n, left_idx + POST_WINDOW + 1)) if j != left_idx]
					if len(nb) >= 3:
						nb.sort()
						if abs(result[left_idx] - nb[len(nb)//2]) <= POST_THRESH:
							break
					left_idx -= 1
				right_idx = i + 1
				while right_idx < n:
					nb = [result[j] for j in range(max(0, right_idx - POST_WINDOW), min(n, right_idx + POST_WINDOW + 1)) if j != right_idx]
					if len(nb) >= 3:
						nb.sort()
						if abs(result[right_idx] - nb[len(nb)//2]) <= POST_THRESH:
							break
					right_idx += 1
				if left_idx >= 0 and right_idx < n:
					# 线性插值
					frac = (times[i] - times[left_idx]) / max(times[right_idx] - times[left_idx], 1e-6)
					result[i] = result[left_idx] + (result[right_idx] - result[left_idx]) * frac
				elif left_idx >= 0:
					result[i] = result[left_idx]
				elif right_idx < n:
					result[i] = result[right_idx]
				# 否则保持原值

	return result


def correct_speed_series_v2(
	samples: list[SpeedObservation],
	max_speed_kmh: float,
	max_accel_mps2: float,
	fps: float = 0.0,
	div: int = 1,
) -> list[float]:
	"""Improved physical-constraint correction (v2).

	Key improvements over v1:
	1. Adaptive reachability window — scales with effective sampling rate (fps/div)
	2. Time-based display update detection — uses dt, not frame count
	3. Auto-detection of boost zones — locally raises acceleration tolerance
	4. Adaptive outlier threshold — scales with max_accel × dt
	5. Wider temporal context for candidate generation

	Args:
		samples: OCR speed observations
		max_speed_kmh: Maximum plausible speed
		max_accel_mps2: Maximum plausible acceleration (m/s²) in NORMAL zones
		fps: Video frame rate (for adaptive window sizing)
		div: Frame sampling divisor
	"""
	if not samples:
		return []

	if max_speed_kmh <= 0 or max_accel_mps2 <= 0:
		return [s.raw_speed_kmh for s in samples]

	n = len(samples)
	if n < 2:
		return [s.raw_speed_kmh for s in samples]

	raw_vals = [s.raw_speed_kmh for s in samples]
	times = [s.timestamp for s in samples]

	# ── Adaptive parameters based on effective sampling rate ──
	effective_fps = fps / max(div, 1) if fps > 0 else 1.0
	typical_dt = 1.0 / max(effective_fps, 0.1)

	# Reachability window: cover ~0.5s in both directions
	reach_window = max(3, int(0.5 / max(typical_dt, 0.01)))
	reach_window = min(reach_window, 15)

	# Display update detection
	display_hold_dt = 0.15

	# Post-processing outlier threshold
	post_thresh = max(15.0, max_accel_mps2 * typical_dt * 3.6 * 5)

	# ── Detect boost zones: regions where acceleration consistently exceeds nominal ──
	# In boost zones, we use 2× the nominal max_accel for correction
	boost_multiplier = _detect_boost_zones(raw_vals, times, max_accel_mps2, typical_dt)

	# ── Step 0: Spike detection — catch isolated single-frame OCR errors ──
	# These can slip through the reachability scan because the wide window
	# may still connect through the spike. We detect them by checking
	# neighbor consistency: a spike differs from both neighbors by more
	# than physics allows, while the neighbors are consistent with each other.
	spike_suspect = [False] * n
	for i in range(2, n - 2):
		if raw_vals[i] < 0 or raw_vals[i] > max_speed_kmh:
			continue
		left_v = raw_vals[i - 1] if raw_vals[i - 1] >= 0 else raw_vals[i - 2] if i >= 2 and raw_vals[i - 2] >= 0 else None
		right_v = raw_vals[i + 1] if raw_vals[i + 1] >= 0 else raw_vals[i + 2] if i + 2 < n and raw_vals[i + 2] >= 0 else None
		if left_v is None or right_v is None:
			continue
		# Check if neighbors are consistent with each other
		dt_left = max(times[i] - times[i - 1], 0.001)
		dt_right = max(times[i + 1] - times[i], 0.001)
		dt_cross = dt_left + dt_right
		dv_cross = abs(right_v - left_v)
		max_dv_cross = max_accel_mps2 * dt_cross * 3.6 * 2.0  # generous cross tolerance
		if dv_cross > max_dv_cross:
			continue  # neighbors themselves are inconsistent; broader issue
		# Neighbors are consistent — check if this frame is an outlier
		dv_left = abs(raw_vals[i] - left_v)
		dv_right = abs(raw_vals[i] - right_v)
		max_dv_left = max_accel_mps2 * dt_left * 3.6 * 1.5
		max_dv_right = max_accel_mps2 * dt_right * 3.6 * 1.5
		if dv_left > max_dv_left and dv_right > max_dv_right:
			spike_suspect[i] = True

	# ── Step 1: Multi-frame reachability scan ──
	can_reach_fwd = [False] * n
	can_reach_bwd = [False] * n
	can_reach_fwd[0] = True
	can_reach_bwd[-1] = True

	# Forward
	for i in range(1, n):
		if raw_vals[i] > max_speed_kmh:
			continue
		for j in range(i - 1, max(-1, i - reach_window - 1), -1):
			if can_reach_fwd[j]:
				dt = times[i] - times[j]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6 * boost_multiplier[i]
				# Display update boost: if current frame is part of a display hold
				if _is_display_hold(raw_vals, i, n, dt, display_hold_dt, times):
					max_dv = max_dv * 1.5 + 5.0
				# Mid-frame consistency check
				if j < i - 1:
					mid_ok = True
					for k in range(j + 1, i):
						if raw_vals[k] > max_speed_kmh:
							continue
						dt_k = times[k] - times[j]
						if dt_k > 0:
							mid_max = max_accel_mps2 * dt_k * 3.6 * boost_multiplier[k]
							if _is_display_hold(raw_vals, k, n, dt_k, display_hold_dt, times):
								mid_max = mid_max * 1.5 + 5.0
							if abs(raw_vals[k] - raw_vals[j]) > mid_max:
								mid_ok = False
								break
					if not mid_ok:
						continue
				if abs(raw_vals[i] - raw_vals[j]) <= max_dv:
					can_reach_fwd[i] = True
					break

	# Backward
	for i in range(n - 2, -1, -1):
		if raw_vals[i] > max_speed_kmh:
			continue
		for j in range(i + 1, min(n, i + reach_window + 1)):
			if can_reach_bwd[j]:
				dt = times[j] - times[i]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6 * boost_multiplier[i]
				if _is_display_hold(raw_vals, i, n, dt, display_hold_dt, times):
					max_dv = max_dv * 1.5 + 5.0
				if j > i + 1:
					mid_ok = True
					for k in range(i + 1, j):
						if raw_vals[k] > max_speed_kmh:
							continue
						dt_k = times[j] - times[k]
						if dt_k > 0:
							mid_max = max_accel_mps2 * dt_k * 3.6 * boost_multiplier[k]
							if _is_display_hold(raw_vals, k, n, dt_k, display_hold_dt, times):
								mid_max = mid_max * 1.5 + 5.0
							if abs(raw_vals[j] - raw_vals[k]) > mid_max:
								mid_ok = False
								break
					if not mid_ok:
						continue
				if abs(raw_vals[j] - raw_vals[i]) <= max_dv:
					can_reach_bwd[i] = True
					break

	# Mark suspect frames (combine reachability failures + spike detection)
	suspect = [False] * n
	for i in range(n):
		# Spike-detected frames are always suspect
		if spike_suspect[i]:
			suspect[i] = True
			continue
		reach_fail = not can_reach_fwd[i] and not can_reach_bwd[i]
		if not reach_fail:
			continue
		# Display-stable frames are protected (unless spike-detected)
		stable = _is_display_hold(raw_vals, i, n, typical_dt, display_hold_dt, times)
		if not stable:
			suspect[i] = True

	# ── Adjacent jump detection ──
	for i in range(1, n):
		dt = times[i] - times[i - 1]
		if dt <= 0:
			continue
		max_dv = max_accel_mps2 * dt * 3.6 * max(boost_multiplier[i], boost_multiplier[i-1])
		dv = abs(raw_vals[i] - raw_vals[i - 1])
		if dv <= max_dv:
			continue
		# Detect which side is the isolated error
		i_left_isolated = (i - 2 >= 0 and raw_vals[i - 1] != raw_vals[i - 2]
							and not _is_display_hold(raw_vals, i-1, n, dt, display_hold_dt, times))
		i_right_isolated = (i + 1 < n and raw_vals[i] != raw_vals[i + 1]
							and not _is_display_hold(raw_vals, i, n, dt, display_hold_dt, times))
		if i_left_isolated and not i_right_isolated:
			suspect[i - 1] = True
		elif i_right_isolated and not i_left_isolated:
			suspect[i] = True

	# ── Step 2: Segment grouping ──
	segments: list[tuple[int, int]] = []
	i = 0
	while i < n:
		if suspect[i]:
			j = i
			while j < n and suspect[j]:
				j += 1
			seg_start = max(0, i - 1)
			seg_end = min(n - 1, j)
			if j < n and not suspect[j]:
				seg_end = j
			segments.append((seg_start, seg_end))
			i = j + 1
		else:
			i += 1

	if not segments:
		return [s.raw_speed_kmh for s in samples]

	# ── Step 3: DP correction per segment ──
	result = [s.raw_speed_kmh for s in samples]

	for seg_start, seg_end in segments:
		seg_n = seg_end - seg_start + 1
		seg_candidates: list[list[float]] = []
		for idx in range(seg_start, seg_end + 1):
			sample = samples[idx]
			cands = build_speed_candidates(sample.raw_text, max_speed_kmh)
			if sample.raw_speed_kmh <= max_speed_kmh and sample.raw_speed_kmh > 0:
				cands.append(float(sample.raw_speed_kmh))
			# For OCR-failure frames (raw <= 0) inside suspect segments,
			# build candidates from interpolation between segment anchors
			if sample.raw_speed_kmh <= 0 and seg_n > 2:
				local_i = idx - seg_start
				frac = local_i / max(seg_n - 1, 1)
				# Use segment boundary raw values as interpolation anchors
				left_v = samples[seg_start].raw_speed_kmh
				right_v = samples[seg_end].raw_speed_kmh
				if left_v > 0 and right_v > 0:
					interp_v = left_v + (right_v - left_v) * frac
				elif left_v > 0:
					interp_v = left_v
				elif right_v > 0:
					interp_v = right_v
				else:
					interp_v = max_speed_kmh / 2  # fallback
				for dv in range(-25, 26, 2):
					v = interp_v + dv
					if 0 <= v <= max_speed_kmh:
						cands.append(float(v))
			if not cands:
				cands = [min(max(sample.raw_speed_kmh, 0.0), max_speed_kmh)]
			seg_candidates.append(sorted(set(cands)))

		seg_n = seg_end - seg_start + 1

		# Anchor frames: lock to raw values
		if not suspect[seg_start] and seg_candidates[0]:
			seg_candidates[0] = [float(samples[seg_start].raw_speed_kmh)]
		if not suspect[seg_end] and seg_candidates[-1]:
			seg_candidates[-1] = [float(samples[seg_end].raw_speed_kmh)]

		# ── Extended interpolation candidates ──
		anchor_left = seg_candidates[0][0]
		anchor_right = seg_candidates[-1][0]
		seg_duration = times[seg_end] - times[seg_start]

		# Wider context: look ±2s for reference values
		ref_before = None
		for k in range(seg_start - 1, -1, -1):
			if not suspect[k] and raw_vals[k] <= max_speed_kmh:
				if times[seg_start] - times[k] < 2.0:
					ref_before = raw_vals[k]
				break
		ref_after = None
		for k in range(seg_end + 1, n):
			if not suspect[k] and raw_vals[k] <= max_speed_kmh:
				if times[k] - times[seg_end] < 2.0:
					ref_after = raw_vals[k]
				break

		for idx in range(seg_start, seg_end + 1):
			if idx == seg_start or idx == seg_end:
				continue
			local_i = idx - seg_start

			# Linear interpolation ± adaptive range
			if seg_duration > 0:
				frac = (times[idx] - times[seg_start]) / seg_duration
				interp_val = anchor_left + (anchor_right - anchor_left) * frac
				# Adaptive range: wider for larger segments
				interp_range = min(30.0, 10.0 + seg_duration * max_accel_mps2 * 3.6)
				lo = max(0.0, interp_val - interp_range)
				hi = min(max_speed_kmh, interp_val + interp_range)
				step = max(1, int(interp_range / 10))
				for v in range(int(lo), int(hi) + 1, step):
					if v <= max_speed_kmh:
						seg_candidates[local_i].append(float(v))

			# Context reference candidates
			for ref_val in (ref_before, ref_after):
				if ref_val is not None:
					lo = max(0.0, ref_val - 10.0)
					hi = min(max_speed_kmh, ref_val + 10.0)
					for v in range(int(lo), int(hi) + 1, 2):
						if v <= max_speed_kmh:
							seg_candidates[local_i].append(float(v))

			seg_candidates[local_i] = sorted(set(seg_candidates[local_i]))

		# DP
		states: list[tuple[float, float, int | None]] = []
		first_cands = seg_candidates[0]
		for c in first_cands:
			cost = 0.0 if not suspect[seg_start] else abs(c - samples[seg_start].raw_speed_kmh)
			states.append((cost, c, None))

		bp: list[list[int]] = [[] for _ in range(seg_n)]
		bp[0] = [-1] * len(first_cands)

		for t in range(1, seg_n):
			abs_idx = seg_start + t
			cur_cands = seg_candidates[t]
			prev_cands = seg_candidates[t - 1]
			dt = max(samples[abs_idx].timestamp - samples[abs_idx - 1].timestamp, 1e-6)
			max_dv = max_accel_mps2 * dt * 3.6 * boost_multiplier[abs_idx]

			cur_states: list[tuple[float, float, int]] = []
			cur_bp: list[int] = []

			for ci, cv in enumerate(cur_cands):
				best_cost = float("inf")
				best_pi = 0
				for pi, pv in enumerate(prev_cands):
					delta = abs(cv - pv)
					if delta > max_dv:
						continue
					ocr_cost = 0.0
					if suspect[abs_idx]:
						ocr_cost = abs(cv - samples[abs_idx].raw_speed_kmh)
					# Reduced smoothness penalty for boost zones
					smoothness_weight = 0.2 if boost_multiplier[abs_idx] > 1.5 else 0.3
					cost = states[pi][0] + delta * smoothness_weight + ocr_cost
					if cost < best_cost:
						best_cost = cost
						best_pi = pi
				if best_cost == float("inf"):
					for pi, pv in enumerate(prev_cands):
						delta = abs(cv - pv)
						cost = states[pi][0] + delta * 5.0
						if cost < best_cost:
							best_cost = cost
							best_pi = pi
				cur_states.append((best_cost, cv, best_pi))
				cur_bp.append(best_pi)

			states = [(c, v, p) for c, v, p in cur_states]
			bp[t] = cur_bp

		# Traceback
		best_final = min(range(len(states)), key=lambda idx: states[idx][0])
		seg_result = [0.0] * seg_n
		seg_result[-1] = states[best_final][1]
		trace = best_final
		for t in range(seg_n - 1, 0, -1):
			trace = bp[t][trace]
			seg_result[t - 1] = seg_candidates[t - 1][trace]

		for t in range(seg_n):
			abs_idx = seg_start + t
			if suspect[abs_idx]:
				result[abs_idx] = seg_result[t]

	# ── Step 4: Post-processing median outlier detection ──
	# Adaptive window: cover ~0.5s
	post_window = max(3, int(0.5 / max(typical_dt, 0.01)))
	for i in range(n):
		lo = max(0, i - post_window)
		hi = min(n, i + post_window + 1)
		neighbors = [result[j] for j in range(lo, hi) if j != i]
		if len(neighbors) >= 3:
			neighbors.sort()
			median = neighbors[len(neighbors) // 2]
			if abs(result[i] - median) > post_thresh:
				# Find nearest clean neighbors
				left_idx = i - 1
				while left_idx >= 0:
					nb = [result[j] for j in range(max(0, left_idx - post_window),
							min(n, left_idx + post_window + 1)) if j != left_idx]
					if len(nb) >= 3:
						nb.sort()
						if abs(result[left_idx] - nb[len(nb)//2]) <= post_thresh:
							break
					left_idx -= 1
				right_idx = i + 1
				while right_idx < n:
					nb = [result[j] for j in range(max(0, right_idx - post_window),
							min(n, right_idx + post_window + 1)) if j != right_idx]
					if len(nb) >= 3:
						nb.sort()
						if abs(result[right_idx] - nb[len(nb)//2]) <= post_thresh:
							break
					right_idx += 1
				if left_idx >= 0 and right_idx < n:
					frac = (times[i] - times[left_idx]) / max(times[right_idx] - times[left_idx], 1e-6)
					result[i] = result[left_idx] + (result[right_idx] - result[left_idx]) * frac
				elif left_idx >= 0:
					result[i] = result[left_idx]
				elif right_idx < n:
					result[i] = result[right_idx]

	return result


def _is_display_hold(
	raw_vals: list[float],
	i: int,
	n: int,
	dt: float,
	hold_dt: float,
	times: list[float] | None = None,
) -> bool:
	"""Check if frame i is part of a display hold (game HUD refresh).

	A display hold occurs when consecutive frames show the same value
	because the game's speedometer updates slower than the video frame rate.

	Uses TIME-BASED detection: if dt < hold_dt and neighbor has same value,
	it's likely a display hold rather than genuine constant speed.
	"""
	if dt > hold_dt:
		return False
	if i > 0 and raw_vals[i] == raw_vals[i - 1]:
		return True
	if i + 1 < n and raw_vals[i] == raw_vals[i + 1]:
		return True
	return False


def _detect_boost_zones(
	raw_vals: list[float],
	times: list[float],
	max_accel_mps2: float,
	typical_dt: float,
) -> list[float]:
	"""Detect acceleration boost zones and return per-frame multiplier.

	Boost zones are regions where raw OCR values show sustained high
	acceleration. In these zones, the DP correction uses higher tolerance
	to avoid over-constraining legitimate rapid speed changes.

	Returns list of multipliers (1.0 = normal, up to 3.0 = heavy boost).
	"""
	n = len(raw_vals)
	multipliers = [1.0] * n
	if n < 5:
		return multipliers

	# Compute raw acceleration between consecutive frames (km/h per second)
	raw_accel = [0.0] * n
	for i in range(1, n):
		dt = times[i] - times[i - 1]
		if dt > 0:
			raw_accel[i] = abs(raw_vals[i] - raw_vals[i - 1]) / (dt * 3.6)

	nom_thresh = max_accel_mps2 * 3.6  # nominal max accel in km/h/s
	window = max(3, int(0.5 / max(typical_dt, 0.01)))

	for i in range(n):
		lo = max(0, i - window)
		hi = min(n, i + window + 1)
		high_count = sum(1 for j in range(lo, hi)
						if raw_accel[j] > nom_thresh * 1.3)
		total = hi - lo
		if total > 0 and high_count / total > 0.3:
			local_max = max(raw_accel[lo:hi])
			ratio = min(3.0, max(1.0, local_max / max(nom_thresh, 0.1)))
			multipliers[i] = ratio

	return multipliers


class _CancelExport(Exception):
	"""内部异常：用户取消了导出任务。"""
	pass


