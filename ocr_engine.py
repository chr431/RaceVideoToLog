"""OCR engine for RaceVideoToLog.

SpeedObservation, preprocessing, correction algorithms,
model configuration, and supporting utilities.
"""
from __future__ import annotations
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 模块级 Logger ──
logger = logging.getLogger("RaceVideoToLog.ocr_engine")

# ── 导出列表：包含 _ 前缀的私有符号供 RaceVideoToLog.py / headless.py 使用 ──
__all__ = [
	"SpeedObservation", "VideoMetadata", "RapidOCR",
	"extract_speed_value", "convert_speed_to_kmh", "clamp_region",
	"build_speed_candidates",
	"normalize_ocr_text", "format_duration", "codec_from_fourcc",
	"safe_int", "safe_float", "SOURCE_TO_KMH", "OCR_NUMBER_RE",
	"ocr_digital_fallback", "compute_video_hash", "auto_select_anchors",
	"_reset_backend", "_select_backend", "_get_model_kwargs",
	"_gpu_backend", "_gpu_patched", "_CancelExport",
	"_parse_int_or_none", "parse_csv_header", "_estimate_raw_trust", "_savgol_filter_np",
	"_set_rec_keys_path", "Flag", "logger",
]

# ═══════════════════ Flag 枚举：速度数据来源标记 ═══════════════════

class Flag:
	"""速度数据 flag 值 — 统一标记每帧数据的来源和可信度。

	用于 CSV 第 4 列和所有相关判断逻辑，消除散布的魔法数字。
	"""
	RAW: int = 0             # 原始 OCR 输出，未纠错
	REOCR_AUTO: int = 11     # 重 OCR 自动修正
	FILL_INTERP: int = 12    # 级联插值填充
	PARTIAL_AUTO: int = 13   # 部分数字模式自动推断修正
	ANCHOR_AUTO: int = 21    # 自动锚点帧（硬约束）
	ANCHOR_MANUAL: int = 22  # 人工修正锚点帧
	CONFIRMED_SEG: int = 23  # 人工确认段内帧
	FLAGGED_REVIEW: int = 30 # 标记待人工审核

	@classmethod
	def is_corrected(cls, flag: int) -> bool:
		"""是否为自动纠错帧 (10-19)。"""
		return 10 <= flag <= 19

	@classmethod
	def is_anchor(cls, flag: int) -> bool:
		"""是否为锚点帧 (>=20)。"""
		return flag >= 20


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
			"v13.3", "v13.2", "v13.1", "v13.0",
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

		# ── 3. 更新 PATH ──
		_path_extra: list[str] = []
		if _cuda_bin:
			_path_extra.append(_cuda_bin)
		if _cudnn_dir:
			_path_extra.append(_cudnn_dir)
		if _path_extra:
			_existing = _os.environ.get("PATH", "")
			_os.environ["PATH"] = ";".join(_path_extra) + (";" + _existing if _existing else "")

		if _cuda_bin:
			logger.info("CUDA: %s", _cuda_bin)
		else:
			logger.info("CUDA: 未找到 CUDA Toolkit 安装")
		if _cudnn_dlls:
			logger.info("cuDNN: %d 个 DLL 在 %s", len(_cudnn_dlls), _cudnn_dir)
		else:
			logger.info("cuDNN: 未找到")
		logger.info("GPU DLL 预加载: %d 个成功", _loaded)
		if _failed:
			logger.warning("GPU DLL 预加载失败 (%d 个): %s", len(_failed),
			               "; ".join(_failed[:5]))

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
	"CUDA": ("CUDAExecutionProvider", {
	    "device_id": 0,
	    "arena_extend_strategy": "kNextPowerOfTwo",
	    "cudnn_conv_algo_search": "EXHAUSTIVE",
	    "do_copy_in_default_stream": True,
	}),
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


def parse_csv_header(path: str) -> dict[str, str]:
	"""从 CSV 文件头中提取所有 # 注释行的 key=value 参数。

	兼容 ", " 和 "," 两种分隔符，正确处理空值、含逗号的值（如 ROI）。
	Returns: {key: value} dict, e.g. {'roi': '862,945,957,1003', 'max_speed': '400', ...}
	"""
	import re
	_pair = re.compile(r"(\w+)=(.*?)(?=,\s*\w+=|$)")
	settings: dict[str, str] = {}
	try:
		with open(path, "r", encoding="utf-8-sig") as f:
			for line in f:
				line = line.strip()
				if not line.startswith("#"):
					break
				line = line.lstrip("#").strip()
				for m in _pair.finditer(line):
					settings[m.group(1)] = m.group(2).strip()
	except Exception as e:
		logger.warning("解析 CSV 文件头失败 (%s): %s", path, e)
	return settings


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


def extract_speed_value(ocr_result: list | None) -> tuple[float | None, str | None]:
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
	ocr: "RapidOCR", crop_bgr: "np.ndarray", max_speed_kmh: float = 400
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


# ── SG 滤波系数缓存（(window_length, polyorder) → coefficients）──
_sg_coeff_cache: dict[tuple[int, int], np.ndarray] = {}

def _savgol_filter_np(y: "np.ndarray", window_length: int, polyorder: int) -> "np.ndarray":
	"""纯 numpy Savitzky-Golay 滤波 — 预计算卷积系数，O(N) 复杂度。

	等价于 scipy.signal.savgol_filter，但无 scipy 依赖。
	通过预计算伪逆系数 + np.convolve 实现，比逐点 lstsq 快 10-100x。
	"""
	if window_length % 2 == 0 or window_length < 1:
		raise ValueError("window_length must be odd")
	if window_length <= polyorder:
		raise ValueError("window_length must be > polyorder")
	half = window_length // 2
	y = np.asarray(y, dtype=float)
	n = len(y)
	if n < window_length:
		return y.copy()

	# ── 预计算卷积系数（缓存复用）──
	cache_key = (window_length, polyorder)
	if cache_key not in _sg_coeff_cache:
		x = np.arange(-half, half + 1, dtype=float)
		A = np.vander(x, polyorder + 1, increasing=True)
		# pinv(A)[0] = 多项式常数项 a0 的系数 = 中心点的平滑值
		_sg_coeff_cache[cache_key] = np.linalg.pinv(A)[0]
	coeffs = _sg_coeff_cache[cache_key]

	# ── 卷积应用（O(N)）──
	result = np.convolve(y, coeffs[::-1], mode="same")

	# ── 边界处理：用最近的有效滤波值填充 ──
	if half > 0 and n > half:
		result[:half] = result[half]
		result[-half:] = result[-half - 1]

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
	"""Get RapidOCR kwargs for the model. Returns None if files missing."""
	import rapidocr_onnxruntime as rr
	if models_dir is None:
		models_dir = str(Path(rr.__file__).parent / "models")
	cfg = {
		"det_model_path": f"{models_dir}/PP-OCRv6_det_small.onnx",
		"rec_model_path": f"{models_dir}/PP-OCRv6_rec_small.onnx",
		"text_score": 0.6, "use_angle_cls": False, "rec_batch_num": 12,
	}
	for key in ("det_model_path", "rec_model_path"):
		if not Path(cfg[key]).exists():
			return None
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




def auto_select_anchors(observations: list["SpeedObservation"], max_speed_kmh: float = 400.0, window: int = 0, max_dev: float = 4.0, max_accel_mps2: float = 50.0) -> set[int]:
	"""Select reliable OCR frames as Correction B anchors.

	Uses local median filter: for each frame, compute median in an adaptive
	sliding window. If frame value deviates <= max_dev from median, it is reliable.

	If window=0 (default), auto-computes window size to cover ~0.3s of data,
	making the filter robust at both high and low sampling rates.

	Returns set of trusted frame indices."""
	n = len(observations)
	raw_vals = [o.raw_speed_kmh for o in observations]
	anchors: set[int] = set()
	times = [o.timestamp for o in observations]

	# Adaptive window: cover ~0.3s regardless of sampling rate
	if window <= 0:
		typical_dt = (times[-1] - times[0]) / max(n - 1, 1) if n > 1 else 0.017
		window = max(5, int(0.3 / max(typical_dt, 0.001)) | 1)  # odd, min 5
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

	# Post-filter: remove anchors that are extreme outliers vs immediate neighbors
	# An anchor must be within 10 km/h of at least one immediate neighbor
	anchors_filtered = set()
	for i in anchors:
		keep = True
		v = raw_vals[i]
		# Check against both neighbors
		left_ok = (i > 0 and raw_vals[i - 1] > 0 and abs(v - raw_vals[i - 1]) <= 10.0)
		right_ok = (i + 1 < n and raw_vals[i + 1] > 0 and abs(raw_vals[i + 1] - v) <= 10.0)
		# Keep if at least one neighbor is within 10 km/h
		if not left_ok and not right_ok:
			# Extreme outlier: not close to either neighbor
			keep = False
		if keep:
			anchors_filtered.add(i)

	# ═══ Acceleration validation ═══
	# Check acceleration to nearest frame with >2 km/h difference (any frame,
	# not just anchors) — avoids "same-cluster" OCR repeat errors fooling the check.
	max_dv_per_sec = max_accel_mps2 * 3.6 * 2.0  # km/h/s (2x safety margin)
	anchors_validated: set[int] = set()
	for i in anchors_filtered:
		v = raw_vals[i]
		left_fail = right_fail = False

		# Left: nearest frame (any valid speed) with >2 km/h difference
		for j in range(i - 1, -1, -1):
			if raw_vals[j] > 0 and abs(raw_vals[j] - v) > 2.0:
				dt = times[i] - times[j]
				if abs(v - raw_vals[j]) / max(dt, 0.001) > max_dv_per_sec:
					left_fail = True
				break

		# Right: nearest frame (any valid speed) with >2 km/h difference
		for j in range(i + 1, n):
			if raw_vals[j] > 0 and abs(raw_vals[j] - v) > 2.0:
				dt = times[j] - times[i]
				if abs(raw_vals[j] - v) / max(dt, 0.001) > max_dv_per_sec:
					right_fail = True
				break

		if not (left_fail or right_fail):  # keep only if NEITHER side fails accel check
			anchors_validated.add(i)

	return anchors_validated



class _CancelExport(Exception):
	"""内部异常：用户取消了导出任务。"""
	pass


