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

import matplotlib
matplotlib.use("TkAgg")
# 配置中文字体支持
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── 导出列表：包含 _ 前缀的私有符号供 RaceVideoToLog.py / headless.py 使用 ──
__all__ = [
    "SpeedObservation", "VideoMetadata", "RapidOCR",
    "extract_speed_value", "convert_speed_to_kmh", "clamp_region",
    "correct_speed_series", "build_speed_candidates",
    "normalize_ocr_text", "format_duration", "codec_from_fourcc",
    "safe_int", "safe_float", "SOURCE_TO_KMH", "OCR_NUMBER_RE",
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

	# 策略3: 常见 OCR 字符混淆替换
	# 对每位数字尝试替换为视觉相似的字符
	_CONFUSION_MAP = {
		"0": ["8"], "8": ["0", "6", "3"],
		"6": ["8", "5"], "5": ["6"],
		"3": ["8"], "1": ["7"], "7": ["1", "2", "4"],
		"2": ["7"], "4": ["7"], "9": ["8"],
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

	# 前向多帧可达
	for i in range(1, n):
		if raw_vals[i] > max_speed_kmh:
			continue
		# 向前搜索最多 30 帧寻找可信前驱
		for j in range(i - 1, max(-1, i - 31), -1):
			if can_reach_fwd[j]:
				dt = times[i] - times[j]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6
				if abs(raw_vals[i] - raw_vals[j]) <= max_dv:
					can_reach_fwd[i] = True
					break

	# 后向多帧可达
	for i in range(n - 2, -1, -1):
		if raw_vals[i] > max_speed_kmh:
			continue
		for j in range(i + 1, min(n, i + 31)):
			if can_reach_bwd[j]:
				dt = times[j] - times[i]
				if dt <= 0:
					continue
				max_dv = max_accel_mps2 * dt * 3.6
				if abs(raw_vals[j] - raw_vals[i]) <= max_dv:
					can_reach_bwd[i] = True
					break

	# 双向都不可达 → 可疑
	suspect = [False] * n
	for i in range(n):
		suspect[i] = not can_reach_fwd[i] and not can_reach_bwd[i]

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

	return result


class _CancelExport(Exception):
    """内部异常：用户取消了导出任务。"""
    pass


