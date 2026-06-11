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

import matplotlib
matplotlib.use("TkAgg")
# 配置中文字体支持
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
		self._ocr_model_var = tk.StringVar(value="v5 均衡")  # OCR 模型版本

		# 时间轴范围
		self._frame_start_var = tk.StringVar(value="")
		self._frame_end_var = tk.StringVar(value="")
		self._color_threshold_var = tk.StringVar(value="50")  # 色距判定范围 (0=严格, 50=默认, 100=2倍)
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

		# 颜色键值拾取（支持双键值）
		self._key_color1: tuple[int, int, int] | None = None
		self._key_color2: tuple[int, int, int] | None = None
		self._key_color_str1 = tk.StringVar(value="未设置")
		self._key_color_str2 = tk.StringVar(value="未设置")
		self._pick_color_mode = 0  # 0=关闭, 1=拾取键值1, 2=拾取键值2
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
		_MODELS = {"v3": "v3 快速", "v5_mobile": "v5 均衡"}
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
		self.preview_canvas.bind("<ButtonPress-3>", self._on_color_pick)  # 右键拾取颜色

		# 视频帧位置滑动条 + 刷新按钮
		slider_row = ttk.Frame(preview_box)
		slider_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
		slider_row.columnconfigure(0, weight=1)
		self._preview_slider = ttk.Scale(slider_row, from_=0, to=1, variable=self._preview_frame_pos,
			orient="horizontal")
		self._preview_slider.grid(row=0, column=0, sticky="ew")
		ttk.Button(slider_row, text="刷新预览", command=self.refresh_preview).grid(row=0, column=1, padx=(8, 0))

		# 键值颜色拾取工具栏（双键值）
		key_bar = ttk.Frame(preview_box)
		key_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
		ttk.Button(key_bar, text="键值1", command=lambda: self._start_color_pick(1)).grid(row=0, column=0, sticky="w")
		self._key_swatch1 = tk.Canvas(key_bar, width=18, height=18, background="#888888", highlightthickness=1, highlightbackground="#555555")
		self._key_swatch1.grid(row=0, column=1, padx=(4, 2))
		ttk.Label(key_bar, textvariable=self._key_color_str1, foreground="#555555", font=("", 8), width=16, anchor="w").grid(row=0, column=2, sticky="w")
		ttk.Button(key_bar, text="键值2", command=lambda: self._start_color_pick(2)).grid(row=0, column=3, padx=(10, 0))
		self._key_swatch2 = tk.Canvas(key_bar, width=18, height=18, background="#888888", highlightthickness=1, highlightbackground="#555555")
		self._key_swatch2.grid(row=0, column=4, padx=(4, 2))
		ttk.Label(key_bar, textvariable=self._key_color_str2, foreground="#555555", font=("", 8), width=16, anchor="w").grid(row=0, column=5, sticky="w")
		ttk.Button(key_bar, text="清除全部", command=self._clear_key_color).grid(row=0, column=6, padx=(12, 0))
		ttk.Label(key_bar, text="范围").grid(row=0, column=7, padx=(12, 0))
		ttk.Entry(key_bar, textvariable=self._color_threshold_var, width=4, justify="center").grid(row=0, column=8, padx=(2, 0))
		ttk.Label(key_bar, text="(0=严格, 50默认, 100=2倍)", foreground="#555555", font=("", 7)).grid(row=0, column=9, padx=(2, 0))

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

	def _start_color_pick(self, slot: int) -> None:
		"""激活键值颜色拾取模式（slot=1或2），下一次右键点击预览将采样颜色。"""
		self._pick_color_mode = slot
		self.status_var.set(f"请在预览图上右键点击速度数字以拾取键值{slot}...")

	def _clear_key_color(self) -> None:
		"""清除全部键值颜色，恢复默认的 CLAHE+OTSU 预处理。"""
		self._key_color1 = None
		self._key_color2 = None
		self._key_color_str1.set("未设置")
		self._key_color_str2.set("未设置")
		self._key_swatch1.configure(background="#888888")
		self._key_swatch2.configure(background="#888888")
		self.status_var.set("已清除键值颜色，将使用默认预处理。")

	def _on_color_pick(self, event: tk.Event) -> None:
		"""预览图右键采样像素颜色作为 OCR 键值。"""
		if not self._pick_color_mode:
			return
		if self.first_frame_bgr is None:
			return
		slot = self._pick_color_mode
		self._pick_color_mode = 0
		# 将 canvas 坐标映射到原始图像坐标
		cw = max(1, self.preview_canvas.winfo_width())
		ch = max(1, self.preview_canvas.winfo_height())
		if not self.preview_photo:
			return
		img_w = self.preview_photo.width()
		img_h = self.preview_photo.height()
		ox = (cw - img_w) / 2
		oy = (ch - img_h) / 2
		ix = int((event.x - ox) / self._preview_scale)
		iy = int((event.y - oy) / self._preview_scale)
		# 获取当前预览帧像素
		cur_frame = self._get_preview_frame()
		if cur_frame is None:
			cur_frame = self.first_frame_bgr
		ih, iw = cur_frame.shape[:2]
		ix = max(0, min(iw - 1, ix))
		iy = max(0, min(ih - 1, iy))
		b, g, r = (int(c) for c in cur_frame[iy, ix])
		if slot == 1:
			self._key_color1 = (b, g, r)
			self._key_color_str1.set(f"BGR({b},{g},{r})")
			self._key_swatch1.configure(background=f"#{r:02x}{g:02x}{b:02x}")
		else:
			self._key_color2 = (b, g, r)
			self._key_color_str2.set(f"BGR({b},{g},{r})")
			self._key_swatch2.configure(background=f"#{r:02x}{g:02x}{b:02x}")
		self.status_var.set(f"已拾取键值{slot} BGR({b},{g},{r})")

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
		_reset_backend()
		BACKEND_LABELS_REV = {"自动": "auto", "CUDA": "cuda", "CPU": "cpu"}
		selected_label = self.backend_var.get()
		selected_key = BACKEND_LABELS_REV.get(selected_label, "auto")
		actual = _select_backend(selected_key)
		MODEL_REV = {"v3 快速": "v3", "v5 均衡": "v5_mobile"}
		model_key = MODEL_REV.get(self._ocr_model_var.get(), "v5_mobile")
		print(f"[OCR] 后端: {actual}, 模型: {model_key}", flush=True)
		kwargs = _get_model_kwargs(model_key)
		if kwargs is None and model_key != "v3":
			print(f"[OCR] 警告: {model_key} 模型文件不存在，回退到默认 v3")
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
		h, w = crop.shape[:2]
		target_h = max(8.0, float(target_h))
		pad_px = max(0.0, float(pad_px))

		if self._key_color1 is not None or self._key_color2 is not None:
			# 颜色键值：欧式距离，亮色阈值35暗色25
			h, w = crop.shape[:2]
			mask = np.zeros((h, w), dtype=np.uint8)
			cf = crop.astype(np.float32)
			for kc in (self._key_color1, self._key_color2):
				if kc is None:
					continue
				kb, kg, kr = kc
				base = 35.0 if (kb + kg + kr) > 600 else 25.0
				try:
					user_scale = max(0.0, float(self._color_threshold_var.get())) / 50.0
				except ValueError:
					user_scale = 1.0
				thr = max(1.0, base * user_scale) if user_scale > 0 else 1  # 0=严格一致
				d = np.sqrt((cf[:,:,0] - kb)**2 + (cf[:,:,1] - kg)**2 + (cf[:,:,2] - kr)**2)
				mask[d < thr] = 255
			gray = mask
		else:
			# 默认模式：CLAHE + OTSU
			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray = clahe.apply(gray)
			_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		scale = target_h / float(h) if h > 0 else 1.0
		if abs(scale - 1.0) > 0.02:
			gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)

		pad_int = int(pad_px)
		if pad_int > 0:
			gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)

		return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

	def _preprocess_fallback(self, crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
		"""备选预处理：有键值时回退到 CLAHE+OTSU，无键值时回退到纯 OTSU。"""
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		if self._key_color1 is not None or self._key_color2 is not None:
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray = clahe.apply(gray)
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
		self._write_csv_with_retry(output_path, rows, _t_elapsed, total_frames, _accuracy, _gpu_backend)

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
	parser.add_argument("--ocr-model", choices=["v3","v5_mobile"], default="v5_mobile",
		help="OCR 模型: v3(快速)/v5_mobile(均衡,默认)")
	parser.add_argument("-o", "--output", type=str, help="输出 CSV 路径 (默认 视频名_log.csv)")
	parser.add_argument("--analysis", nargs=2, metavar=("CSV1","CSV2"), help="无头分析: 从两个CSV导出v-t/v-x/Δt-x的PNG")
	parser.add_argument("--analysis-out", type=str, help="分析PNG输出前缀 (默认 CSV1所在目录/分析)")
	parser.add_argument("--key-color", type=str, metavar="B,G,R[;B,G,R]",
		help="键值颜色: 逗号分隔的B,G,R, 多个用分号分隔 (如 255,250,247;135,124,121)")
	parser.add_argument("--frame-start", type=int, metavar="N", help="起始帧 (含)")
	parser.add_argument("--frame-end", type=int, metavar="N", help="结束帧 (不含)")
	args = parser.parse_args()

	if args.video:
		run_headless(args)
	elif args.analysis:
		run_analysis_headless(args)
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
	print(f"OCR 后端: {backend_actual}, 模型: {args.ocr_model}")
	model_kwargs = _get_model_kwargs(args.ocr_model)
	if model_kwargs is None and args.ocr_model != "v3":
		print(f"警告: {args.ocr_model} 模型文件不存在，回退到默认 v3")
	ocr = RapidOCR(**(model_kwargs or {}))

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

	# 解析键值颜色
	key_colors: list[tuple[int, int, int]] = []
	if args.key_color:
		for part in args.key_color.split(";"):
			ch = [int(c.strip()) for c in part.split(",")]
			if len(ch) == 3:
				key_colors.append((ch[0], ch[1], ch[2]))
		print(f"键值颜色: {key_colors}")

	raw_frames: list[tuple[float, np.ndarray]] = []
	fi = 0
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			break
		if args.frame_end is not None and fi >= args.frame_end:
			break
		if args.frame_start is not None and fi < args.frame_start:
			fi += 1
			continue
		if fi % frame_step != 0:
			fi += 1
			continue
		ts = fi / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
		crop = frame[y1:y2 + 1, x1:x2 + 1].copy()  # .copy() 断开对整帧的引用
		raw_frames.append((ts, crop))
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
		proc = _preprocess_headless(crop, args.target_h, args.pad, key_colors)
		ocr_result, _ = ocr(proc)
		sv, rt = extract_speed_value(ocr_result)
		if sv is None:
			# 备选预处理
			proc2 = _preprocess_headless_fallback(crop, args.target_h, args.pad, key_colors)
			ocr_result, _ = ocr(proc2)
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

	# 初次纠错 + 物理超限帧重试 OCR
	print(f"识别: {len(observations)} 条, 正在进行物理约束纠错...")
	corrected = correct_speed_series(observations, args.max_speed, args.max_accel)
	observations, corrected = _retry_suspect_frames(
		observations, corrected, raw_frames, ocr, args
	)
	if corrected is None:
		corrected = correct_speed_series(observations, args.max_speed, args.max_accel)

	# 积分 + 写出
	rows: list[tuple[float, float, float, int]] = []
	dist = 0.0
	prev_t, prev_v, prev_cspd = None, None, None
	for obs, cspd in zip(observations, corrected):
		v = cspd / 3.6
		if prev_t is not None and prev_v is not None:
			dt = obs.timestamp - prev_t
			if dt > 0:
				dist += (prev_v + v) * 0.5 * dt
		prev_t, prev_v = obs.timestamp, v
		flag = 1 if abs(obs.raw_speed_kmh - cspd) > 0.01 else 0
		# 纠正后仍超物理极限的也标记
		if not flag and prev_cspd is not None:
			dt_phys = obs.timestamp - (rows[-1][0] if rows else 0)
			if dt_phys > 0 and abs(cspd - prev_cspd) / (dt_phys * 3.6) > args.max_accel:
				flag = 1
		prev_cspd = cspd
		rows.append((obs.timestamp, dist, cspd, flag))

	with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
		w = csv.writer(fh)
		for t, d, s, fl in rows:
			w.writerow([f"{t:.2f}", f"{d:.2f}", f"{s:.2f}", str(fl)])

	corrected_count = sum(r[3] for r in rows)
	print(f"导出完成: {output_path}")
	print(f"共 {len(rows)} 条, 纠错 {corrected_count} 条 (准确率 {100 - corrected_count/len(rows)*100:.1f}%)")


def _preprocess_headless(crop, target_h, pad, key_colors):
	"""无头模式预处理：有键值时颜色分割，否则 CLAHE+OTSU。"""
	if key_colors:
		h, w = crop.shape[:2]
		mask = np.zeros((h, w), dtype=np.uint8)
		cf = crop.astype(np.float32)
		for kb, kg, kr in key_colors:
			thr = 35 if (kb + kg + kr) > 600 else 25
			d = np.sqrt((cf[:,:,0] - kb)**2 + (cf[:,:,1] - kg)**2 + (cf[:,:,2] - kr)**2)
			mask[d < thr] = 255
		gray = mask
	else:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		gray = clahe.apply(gray)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return _finish_preprocess(gray, target_h, pad)


def _preprocess_headless_fallback(crop, target_h, pad, key_colors):
	"""无头模式备选预处理。"""
	if key_colors:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		gray = clahe.apply(gray)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return _finish_preprocess(gray, target_h, pad)


def _finish_preprocess(gray, target_h, pad):
	"""统一的缩放+填充+转BGR。"""
	h, w = gray.shape[:2]
	th = max(8.0, float(target_h))
	scale = th / h if h > 0 else 1.0
	if abs(scale - 1.0) > 0.02:
		gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)), interpolation=cv2.INTER_LINEAR)
	pad_int = int(pad)
	if pad_int > 0:
		gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
	return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _retry_suspect_frames(
	observations: list,
	corrected_first: list[float],
	raw_frames: list,
	ocr,
	args,
) -> tuple[list, list[float] | None]:
	"""对物理超限帧用备选预处理重试 OCR，选最接近邻帧期望值的读数。"""
	if args.max_accel <= 0:
		return observations, None
	max_delta_kmh = args.max_accel * (1.0 / 57.0) * 3.6  # 假设 ~17.5ms 帧间隔
	_suspects = []
	for i in range(1, len(corrected_first)):
		dv = abs(corrected_first[i] - corrected_first[i - 1])
		if dv > max_delta_kmh * 2 and abs(observations[i].raw_speed_kmh - corrected_first[i]) < 0.5:
			# 纠正没改动但跳变超物理极限
			expected = corrected_first[i - 1]  # 期望接近前一帧
			_suspects.append((i, expected))

	if not _suspects:
		return observations, None

	# 对可疑帧重试 OCR
	improved = 0
	new_obs = list(observations)
	for idx, expected in _suspects:
		ts, crop = raw_frames[idx]
		best_speed = new_obs[idx].raw_speed_kmh
		best_text = new_obs[idx].raw_text
		best_diff = abs(best_speed - expected)

		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

		# 变体1: 不缩放，直接灰度图
		proc1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc1, expected, best_speed, best_text, best_diff, args)

		# 变体2: OTSU 二值化 + 缩放
		_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		h2, w2 = th2.shape[:2]
		th = max(8.0, float(args.target_h))
		sc2 = th / h2 if h2 > 0 else 1.0
		if abs(sc2 - 1.0) > 0.02:
			th2 = cv2.resize(th2, (max(1, int(w2 * sc2)), int(th)), interpolation=cv2.INTER_LINEAR)
		proc2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc2, expected, best_speed, best_text, best_diff, args)

		# 变体3: 固定阈值 120（浅色数字）
		_, th3 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
		proc3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc3, expected, best_speed, best_text, best_diff, args)

		# 变体4: 互补引擎 — v3 模型（不同网络权重，错误模式互补）
		best_speed, best_text, best_diff = _ocr_retry_v3(
			crop, expected, best_speed, best_text, best_diff, args)

		# 变体5: 互补引擎 — 反向二值化（黑底白字→白底黑字）
		th_inv = cv2.bitwise_not(th2) if th2.shape == gray.shape else cv2.bitwise_not(
			cv2.resize(th2, (gray.shape[1], gray.shape[0])))
		h5, w5 = th_inv.shape[:2]
		sc5 = max(8.0, float(args.target_h)) / h5 if h5 > 0 else 1.0
		if abs(sc5 - 1.0) > 0.02:
			th_inv = cv2.resize(th_inv, (max(1, int(w5 * sc5)), int(max(8.0, float(args.target_h)))), interpolation=cv2.INTER_LINEAR)
		proc5 = cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc5, expected, best_speed, best_text, best_diff, args)

		if best_diff < abs(new_obs[idx].raw_speed_kmh - expected):
			new_obs[idx] = SpeedObservation(
				timestamp=new_obs[idx].timestamp,
				raw_speed_kmh=best_speed,
				raw_text=best_text,
			)
			improved += 1

	if improved > 0:
		print(f"  重试 OCR 改善 {improved} 帧, 重新纠错...")
		return new_obs, None  # 需要重新跑 DP
	return observations, corrected_first


def _ocr_retry(ocr, proc, expected, best_speed, best_text, best_diff, args):
	"""单次备选 OCR 尝试，返回（可能更新后的）best_* 值。"""
	ocr_result, _ = ocr(proc)
	sv, rt = extract_speed_value(ocr_result)
	if sv is not None and rt is not None:
		spd = sv * SOURCE_TO_KMH[args.format]
		diff = abs(spd - expected)
		if diff < best_diff:
			return spd, rt, diff
	return best_speed, best_text, best_diff


def _ocr_retry_v3(crop, expected, best_speed, best_text, best_diff, args):
	"""互补引擎重试：用 RapidOCR v3 模型（与 v5_mobile 不同网络权重，错误模式互补）。

	v3 和 v5_mobile 使用不同的检测/识别网络，错误模式互补：
	v3 极端错误少但中等偏差多，v5m 反之。
	"""
	try:
		v3_ocr = _get_v3_ocr()
		if v3_ocr is not None:
			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray = clahe.apply(gray)
			_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			h2, w2 = gray.shape[:2]
			th = max(8.0, float(args.target_h))
			sc = th / h2 if h2 > 0 else 1.0
			gray = cv2.resize(gray, (max(1, int(w2 * sc)), int(th)))
			proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
			ocr_result, _ = v3_ocr(proc)
			sv, rt = extract_speed_value(ocr_result)
			if sv is not None and rt is not None:
				spd = sv * SOURCE_TO_KMH[args.format]
				diff = abs(spd - expected)
				if diff < best_diff:
					return spd, rt, diff
	except Exception:
		pass
	return best_speed, best_text, best_diff


_v3_ocr_cache = None


def _get_v3_ocr():
	"""懒加载 v3 模型 OCR 引擎（全局单例）。"""
	global _v3_ocr_cache
	if _v3_ocr_cache is None:
		try:
			from rapidocr_onnxruntime import RapidOCR
			_v3_ocr_cache = RapidOCR()  # 默认 v3
		except Exception:
			_v3_ocr_cache = False  # 标记失败，不再重试
	return _v3_ocr_cache if _v3_ocr_cache is not False else None


def run_analysis_headless(args: argparse.Namespace) -> None:
	"""无头数据分析：从两个 CSV 导出 v-t、v-x、Δt-x 三张 PNG。"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	csv1, csv2 = Path(args.analysis[0]), Path(args.analysis[1])
	if not csv1.exists() or not csv2.exists():
		print("错误: CSV 文件不存在")
		sys.exit(1)

	out_prefix = Path(args.analysis_out) if args.analysis_out else csv1.parent / "分析"
	out_prefix.parent.mkdir(parents=True, exist_ok=True)

	# 解析 CSV
	def _read_csv(p: Path):
		times, dists, speeds, flags = [], [], [], []
		with open(p, "r", encoding="utf-8-sig") as f:
			for line in f:
				parts = line.strip().split(",")
				if len(parts) >= 3:
					times.append(float(parts[0]))
					dists.append(float(parts[1]))
					speeds.append(float(parts[2]))
					flags.append(int(parts[3]) if len(parts) > 3 else 0)
		# 去开头静止帧
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

	t1, d1, s1, f1 = _read_csv(csv1)
	t2, d2, s2, f2 = _read_csv(csv2)
	name1, name2 = csv1.stem, csv2.stem

	from scipy.signal import savgol_filter
	def _smooth(yv, strength=25):
		if len(yv) < 5:
			return yv
		win = int(len(yv) * strength / 100.0 * 0.0175)
		win = max(5, min(win, len(yv) - 2))
		if win % 2 == 0:
			win += 1
		return _savgol_filter_np(np.array(yv, dtype=float), win, min(3, win - 1))

	# ── v-t ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, name, c in [(s1, name1, "#2196F3"), (s2, name2, "#FF5722")]:
		ax.plot(t1 if data is s1 else t2, _smooth(data), color=c, linewidth=0.8, label=name)
	ax.set_xlabel("时间 (s)"); ax.set_ylabel("速度 (km/h)")
	ax.set_title("速度-时间曲线"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_v-t.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"v-t: {out_prefix}_v-t.png")

	# ── v-x ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, name, c in [(s1, name1, "#2196F3"), (s2, name2, "#FF5722")]:
		ax.plot(d1 if data is s1 else d2, _smooth(data), color=c, linewidth=0.8, label=name)
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
	ax.plot(d1, _smooth(dt), color="#2196F3", linewidth=0.8, label=f"{name1} - {name2}")
	ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)
	ax.set_xlabel("距离 (m)"); ax.set_ylabel("Δt (s)")
	ax.set_title(f"时间差-距离 ({name1} vs {name2})"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_Δt-x.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Δt-x: {out_prefix}_Δt-x.png")

	print("分析完成。")


if __name__ == "__main__":
	main()
