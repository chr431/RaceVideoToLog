"""OCR engine for RaceVideoToLog.

SpeedObservation, preprocessing, correction algorithms,
model configuration, and supporting utilities.
"""
from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from config import MPS_TO_KMH, SOURCE_TO_KMH

# Lazy matplotlib font config (no import-time side effects)
_matplotlib_configured = False

def _ensure_matplotlib_fonts() -> None:
	"""配置 matplotlib 中文字体支持（幂等，可多次调用）。"""
	global _matplotlib_configured
	if not _matplotlib_configured:
		try:
			import matplotlib
			matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
			matplotlib.rcParams["axes.unicode_minus"] = False
		except ImportError:
			pass
		_matplotlib_configured = True

# 确保字体配置在 import 时生效（所有模块在创建 Figure 前都已导入 ocr_engine）
_ensure_matplotlib_fonts()

# ── 模块级 Logger ──
logger = logging.getLogger("RaceVideoToLog.ocr_engine")

# ── 导出列表：包含 _ 前缀的私有符号供 RaceVideoToLog.py / headless.py 使用 ──
__all__ = [
	"SpeedObservation", "VideoMetadata", "RapidOCR",
	"extract_speed_value", "convert_speed_to_kmh", "clamp_region",
	"build_speed_candidates",
	"normalize_ocr_text", "format_duration", "codec_from_fourcc",
	"safe_int", "safe_float", "SOURCE_TO_KMH", "OCR_NUMBER_RE",
	"compute_video_hash", "auto_select_anchors",
	"_reset_backend", "_select_backend", "_get_model_kwargs",
	"_CancelExport",
	"_parse_int_or_none", "parse_csv_header", "_savgol_filter_np",
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


# ═══════════════════ GPU 加速：由 gpu_setup.py 延迟初始化 ═══════════════════
from gpu_setup import select_backend as _select_backend, reset_backend as _reset_backend

from rapidocr_onnxruntime import RapidOCR
# ═══════════════════════════════════════════════════════════

# SOURCE_TO_KMH 已从 config 导入

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


def _get_model_kwargs(variant: str, models_dir: str | None = None) -> dict | None:
	"""Get RapidOCR kwargs for the model. Returns None if files missing.

	variant: "v6_small" | "v6_medium" — 去掉 v6_ 前缀后匹配模型文件名。
	"""
	import rapidocr_onnxruntime as rr
	if models_dir is None:
		models_dir = str(Path(rr.__file__).parent / "models")
	# 将 v6_small → small, v6_medium → medium
	size = variant.replace("v6_", "")
	cfg = {
		"det_model_path": f"{models_dir}/PP-OCRv6_det_{size}.onnx",
		"rec_model_path": f"{models_dir}/PP-OCRv6_rec_{size}.onnx",
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




def _anchor_adaptive_window(times: list[float], n: int) -> int:
	"""计算自适应窗口大小：覆盖约 0.3s 的数据，最小 5 帧。"""
	typical_dt = (times[-1] - times[0]) / max(n - 1, 1) if n > 1 else 0.017
	return max(5, int(0.3 / max(typical_dt, 0.001)) | 1)


def _anchor_select_center(raw_vals: list[float], times: list[float], n: int,
                           window: int, max_speed_kmh: float,
                           max_dev: float) -> set[int]:
	"""Center region: median filter anchor selection."""
	half = window // 2
	anchors: set[int] = set()
	for i in range(half, n - half):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(i - half, i + half + 1)
		          if j != i and 0 < raw_vals[j] <= max_speed_kmh]
		if len(local) < 3:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)
	return anchors


def _anchor_select_boundaries(raw_vals: list[float], n: int, window: int,
                               max_speed_kmh: float, max_dev: float) -> set[int]:
	"""Head and tail boundary anchor selection."""
	anchors: set[int] = set()
	# Head
	for i in range(0, window // 2):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(0, min(window, n))
		          if j != i and 0 < raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)
	# Tail
	for i in range(n - window // 2, n):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(max(0, n - window), n)
		          if j != i and 0 < raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)
	return anchors


def _anchor_validate_neighbors(anchors: set[int], raw_vals: list[float],
                                n: int) -> set[int]:
	"""Remove anchors that are extreme outliers vs immediate neighbors."""
	filtered: set[int] = set()
	for i in anchors:
		v = raw_vals[i]
		left_ok = (i > 0 and raw_vals[i - 1] > 0
		           and abs(v - raw_vals[i - 1]) <= 10.0)
		right_ok = (i + 1 < n and raw_vals[i + 1] > 0
		            and abs(raw_vals[i + 1] - v) <= 10.0)
		if left_ok or right_ok:
			filtered.add(i)
	return filtered


def _anchor_wide_window(anchors: set[int], raw_vals: list[float],
                          n: int, max_dev: float = 20.0) -> set[int]:
	"""宽窗口一致性：剔除偏离 ±30 帧邻域中位数超过 max_dev 的锚点。

	解决局部验证无法检测的"多步渐进漂移"——每步变化都在容限内，
	但累计偏移物理不可能（例如 216→117 在 0.12s 内的 99 km/h 变化）。
	"""
	if len(anchors) <= 5:
		return anchors
	WINDOW = 30
	validated: set[int] = set()
	for i in anchors:
		v = raw_vals[i]
		# 收集 ±WINDOW 内所有有效值（排除自身和紧邻 ±5 帧以检查更广趋势）
		context = []
		for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1)):
			if abs(j - i) > 5 and raw_vals[j] > 0:
				context.append(raw_vals[j])
		if len(context) < 10:
			validated.add(i)  # 上下文不足则保留
			continue
		context.sort()
		median = context[len(context) // 2]
		if abs(v - median) <= max_dev:
			validated.add(i)
	return validated


def _anchor_graph_consistency(anchors: set[int], raw_vals: list[float],
                               times: list[float], n: int,
                               max_accel_mps2: float) -> set[int]:
	"""图连通性验证：仅保留可在物理约束下互相到达的最大锚点集合。

	将候选锚点建为图节点，若两锚点间加速度不超限则连边。
	保留最大连通分量，自动剔除与多数锚点物理矛盾的孤立异常点。
	"""
	if len(anchors) <= 2:
		return anchors
	sorted_anchors = sorted(anchors)
	max_dv_per_sec = max_accel_mps2 * MPS_TO_KMH * 2.0

	# 建邻接表（仅连相邻锚点以减少 O(N²)，连通性传递保证全局一致）
	adj: dict[int, list[int]] = {a: [] for a in sorted_anchors}
	for k in range(len(sorted_anchors) - 1):
		i = sorted_anchors[k]
		for j in sorted_anchors[k + 1:]:
			dt = times[j] - times[i]
			if dt <= 0:
				continue
			dv = abs(raw_vals[j] - raw_vals[i])
			if dv / dt <= max_dv_per_sec:
				adj[i].append(j)
				adj[j].append(i)
			# 若间距过大（>5s），不再连更远的（物理相关性弱）
			if dt > 5.0:
				break

	# DFS 找最大连通分量
	visited: set[int] = set()
	largest: set[int] = set()
	for a in sorted_anchors:
		if a in visited:
			continue
		stack = [a]
		comp: set[int] = set()
		while stack:
			node = stack.pop()
			if node in comp:
				continue
			comp.add(node)
			visited.add(node)
			stack.extend(nb for nb in adj[node] if nb not in comp)
		if len(comp) > len(largest):
			largest = comp

	return largest


def auto_select_anchors(observations: list["SpeedObservation"],
                         max_speed_kmh: float = 400.0, window: int = 0,
                         max_dev: float = 4.0,
                         max_accel_mps2: float = 50.0) -> set[int]:
	"""Select reliable OCR frames as Correction B anchors.

	5 阶段流水线：自适应窗口 → 中位数筛选 → 邻居验证 → 宽窗口去漂移 → 图连通性验证。
	Returns: trusted frame indices.
	"""
	n = len(observations)
	raw_vals = [o.raw_speed_kmh for o in observations]
	times = [o.timestamp for o in observations]

	# 阶段 1: 自适应窗口
	if window <= 0:
		window = _anchor_adaptive_window(times, n)

	# 阶段 2: 中位数筛选 (center + boundaries)
	anchors = _anchor_select_center(raw_vals, times, n, window,
                                    max_speed_kmh, max_dev)
	anchors |= _anchor_select_boundaries(raw_vals, n, window,
                                         max_speed_kmh, max_dev)

	# 阶段 3: 邻居验证
	anchors = _anchor_validate_neighbors(anchors, raw_vals, n)

	# 阶段 4: 宽窗口漂移检查（防多步渐进漂移）
	anchors = _anchor_wide_window(anchors, raw_vals, n, max_dev=20.0)

	# 阶段 5: 图连通性验证
	anchors = _anchor_graph_consistency(anchors, raw_vals, times, n,
                                     max_accel_mps2)

	return anchors

class _CancelExport(Exception):
	"""内部异常：用户取消了导出任务。"""
	pass


