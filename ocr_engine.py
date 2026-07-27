"""OCR engine for RaceVideoToLog.

SpeedObservation, preprocessing, neighbor consistency scoring, Flag enum,
candidate generation, SG filtering, and OCR/parsing utilities.
"""
from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from config import (MPS_TO_KMH, SOURCE_TO_KMH,
    CONSISTENCY_TIME_WINDOW, CONSISTENCY_DECAY_TAU, CONSISTENCY_PINNED_WEIGHT)

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
    "SpeedObservation", "VideoMetadata",
    "extract_speed_value", "convert_speed_to_kmh", "clamp_region",
    "build_speed_candidates",
    "normalize_ocr_text", "format_duration", "codec_from_fourcc",
    "safe_int", "safe_float", "SOURCE_TO_KMH", "OCR_NUMBER_RE",
    "compute_video_hash",
    "_neighbor_consistency_score", "_neighbor_consistency_score_lr",
    "_reset_backend", "_select_backend", "_get_model_params",
    "_CancelExport",
    "_parse_int_or_none", "parse_csv_header", "_savgol_filter_np",
    "Flag", "logger",
]

# ═══════════════════ Flag 枚举：速度数据来源标记 ═══════════════════

class Flag:
    """速度数据 flag 值 — 统一标记每帧数据的来源和可信度。

    信任层级（由低到高）:
        RAW (0)           — 原始 OCR 值，未经修正
        REOCR_AUTO (11)   — 重 OCR 自动修正
        FILL_INTERP (12)  — 物理插值填充
        PARTIAL_AUTO (13) — 部分数字模式推断修正
        HIGH_TRUST (21)   — Viterbi+物理验证，自动高可信帧
        PINNED (22)       — 用户手动修正，绝对真值
        CONFIRMED_SEG (23)— 用户确认的段内帧
    """
    RAW: int = 0
    REOCR_AUTO: int = 11
    FILL_INTERP: int = 12
    PARTIAL_AUTO: int = 13
    HIGH_TRUST: int = 21
    PINNED: int = 22
    CONFIRMED_SEG: int = 23

    @classmethod
    def is_corrected(cls, flag: int) -> bool:
        """是否为自动纠错帧 (10-19)。"""
        return 10 <= flag <= 19

    @classmethod
    def is_trusted(cls, flag: int) -> bool:
        """是否为高可信帧 — HIGH_TRUST / PINNED / CONFIRMED_SEG (20-29)。"""
        return 20 <= flag <= 29

    @classmethod
    def is_anchor(cls, flag: int) -> bool:
        """[Deprecated] Backward-compat alias for is_trusted()."""
        return cls.is_trusted(flag)


# ═══════════════════ GPU 加速：由 gpu_setup.py 延迟初始化 ═══════════════════
from gpu_setup import select_backend as _select_backend, reset_backend as _reset_backend

# ── rapidocr 延迟导入缓存 ──
# 必须在 gpu_setup 注册 CUDA/TensorRT DLL 之后才能 import rapidocr，
# 否则 tensorrt find_lib() 找不到 DLL 会导致初始化失败。
# _init_rapidocr() 在首次调用时完成导入 + monkey-patch，后续调用直接返回缓存。
_rapidocr_imported = False
_RapidOCR: "type | None" = None       # rapidocr.RapidOCR 类
_EngineType: "type | None" = None     # rapidocr.EngineType 枚举
_ModelType: "type | None" = None      # rapidocr.ModelType 枚举
_OCRVersion: "type | None" = None     # rapidocr.OCRVersion 枚举


def _init_rapidocr() -> None:
    """初始化 rapidocr：导入 + monkey-patch det/cls 跳过。

    幂等 — 第二次及后续调用立即返回。gpu_setup 的 DLL 注册
    必须在此函数调用之前完成（通常由 gui._create_ocr 或
    pipeline._ensure_ocr 中的 _select_backend() 触发）。
    """
    global _rapidocr_imported, _RapidOCR, _EngineType, _ModelType, _OCRVersion
    if _rapidocr_imported:
        return
    _rapidocr_imported = True

    from rapidocr import RapidOCR as _R, EngineType as _ET, ModelType as _MT, OCRVersion as _OV
    _RapidOCR = _R
    _EngineType = _ET
    _ModelType = _MT
    _OCRVersion = _OV

    # ── Monkey-patch: 条件加载 det/cls 模型，跳过未使用的 ONNX 模型加载 ──
    _patch_rapidocr_init()


def _patch_rapidocr_init() -> None:
    """Monkey-patch RapidOCR._initialize 以跳过未使用的 det/cls 模型加载。

    rapidocr 3.9.x 在 _initialize 中无条件创建 TextDetector 和 TextClassifier，
    即使 use_det=False / use_cls=False 也会加载对应的 ONNX 模型
    （浪费 GPU 显存 ~200MB 和初始化时间 ~2s）。
    """
    assert _RapidOCR is not None, "call _init_rapidocr() first"

    from rapidocr.ch_ppocr_det import TextDetector
    from rapidocr.ch_ppocr_cls import TextClassifier
    from rapidocr.ch_ppocr_rec import TextRecognizer
    from rapidocr.cal_rec_boxes import CalRecBoxes
    from rapidocr.utils.load_image import LoadImage

    def _patched_initialize(self, cfg):
        self.text_score = cfg.Global.text_score
        self.min_height = cfg.Global.min_height
        self.width_height_ratio = cfg.Global.width_height_ratio

        self.use_det = cfg.Global.use_det
        if self.use_det:
            cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
            cfg.Det.model_root_dir = cfg.Global.model_root_dir
            self.text_det = TextDetector(cfg.Det)
        else:
            self.text_det = None

        self.use_cls = cfg.Global.use_cls
        if self.use_cls:
            cfg.Cls.engine_cfg = cfg.EngineConfig[cfg.Cls.engine_type.value]
            cfg.Cls.model_root_dir = cfg.Global.model_root_dir
            self.text_cls = TextClassifier(cfg.Cls)
        else:
            self.text_cls = None

        self.use_rec = cfg.Global.use_rec
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.font_path = cfg.Global.font_path
        cfg.Rec.model_root_dir = cfg.Global.model_root_dir
        self.text_rec = TextRecognizer(cfg.Rec)

        self.load_img = LoadImage()
        self.max_side_len = cfg.Global.max_side_len
        self.min_side_len = cfg.Global.min_side_len
        self.return_word_box = cfg.Global.return_word_box
        self.return_single_char_box = cfg.Global.return_single_char_box
        self.cal_rec_boxes = CalRecBoxes()

        self.cfg = cfg

    _RapidOCR._initialize = _patched_initialize  # type: ignore[method-assign]
    logger.info("RapidOCR patched: det/cls model loading conditional on use_det/use_cls")
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
    raw_speed_kmh: int
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


# ═══════════════════ CSV 头字段定义（CLI/GUI 共用） ═══════════════════
# CSV key → (argparse_dest, value_type)
# value_type: "roi" | "int" | "float" | "str"
# 新增字段时在此添加即可，CLI 和 GUI 自动同步。
_CSV_FIELD_MAP: dict[str, tuple[str, str]] = {
    "roi":           ("roi",           "roi"),
    "format":        ("format",        "str"),
    "max_speed":     ("max_speed",     "float"),
    "max_accel":     ("max_accel",     "float"),
    "div":           ("div",           "int"),
    "target_h":      ("target_h",      "int"),
    "pad":           ("pad",           "int"),
    "backend":       ("backend",       "str"),
    "buffer":        ("buffer",        "int"),
    "frame_start":   ("frame_start",   "int"),
    "frame_end":     ("frame_end",     "int"),
    "model":         ("ocr_model",     "str"),
    "reocr_model":   ("reocr_model",   "str"),
    "video_backend": ("video_backend", "str"),
}


def parse_csv_setting(key: str, raw_value: str):
    """Parse a single CSV header value according to its declared type.

    Returns the parsed value, or None if parsing fails silently.
    CLI and GUI both call this to avoid duplicating type-cast logic.
    """
    field = _CSV_FIELD_MAP.get(key)
    if field is None:
        return None  # unknown key — caller should skip
    vtype = field[1]
    if vtype == "roi":
        try:
            parts = [int(x.strip()) for x in raw_value.split(",")]
            return parts if len(parts) == 4 else None
        except ValueError:
            return None
    elif vtype == "int":
        try:
            return int(raw_value)
        except ValueError:
            return None
    elif vtype == "float":
        try:
            return float(raw_value)
        except ValueError:
            return None
    else:
        return raw_value


def csv_field_dest(key: str) -> str | None:
    """Return the argparse dest name for a CSV header key, or None if unknown."""
    field = _CSV_FIELD_MAP.get(key)
    return field[0] if field else None


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


def extract_speed_value(ocr_result: "object | None") -> tuple[float | None, str | None, float]:
    """从 RapidOCR 3.x 结果中提取速度值和置信度。

    Returns: (speed_value, raw_text, confidence)
        confidence: 0.0-1.0 (含), 0.0 表示置信度不可用
    """
    if not ocr_result:
        return None, None, 0.0

    # RapidOCR 3.x TextRecOutput (use_det=False 时返回)
    if hasattr(ocr_result, "txts"):
        txts = ocr_result.txts  # type: ignore[attr-defined]
        if not txts or not txts[0]:
            return None, None, 0.0
        scores = getattr(ocr_result, "scores", [])
        conf = float(scores[0]) if scores else 0.0
        text = str(txts[0]).strip()
        if not text:
            return None, None, conf
        normalized = normalize_ocr_text(text).replace(" ", "")
        match = OCR_NUMBER_RE.search(normalized)
        if not match:
            return None, None, conf
        raw_text = re.sub(r"\D", "", match.group(0))
        if not raw_text:
            return None, None, conf
        try:
            return int(float(raw_text)), raw_text, conf
        except ValueError:
            return None, None, conf

    # RapidOCR 3.x 带检测时返回 tuple (dt_boxes, rec_res, elapse)
    if isinstance(ocr_result, (tuple, list)) and len(ocr_result) >= 2:
        rec = ocr_result[1]
        if rec is None:
            return None, None, 0.0
        candidates: list[str] = []
        if isinstance(rec, list):
            for item in rec:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[1]).strip()
                elif hasattr(item, "txts"):
                    if item.txts and item.txts[0]:  # type: ignore[attr-defined]
                        text = str(item.txts[0]).strip()  # type: ignore[attr-defined]
                    else:
                        continue
                elif hasattr(item, "text"):
                    text = str(item.text).strip()
                else:
                    text = str(item).strip()
                if text:
                    candidates.append(text)
        elif hasattr(rec, "txts"):
            if rec.txts and rec.txts[0]:  # type: ignore[attr-defined]
                candidates.append(str(rec.txts[0]).strip())  # type: ignore[attr-defined]
        elif hasattr(rec, "text"):
            candidates.append(str(rec.text).strip())  # type: ignore[attr-defined]
        elif isinstance(rec, str):
            candidates.append(rec.strip())

        if not candidates:
            return None, None, 0.0
        joined = normalize_ocr_text(" ".join(candidates)).replace(" ", "")
        match = OCR_NUMBER_RE.search(joined)
        if not match:
            return None, None, 0.0
        raw_text = re.sub(r"\D", "", match.group(0))
        if not raw_text:
            return None, None, 0.0
        try:
            return float(raw_text), raw_text, 0.0
        except ValueError:
            return None, None, 0.0

    return None, None, 0.0


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


def build_speed_candidates(raw_text: str, max_speed_kmh: float) -> list[int]:
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

    candidates: set[int] = set()

    # 策略1: 保留原始值
    try:
        val = int(text)
        if val <= max_speed_int:
            candidates.add(int(val))
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
            candidates.add(int(candidate))

    # 策略3: 常见 OCR 字符混淆替换（对称映射）
    _CONFUSION_MAP = {
        "0": ["8", "6", "9"],
        "1": ["7", "2", "4", "9"],
        "2": ["7", "1", "3", "9"],
        "3": ["8", "9", "2", "5"],
        "4": ["7", "9", "1"],
        "5": ["6", "3", "8", "9"],
        "6": ["8", "5", "0", "2"],
        "7": ["1", "2", "4"],
        "8": ["0", "6", "3", "5", "9"],
        "9": ["8", "3", "5", "0", "4", "1", "2"],
    }
    for i, ch in enumerate(text):
        for alt in _CONFUSION_MAP.get(ch, []):
            altered = text[:i] + alt + text[i+1:]
            try:
                val = int(altered)
                if val <= max_speed_int:
                    candidates.add(int(val))
            except ValueError:
                pass

    return sorted(candidates)


def _get_model_params(variant: str, engine_type: str = "onnxruntime") -> dict | None:
    """Get RapidOCR params dict for the model variant. Returns None if unsupported.

    variant: "v6_tiny" | "v6_small"
    engine_type: "onnxruntime" | "tensorrt"
    """
    _init_rapidocr()
    assert _ModelType is not None and _EngineType is not None and _OCRVersion is not None
    size = variant.replace("v6_", "")
    model_map = {"tiny": _ModelType.TINY, "small": _ModelType.SMALL}
    model_type = model_map.get(size)
    if model_type is None:
        return None
    _et = {"tensorrt": _EngineType.TENSORRT}.get(engine_type, _EngineType.ONNXRUNTIME)
    return {
        "Global.use_det": False,
        "Global.use_cls": False,
        "Det.model_type": model_type,
        "Rec.model_type": model_type,
        "Det.ocr_version": _OCRVersion.PPOCRV6,
        "Rec.ocr_version": _OCRVersion.PPOCRV6,
        "Det.engine_type": _et,
        "Rec.engine_type": _et,
        "Rec.rec_batch_num": config.OCR_REC_BATCH_NUM,
    }


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






def _neighbor_consistency_score_lr(i: int, v: float, rows: list, times: list[float],
                    max_speed_kmh: float, max_accel_mps2: float,
                    time_window: float = CONSISTENCY_TIME_WINDOW, tau: float = CONSISTENCY_DECAY_TAU,
                    high_weight: set[int] | None = None) -> tuple[float, float]:
    """邻域左右分侧一致性评分。权重 = exp(-dt/tau)。

    Returns: (left_score, right_score)
    - 单侧无邻居时默认 1.0（不惩罚边界帧）
    - high_weight 中的帧额外 ×CONSISTENCY_PINNED_WEIGHT 权重
    """
    import math
    n = len(rows)
    if v < 0 or v > max_speed_kmh:
        return 0.0, 0.0
    t_i = times[i]
    hw = high_weight or set()

    def _scan(start: int, stop: int, step: int) -> float:
        votes = 0.0
        total = 0.0
        for j in range(start, stop, step):
            dt = abs(times[j] - t_i)
            if dt > time_window:
                break
            v_j = rows[j][2]
            if v_j < 0 or v_j > max_speed_kmh:
                continue
            max_dv = max_accel_mps2 * dt * MPS_TO_KMH
            exp_w = math.exp(-dt / tau)
            pin_w = CONSISTENCY_PINNED_WEIGHT if j in hw else 1.0
            total += exp_w * pin_w
            if abs(v - v_j) <= max_dv:
                votes += exp_w * pin_w
        return votes / total if total > 0 else 1.0

    left = _scan(i - 1, -1, -1)
    right = _scan(i + 1, n, 1)
    return left, right


def _neighbor_consistency_score(i: int, v: float, rows: list, times: list[float],
                            max_speed_kmh: float, max_accel_mps2: float,
                            time_window: float = CONSISTENCY_TIME_WINDOW, tau: float = CONSISTENCY_DECAY_TAU,
                            high_weight: set[int] | None = None) -> float:
    """邻域一致性合并分数（向后兼容）。左右侧权重合并后的单值分数。"""
    left, right = _neighbor_consistency_score_lr(i, v, rows, times, max_speed_kmh, max_accel_mps2,
                                    time_window, tau, high_weight)
    return (left + right) / 2.0


class _CancelExport(Exception):
    """内部异常：用户取消了导出任务。"""
    pass
