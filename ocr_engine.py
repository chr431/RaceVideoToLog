"""OCR engine — RapidOCR initialization, batch recognition, model params.

Flag/candidates/CSV/utilities live in their own modules (constants,
ocr_text, csv_io, signals, video_utils); this module re-exports them
for backward compatibility.
"""
from __future__ import annotations
import logging
from pathlib import Path

import config

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
    "_parse_int_or_none", "parse_csv_header", "_savgol_filter_np",
    "Flag", "logger",
]

# ── 新模块 re-export（向后兼容：消费者可直接 from ocr_engine import ...）──
from constants import Flag, OCR_NUMBER_RE  # noqa: F401
from csv_io import (  # noqa: F401
    _CSV_FIELD_MAP, parse_csv_setting, csv_field_dest, parse_csv_header,
)
from ocr_text import (  # noqa: F401
    normalize_ocr_text, _extract_speed_from_text,
    extract_speed_value, convert_speed_to_kmh, build_speed_candidates,
)
from signals import (  # noqa: F401
    _savgol_filter_np, _neighbor_consistency_score_lr,
    _neighbor_consistency_score,
)
from video_utils import (  # noqa: F401
    VideoMetadata, SpeedObservation, format_duration, codec_from_fourcc,
    safe_int, safe_float, _parse_int_or_none, clamp_region,
    compute_video_hash,
)
from config import SOURCE_TO_KMH  # noqa: F401

# ═══════════════════ Flag 枚举：速度数据来源标记 ═══════════════════



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


def ocr_rec_batch(ocr: "object", img_list: list) -> list:
    """批量识别：一次 session.run 处理多帧，按输入顺序返回结果。

    通过 RapidOCR 的 text_rec 直接批处理（跳过 __call__ 的
    load_img/preprocess/build_final_output 包装开销，实测 ~3.3x 提速）。
    任一步骤失败时回退为逐帧调用，保证功能不受影响。

    Returns: list，每项兼容 extract_speed_value()（TextRecOutput 或等效对象）。
    """
    if not img_list:
        return []
    try:
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        out = ocr.text_rec(TextRecInput(img=list(img_list), return_word_box=False))
        txts = out.txts or ()
        scores = out.scores or [0.0] * len(img_list)
        # 构造轻量对象，兼容 extract_speed_value 的 hasattr("txts") 分支
        results: list = []
        for i in range(len(img_list)):
            item = type("_BatchRecOut", (), {})()
            item.txts = (txts[i],) if i < len(txts) else ()  # type: ignore[attr-defined]
            item.scores = [float(scores[i])] if i < len(scores) else []  # type: ignore[attr-defined]
            results.append(item)
        return results
    except Exception:
        # 回退：逐帧调用标准路径（与旧行为完全一致）
        return [ocr(im) for im in img_list]








# ── SG 滤波系数缓存（(window_length, polyorder) → coefficients）──
_sg_coeff_cache: dict[tuple[int, int], np.ndarray] = {}



# 数字混淆映射已移至 constants.py（CONFUSION_MAP）

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











