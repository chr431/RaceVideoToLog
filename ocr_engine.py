"""OCR engine — 批识别入口 + 旧模块 re-export（向后兼容）。

识别实现已迁移至 ocr_native.OcrEngine（ONNX / TensorRT 直连，无 rapidocr）。
Flag/candidates/CSV/utilities 在 constants/ocr_text/csv_io/signals/video_utils。
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

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

def ocr_rec_batch(ocr: "OcrEngine", img_list: list) -> list:
    """批量识别（OcrEngine 直通）：按输入顺序返回结果。

    输出对象带 .txts / .scores，兼容 extract_speed_value()。
    TRT 的 batch 上限分片由 OcrEngine 内部处理。
    """
    return ocr(img_list)


# ── SG 滤波系数缓存（(window_length, polyorder) → coefficients）──
_sg_coeff_cache: dict[tuple[int, int], np.ndarray] = {}



# 数字混淆映射已移至 constants.py（CONFUSION_MAP）
