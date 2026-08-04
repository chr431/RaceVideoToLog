"""OCR engine — 批识别入口 + 公共符号 re-export。

识别实现为 ocr_native.OcrEngine（ONNX / TensorRT 直连，无 rapidocr）。
Flag/candidates/CSV/utilities 在 constants/ocr_text/csv_io/signals/video_utils。
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

# ── 公共符号 re-export（消费者统一 from ocr_engine import ...）──
from constants import Flag  # noqa: F401
from csv_io import (  # noqa: F401
    parse_csv_setting, csv_field_dest, parse_csv_header,
)
from ocr_text import (  # noqa: F401
    normalize_ocr_text, extract_speed_value, build_speed_candidates,
)
from signals import (  # noqa: F401
    _savgol_filter_np, _neighbor_consistency_score,
)
from video_utils import (  # noqa: F401
    VideoMetadata, SpeedObservation, format_duration,
    _parse_int_or_none, clamp_region, compute_video_hash,
)
from config import SOURCE_TO_KMH  # noqa: F401

# ═══════════════════ GPU 加速：由 gpu_setup.py 延迟初始化 ═══════════════════
from gpu_setup import select_backend as _select_backend, reset_backend as _reset_backend

def ocr_rec_batch(ocr: "OcrEngine", img_list: list) -> list:
    """批量识别（OcrEngine 直通）：按输入顺序返回结果。

    输出对象带 .txts / .scores，兼容 extract_speed_value()。
    TRT 的 batch 上限分片由 OcrEngine 内部处理。
    """
    return ocr(img_list)
