"""OCR engine — 批识别入口 + 公共符号 re-export。

识别实现为 ocr_native.OcrEngine（ONNX / TensorRT 直连，无 rapidocr）。
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

# ── 公共符号 re-export（只保留实际有消费者的 API）──
from csv_io import (  # noqa: F401
    parse_csv_setting, csv_field_dest, parse_csv_header,
    normalize_ocr_backend,
)
from ocr_text import (  # noqa: F401
    extract_speed_value,
)
from signals import (  # noqa: F401
    _savgol_filter_np,
)
from video_utils import (  # noqa: F401
    VideoMetadata, format_duration,
)
