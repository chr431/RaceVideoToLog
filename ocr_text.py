"""OCR 文本提取（速度数字解析）。"""
from __future__ import annotations
import re

from constants import OCR_NUMBER_RE


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


def _extract_speed_from_text(raw_text: str, conf: float) -> tuple[float | None, str | None, float]:
    """从单行识别文本提取速度值（extract_speed_value 的核心逻辑）。

    Returns: (speed_value, raw_text, confidence)
    """
    text = raw_text.strip()
    if not text:
        return None, None, conf
    normalized = normalize_ocr_text(text).replace(" ", "")
    match = OCR_NUMBER_RE.search(normalized)
    if not match:
        return None, None, conf
    digits = re.sub(r"\D", "", match.group(0))
    if not digits:
        return None, None, conf
    try:
        return int(float(digits)), digits, conf
    except ValueError:
        return None, None, conf


def extract_speed_value(ocr_result: "object | None") -> tuple[float | None, str | None, float]:
    """从 OCR 结果（OcrEngine RecOut，带 .txts/.scores）中提取速度值和置信度。

    Returns: (speed_value, raw_text, confidence)
        confidence: 0.0-1.0 (含), 0.0 表示置信度不可用
    """
    if not ocr_result:
        return None, None, 0.0

    if hasattr(ocr_result, "txts"):
        txts = ocr_result.txts  # type: ignore[attr-defined]
        if not txts or not txts[0]:
            return None, None, 0.0
        scores = getattr(ocr_result, "scores", [])
        conf = float(scores[0]) if scores else 0.0
        return _extract_speed_from_text(str(txts[0]), conf)

    return None, None, 0.0
