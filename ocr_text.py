"""OCR 文本提取与候选生成。"""
from __future__ import annotations
import math
import re

from config import SOURCE_TO_KMH
from constants import OCR_NUMBER_RE, CONFUSION_MAP as _CONFUSION_MAP


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
        return _extract_speed_from_text(str(txts[0]), conf)

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
