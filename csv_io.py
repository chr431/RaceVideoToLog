"""CSV 头部字段解析（CLI/GUI 共用）。"""
from __future__ import annotations
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)




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
    "max_width":     ("max_width",     "int"),
    "pad":           ("pad",           "int"),
    "backend":       ("backend",       "str"),
    "buffer":        ("buffer",        "int"),
    "frame_start":   ("frame_start",   "int"),
    "frame_end":     ("frame_end",     "int"),
    "model":         ("ocr_model",     "str"),
    "reocr_model":   ("reocr_model",   "str"),
    "fps":          ("fps",          "float"),
    "codec":        ("codec",        "str"),
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
