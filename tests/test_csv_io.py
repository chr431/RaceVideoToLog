"""csv_io 单元测试：CSV 头解析边界、类型转换失败路径、字段映射。"""
from __future__ import annotations

import pytest

from csv_io import (parse_csv_header, parse_csv_setting, csv_field_dest,
                    normalize_ocr_backend)


# ═══════════════ parse_csv_header ═══════════════

def test_parse_header_basic(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(
        "# video=test5.mp4, fps=59.767\n"
        "# roi=843,993,948,1025, format=km/h, frame_start=362\n"
        "362,0.00,257,21\n",
        encoding="utf-8")
    s = parse_csv_header(str(p))
    assert s == {"video": "test5.mp4", "fps": "59.767",
                 "roi": "843,993,948,1025", "format": "km/h",
                 "frame_start": "362"}


def test_parse_header_stops_at_data(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("# a=1\n100,0.0,100,0\n# b=2\n", encoding="utf-8")
    s = parse_csv_header(str(p))
    assert s == {"a": "1"}, "数据行后的 # 注释不再当头部（防误导入）"


def test_parse_header_empty_and_missing(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("# video=, fps=\n# roi=1,2,3,4\n", encoding="utf-8")
    s = parse_csv_header(str(p))
    assert s["video"] == ""
    assert s["fps"] == ""
    assert s["roi"] == "1,2,3,4"


def test_parse_header_no_header(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("0,0.0,100,0\n", encoding="utf-8")
    assert parse_csv_header(str(p)) == {}


def test_parse_header_missing_file(tmp_path):
    s = parse_csv_header(str(tmp_path / "nope.csv"))
    assert s == {}, "文件不存在应静默返回空 dict（调用方自行处理）"


def test_parse_header_bom(tmp_path):
    p = tmp_path / "h.csv"
    p.write_bytes("\ufeff# max_speed=400.0\n".encode("utf-8"))
    s = parse_csv_header(str(p))
    assert s == {"max_speed": "400.0"}, "utf-8-sig BOM 应被剥掉"


# ═══════════════ parse_csv_setting 类型转换 ═══════════════

def test_parse_setting_types():
    assert parse_csv_setting("roi", "1, 2,3,4") == [1, 2, 3, 4]
    assert parse_csv_setting("max_speed", "400.0") == 400.0
    assert parse_csv_setting("max_accel", "50") == 50.0
    assert parse_csv_setting("fill_width", "224") == 224
    assert parse_csv_setting("format", "km/h") == "km/h"
    assert parse_csv_setting("model", "v6_small") == "v6_small"
    assert parse_csv_setting("ocr_backend", "onnxruntime") == "onnxruntime"


def test_parse_setting_failures():
    assert parse_csv_setting("roi", "1,2,3") is None       # 缺一维
    assert parse_csv_setting("roi", "1,2,x,4") is None     # 非数字
    assert parse_csv_setting("fill_width", "abc") is None
    assert parse_csv_setting("max_speed", "") is None
    assert parse_csv_setting("unknown_key", "1") is None


# ═══════════════ csv_field_dest 映射 ═══════════════

def test_field_dest_mapping():
    assert csv_field_dest("roi") == "roi"
    assert csv_field_dest("max_speed") == "max_speed"
    assert csv_field_dest("model") == "ocr_model"          # key ≠ dest
    assert csv_field_dest("fps") == "fps"
    assert csv_field_dest("codec") == "codec"
    assert csv_field_dest("ocr_backend") == "ocr_backend"  # 实际引擎 → argparse dest
    assert csv_field_dest("no_such") is None


# ═══════════════ ocr_backend 实际引擎归一化 ═══════════════

def test_normalize_ocr_backend_actual_engine():
    assert normalize_ocr_backend("onnxruntime") == "cpu"
    assert normalize_ocr_backend("tensorrt") == "tensorrt"
    assert normalize_ocr_backend("cpu") == "cpu"
    assert normalize_ocr_backend("auto") == "auto"
    # 实验混合（env 开关）：GUI/CLI 无对应项，归一化到 auto
    assert normalize_ocr_backend("tensorrt+onnxruntime") == "auto"
    assert normalize_ocr_backend(" Unknown ") is None
