"""from-csv 设置合并优先级测试（apply_csv_settings 纯函数）。

锁定语义：命令行显式写出的参数（即使等于默认值）优先于 CSV 头 ——
历史 bug：值==默认值被误判为"未指定"，被 CSV 静默覆盖（引擎静默换模型）。
"""
from __future__ import annotations

from types import SimpleNamespace

from RaceVideoToLog import apply_csv_settings

DEFAULTS = dict(
    roi=None, format="km/h", max_speed=400.0, max_accel=50.0,
    fill_width=224, force_aspect=0.0, buffer=128, decode_backend="auto",
    ocr_backend="auto", frame_start=None, frame_end=None, ocr_model="v6_small",
    output=None,
)


def _args(**kw):
    d = dict(video=None, from_csv=None, **DEFAULTS)
    d.update(kw)
    return SimpleNamespace(**d)


def _csv(tmp_path, header: str) -> str:
    p = tmp_path / "in.csv"
    p.write_text(header + "\n0,0.0,100,0\n", encoding="utf-8")
    return str(p)


# ═══════════════ CSV 覆盖默认值（无显式参数） ═══════════════

def test_csv_fills_defaults(tmp_path):
    csv = _csv(tmp_path, "# max_speed=320.0, fill_width=160, "
               "roi=10,20,30,40, max_accel=45")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    assert args.max_speed == 320.0
    assert args.fill_width == 160
    assert args.roi == [10, 20, 30, 40]
    assert args.max_accel == 45.0
    assert args.format == "km/h", "CSV 未提的字段保持默认"


# ═══════════════ 显式参数优先（核心回归） ═══════════════

def test_explicit_arg_equal_to_default_wins(tmp_path):
    """历史 bug：--fill-width 224（恰等于默认值）曾被 CSV 的 160 静默覆盖。"""
    csv = _csv(tmp_path, "# fill_width=160")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog", "--fill-width", "224"])
    assert args.fill_width == 224, "显式参数即使等于默认值也必须赢 CSV"


def test_explicit_arg_differs_wins(tmp_path):
    csv = _csv(tmp_path, "# max_speed=320.0")
    # args 反映 argparse 已解析的 argv（--max-speed 300 → max_speed=300）
    args = _args(from_csv=csv, max_speed=300.0)
    apply_csv_settings(args, DEFAULTS, argv=["prog", "--max-speed", "300"])
    assert args.max_speed == 300.0


def test_explicit_arg_equals_form_wins(tmp_path):
    csv = _csv(tmp_path, "# buffer=64")
    args = _args(from_csv=csv, buffer=32)
    apply_csv_settings(args, DEFAULTS, argv=["prog", "--buffer=32"])
    assert args.buffer == 32, "--flag=value 形式也应被识别为显式指定"


# ═══════════════ ocr_backend 实际引擎导入 ═══════════════

def test_csv_ocr_backend_actual_engine_normalized(tmp_path):
    """CSV 只写实际引擎：onnxruntime 应归一化为可请求的 cpu。"""
    csv = _csv(tmp_path, "# ocr_backend=onnxruntime")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    assert args.ocr_backend == "cpu"


def test_csv_ocr_backend_tensorrt_roundtrip(tmp_path):
    csv = _csv(tmp_path, "# ocr_backend=tensorrt")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    assert args.ocr_backend == "tensorrt"


def test_csv_ocr_backend_hybrid_maps_auto(tmp_path):
    csv = _csv(tmp_path, "# ocr_backend=tensorrt+onnxruntime")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    assert args.ocr_backend == "auto"


def test_explicit_ocr_backend_wins_over_csv(tmp_path):
    csv = _csv(tmp_path, "# ocr_backend=onnxruntime")
    args = _args(from_csv=csv, ocr_backend="tensorrt")
    apply_csv_settings(args, DEFAULTS, argv=["prog", "--ocr-backend", "tensorrt"])
    assert args.ocr_backend == "tensorrt"


# ═══════════════ 只读字段与坏值 ═══════════════

def test_readonly_fields_skipped(tmp_path):
    csv = _csv(tmp_path, "# fps=59.767, codec=h264, video=test5.mp4")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    # fps/codec/video 无 argparse dest —— 静默跳过，不抛异常
    assert not hasattr(args, "fps")


def test_invalid_values_skipped(tmp_path):
    csv = _csv(tmp_path, "# fill_width=abc, max_speed=, roi=1,2,3")
    args = _args(from_csv=csv)
    apply_csv_settings(args, DEFAULTS, argv=["prog"])
    assert args.fill_width == 224
    assert args.max_speed == 400.0
    assert args.roi is None


def test_no_from_csv_noop(tmp_path):
    # 无 from_csv → 函数不得触碰 args（即使 argv 有显式参数）
    args = _args(from_csv=None, max_speed=300.0)
    apply_csv_settings(args, DEFAULTS, argv=["prog", "--max-speed", "300"])
    assert args.max_speed == 300.0
