"""最终错误案例代表帧的 OCR 行为锁定（CI 可跑：无需 decord / GPU）。

夹具由 tools/make_regression_fixtures.py 生成：
- tests/fixtures/ocr_frames/*.npy —— 最终错误段代表帧的 ROI 裁剪
  （YUV420 生产模式下存 Y 平面 H,W,1，即 OCR 实际输入）
- tests/fixtures/ocr_frames/manifest.json —— 每帧期望的 OCR raw 值

本测试锁定 OCR 预处理 + 模型 + 文本解析的基线行为：若任一案例读数改变
（无论更好或更坏），测试失败。此时该改动属于有意改动：先在本机跑完整
漏斗（tools/accuracy_breakdown.py）确认 5 错误基线无回归，再用
make_regression_fixtures.py 重新生成本夹具与 seg_series 夹具。

确定性：onnxruntime CPU + RVTOL_OCR_THREADS=1（单线程推理，跨机器
逐位一致）。若 onnxruntime 版本升级导致浮点内核差异翻转某案例读数，
优先考虑在 pyproject 固定 onnxruntime 版本。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

FRAMES = Path(__file__).parent / "fixtures" / "ocr_frames"


def _manifest() -> dict:
    with open(FRAMES / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _cases():
    return _manifest()["cases"]


@pytest.fixture(scope="module")
def engine():
    # 单线程推理：确定性（CI 与本机逐位一致）
    os.environ.setdefault("RVTOL_OCR_THREADS", "1")
    from ocr_native import OcrEngine
    return OcrEngine("v6_small", "onnxruntime", fill_width=224)


@pytest.mark.parametrize(
    "case", _cases(),
    ids=lambda c: f"{c['video']}_f{c['rep_frame']}_raw{c['expected_raw']}")
def test_ocr_frame_matches_baseline(engine, case):
    from video_utils import _preprocess_standard
    from ocr_text import extract_speed_value
    crop = np.load(FRAMES / case["file"])
    proc = _preprocess_standard(crop, force_aspect=0.0)
    res = engine([proc])[0]
    sv, _txt, _conf = extract_speed_value(res)
    sv = int(sv) if sv is not None else None
    assert sv == case["expected_raw"], (
        f"{case['video']} 帧 {case['rep_frame']}: OCR 读出 {sv}，"
        f"基线为 {case['expected_raw']}（truth={case['truth']}）。"
        "若改动有意，先跑完整漏斗确认无回归，再重新生成夹具。")


def test_manifest_has_error_cases():
    """夹具完整性：0 错误案例口径（全部视频 final=0，v2.16 truth 晋升后）。"""
    cases = _cases()
    assert len(cases) == 0
    by_video = {}
    for c in cases:
        by_video[c["video"]] = by_video.get(c["video"], 0) + 1
    assert by_video == {}


def test_all_frames_exist():
    for c in _cases():
        assert (FRAMES / c["file"]).exists(), f"缺夹具帧: {c['file']}"
