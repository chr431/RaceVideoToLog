"""段级纠错链回归（CI 可跑：无需视频 / decord / OCR 引擎）。

夹具 = 生产 run() 的全量段级序列（tests/fixtures/seg_series/<video>.json，
由 tools/make_regression_fixtures.py 生成）。测试用 SegmentPipeline 重构
_confidence + _dense_correct，断言：

1. 逐段纠正输出与基线完全一致 —— 任何 _confidence/_dense_correct/DP
   参数改动导致输出变化都会在此失败。属于有意改动时：本机先跑完整漏斗
   （tools/accuracy_breakdown.py）确认 12 错误基线无回归，再重新生成夹具。
2. 最终错误（|corr-truth|>tol）的集合与基线一致（当前口径：test 4 /
   test2 8 / test3/5/6 0，合计 12 —— 与 CLAUDE.md 基线同步）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from segment_flow import SegmentPipeline

SERIES = Path(__file__).parent / "fixtures" / "seg_series"
VIDEOS = ["test", "test2", "test3", "test5", "test6"]
# 每视频基线最终错误数（漏斗口径：跳过 raw 为 None 的段；与
# tools/baseline.json 保持一致）
BASELINE_FINAL = {"test": 3, "test2": 8, "test3": 0, "test5": 0, "test6": 0}


def _load(v: str) -> dict:
    with open(SERIES / f"{v}.json", encoding="utf-8") as f:
        return json.load(f)


def _error_indices(segs, corr, tol) -> set:
    """漏斗口径：raw None 的段不计（与 tools/accuracy_breakdown.py 一致）。"""
    return {i for i, s in enumerate(segs)
            if s["raw"] is not None and s["truth"] is not None
            and corr[i] is not None and abs(corr[i] - s["truth"]) > tol}


@pytest.mark.parametrize("video", VIDEOS)
def test_correction_chain_matches_baseline(video):
    fx = _load(video)
    meta = fx["meta"]
    p = SegmentPipeline(video_path="x", roi=tuple(meta["roi"]),
                        max_speed_kmh=meta["max_speed"],
                        max_accel_mps2=meta["max_accel"],
                        fps=meta["fps"],
                        frame_start=meta["frame_start"],
                        frame_end=meta["frame_end"],
                        force_aspect=meta["force_aspect"])
    segs = fx["segments"]
    raw = [s["raw"] for s in segs]
    times = [s["mid"] for s in segs]
    lens = [s["len"] for s in segs]

    conf = p._confidence(raw, times, lens)
    corr, _n = p._dense_correct(raw, times, conf)

    baseline_corr = [s["corr"] for s in segs]
    assert corr == baseline_corr, (
        f"{video}: 纠错输出与基线不一致（共 "
        f"{sum(1 for a, b in zip(corr, baseline_corr) if a != b)} 段不同）。"
        "若改动有意，先跑完整漏斗确认无回归，再重新生成夹具。")


@pytest.mark.parametrize("video", VIDEOS)
def test_final_error_set_matches_baseline(video):
    fx = _load(video)
    meta = fx["meta"]
    p = SegmentPipeline(video_path="x", roi=tuple(meta["roi"]),
                        max_speed_kmh=meta["max_speed"],
                        max_accel_mps2=meta["max_accel"],
                        fps=meta["fps"],
                        frame_start=meta["frame_start"],
                        frame_end=meta["frame_end"],
                        force_aspect=meta["force_aspect"])
    segs = fx["segments"]
    raw = [s["raw"] for s in segs]
    times = [s["mid"] for s in segs]
    lens = [s["len"] for s in segs]
    conf = p._confidence(raw, times, lens)
    corr, _n = p._dense_correct(raw, times, conf)

    err = _error_indices(segs, corr, fx["tol"])
    err_base = _error_indices(segs, [s["corr"] for s in segs], fx["tol"])
    assert err == err_base, (
        f"{video}: 最终错误集变化（基线 {len(err_base)} → 本次 {len(err)}）")
    assert len(err) == BASELINE_FINAL[video], (
        f"{video}: 最终错误数 {len(err)} ≠ 基线 {BASELINE_FINAL[video]}。"
        "与 CLAUDE.md/tools/baseline.json 的 11 错误口径不一致——"
        "先跑完整漏斗（tools/accuracy_breakdown.py）定位。")


def test_fixture_versions_consistent():
    """夹具与 baseline.json 版本一致（版本漂移 = 夹具过期信号）。"""
    import config
    for v in VIDEOS:
        msg = (f"{v} 夹具版本漂移（夹具 {_load(v)['version']} vs "
               f"代码 {config.__version__}），需重新生成夹具")
        assert _load(v)["version"] == config.__version__, msg
