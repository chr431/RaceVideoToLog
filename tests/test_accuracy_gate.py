"""准确率漏斗门禁逻辑单测（tools/accuracy_breakdown.py 的对比与基线存取）。

门禁语义：任一视频或总量的 final 错误数增加 = 回归；基线视频未全部
覆盖（子集运行）= 失败。全视频漏斗跑测试视频太慢，CI 用本单测锁住
对比逻辑本身（完整门禁在本机跑 tools/accuracy_breakdown.py）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from tools.accuracy_breakdown import (  # noqa: E402
    check_baseline, load_baseline, save_baseline,
)


def _res(videos: dict, total_final: int) -> dict:
    return {"videos": videos,
            "total": {"seg": 0, "raw": 0, "fix": 0, "fix_wrong": 0,
                      "missed": 0, "harm": 0, "final": total_final}}


def _base() -> dict:
    return _res({"test": {"final": 4}, "test2": {"final": 8},
                 "test3": {"final": 0}}, 12)


def test_no_regression_passes():
    assert check_baseline(_res({"test": {"final": 4}, "test2": {"final": 8},
                                "test3": {"final": 0}}, 12), _base())


def test_improvement_passes():
    assert check_baseline(_res({"test": {"final": 4}, "test2": {"final": 7},
                                "test3": {"final": 0}}, 11), _base())


def test_per_video_regression_fails():
    assert not check_baseline(_res({"test": {"final": 5}, "test2": {"final": 8},
                                    "test3": {"final": 0}}, 13), _base())


def test_total_regression_fails():
    assert not check_baseline(_res({"test": {"final": 4}, "test2": {"final": 8},
                                    "test3": {"final": 1}}, 13), _base())


def test_missing_video_fails():
    """子集运行（只跑了部分视频）必须失败——防部分运行误通过门禁。"""
    assert not check_baseline(_res({"test": {"final": 4}}, 4), _base())


def test_save_load_roundtrip(tmp_path):
    bp = tmp_path / "baseline.json"
    save_baseline(bp, _base(), tol=1.0, version="2.14.0")
    loaded = load_baseline(bp)
    assert loaded["tol"] == 1.0
    assert loaded["version"] == "2.14.0"
    assert loaded["total"]["final"] == 12
    assert loaded["videos"]["test2"]["final"] == 8
