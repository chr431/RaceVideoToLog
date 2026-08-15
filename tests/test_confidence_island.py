"""一致性孤岛（连续相同值短段）的后处理保护测试。"""
from __future__ import annotations

from seg_correction import _consistent_run_frames


def test_consistent_run_frames_counts_consecutive_equal_segments():
    vals = [1, 1, 2, 2, 2, 3, None, 3]
    lens = [2, 3, 1, 2, 1, 1, 1, 1]
    # 索引 0/1 都是 1：累计 2+3=5 帧
    assert _consistent_run_frames(vals, lens, 0) == 5
    assert _consistent_run_frames(vals, lens, 1) == 5
    # 索引 2/3/4 都是 2：累计 1+2+1=4 帧
    assert _consistent_run_frames(vals, lens, 2) == 4
    assert _consistent_run_frames(vals, lens, 4) == 4
    # None 不算
    assert _consistent_run_frames(vals, lens, 6) == 0


def test_consistent_run_frames_defaults_to_one_per_segment():
    vals = [5, 5, 6]
    assert _consistent_run_frames(vals, None, 0) == 2
    assert _consistent_run_frames(vals, None, 2) == 1
