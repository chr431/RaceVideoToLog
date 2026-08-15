"""一致性孤岛（连续近似相同值短段）的后处理保护测试。"""
from __future__ import annotations

from seg_correction import (
    _consistent_run_bounds,
    _consistent_run_frames,
    confidence_scores,
)


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


def test_consistent_run_frames_tolerance_groups_small_fluctuation():
    vals = [160, 127, 128, 127, 151]
    # tol=0：127/128 各自都不是“完全相同”的连续段
    assert _consistent_run_frames(vals, None, 1) == 1
    assert _consistent_run_frames(vals, None, 2) == 1
    # tol=2：中间 127,128,127 是一个 max-min=1 的近似一致 run
    assert _consistent_run_frames(vals, None, 1, tol=2.0) == 3
    assert _consistent_run_frames(vals, None, 2, tol=2.0) == 3


def test_consistent_run_bounds_returns_median_and_bounds():
    vals = [160, 127, 128, 127, 151]
    l, r, frames, med = _consistent_run_bounds(vals, None, 1, tol=2.0)
    assert (l, r, frames) == (1, 3, 3)
    assert med == 127.0


def test_confidence_scores_caps_short_fluctuating_island():
    # 中间的 150,152 是两个“近似相同”的短孤岛；旧逻辑只看完全相同时，
    # 第一个 150 会被局部一致性打成 100，新逻辑把整个 run 一起封顶。
    vals = [110, 113, 114, 113, 114, 120, 128, 124, 118, 128,
            150, 152, 184, 187, 188, 193, 188, 199, 202, 205, 200, 202]
    times = list(range(len(vals)))
    conf = confidence_scores(vals, times, [1] * len(vals))
    assert conf[10] < 20
    assert conf[11] < 20
    # 旧逻辑（tol=0）确实会放过第一个 150
    old_conf = confidence_scores(vals, times, [1] * len(vals), island_tol=0.0)
    assert old_conf[10] >= 20


def test_confidence_scores_caps_short_exact_four_frame_island():
    # 用户实际案例：4 帧完全相同的 127 短促平坦孤岛，旧逻辑因“≥3 帧可信”
    # 放过了中间两帧；现在完全相同 run 需要 ≥5 帧才允许高置信。
    vals = [197, 197, 197, 197, 127, 127, 127, 127, 178, 198, 198, 198, 198]
    times = list(range(len(vals)))
    conf = confidence_scores(vals, times, [1] * len(vals))
    assert conf[5] < 20
    assert conf[6] < 20
