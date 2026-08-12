"""生产路径单测：置信度/稠密 DP 纠错、rows 构建与距离积分、CSV 头解析往返、
OCR 文本提取、标准预处理。

覆盖 run() 实际调用的链路（_detect/_correct 之外的部分），防止精度行为回归。
"""
from __future__ import annotations
import numpy as np

from segment_flow import SegmentPipeline
from video_utils import _preprocess_standard


def _pipe(**kw):
    defaults = dict(video_path="x", roi=(0, 0, 10, 10), max_speed_kmh=400.0,
                    max_accel_mps2=50.0, fps=30.0, frame_start=None,
                    frame_end=None, force_aspect=0.0)
    defaults.update(kw)
    return SegmentPipeline(**defaults)


# ═══════════════ 置信度 _confidence（门控急动度） ═══════════════

def test_confidence_smooth_high():
    p = _pipe()
    vals = [100, 100, 100, 100, 100]
    times = [0, 3, 6, 9, 12]
    conf = p._confidence(vals, times)
    assert conf[2] == 100.0, "贴合曲线的正确段 conf 应满分（门控不被污染）"


def test_confidence_none_zero():
    p = _pipe()
    vals = [100, None, 100]
    times = [0, 3, 6]
    conf = p._confidence(vals, times)
    assert conf[1] == 0.0, "None 段 conf=0（必纠正）"


def test_confidence_spike_low():
    p = _pipe()
    vals = [100, 100, 92, 100, 100]  # 尖峰：med 100 vs 92（8-off）
    times = [0, 3, 6, 9, 12]
    conf = p._confidence(vals, times)
    assert conf[2] < 50.0, "尖峰段偏离曲线 → 低 conf"


# ═══════════════ 稠密 DP 纠正 _dense_correct ═══════════════

def test_dense_correct_fixes_spike():
    p = _pipe()
    vals = [100, 100, 92, 100, 100]   # 8-off 尖峰（> change_threshold 3）
    times = [0, 3, 6, 9, 12]
    conf = [90.0, 90.0, 0.0, 90.0, 90.0]  # 尖峰低 conf → 非锚点
    out, n = p._dense_correct(vals, times, conf)
    assert out[2] == 100, "DP 应把尖峰拉回锚点插值"
    assert n == 1


def test_dense_correct_all_anchors_untouched():
    p = _pipe()
    vals = [100, 101, 102, 103, 104]
    times = [0, 3, 6, 9, 12]
    conf = [90.0] * 5  # 全部锚定
    out, n = p._dense_correct(vals, times, conf)
    assert out == vals and n == 0, "全锚定段不应被改动"


def test_dense_correct_small_change_not_committed():
    p = _pipe()
    vals = [100, 100, 98, 100, 100]   # 2-off < change_threshold 3
    times = [0, 3, 6, 9, 12]
    conf = [90.0, 90.0, 0.0, 90.0, 90.0]
    out, n = p._dense_correct(vals, times, conf)
    assert out[2] == 98 and n == 0, "阈值内微调不提交（防把正确的改错）"


def test_dense_correct_none_filled():
    p = _pipe()
    vals = [100, None, 100]
    times = [0, 3, 6]
    conf = [90.0, 0.0, 90.0]
    out, n = p._dense_correct(vals, times, conf)
    assert out[1] == 100 and n == 1, "None 段应填向锚点插值"


def test_dense_correct_jerk_spike_deanchored():
    """A4 豁免：孤立尖峰误读（jerk 中等、conf∈[20,50)）不应被锚定保留。

    test#74 同构：raw=107 夹在 103/104 之间（jerk=9 ∈ [5,40]），conf=27
    锚定会保留误读 → 豁免后 DP 拉正。
    """
    p = _pipe()
    vals = [103, 101, 107, 104, 104]   # #74 邻域：101/107 双误读
    times = [0, 3, 6, 9, 12]
    conf = [72.0, 51.0, 27.0, 100.0, 72.0]  # 尖峰 107 conf=27 但 jerk=9
    out, n = p._dense_correct(vals, times, conf)
    # 尖峰 107 被 DP 拉向 101/104 插值（≈102-103），不再原样保留
    assert out[2] != 107, "jerk 带通内的尖峰必须被解锚纠正"
    assert abs(out[2] - 102.5) <= 1, f"应接近插值 102.5，实际 {out[2]}"


def test_dense_correct_braking_anchor_kept():
    """A4 反例：平滑变化（jerk≈0）的 conf∈[20,50) 段必须保持锚定。

    刹车段 jerk≈0 不在带通 [5,40] 内 → 不被豁免（下界 0 是灾难：78 误改）。
    """
    p = _pipe()
    vals = [100, 90, 80, 70, 60]       # 匀速刹车：jerk=0
    times = [0, 3, 6, 9, 12]
    conf = [100.0, 30.0, 30.0, 30.0, 100.0]  # 中间段 conf=30（jerk 分支）
    out, n = p._dense_correct(vals, times, conf)
    assert out == vals and n == 0, "刹车段（jerk=0）锚定不动，不允许 DP 拉偏"


# ═══════════════ rows 构建 _build_rows（flag 判定 + 距离积分） ═══════════════

def test_build_rows_flags():
    p = _pipe()
    p._fps = 30.0
    segs = [[0], [1], [2], [3], [4]]
    corr = [100, 101, 102, 99, 104]
    raw = [100, None, 102, 103, 90]
    conf = [30.0, 30.0, 30.0, 5.0, 30.0]
    pinned = {4}
    rows = p._build_rows([0, 1, 2, 3, 4], segs, corr, raw=raw, conf=conf,
                         pinned=pinned)
    flags = [r[3] for r in rows]
    assert flags == [21, 12, 21, 11, 22], f"flag 判定错误: {flags}"


def test_build_rows_distance_integration():
    p = _pipe()
    p._fps = 30.0
    segs = [[0, 1], [2, 3]]
    corr = [100, 100]  # km/h 匀速
    rows = p._build_rows([0, 1, 2, 3], segs, corr)
    dists = [r[1] for r in rows]
    assert dists[0] == 0.0
    # 每帧 dt=1/30s，100 km/h = 27.78 m/s → 每帧 +0.926m
    assert dists[1] == round(100 / 3.6 / 30, 2)
    assert dists[3] == round(3 * 100 / 3.6 / 30, 2)
    # 帧序稳定 + 距离单调不减
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    assert dists == sorted(dists)


# ═══════════════ CSV 头解析往返（GUI 导入 / CLI from-csv 依赖） ═══════════════

def test_csv_header_roundtrip(tmp_path):
    from ocr_engine import (parse_csv_header, parse_csv_setting, csv_field_dest)
    p = _pipe(frame_start=362, frame_end=7585)
    p._fps = 30.0
    p._backend = "decord/GPU"
    p._n_segments = 5
    p._n_corr = 1
    p.timing = {"decode": 1.0, "correction": 0.1, "total": 2.0}
    out = tmp_path / "t.csv"
    p._write_csv([[0, 0.0, 100, 0]], out)
    s = parse_csv_header(str(out))
    assert parse_csv_setting("roi", s["roi"]) == [0, 0, 10, 10]
    assert parse_csv_setting("max_speed", s["max_speed"]) == 400.0
    assert parse_csv_setting("max_accel", s["max_accel"]) == 50.0
    assert parse_csv_setting("fill_width", s["fill_width"]) == p._fill_width
    assert parse_csv_setting("format", s["format"]) == "km/h"
    assert parse_csv_setting("frame_start", s["frame_start"]) == 362
    assert parse_csv_setting("frame_end", s["frame_end"]) == 7585
    assert float(s["correction"]) == 0.10, "timing 行应含 correction 阶段（parse_csv_header 展平为独立 key）"


def test_csv_field_dest_unknown():
    from ocr_engine import csv_field_dest, parse_csv_setting
    assert csv_field_dest("roi") == "roi"
    assert csv_field_dest("fps") == "fps"
    assert csv_field_dest("no_such_key") is None
    assert parse_csv_setting("no_such_key", "1") is None


# ═══════════════ OCR 文本提取 ═══════════════

def test_normalize_ocr_text():
    from ocr_text import normalize_ocr_text
    assert normalize_ocr_text("O5B") == "058"
    assert normalize_ocr_text("12l") == "121"
    assert normalize_ocr_text("3,5") == "3.5"


def test_extract_speed_value():
    from ocr_native import RecOut
    from ocr_text import extract_speed_value
    v, text, conf = extract_speed_value(RecOut("12O", 0.9))
    assert v == 120 and text == "120" and conf == 0.9
    v, _t, _c = extract_speed_value(RecOut("", 0.9))
    assert v is None
    v, _t, _c = extract_speed_value(None)
    assert v is None


# ═══════════════ 标准预处理 _preprocess_standard ═══════════════

def test_preprocess_standard_deterministic():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (48, 78, 3), dtype=np.uint8)
    a = _preprocess_standard(crop)
    b = _preprocess_standard(crop)
    assert np.array_equal(a, b), "同一输入两次预处理必须逐位一致"
    assert a.shape == (48, 78, 3) and a.dtype == np.float32


def test_preprocess_standard_force_aspect():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (48, 40, 3), dtype=np.uint8)
    out = _preprocess_standard(crop, force_aspect=1.5)
    assert out.shape == (48, 72, 3), "force_aspect=1.5 → 宽度 48×1.5"


def test_preprocess_standard_gamma_gray():
    # gamma>0 输出灰度（三通道相等）；gamma<=0 保留 RGB
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (48, 78, 3), dtype=np.uint8)
    g = _preprocess_standard(crop, gamma=2.0)
    assert np.array_equal(g[:, :, 0], g[:, :, 1])
    assert np.array_equal(g[:, :, 1], g[:, :, 2])
    rgb = _preprocess_standard(crop, gamma=0.0)
    assert not np.array_equal(rgb[:, :, 0], rgb[:, :, 1])


# ═══════════════ 灰度一致性（两模块共用） ═══════════════

def test_gray_consistency_between_modules():
    from video_utils import _gray as vu_gray
    from segment_flow import _gray as sf_gray
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (12, 20, 3), dtype=np.uint8)
    assert np.array_equal(vu_gray(crop), sf_gray(crop)), \
        "segment_flow 与 video_utils 的灰度实现必须一致"
