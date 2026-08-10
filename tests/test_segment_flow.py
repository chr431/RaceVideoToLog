"""段管线单测：聚类判别、中值滤波检测、锚点插值纠正。"""
from __future__ import annotations
import numpy as np

from segment_flow import SegmentPipeline, _cluster_win3, _otsu, _gray


def _pipe(**kw):
    defaults = dict(video_path="x", roi=(0, 0, 10, 10), max_speed_kmh=400.0,
                    max_accel_mps2=50.0, fps=30.0, frame_start=None,
                    frame_end=None, target_h=48, max_width=0,
                    ocr_model="v6_small")
    defaults.update(kw)
    return SegmentPipeline(**defaults)


# ═══════════════ 聚类判别 _cluster_win3 ═══════════════

def test_cluster_win3_empty():
    assert _cluster_win3(np.zeros((10, 10), dtype=bool)) == 0.0


def test_cluster_win3_dense_vs_scattered():
    # 密集 3×3 块 → 窗口和 ≥ 5（真实数字变化）
    dense = np.zeros((12, 12), dtype=bool)
    dense[4:8, 4:8] = True
    assert _cluster_win3(dense) >= 5
    # 稀疏孤立像素 → 最大 3×3 窗口和 < 5（噪声）
    scattered = np.zeros((12, 12), dtype=bool)
    scattered[::3, ::3] = True  # 每 3 像素一个 → 3×3 窗口最多 2 个
    assert _cluster_win3(scattered) < 5


# ═══════════════ 段级检测 _detect（中值滤波） ═══════════════

def test_detect_flags_spike():
    p = _pipe()
    # 平滑曲线（15 段）+ 一个 12-off 尖峰（段 7），bw 不被尖峰撑大
    vals = list(range(100, 115))  # 100..114
    vals[7] = 94  # 尖峰：应约为 107，实际 94（-13 off > 门限 8）
    times = [i * 3 for i in range(15)]
    sus = p._detect(vals, times)
    assert sus[7], "尖峰段应被 flag"


def test_detect_keeps_smooth():
    p = _pipe()
    vals = [100, 101, 102, 103, 104, 105]
    times = [0, 3, 6, 9, 12, 15]
    sus = p._detect(vals, times)
    assert not any(sus), "平滑曲线不应被 flag"


def test_detect_edge_not_flagged():
    p = _pipe()
    # 视频起始低值单调上升：段 0 无左上下文，即使偏离窗口也不 flag（边缘保守）
    vals = [90, 100, 102, 104, 106]
    times = [0, 3, 6, 9, 12]
    sus = p._detect(vals, times)
    assert not sus[0], "起始边缘段不应被 flag"


def test_detect_none_is_suspect():
    p = _pipe()
    vals = [100, None, 102, 103, 104]
    times = [0, 3, 6, 9, 12]
    sus = p._detect(vals, times)
    assert sus[1], "None 段恒 suspect"


# ═══════════════ 纠正 _correct（锚点插值 + 距离上界） ═══════════════

def test_correct_none_interpolated():
    p = _pipe()
    vals = [100, None, 104]
    times = [0, 3, 6]
    sus = [False, True, False]
    out, n = p._correct(vals, times, sus)
    assert out[1] == 102, "None 段应插值"


def test_correct_suspect_interpolated():
    p = _pipe()
    vals = [100, 90, 104]
    times = [0, 3, 6]
    sus = [False, True, False]
    out, n = p._correct(vals, times, sus)
    assert out[1] == 102, "suspect 段应插值"


def test_correct_anchor_too_far_not_corrected():
    # 锚点超过 anchor_max 帧距离 → 不插值（防远锚点误插值）
    p = _pipe(anchor_max=10.0)
    vals = [100, None, None, None, None, None, 200]
    times = [0, 50, 100, 150, 200, 250, 300]
    sus = [False, True, True, True, True, True, False]
    out, n = p._correct(vals, times, sus)
    # 段 1-5 距最近可信锚点（段0@0 或 段6@300）> 10 帧 → 不插值（保持 None/-1 语义）
    assert all(out[i] is None for i in range(1, 6))


def test_correct_small_dev_not_changed():
    p = _pipe(min_dev=8.0)
    # 偏差 4 < min_dev → 不纠正（±1-2 噪声不动）
    vals = [100, 104, 108]
    times = [0, 3, 6]
    sus = [False, True, False]
    out, n = p._correct(vals, times, sus)
    assert out[1] == 104, "小偏差段不应纠正"


# ═══════════════ 基础行为 ═══════════════

def test_gray_dims():
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    g = _gray(crop)
    assert g.shape == (10, 20) and g.dtype == np.uint8


def test_otsu_bimodal():
    # 双高斯（有分布），otsu 应取谷值分隔两峰
    rng = np.random.default_rng(0)
    g = np.zeros((100, 50), dtype=np.uint8)
    g[:50, :] = np.clip(rng.normal(40, 6, (50, 50)), 0, 255).astype(np.uint8)
    g[50:, :] = np.clip(rng.normal(200, 6, (50, 50)), 0, 255).astype(np.uint8)
    th = _otsu(g)
    assert 50 < th < 180, f"otsu 应在谷值，实际 {th}"


def test_write_csv_header_parsable(tmp_path):
    """CSV 头必须能被 parse_csv_header 解析（GUI 导入 / CLI from-csv 依赖）。

    回归：头行曾用 csv.writer 写入，含逗号注释被加引号（行首变 "）导致
    parse_csv_header 返回空 dict。
    """
    from ocr_engine import parse_csv_header, parse_csv_setting
    p = _pipe()
    p._fps = 30.0
    p._backend = "decord/GPU"
    p._n_segments = 5
    p._n_corr = 1
    p.timing = {"decode": 1.0, "total": 2.0}
    out = tmp_path / "t.csv"
    p._write_csv([[0, 0.0, 100, 0], [1, 0.0, 100, 0]], out)
    s = parse_csv_header(str(out))
    assert s, "CSV 头应能被解析"
    assert parse_csv_setting("roi", s["roi"]) == [0, 0, 10, 10]
    assert float(s["max_speed"]) == 400.0
