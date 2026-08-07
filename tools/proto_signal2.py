"""Signal 2 原型：中位数插值残差 × 局部真实带宽归一化。

目标修复 linearity 的两个失败模式：
1. 高速度漏检（相对归一化过宽）→ 用绝对残差
2. 真实动力学误报（起步/刹车）→ 用局部带宽归一化

引擎 = median-of-pairs 插值（error_detection._signal_linearity 同款，验证过最优）；
评分 = 100*exp(-|raw − expected| / max(bw, floor))。

用法：python tools/proto_signal2.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
import numpy as np
from tools.eval_phase1 import build_arrays, evaluate, fmt


def median_pairs_expected(vals: np.ndarray, times: np.ndarray,
                          time_window: float, max_neighbors: int) -> np.ndarray:
    """median-of-pairs 插值期望值（与 error_detection._signal_linearity 同公式）。"""
    n = len(vals)
    expected = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v = vals[i]
        if v < 0:
            continue
        t_i = times[i]
        left = [j for j in range(i - 1, -1, -1)
                if t_i - times[j] <= time_window][:max_neighbors]
        right = [j for j in range(i + 1, n)
                 if times[j] - t_i <= time_window][:max_neighbors]
        if not left or not right:
            continue
        lv = np.fromiter((vals[j] for j in left), dtype=np.float64)
        lt = np.fromiter((times[j] for j in left), dtype=np.float64)
        rv = np.fromiter((vals[j] for j in right), dtype=np.float64)
        rt = np.fromiter((times[j] for j in right), dtype=np.float64)
        span = rt[None, :] - lt[:, None]
        mask = span >= 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = (t_i - lt[:, None]) / span
        e = lv[:, None] + (rv[None, :] - lv[:, None]) * frac
        ef = e[mask]
        ef = ef[ef > 0]
        if ef.size:
            ef.sort()
            expected[i] = float(ef[ef.size // 2])
    return expected


def sliding_median(x: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    p = np.pad(x, (half, half), mode="edge")
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = np.median(p[i:i + w])
    return out


def sliding_mean(x: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    p = np.pad(x, (half, half), mode="edge")
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = p[i:i + w].mean()
    return out


def sig_abs_bw(vals: np.ndarray, times: np.ndarray,
               exp: np.ndarray, floor: float = 1.0,
               w: int = 15) -> np.ndarray:
    """绝对残差 + 局部带宽（raw 一阶差分滑动均值）归一化。"""
    resid = np.abs(vals - exp)
    rd = np.abs(np.diff(vals, prepend=vals[0]))
    bw = np.maximum(sliding_mean(rd, w), floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 100.0 * np.exp(-resid / bw)
    scores = np.where(np.isnan(exp), 50.0, scores)
    scores = np.where(vals < 0, 0.0, scores)
    return np.round(scores, 1)


def sig_abs_residual(vals: np.ndarray, times: np.ndarray,
                     expected: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """绝对残差 + 带宽归一化评分。无期望值的帧 → 50 中性。"""
    resid = np.abs(vals - expected)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 100.0 * np.exp(-resid / np.maximum(bw, 1e-9))
    scores = np.where(np.isnan(expected), 50.0, scores)
    scores = np.where(vals < 0, 0.0, scores)
    return np.round(scores, 1)


def run() -> None:
    tests = [
        ("test4", "outputs/bench_test4_r1_stage_report.csv",
         "ground_truth_csv/test4_truth.csv"),
        ("test5", "outputs/bench_test5_r1_stage_report.csv",
         "ground_truth_csv/test5_ref.csv"),
        ("test6", "outputs/bench_test6_r1_stage_report.csv",
         "ground_truth_csv/test6_ref.csv"),
    ]
    TW = 0.25   # ±0.25s 时间窗（同 linearity）
    MN = 10     # 每侧最大邻居数
    results: dict[tuple, dict] = {}
    for name, sp, tp in tests:
        d = build_arrays(sp, tp)
        exp = median_pairs_expected(d["vals"], d["times"], TW, MN)
        # 带宽定义候选
        diffs = np.abs(np.diff(d["vals"], prepend=d["vals"][0]))
        ediffs = np.abs(np.diff(exp, prepend=exp[0]))
        bws = {
            "rawdiff_med": sliding_median(diffs, 15),
            "exp_med": sliding_median(ediffs, 15),
            "exp_std": np.array([np.std(exp[max(0, i-7):i+8]) for i in range(len(exp))]),
        }
        for bw_name, bw_raw in bws.items():
            for floor in (1.0, 2.0, 3.0):
                bw = np.nan_to_num(bw_raw, nan=floor)
                bw = np.maximum(bw, floor)
                s = sig_abs_residual(d["vals"], d["times"], exp, bw)
                key = (bw_name, floor)
                r = evaluate(s, d, 70.0)
                results.setdefault(key, {})[name] = r
                print(f"[{name}] bw={bw_name} floor={floor}: {fmt(r)}")
        print()


if __name__ == "__main__":
    run()
