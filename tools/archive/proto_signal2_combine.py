"""Phase 1 信号体系重设计评估 — 忠实复刻现有组合逻辑。

用当前 error_detection 的信号值（physics/linearity/accel/freq）+ 离线重算的
abs（带宽归一化绝对残差），按现有 _combine_confidence 结构（加权 + min-floor
+ freq-floor）组合，评估检出率/误报率。

信号列已与当前代码逐位一致（combined_conf 列是旧配置产物，不用）。

用法：python tools/proto_signal2_combine.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
import numpy as np
from tools.archive.eval_phase1 import build_arrays, evaluate, fmt
from tools.archive.proto_signal2 import median_pairs_expected, sig_abs_bw


def combine_faithful(signals: dict, weights: dict, floor_sigs: list,
                     freq_floor: float = 50.0, freq_cap: float = 50.0,
                     use_freq_floor: bool = True,
                     freq_corroborate: bool = False,
                     corr_threshold: float = 80.0) -> np.ndarray:
    """忠实复刻 error_detection._combine_confidence（+可选的协同 floor）。

    freq_corroborate=True：freq 低仅当 min(floor_sigs) < corr_threshold 才压顶。
    """
    total_w = sum(weights.values())
    ws = {k: v / total_w for k, v in weights.items()}
    score = np.zeros(len(signals["ocr_conf"]))
    for k, w in ws.items():
        score += w * signals[k]
    score = np.clip(score, 0, 100)
    # 通用 min-floor：min(floor_sigs) < thr → cap
    minsig = None
    for s in floor_sigs:
        minsig = signals[s] if minsig is None else np.minimum(minsig, signals[s])
    for thr, cap in sorted({30.0: 25.0, 50.0: 50.0, 70.0: 69.0}.items()):
        m = minsig < thr
        score[m] = np.minimum(score[m], cap)
    # freq 专用 floor（可协同）
    if use_freq_floor and "freq" in signals:
        m = signals["freq"] < freq_floor
        if freq_corroborate and minsig is not None:
            m = m & (minsig < corr_threshold)
        score[m] = np.minimum(score[m], freq_cap)
    return np.round(score, 1)


def run() -> None:
    tests = [
        ("test4", "outputs/bench_test4_r1_stage_report.csv",
         "ground_truth_csv/test4_truth.csv"),
        ("test5", "outputs/bench_test5_r1_stage_report.csv",
         "ground_truth_csv/test5_ref.csv"),
        ("test6", "outputs/bench_test6_r1_stage_report.csv",
         "ground_truth_csv/test6_ref.csv"),
    ]
    # 配置：weights / floor_sigs / use_freq_floor / abs_floor
    configs = {
        "基线 5信号": dict(
            weights={"ocr_conf": .01, "phy": .15, "lin": .15, "accel": .50, "freq": .10},
            floor_sigs=["phy", "lin", "accel"], use_freq_floor=True),
        "abs替phy去freq": dict(
            weights={"ocr_conf": .01, "abs": .15, "lin": .15, "accel": .50},
            floor_sigs=["abs", "lin", "accel"], use_freq_floor=False, abs_floor=3.0),
        "abs替phy+freq协同floor": dict(
            weights={"ocr_conf": .01, "abs": .15, "lin": .15, "accel": .50, "freq": .10},
            floor_sigs=["abs", "lin", "accel"], use_freq_floor=True,
            freq_corroborate=True, corr_threshold=80.0, abs_floor=3.0),
        "abs替phy+freq协同floor(corr85)": dict(
            weights={"ocr_conf": .01, "abs": .15, "lin": .15, "accel": .50, "freq": .10},
            floor_sigs=["abs", "lin", "accel"], use_freq_floor=True,
            freq_corroborate=True, corr_threshold=85.0, abs_floor=3.0),
        "abs替phy+freq协同floor(corr90)": dict(
            weights={"ocr_conf": .01, "abs": .15, "lin": .15, "accel": .50, "freq": .10},
            floor_sigs=["abs", "lin", "accel"], use_freq_floor=True,
            freq_corroborate=True, corr_threshold=90.0, abs_floor=3.0),
    }
    for name, sp, tp in tests:
        d = build_arrays(sp, tp)
        exp = median_pairs_expected(d["vals"], d["times"], 0.25, 10)
        print(f"\n════════ {name} ════════")
        for label, cfg in configs.items():
            abs_floor = cfg.get("abs_floor", 1.0)
            abs_s = sig_abs_bw(d["vals"], d["times"], exp, floor=abs_floor)
            signals = {"ocr_conf": d["ocr_conf"], "phy": d["signals"]["physics"],
                       "lin": d["signals"]["linearity"],
                       "accel": d["signals"]["accel"], "freq": d["signals"]["freq"],
                       "abs": abs_s}
            s = combine_faithful(signals, cfg["weights"], cfg["floor_sigs"],
                                 use_freq_floor=cfg["use_freq_floor"],
                                 freq_corroborate=cfg.get("freq_corroborate", False),
                                 corr_threshold=cfg.get("corr_threshold", 80.0))
            print(f"  {label:30s} {fmt(evaluate(s, d, 70.0))}")


if __name__ == "__main__":
    run()
