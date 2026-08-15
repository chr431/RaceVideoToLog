"""孤岛（加速度尖峰对）信号 A/B：段值曲线上的相反跳变检测。

旧逐帧信号 3：误读连段两端产生相反方向的跳变（进入/退出），中间是
"一致性孤岛"。阈值 = max_accel × dt × 3.6 × MULT。测 MULT=2/1.5 能否
捕捉 -5/+5 模式，且不伤整体。

用法：python tools/_island_ab.py [--tol 1] [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from _detect_ab import correct_anchor_bound  # noqa: E402


def load_meta(v: str):
    tpath = PROJECT / f"ground_truth_csv/{v}_truth.csv"
    if not tpath.exists():
        tpath = PROJECT / f"ground_truth_csv/{v}_ref.csv"
    roi = f_start = f_end = fps = None
    max_speed = 400.0
    max_accel = 50.0
    force_aspect = 0.0
    for line in open(tpath, encoding="utf-8-sig"):
        m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
        if m:
            roi = tuple(int(x) for x in m.groups())
        m = re.search(r"fps=([\d.]+)", line)
        if m:
            fps = float(m.group(1))
        m = re.search(r"frame_start=(\d+)", line)
        if m:
            f_start = int(m.group(1))
        m = re.search(r"frame_end=(\d+)", line)
        if m:
            f_end = int(m.group(1))
        m = re.search(r"max_speed=([\d.]+)", line)
        if m:
            max_speed = float(m.group(1))
        m = re.search(r"max_accel=([\d.]+)", line)
        if m:
            max_accel = float(m.group(1))
        m = re.search(r"force_aspect=([\d.]+)", line)
        if m:
            force_aspect = float(m.group(1))
        else:
            m = re.search(r"max_width=(\d+)", line)
            if m:
                force_aspect = round(int(m.group(1)) / 48.0, 2)
    truth = {}
    for line in open(tpath, encoding="utf-8-sig"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.strip().split(",")
        try:
            truth[int(float(p[0]))] = float(p[2])
        except (ValueError, IndexError):
            pass
    return roi, f_start, f_end, fps, max_speed, max_accel, force_aspect, truth


def island_suspect(seg_vals, seg_times, fps, max_accel, mult, look=15):
    """段值曲线上的一致性孤岛：两端相反方向跳变（>max_accel×dt×3.6×mult）夹住的段。"""
    n = len(seg_vals)
    violation = [False] * n
    vsign = [0] * n
    for i in range(1, n):
        vi, vp = seg_vals[i], seg_vals[i - 1]
        if vi is None or vp is None:
            continue
        d = vi - vp
        dt = abs(seg_times[i] - seg_times[i - 1]) / max(fps, 1.0)
        threshold = max_accel * dt * 3.6 * mult
        if abs(d) > threshold:
            violation[i] = True
            vsign[i] = 1 if d > 0 else -1
    island = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            continue
        # 跳变段自身（旧代码 ACCEL_SCORE_VIOLATION）
        if violation[i]:
            island[i] = True
            continue
        # 两相反跳变之间的段（一致性孤岛内部）
        left_signs = [vsign[j] for j in range(max(0, i - look), i) if violation[j]]
        right_signs = [vsign[j] for j in range(i + 1, min(n, i + look + 1)) if violation[j]]
        if left_signs and right_signs and left_signs[-1] != right_signs[0]:
            island[i] = True
    return island


_MED_PIPE = SegmentPipeline("x", (0, 0, 10, 10), 400, 50, 30, None, None,
                            48, 0, "v6_small")


def median_suspect(seg_vals, seg_times, seg_lens):
    """现行中值滤波检测（复用 SegmentPipeline._detect）。"""
    return _MED_PIPE._detect(seg_vals, seg_times, seg_lens)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    # 收集数据
    data = []  # (v, seg_vals, seg_times, seg_lens, is_err, tbf)
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        sv, rp = pipe._ocr_segments(segs, crops, sharp)
        st = [seg[len(seg) // 2] for seg in segs]
        sl = [len(s) for s in segs]
        is_err = [False] * len(sv)
        tbf = [None] * len(sv)
        for i, seg in enumerate(segs):
            t = truth.get(rp[i])
            ov = sv[i]
            if t is None or ov is None:
                continue
            tbf[i] = t
            is_err[i] = abs(ov - t) > TOL
        data.append((v, sv, st, sl, is_err, tbf, fps, ma))
        print(f"加载 {v}: {len(sv)} 段, 误读(>±{TOL:.0f}) {sum(is_err)}")

    # 各种检测组合
    print(f"\n{'组合':<22} | {'召回率':>7} {'误报率':>7} {'TP':>4} {'FN':>3} {'FP':>4} {'纠错后':>6}")
    configs = [
        ("中值(现行)", "med", 0),
        ("中值+岛M2", "med+island", 2.0),
        ("中值+岛M1.5", "med+island", 1.5),
        ("中值+岛M1.2", "med+island", 1.2),
        ("孤岛M2单独", "island", 2.0),
        ("孤岛M1.5单独", "island", 1.5),
    ]
    for label, kind, mult in configs:
        tp = fn = fp = ok = err = final_err = 0
        for v, sv, st, sl, is_err, tbf, fps, ma in data:
            if kind == "med":
                sus = median_suspect(sv, st, sl)
            elif kind == "island":
                sus = island_suspect(sv, st, fps, ma, mult)
            else:  # med+island
                sus = [a or b for a, b in
                       zip(median_suspect(sv, st, sl),
                           island_suspect(sv, st, fps, ma, mult))]
            for i in range(len(sv)):
                if sv[i] is None:
                    continue
                if is_err[i]:
                    err += 1
                    if sus[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    ok += 1
                    if sus[i]:
                        fp += 1
            corr, _ = correct_anchor_bound(sv, st, sus, 6.0, 120.0)
            for i in range(len(sv)):
                if corr[i] is None or tbf[i] is None:
                    continue
                if abs(corr[i] - tbf[i]) > TOL:
                    final_err += 1
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(ok, 1)
        print(f"{label:<22} | {rec*100:>6.1f}% {fpr*100:>6.2f}% "
              f"{tp:>4} {fn:>3} {fp:>4} {final_err:>6}")


if __name__ == "__main__":
    main()
