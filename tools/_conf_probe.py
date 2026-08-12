"""置信度（中值偏差 + 急动度加权）探针：扫 conf 阈值出 PR。

急动度 = |v[i+1] - 2v[i] + v[i-1]|（二阶差分，accel 变化率）：
- 真实急刹/油门：速度平滑斜坡 → 二阶差分小 → 急动度小（高 conf）
- 误读尖 V：-5 后立刻 +5 → 二阶差分 = 10 → 急动度大（低 conf）
中值偏差抓"偏离曲线"，急动度抓"尖锐性"——急刹区被中值误伤时急动度
救回（平滑 → 高急动度分）。

用法：python tools/_conf_probe.py [--tol 1] [videos...]
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


def confidence(seg_vals, seg_times, med_k=10, win=30, floor=3.0,
               single_floor=2.0, mult=2.0, w_med=0.5, w_jerk=0.5,
               jerk_scale=5.0):
    """中值偏差 + 急动度加权 → conf[0,100]。None 段 → 0。"""
    n = len(seg_vals)
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(win * max(med_gap, 1.0), 120.0)
    st = np.asarray(seg_times, dtype=np.float64)
    bw_raw = [0.0] * n
    for i in range(n):
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        bw_raw[i] = float(np.median(dvs)) if dvs else 0.0
    conf = [0.0] * n
    for i in range(n):
        if seg_vals[i] is None:
            conf[i] = 0.0
            continue
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
        if len(nbrs) < 3:
            conf[i] = 30.0
            continue
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            conf[i] = 100.0  # 边缘段保守：高 conf 不 flag
            continue
        med = float(np.median(nbrs))
        dev = abs(seg_vals[i] - med)
        bw = max(bw_raw[i], floor)
        med_score = 100.0 * np.exp(-dev / bw)
        # 急动度：二阶差分（accel 变化率）
        jl = seg_vals[i - 1] if i - 1 >= 0 else None
        jr = seg_vals[i + 1] if i + 1 < n else None
        if jl is not None and jr is not None:
            jerk = abs(jr - 2 * seg_vals[i] + jl)
        else:
            jerk = 0.0
        jerk_score = 100.0 * np.exp(-jerk / jerk_scale)
        conf[i] = w_med * med_score + w_jerk * jerk_score
    return conf


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    data = []
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
        data.append((v, sv, st, sl, is_err, tbf))
        print(f"加载 {v}: {len(sv)} 段, 误读 {sum(is_err)}")

    # 扫 conf 阈值
    print(f"\n{'conf阈值':>7} | {'召回率':>7} {'误报率':>7} {'TP':>4} {'FN':>3} {'FP':>4}")
    for T in (100, 90, 80, 70, 60, 50, 40, 30, 20, 10):
        tp = fn = fp = ok = err = 0
        for v, sv, st, sl, is_err, tbf in data:
            conf = confidence(sv, st)
            for i in range(len(sv)):
                if sv[i] is None:
                    continue
                sus = conf[i] < T
                if is_err[i]:
                    err += 1
                    if sus:
                        tp += 1
                    else:
                        fn += 1
                else:
                    ok += 1
                    if sus:
                        fp += 1
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(ok, 1)
        print(f"{T:>7} | {rec*100:>6.1f}% {fpr*100:>6.2f}% {tp:>4} {fn:>3} {fp:>4}")

    # 对照：现行二值检测
    med_pipe = SegmentPipeline("x", (0, 0, 10, 10), 400, 50, 30, None, None,
                               48, 0, "v6_small")
    tp = fn = fp = 0
    for v, sv, st, sl, is_err, tbf in data:
        sus = med_pipe._detect(sv, st, sl)
        for i in range(len(sv)):
            if sv[i] is None:
                continue
            if is_err[i]:
                if sus[i]:
                    tp += 1
                else:
                    fn += 1
            else:
                if sus[i]:
                    fp += 1
    print(f"\n现行二值检测: 召回 {tp/(tp+fn)*100:.1f}% FN {fn} FP {fp}")


if __name__ == "__main__":
    main()
