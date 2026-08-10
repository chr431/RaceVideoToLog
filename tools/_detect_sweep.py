"""±1 口径下检测参数扫参：floor × single_floor × mult → 召回率 / 误报率。

单次解码+分段+OCR，复用段数据扫检测参数（检测本身便宜）。
错误 = |ocr - truth| > tol（默认 1）。放宽误报率优先提升召回。

用法：python tools/_detect_sweep.py [--tol 1] [videos...]
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
    max_width = 0
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
        m = re.search(r"max_width=(\d+)", line)
        if m:
            max_width = int(m.group(1))
    truth = {}
    for line in open(tpath, encoding="utf-8-sig"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.strip().split(",")
        try:
            truth[int(float(p[0]))] = float(p[2])
        except (ValueError, IndexError):
            pass
    return roi, f_start, f_end, fps, max_speed, max_accel, max_width, truth


def detect_adapt(seg_vals, seg_times, seg_lens, med_k, win, floor, single_floor, mult):
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
    suspect = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
        if len(nbrs) < 3:
            suspect[i] = True
            continue
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            continue
        med = float(np.median(nbrs))
        fl = single_floor if seg_lens[i] == 1 else floor
        if abs(seg_vals[i] - med) > max(bw_raw[i], fl) * mult:
            suspect[i] = True
    return suspect


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    # 收集所有视频的段数据
    all_data = []
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        seg_vals, rep_frames = pipe._ocr_segments(segs, crops, sharp)
        seg_times = [seg[len(seg) // 2] for seg in segs]
        seg_lens = [len(s) for s in segs]
        is_err = [False] * len(seg_vals)
        tbf = [None] * len(seg_vals)
        for i, seg in enumerate(segs):
            t = truth.get(rep_frames[i])
            ov = seg_vals[i]
            if t is None or ov is None:
                continue
            tbf[i] = t
            is_err[i] = abs(ov - t) > TOL
        all_data.append((v, seg_vals, seg_times, seg_lens, is_err, tbf))
        print(f"加载 {v}: {len(seg_vals)} 段, 误读(>±{TOL:.0f}) {sum(is_err)}")

    # 扫参
    print(f"\n{'floor':>5} {'s_floor':>7} {'mult':>4} | {'召回率':>7} {'误报率':>7} "
          f"{'TP':>4} {'FN':>3} {'FP':>4} {'纠错后':>6}")
    results = []
    for floor in (1.0, 1.5, 2.0, 3.0):
        for s_floor in (1.0, 1.5, 2.0):
            for mult in (1.5, 2.0):
                tp = fn = fp = ok = err = 0
                final_err = 0
                for v, sv, st, sl, is_err, tbf in all_data:
                    sus = detect_adapt(sv, st, sl, 10, 30, floor, s_floor, mult)
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
                        if corr[i] is None:
                            continue
                        t = tbf[i]
                        if t is not None and abs(corr[i] - t) > TOL:
                            final_err += 1
                rec = tp / max(tp + fn, 1)
                fpr = fp / max(ok, 1)
                print(f"{floor:>5.1f} {s_floor:>7.1f} {mult:>4.1f} | {rec*100:>6.1f}% "
                      f"{fpr*100:>6.2f}% {tp:>4} {fn:>3} {fp:>4} {final_err:>6}")
                results.append((rec, fpr, tp, fn, fp, floor, s_floor, mult,
                                final_err))

    results.sort(key=lambda r: -r[0])
    print("\n最佳召回组合（召回降序，看纠错后是否≤17）:")
    for rec, fpr, tp, fn, fp, floor, s_floor, mult, fe in results[:5]:
        print(f"  floor={floor} single={s_floor} mult={mult}: 召回 {rec*100:.1f}% "
              f"误报 {fpr*100:.2f}% 纠错后 {fe} (TP {tp} FN {fn} FP {fp})")


if __name__ == "__main__":
    main()
