"""段级检测算法评估：召回率 + 误报率。

对每视频：解码+分段+OCR → 每段 (ocr值, truth值)。
- 误读段（真错误）= |ocr - truth| > tol（默认 1，容忍 ±1：ref 有 ±1 容差，
  ±1 off 不算错误）
- _detect 的 suspect 标记 → TP(误读被flag) / FN(误读漏) / FP(正确被flag)
- 召回率 Recall = TP/(TP+FN)，误报率 FPR = FP/正确段数

用法：python tools/_detect_eval.py [--tol 1] [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402


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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0, help="±容差，默认 1")
    args = ap.parse_args()
    TOL = args.tol

    agg = {"seg": 0, "err": 0, "ok": 0, "tp": 0, "fn": 0, "fp": 0,
           "final_err": 0}
    missed_by_mag: dict[int, int] = {}
    print(f"{'视频':<6} {'段数':>6} {'误读':>5} {'TP':>4} {'FN':>4} {'FP':>5} "
          f"{'召回率':>7} {'误报率':>7} {'纠错后':>6}")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        seg_vals, rep_frames = pipe._ocr_segments(segs, crops, sharp)
        seg_times = [seg[len(seg) // 2] for seg in segs]
        sus = pipe._detect(seg_vals, seg_times, [len(s) for s in segs])

        tp = fn = fp = ok = err = 0
        for i, seg in enumerate(segs):
            t = truth.get(rep_frames[i])
            ov = seg_vals[i]
            if t is None or ov is None:
                continue
            is_err = abs(ov - t) > TOL
            if is_err:
                err += 1
                if sus[i]:
                    tp += 1
                else:
                    fn += 1
                    mag = abs(ov - int(t))
                    missed_by_mag[mag] = missed_by_mag.get(mag, 0) + 1
            else:
                ok += 1
                if sus[i]:
                    fp += 1
        recall = tp / max(err, 1)
        fpr = fp / max(ok, 1)
        corr, _n = pipe._correct(seg_vals, seg_times, sus)
        final_err = sum(1 for i, seg in enumerate(segs)
                        if corr[i] is not None
                        and truth.get(rep_frames[i]) is not None
                        and abs(corr[i] - truth[rep_frames[i]]) > TOL)
        print(f"{v:<6} {len(segs):>6} {err:>5} {tp:>4} {fn:>4} {fp:>5} "
              f"{recall*100:>6.1f}% {fpr*100:>6.2f}% {final_err:>6}")
        agg["seg"] += len(segs); agg["err"] += err; agg["ok"] += ok
        agg["tp"] += tp; agg["fn"] += fn; agg["fp"] += fp
        agg["final_err"] += final_err

    print(f"\n[tol=±{TOL:.0f}] 合计: 段 {agg['seg']} 误读 {agg['err']} 正确 {agg['ok']}")
    print(f"召回率 = {agg['tp']}/{agg['tp']+agg['fn']} "
          f"({agg['tp']/max(agg['tp']+agg['fn'],1)*100:.1f}%)  误读漏 {agg['fn']}")
    print(f"误报率 = {agg['fp']}/{agg['ok']} "
          f"({agg['fp']/max(agg['ok'],1)*100:.2f}%)  正确误flag {agg['fp']}")
    print(f"纠错后最终错误: {agg['final_err']}")
    if missed_by_mag:
        print("漏检误差幅度分布:", dict(sorted(missed_by_mag.items())))


if __name__ == "__main__":
    main()

