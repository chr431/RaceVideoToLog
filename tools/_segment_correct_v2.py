"""分段纠错：检测 + 可信锚点插值。

段级 abs 检测（median-of-pairs）→ 可疑段。纠正：可疑段取最近非可疑左/右
锚点线性插值（时间），段内全帧应用。

评估帧级准确率 vs 逐帧 raw / 完整 pipeline。

用法：python tools/_segment_correct_v2.py [videos...] [--model tiny|small]
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from tools._segment_detect_proto import (  # noqa: E402
    load, read_crops, calibrate, segment, sharpness, detect)


def correct_anchors(seg_vals, seg_times, suspect, min_dev=5.0) -> tuple[list, int]:
    """可疑段取最近非可疑左右锚点线性插值。min_dev：仅当 |当前-插值|>min_dev
    才纠正（跳过小偏差误报段，防 test5 式回归）。返回 (纠正后段值, 纠正段数)。"""
    n = len(seg_vals)
    out = list(seg_vals)
    n_corr = 0
    for i in range(n):
        if not suspect[i] or seg_vals[i] is None:
            continue
        la = None
        for j in range(i - 1, -1, -1):
            if not suspect[j] and seg_vals[j] is not None:
                la = j
                break
        ra = None
        for j in range(i + 1, n):
            if not suspect[j] and seg_vals[j] is not None:
                ra = j
                break
        if la is not None and ra is not None:
            span = seg_times[ra] - seg_times[la]
            frac = (seg_times[i] - seg_times[la]) / span if span > 1e-3 else 0.5
            newv = round(seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac)
            if abs(newv - seg_vals[i]) > min_dev:
                out[i] = newv
                n_corr += 1
        elif la is not None and abs(seg_vals[la] - seg_vals[i]) > min_dev:
            out[i] = seg_vals[la]
            n_corr += 1
        elif ra is not None and abs(seg_vals[ra] - seg_vals[i]) > min_dev:
            out[i] = seg_vals[ra]
            n_corr += 1
    return out, n_corr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--min-dev", type=float, default=5.0, help="仅纠正 |当前-插值|>此值 的粗错段")
    ap.add_argument("--C", type=float, default=5.0)
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine(args.model, "onnxruntime")
    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        crops = read_crops(vdir, frames)
        th = calibrate(crops, frames)
        segs = segment(crops, frames, th, args.C)

        seg_vals = []
        seg_mid = []
        for seg in segs:
            rep = max(seg, key=lambda fi: sharpness(crops[fi]))
            proc = _preprocess_standard(crops[rep], 0, force_aspect=max_width)
            sv, _rt, _c = extract_speed_value(eng([proc])[0])
            seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
            seg_mid.append(seg[len(seg) // 2])

        seg_times = list(seg_mid)
        suspect, _ = detect(seg_vals, seg_times, win=args.win, mult=args.mult)
        corr, n_corr = correct_anchors(seg_vals, seg_times, suspect,
                                       min_dev=args.min_dev)

        # 帧级准确率：纠正前 vs 纠正后
        def frame_acc(vals):
            ok = tot = 0
            for seg, val in zip(segs, vals):
                for fi in seg:
                    ti = int(float(truth[fi])) if truth.get(fi) else None
                    if ti is not None:
                        tot += 1
                        if val == ti:
                            ok += 1
            return ok, tot
        ok_b, tot = frame_acc(seg_vals)
        ok_c, _ = frame_acc(corr)
        print(f"--- {v}: {len(segs)}段, 可疑 {sum(suspect)} ({sum(suspect)/len(segs)*100:.0f}%), "
              f"纠正 {n_corr} ---")
        print(f"  帧级准确率: 纠正前 {ok_b}/{tot} ({ok_b/tot*100:.2f}%) | "
              f"纠正后 {ok_c}/{tot} ({ok_c/tot*100:.2f}%) | "
              f"Δ {ok_c-ok_b:+d}")


if __name__ == "__main__":
    main()
