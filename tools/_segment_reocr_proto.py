"""段内重OCR恢复潜力探针：错误段里有多少不同帧能 OCR 对。

检测出可疑段后，段内重 OCR 不同帧（top-K 最清晰）投票。量化：
- 代表帧(最清晰)错误但段内某帧正确的比例
- top-K 帧投票多数正确的比例

用法：python tools/_segment_reocr_proto.py [videos...] [--model tiny|small] [--K 5]
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
    load, read_crops, calibrate, segment, sharpness, gray,
    detect)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--K", type=int, default=5, help="每段重 OCR 的帧数")
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--mult", type=float, default=3.0)
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
        segs = segment(crops, frames, th, 5.0)

        # 每段 OCR top-K 最清晰帧
        seg_vals = []
        seg_mid = []
        all_frame_vals = {}  # (seg_idx, frame) -> val
        for si, seg in enumerate(segs):
            ranked = sorted(seg, key=lambda fi: sharpness(crops[fi]), reverse=True)
            vals = []
            for fi in ranked[:args.K]:
                proc = _preprocess_standard(crops[fi], 48, 0, max_width=max_width)
                sv, _rt, _c = extract_speed_value(eng([proc])[0])
                vals.append(int(sv) if sv is not None and sv >= 0 else None)
            seg_vals.append(vals[0] if vals else None)  # 最清晰 = 代表值
            seg_mid.append(seg[len(seg) // 2])
            all_frame_vals[si] = vals

        seg_times = list(seg_mid)
        suspect, _ = detect(seg_vals, seg_times, win=args.win, mult=args.mult)

        # 恢复评估：真错段 vs 重OCR
        n_wrong = 0          # 代表帧错的段（对真值）
        n_any_correct = 0    # 段内任一帧 OCR 对
        n_vote_correct = 0   # top-K 投票多数对（且代表帧错）
        for si, (val, mid) in enumerate(zip(seg_vals, seg_mid)):
            ti = int(float(truth[mid])) if truth.get(mid) else None
            if ti is None:
                continue
            vals = all_frame_vals[si]
            rep_wrong = val != ti
            if not rep_wrong:
                continue
            n_wrong += 1
            any_ok = any(x == ti for x in vals if x is not None)
            if any_ok:
                n_any_correct += 1
            # 投票（多数，去 None）
            ok_votes = sum(1 for x in vals if x == ti)
            other_votes = sum(1 for x in vals if x is not None and x != ti)
            if ok_votes > other_votes:
                n_vote_correct += 1
        print(f"--- {v}: 代表帧错 {n_wrong} 段 ---")
        if n_wrong:
            print(f"  段内任一帧对: {n_any_correct} ({n_any_correct/n_wrong*100:.0f}%) | "
                  f"top-{args.K}投票对: {n_vote_correct} ({n_vote_correct/n_wrong*100:.0f}%)")
        else:
            print("  无错误段")


if __name__ == "__main__":
    main()
