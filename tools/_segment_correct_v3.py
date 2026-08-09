"""分段纠错 v3：检测 + tiny→small 同帧重OCR交叉验证。

降误报策略：检测出的可疑段，用 small 重 OCR 同一代表帧——
- small == tiny：值被交叉验证 → 误报，不纠正
- small != tiny：真错，用 small 值（若物理一致）

用法：python tools/_segment_correct_v3.py [videos...] [--model tiny|small]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"],
                    help="初次 OCR 模型（small 则无重 OCR）")
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--C", type=float, default=5.0)
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine(args.model, "onnxruntime")
    eng_small = OcrEngine("v6_small", "onnxruntime") if args.model == "tiny" else None
    min_dev = 15.0  # 无重OCR时锚点插值的粗错门（v2 甜点）

    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        crops = read_crops(vdir, frames)
        th = calibrate(crops, frames)
        segs = segment(crops, frames, th, args.C)

        seg_vals = []
        seg_mid = []
        rep_frames = []
        for seg in segs:
            rep = max(seg, key=lambda fi: sharpness(crops[fi]))
            proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
            sv, _rt, _c = extract_speed_value(eng([proc])[0])
            seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
            seg_mid.append(seg[len(seg) // 2])
            rep_frames.append(rep)

        seg_times = list(seg_mid)
        suspect, _ = detect(seg_vals, seg_times, win=args.win, mult=args.mult)

        # 纠正：tiny → small 交叉验证；small（无重OCR）→ 锚点插值(min_dev 门)
        corr = list(seg_vals)
        n_reocr = n_dismiss = n_fix = 0
        for i in range(len(segs)):
            if not suspect[i] or seg_vals[i] is None:
                continue
            la = None
            for j in range(i - 1, -1, -1):
                if not suspect[j] and seg_vals[j] is not None:
                    la = j
                    break
            ra = None
            for j in range(i + 1, len(segs)):
                if not suspect[j] and seg_vals[j] is not None:
                    ra = j
                    break
            interp = None
            if la is not None and ra is not None:
                span = seg_times[ra] - seg_times[la]
                frac = (seg_times[i] - seg_times[la]) / span if span > 1e-3 else 0.5
                interp = seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac
            elif la is not None:
                interp = seg_vals[la]
            elif ra is not None:
                interp = seg_vals[ra]

            if eng_small is not None:
                # tiny → small 同帧交叉验证
                rep = rep_frames[i]
                proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
                sv_s, _rt_s, _c_s = extract_speed_value(eng_small([proc])[0])
                sv_s = int(sv_s) if sv_s is not None and sv_s >= 0 else None
                n_reocr += 1
                if sv_s == seg_vals[i]:
                    n_dismiss += 1  # 交叉一致 → 误报，不纠正
                    continue
                if sv_s is not None and interp is not None and abs(sv_s - interp) <= 20:
                    corr[i] = sv_s
                    n_fix += 1
            else:
                # small 无重 OCR → 锚点插值（min_dev 门，防误纠正曲线正确段）
                if interp is not None and abs(interp - seg_vals[i]) > min_dev:
                    corr[i] = round(interp)
                    n_fix += 1

        # 帧级准确率
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
        print(f"--- {v}: {len(segs)}段, 可疑 {sum(suspect)}, "
              f"重OCR {n_reocr}, 确认误报 {n_dismiss}, 修正 {n_fix} ---")
        print(f"  帧级准确率: 纠正前 {ok_b}/{tot} ({ok_b/tot*100:.2f}%) | "
              f"纠正后 {ok_c}/{tot} ({ok_c/tot*100:.2f}%) | Δ {ok_c-ok_b:+d}")


if __name__ == "__main__":
    main()
