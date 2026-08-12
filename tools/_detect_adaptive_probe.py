"""自适应门限探针：单帧段 vs 多帧段用不同检测门限。

单帧段误读率 4.2% vs 多帧 0.3%（12.6×），80% 误读是单帧段。假设：对单帧段
用更紧门限（dev > th_single 即 flag）能提升召回，多帧段保持现行门限防误报。

用法：python tools/_detect_adaptive_probe.py [videos...]
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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    args = ap.parse_args()

    # 收集所有段的 (dev, len, is_error)
    rows = []  # (dev_from_median, seg_len, is_error)
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        seg_vals, rep_frames = pipe._ocr_segments(segs, crops, sharp)
        seg_times = [seg[len(seg) // 2] for seg in segs]
        n = len(seg_vals)
        # 每段中值偏差（复用 _detect 逻辑）
        for i in range(n):
            if seg_vals[i] is None:
                continue
            lo = max(0, i - pipe._med_k)
            hi = min(n, i + pipe._med_k + 1)
            nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
            med = float(np.median(nbrs))
            dev = abs(seg_vals[i] - med)
            t = truth.get(rep_frames[i])
            if t is None:
                continue
            is_err = abs(seg_vals[i] - t) >= 0.5
            rows.append((dev, len(segs[i]), is_err))

    rows = np.array(rows)
    dev, ln, err = rows[:, 0], rows[:, 1], rows[:, 2].astype(bool)
    single = ln == 1
    multi = ~single

    print(f"总计 {len(rows)} 段：单帧 {single.sum()} ({single.mean()*100:.0f}%), "
          f"多帧 {multi.sum()} ({multi.mean()*100:.0f}%)")
    print(f"误读 {err.sum()}：单帧 {err[single].sum()} / {single.sum()} "
          f"({err[single].sum()/max(single.sum(),1)*100:.1f}%), "
          f"多帧 {err[multi].sum()} / {multi.sum()} "
          f"({err[multi].sum()/max(multi.sum(),1)*100:.1f}%)")
    print()

    print(f"{'th_single':>9} {'多帧门限':>8} | {'单帧召回':>9} {'多帧召回':>9} "
          f"{'总召回':>7} {'单帧FP':>7} {'多帧FP':>8} {'总误报':>7}")
    # 基线：现行统一门限
    for th_s in (2.0, 3.0, 4.0, 5.0, 6.0, 99.0):
        # 多帧门限 = 现行 6（max(bw,3)*2，平缓区）
        th_m = 6.0
        sus_s = dev[single] > th_s
        sus_m = dev[multi] > th_m
        tp_s = (sus_s & err[single]).sum()
        tp_m = (sus_m & err[multi]).sum()
        fn_s = err[single].sum() - tp_s
        fn_m = err[multi].sum() - tp_m
        fp_s = (sus_s & ~err[single]).sum()
        fp_m = (sus_m & ~err[multi]).sum()
        rec = (tp_s + tp_m) / max(err.sum(), 1)
        fpr = (fp_s + fp_m) / max((~err).sum(), 1)
        print(f"{th_s:>9.0f} {th_m:>8.0f} | {tp_s/max(tp_s+fn_s,1)*100:>8.1f}% "
              f"{tp_m/max(tp_m+fn_m,1)*100:>8.1f}% {rec*100:>6.1f}% "
              f"{fp_s:>7} {fp_m:>8} {fpr*100:>6.2f}%")

    # 单帧段的 FP 代价明细：dev 阈值下正确单帧段的偏差分布
    ok_single_dev = dev[single & ~err]
    err_single_dev = dev[single & err]
    print(f"\n单帧段偏差分布: 正确段 p90={np.percentile(ok_single_dev,90):.1f} "
          f"p99={np.percentile(ok_single_dev,99):.1f} max={ok_single_dev.max():.1f}")
    print(f"单帧段误读偏差: min={err_single_dev.min():.1f} "
          f"med={np.median(err_single_dev):.1f} max={err_single_dev.max():.1f}")


if __name__ == "__main__":
    main()
