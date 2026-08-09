"""假设验证：初次误读的段是否更可能单独成段（段长=1）。

逻辑：过渡帧/模糊帧既难 OCR 又易与邻帧显示不同被切成单帧段。
若单帧段的误读率显著高于多帧段，则"段长=1"可作为检测的强先验。

用法：python tools/_seg_len_probe.py [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

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
    args = ap.parse_args()

    # 按段长分组：段数 / 误读数 / 误读率；误读的段长分布
    by_len: dict[int, list[int]] = {}  # len -> [段数, 误读数]
    mis_len: list[int] = []
    corr_len: list[int] = []
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        seg_vals, rep_frames = pipe._ocr_segments(segs, crops, sharp)
        for i, seg in enumerate(segs):
            t = truth.get(rep_frames[i])
            ov = seg_vals[i]
            if t is None or ov is None:
                continue
            ln = len(seg)
            is_err = abs(ov - t) >= 0.5
            a = by_len.setdefault(ln, [0, 0])
            a[0] += 1
            if is_err:
                a[1] += 1
                mis_len.append(ln)
            else:
                corr_len.append(ln)

    print(f"{'段长':>4} {'段数':>6} {'误读':>5} {'误读率':>8}")
    singles = by_len.get(1, [0, 0])
    multis = [0, 0]
    for ln, a in sorted(by_len.items()):
        if ln > 1:
            multis[0] += a[0]; multis[1] += a[1]
        print(f"{ln:>4} {a[0]:>6} {a[1]:>5} {a[1]/a[0]*100:>7.1f}%")
    sr = singles[1] / max(singles[0], 1)
    mr = multis[1] / max(multis[0], 1)
    print(f"\n单帧段: {singles[0]} 段, 误读 {singles[1]} ({sr*100:.1f}%)")
    print(f"多帧段: {multis[0]} 段, 误读 {multis[1]} ({mr*100:.1f}%)")
    print(f"误读率倍率: 单帧/多帧 = {sr/max(mr,1e-9):.1f}x")
    # 误读的段长分布（累积覆盖）
    import numpy as np
    mis_arr = np.array(mis_len)
    if len(mis_arr):
        for L in (1, 2, 3, 5):
            print(f"误读中段长≤{L}: {(mis_arr <= L).mean()*100:.0f}%")
        print(f"误读段长中位数: {int(np.median(mis_arr))}, 均值: {np.mean(mis_arr):.1f}")
    corr_arr = np.array(corr_len)
    if len(corr_arr):
        print(f"正确段长中位数: {int(np.median(corr_arr))}, 均值: {np.mean(corr_arr):.1f}")


if __name__ == "__main__":
    main()
