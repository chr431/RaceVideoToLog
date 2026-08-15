"""Extract ROI crops of all test videos to ground_truth_roi/ for manual review.

每个视频的 ROI 区域逐帧存为 PNG（文件名=帧号），方便人工复查 truth/ref
的帧同步（显示转换帧的 ±1 帧模糊）。同时存 truth.csv 供对照。

用法：python tools/extract_roi.py [--scale 2] [--max-frames N]
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / "ground_truth_roi"

VIDEOS = ["test", "test2", "test3", "test4", "test5", "test6"]


def load_meta(video_name: str) -> tuple[Path, tuple, int, int]:
    truth = PROJECT / f"ground_truth_csv/{video_name}_truth.csv"
    if not truth.exists():
        truth = PROJECT / f"ground_truth_csv/{video_name}_ref.csv"
    roi = None
    f_start = f_end = None
    with open(truth, encoding="utf-8-sig") as f:
        for line in f:
            m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if m:
                roi = tuple(int(x) for x in m.groups())
            m = re.search(r"frame_start=(\d+)", line)
            if m:
                f_start = int(m.group(1))
            m = re.search(r"frame_end=(\d+)", line)
            if m:
                f_end = int(m.group(1))
    return truth, roi, f_start, f_end


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=VIDEOS)
    ap.add_argument("--scale", type=int, default=2, help="放大倍数（数字可读性）")
    ap.add_argument("--max-frames", type=int, default=0, help="每视频最多帧（调试）")
    args = ap.parse_args()

    from decord import VideoReader, cpu
    from PySide6.QtGui import QImage, Qt

    OUT.mkdir(exist_ok=True)
    for v in args.videos:
        truth, roi, f_start, f_end = load_meta(v)
        if roi is None or f_start is None:
            print(f"!! {v}: missing roi/frame_start in {truth.name}")
            continue
        x1, y1, x2, y2 = roi
        vout = OUT / v
        vout.mkdir(exist_ok=True)
        video = Path(f"D:/Videos/racelog_test/{v}.mp4")
        vr = VideoReader(str(video), ctx=cpu(0))
        frames = list(range(f_start, f_end + 1))
        if args.max_frames:
            frames = frames[: args.max_frames]
        # truth values
        tv: dict[int, float] = {}
        with open(truth, encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].isdigit():
                    try:
                        tv[int(float(row[0]))] = float(row[2])
                    except ValueError:
                        pass
        vr.seek_accurate(frames[0])
        s = args.scale
        n = 0
        for fi in frames:
            crop = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if crop.shape[0] != y2 - y1 + 1 or crop.shape[1] != x2 - x1 + 1:
                crop = crop[y1:y2 + 1, x1:x2 + 1]
            h, w = crop.shape[:2]
            if s != 1:
                img = QImage(crop.data, w, h, 3 * w,
                             QImage.Format.Format_RGB888).scaled(
                    w * s, h * s, Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation)
            else:
                img = QImage(crop.data, w, h, 3 * w,
                             QImage.Format.Format_RGB888)
            img.save(str(vout / f"frame_{fi:05d}.png"))
            n += 1
        with open(vout / "truth.csv", "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "speed_kmh"])
            for fi in frames:
                wr.writerow([fi, tv.get(fi, "")])
        print(f"{v}: {n} frames → {vout}")
        del vr


if __name__ == "__main__":
    main()
