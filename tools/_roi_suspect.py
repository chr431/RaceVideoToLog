"""ROI 可疑帧汇总：真值转换帧 vs 像素证据对照。

逐视频统计：
- 转换帧（truth 变了）中 ROI diff < 阈值的帧 —— 显示没动却标了变 = 帧同步可疑
- 分级：diff<1.0（极可疑，像素几乎相同）、<2.0、<3.0（轻微变动，真值转换帧或微小变化）
- 反向：truth 未变但 ROI diff 很大（>20）—— 可能的漏标转换（受模糊噪声干扰，仅参考）

用法：python tools/_roi_suspect.py
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
VIDEOS = ["test", "test2", "test3", "test4", "test5", "test6"]


def load(video_name: str):
    truth = PROJECT / f"ground_truth_csv/{video_name}_truth.csv"
    if not truth.exists():
        truth = PROJECT / f"ground_truth_csv/{video_name}_ref.csv"
    rows: dict[int, float] = {}
    roi = None
    with open(truth, encoding="utf-8-sig") as f:
        for line in f:
            m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if m:
                roi = tuple(int(x) for x in m.groups())
            if line.startswith("#"):
                continue
            p = line.strip().split(",")
            if len(p) >= 3 and p[0].isdigit():
                try:
                    rows[int(float(p[0]))] = float(p[2])
                except ValueError:
                    pass
    return rows, roi, str(truth)


def main() -> None:
    from decord import VideoReader, cpu

    print(f"{'video':6s} {'帧':>6s} {'转换':>5s} | "
          f"可疑diff<1.0 {'diff<2.0':>7s} {'diff<3.0':>7s} | 漏标候选(diff>20)")
    for v in VIDEOS:
        truth, roi, tpath = load(v)
        x1, y1, x2, y2 = roi
        frames = sorted(truth)
        vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0))
        vr.seek_accurate(frames[0])
        prev = None
        prev_fi = None
        sus: dict[float, list] = {1.0: [], 2.0: [], 3.0: []}
        missed: list = []
        n_trans = 0
        for fi in frames:
            crop = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if crop.shape[0] != y2 - y1 + 1 or crop.shape[1] != x2 - x1 + 1:
                crop = crop[y1:y2 + 1, x1:x2 + 1]
            if prev is not None:
                dv = abs(truth[fi] - truth[prev_fi])
                if dv >= 0.5:
                    n_trans += 1
                    d = float(np.abs(crop.astype(np.int16)
                                     - prev.astype(np.int16)).mean())
                    for T in (1.0, 2.0, 3.0):
                        if d < T:
                            sus[T].append((fi, truth[prev_fi], truth[fi], round(d, 3)))
                else:
                    d = float(np.abs(crop.astype(np.int16)
                                     - prev.astype(np.int16)).mean())
                    if d > 20:
                        missed.append(fi)
            prev, prev_fi = crop, fi
        del vr
        # 明细：diff<2.0 的全部（含帧号、前后值、diff）
        sus_csv = PROJECT / "ground_truth_roi" / f"{v}_suspicious.csv"
        with open(sus_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "prev_value", "new_value", "roi_diff"])
            seen = set()
            for T in (1.0, 2.0):
                for fi, pv, nv, d in sus[T]:
                    if fi not in seen:
                        seen.add(fi)
                        wr.writerow([fi, pv, nv, d])
        def fmt(l):
            return f"{len(l):>7d}"
        print(f"{v:6s} {len(frames):>6d} {n_trans:>5d} | "
              f"{fmt(sus[1.0])} {fmt(sus[2.0])} {fmt(sus[3.0])} | {len(missed):>6d}  → {sus_csv.name}")
        # 明细：diff<1.0 全部 + diff 1-2 前 10
        if sus[1.0]:
            pass  # 明细另存文件


if __name__ == "__main__":
    main()
