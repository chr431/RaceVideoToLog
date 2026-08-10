"""diff 聚类判别验证：数字变化聚集、噪声分散。

假设：数字变化产生的 diff 像素聚集（数字笔画连通块），噪声产生随机孤立像素。
指标：diff 掩码的连通分量 → max 分量大小 / 聚集占比（≥4 像素分量中的像素占比）。

若未变帧 max 分量小、聚集占比低，变帧 max 分量大、聚集占比高 → 聚类比总占比更可分。

用法：python tools/_diff_cluster_probe.py [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


def gray_of(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def otsu(g: np.ndarray) -> int:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = int(g.size)
    sum_total = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = 127
    vmax = -1.0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb = sb / wb
        mf = (sum_total - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def cluster_stats(diff: np.ndarray) -> tuple[float, float]:
    """(max 分量大小, 聚集占比)。8-连通。"""
    n_changed = int(diff.sum())
    if n_changed == 0:
        return 0.0, 0.0
    labels, n = ndimage.label(diff, structure=np.ones((3, 3)))
    sizes = ndimage.sum(diff, labels, range(1, n + 1))
    maxsz = float(sizes.max()) if n > 0 else 0.0
    # 聚集占比：≥4 像素分量中的像素占比
    clustered = float(sizes[sizes >= 4].sum()) / n_changed
    return maxsz, clustered


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    args = ap.parse_args()
    from decord import VideoReader, cpu

    for v in args.videos:
        tpath = PROJECT / f"ground_truth_csv/{v}_truth.csv"
        if not tpath.exists():
            tpath = PROJECT / f"ground_truth_csv/{v}_ref.csv"
        truth = {}
        roi = None
        for line in open(tpath, encoding="utf-8-sig"):
            m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if m:
                roi = tuple(int(x) for x in m.groups())
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            try:
                truth[int(float(p[0]))] = float(p[2])
            except (ValueError, IndexError):
                pass
        x1, y1, x2, y2 = roi
        frames = sorted(truth)
        vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0))
        vr.seek_accurate(frames[0])
        # 校准阈值
        step = max(1, len(frames) // 50)
        ths = []
        for fi in frames[::step][:50]:
            c = vr[fi].asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            ths.append(otsu(gray_of(c)))
        th = int(np.median(ths))
        vr.seek_accurate(frames[0])
        u_max, u_cl, c_max, c_cl = [], [], [], []
        prev = None
        prev_fi = None
        for fi in frames:
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            if prev is not None:
                dv = abs(truth[fi] - truth[prev_fi])
                bi = gray_of(c) > th
                bp = gray_of(prev) > th
                mx, cl = cluster_stats(bi != bp)
                if dv < 0.5:
                    u_max.append(mx); u_cl.append(cl)
                else:
                    c_max.append(mx); c_cl.append(cl)
            prev, prev_fi = c, fi
        del vr
        u_max, u_cl = np.array(u_max), np.array(u_cl)
        c_max, c_cl = np.array(c_max), np.array(c_cl)
        print(f"--- {v} (th={th}) ---")
        print(f"  max分量大小: 未变 med={np.median(u_max):.0f} p90={np.percentile(u_max,90):.0f} "
              f"max={u_max.max():.0f} | 变 med={np.median(c_max):.0f} min={c_max.min():.0f}")
        print(f"  聚集占比:    未变 med={np.median(u_cl)*100:.1f}% p90={np.percentile(u_cl,90)*100:.1f}% "
              f"| 变 med={np.median(c_cl)*100:.1f}% min={c_cl.min()*100:.1f}%")


if __name__ == "__main__":
    main()
