"""diff 计算对比：直接二值化 vs 灰度+gamma+二值化。

判别力指标：未变帧 diff 地板（med/p90/max）越低、变帧 min 越高 → 分离越好。
gamma 把黄底压暗，二值化可分离黄/白 → diff 只捕获数字变化，背景波动不产生假 diff。

用法：python tools/_diff_gamma_probe.py [videos...]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

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


def binarize(gray: np.ndarray, th: int) -> np.ndarray:
    return gray > th


def gamma_binarize(gray: np.ndarray, th: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    e = (255.0 * np.power(gray.astype(np.float32) / 255.0, gamma)).astype(np.uint8)
    return e, e > th


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()

    import re
    import csv
    from decord import VideoReader, cpu

    for v in args.videos:
        # truth + roi
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

        # 采样校准两个阈值（随机访问，避免全量预解码）
        gs = []
        step = max(1, len(frames) // 50)
        for fi in frames[::step][:50]:
            c = vr[fi].asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            gs.append(gray_of(c))
        th_cur = int(np.median([otsu(g) for g in gs]))
        th_gam = int(np.median([otsu(gamma_binarize(g, 0, args.gamma)[0]) for g in gs]))
        print(f"--- {v} ({len(frames)}帧) th_cur={th_cur} th_gam={th_gam} ---")
        vr.seek_accurate(frames[0])  # 校准 pass 消耗了 reader → 重 seek

        # 逐对 diff
        prev = None
        prev_fi = None
        res = {"cur": {"u": [], "c": []}, "gam": {"u": [], "c": []}}
        for k, fi in enumerate(frames):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            if prev is not None:
                dv = abs(truth[fi] - truth[prev_fi])
                g = gray_of(c)
                gp = gray_of(prev)
                # 当前：直接二值化
                f_cur = float((binarize(g, th_cur) != binarize(gp, th_cur)).mean())
                # gamma+二值化
                eg, bg = gamma_binarize(g, th_gam, args.gamma)
                ep, bp = gamma_binarize(gp, th_gam, args.gamma)
                f_gam = float((bg != bp).mean())
                (res["cur"]["u"] if dv < 0.5 else res["cur"]["c"]).append(f_cur)
                (res["gam"]["u"] if dv < 0.5 else res["gam"]["c"]).append(f_gam)
            prev, prev_fi = c, fi
        del vr

        for name, r in (("直接二值化", res["cur"]), ("gamma+二值化", res["gam"])):
            u, c = np.array(r["u"]), np.array(r["c"])
            print(f"  {name}: 未变 med={np.median(u)*100:.3f}% p90={np.percentile(u,90)*100:.3f}% "
                  f"max={u.max()*100:.2f}% | 变 min={c.min()*100:.3f}% med={np.median(c)*100:.2f}%")


if __name__ == "__main__":
    main()
