"""二值化 ROI diff 探针：raw diff vs 二值化 diff 的未变/变分离度对比。

思路 2 改进尝试：先 Otsu 二值化（隔离数字像素，消除背景/光照/模糊噪声），
再算两帧 diff。用当前 truth（test 用修正后的 test_truth_v2）。

用法：python tools/_roi_binary_probe.py [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
VIDEOS = ["test", "test5", "test6"]


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
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            try:
                rows[int(float(p[0]))] = float(p[2])
            except (ValueError, IndexError):
                pass
    return rows, roi


def otsu_thresh(gray: np.ndarray) -> int:
    """Otsu 二值化阈值（最大化类间方差）。"""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = int(gray.size)
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    var_max = -1.0
    best = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            best = t
    return best


def binary_diff(a_rgb: np.ndarray, b_rgb: np.ndarray) -> tuple[float, float]:
    """(raw mean diff, 二值化后 fraction diff)。"""
    raw = float(np.abs(a_rgb.astype(np.int16) - b_rgb.astype(np.int16)).mean())
    ga = (a_rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    gb = (b_rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    ta, tb = otsu_thresh(ga), otsu_thresh(gb)
    ba = ga > ta
    bb = gb > tb
    frac = float((ba != bb).mean())
    return raw, frac


def main() -> None:
    from decord import VideoReader, cpu

    for v in VIDEOS:
        truth, roi = load(v)
        x1, y1, x2, y2 = roi
        frames = sorted(truth)
        vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0))
        vr.seek_accurate(frames[0])
        prev_raw = prev_frac = None
        prev_fi = None
        u_raw, u_frac, c_raw, c_frac = [], [], [], []
        for fi in frames:
            crop = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if crop.shape[0] != y2 - y1 + 1 or crop.shape[1] != x2 - x1 + 1:
                crop = crop[y1:y2 + 1, x1:x2 + 1]
            if prev_raw is not None:
                dv = abs(truth[fi] - truth[prev_fi])
                raw, frac = binary_diff(prev_raw, crop)
                if dv < 0.5:
                    u_raw.append(raw); u_frac.append(frac)
                else:
                    c_raw.append(raw); c_frac.append(frac)
            prev_raw, prev_fi = crop, fi
        del vr
        u_r, c_r = np.array(u_raw), np.array(c_raw)
        u_f, c_f = np.array(u_frac), np.array(c_frac)
        print(f"--- {v} (truth {len(frames)} 帧) ---")
        print(f"  RAW:     未变 med={np.median(u_r):.3f} p90={np.percentile(u_r,90):.3f} | "
              f"变 med={np.median(c_r):.3f} min={c_r.min():.3f}")
        print(f"  二值化:  未变 med={np.median(u_f)*100:.3f}% p90={np.percentile(u_f,90)*100:.3f}% | "
              f"变 med={np.median(c_f)*100:.3f}% min={c_f.min()*100:.3f}%")
        # 分离度：把未变全拦下的阈值下，误把变判为未变（危险）的比例
        for label, u, c in (("RAW", u_r, c_r), ("二值化", u_f, c_f)):
            best_fn = None
            for T in (0.5, 1.0, 2.0, 3.0):
                fn = (c <= T).mean()
                if best_fn is None or fn < best_fn:
                    best_fn = fn
            print(f"  {label} 最低危险FN(变判未变): {best_fn*100:.2f}%")


if __name__ == "__main__":
    main()
