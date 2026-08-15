"""分段错误检测算法原型。

段值序列 + 段级 abs 信号（镜像 Phase 1 但以段为单位）：
- 可信锚点：段 OCR 高置信且与物理一致
- expected[i] = 最近可信左/右锚点的线性插值（时间）
- 残差 = |v[i] - expected[i]|，按局部变速带（相邻段 |Δv| 均值）归一化
- 可疑：残差/带宽 > 阈值 或 低置信且期望可得

评估：检测精确率/召回率 vs 真值（段值≠真值=真实错误段）。

用法：python tools/_segment_detect_proto.py [videos...] [--model tiny|small]
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import time  # noqa: E402


def load(video: str):
    vdir = PROJECT / "ground_truth_roi" / video
    truth = {}
    with open(vdir / "truth.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].isdigit():
                truth[int(row[0])] = row[1]
    tpath = PROJECT / f"ground_truth_csv/{video}_truth.csv"
    if not tpath.exists():
        tpath = PROJECT / f"ground_truth_csv/{video}_ref.csv"
    max_width = 0
    for line in open(tpath, encoding="utf-8-sig"):
        if "max_width=" in line:
            try:
                max_width = int(line.split("max_width=")[1].split(",")[0])
            except (ValueError, IndexError):
                pass
            break
    return truth, vdir, max_width


def read_crops(vdir: Path, frames: list) -> dict:
    from PySide6.QtGui import QImage
    out = {}
    for fi in frames:
        img = QImage(str(vdir / f"frame_{fi:05d}.png")).convertToFormat(
            QImage.Format.Format_RGB888)
        data = img.constBits().tobytes()
        bpl = img.bytesPerLine()
        rgb = np.empty((img.height(), img.width(), 3), dtype=np.uint8)
        for r in range(img.height()):
            row = np.frombuffer(data, np.uint8, count=bpl, offset=r * bpl)
            rgb[r] = row[: img.width() * 3].reshape(img.width(), 3)
        out[fi] = rgb
    return out


def gray(crop) -> np.ndarray:
    return (crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def sharpness(crop) -> float:
    return float(gray(crop).std())


def otsu(g: np.ndarray) -> int:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = int(g.size)
    st = float((np.arange(256) * hist).sum())
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
        mf = (st - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def calibrate(crops: dict, frames: list) -> int:
    step = max(1, len(frames) // 50)
    ths = [otsu(gray(crops[fi])) for fi in frames[::step][:50]]
    return int(np.median(ths))


def cluster_max(diff: np.ndarray) -> float:
    from scipy import ndimage
    n = int(diff.sum())
    if n == 0:
        return 0.0
    labels, nl = ndimage.label(diff, structure=np.ones((3, 3)))
    return float(ndimage.sum(diff, labels, range(1, nl + 1)).max()) if nl else 0.0


def segment(crops: dict, frames: list, thresh: int, C: float) -> list[list]:
    edges = [cluster_max((gray(crops[frames[i]]) > thresh)
                         != (gray(crops[frames[i + 1]]) > thresh)) < C
             for i in range(len(frames) - 1)]
    segs = []
    s = 0
    for i in range(len(frames) - 1):
        if not edges[i]:
            segs.append(frames[s:i + 1])
            s = i + 1
    segs.append(frames[s:])
    return segs


def detect(seg_vals: list, seg_times: list, max_accel=None, fps=None,
           win=30, mult=1.5, floor=6.0) -> tuple[list[bool], list[float]]:
    """段级 abs 检测（镜像 Phase 1，无需 OCR 置信度——实测恒为 1 不可用）。

    每段 expected = 最近 win 段内 左×右 线性插值的中位数（稳健抗错误邻居），
    bandwidth = 窗口内相邻段 |Δv| 中位数（局部变速带）。残差/带宽 > mult → 可疑。
    """
    n = len(seg_vals)
    suspect = [False] * n
    residual = [0.0] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        lefts = [j for j in range(lo, i) if seg_vals[j] is not None]
        rights = [j for j in range(i + 1, hi) if seg_vals[j] is not None]
        exps = []
        for l in lefts:
            for r in rights:
                span = seg_times[r] - seg_times[l]
                if span < 1e-3:
                    continue
                frac = (seg_times[i] - seg_times[l]) / span
                exps.append(seg_vals[l] + (seg_vals[r] - seg_vals[l]) * frac)
        if not exps:
            suspect[i] = True
            continue
        exp = float(np.median(exps))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        bw = max(float(np.median(dvs)) if dvs else 0.0, floor)
        resid = abs(seg_vals[i] - exp)
        residual[i] = resid
        if resid > bw * mult:
            suspect[i] = True
    return suspect, residual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--C", type=float, default=5.0)
    ap.add_argument("--win", type=int, default=30, help="median-of-pairs 窗口段数")
    ap.add_argument("--mult", type=float, default=1.5, help="残差/带宽 阈值倍数")
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine(args.model, "onnxruntime")
    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        crops = read_crops(vdir, frames)
        thresh = calibrate(crops, frames)
        segs = segment(crops, frames, thresh, args.C)

        seg_vals, seg_times, seg_conf, seg_mid = [], [], [], []
        for seg in segs:
            rep = max(seg, key=lambda fi: sharpness(crops[fi]))
            proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
            sv, _rt, conf = extract_speed_value(eng([proc])[0])
            seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
            seg_conf.append(conf)
            mid = seg[len(seg) // 2]
            seg_mid.append(mid)
            seg_times.append(mid)
        suspect, residual = detect(seg_vals, seg_times, win=args.win, mult=args.mult)
        # 真值：段值≠真值
        tp = fp = fn = 0
        for i, (val, mid) in enumerate(zip(seg_vals, seg_mid)):
            ti = int(float(truth[mid])) if truth.get(mid) else None
            truly_wrong = ti is not None and (val is None or val != ti)
            det = suspect[i]
            if det and truly_wrong:
                tp += 1
            elif det and not truly_wrong:
                fp += 1
            elif not det and truly_wrong:
                fn += 1
        total_wrong = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / total_wrong if total_wrong else 1.0
        print(f"--- {v}: {len(segs)}段, 真错 {total_wrong} ({total_wrong/len(segs)*100:.1f}%) ---")
        print(f"  检测: TP={tp} FP={fp} FN={fn} | precision={precision*100:.1f}% "
              f"recall={recall*100:.1f}%")


if __name__ == "__main__":
    main()
