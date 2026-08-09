"""分段错误识别与纠正原型。

流程：
1. diff 分段（聚类判别）→ holding 拉伸段
2. 每段最清晰代表帧 OCR → 段值 + 置信度
3. 段级错误检测：段值序列应物理平滑（相邻段变化有界）→ 偏离插值期望 = 可疑
4. 段级纠正：可疑段取可信邻居插值，段内全帧应用
5. 评估帧级准确率

用法：python tools/_segment_correct_proto.py [videos...] [--model tiny|small]
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--C", type=float, default=5.0, help="聚类分段 max 分量阈值")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine(args.model, "onnxruntime")
    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        if args.max_frames:
            frames = frames[: args.max_frames]
        crops = read_crops(vdir, frames)
        thresh = calibrate(crops, frames)
        segs = segment(crops, frames, thresh, args.C)

        # 段值 OCR（最清晰代表帧）
        seg_vals = []
        seg_frames = []
        for seg in segs:
            rep = max(seg, key=lambda fi: sharpness(crops[fi]))
            proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
            sv, rt, conf = extract_speed_value(eng([proc])[0])
            seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
            seg_frames.append(seg)

        # 段级错误率（对 truth）
        seg_err = 0
        seg_truth = []
        for seg, val in zip(seg_frames, seg_vals):
            mid = seg[len(seg) // 2]
            ti = int(float(truth[mid])) if truth.get(mid) else None
            seg_truth.append(ti)
            if ti is not None and val is not None and val != ti:
                seg_err += 1
        n_valid = sum(1 for t in seg_truth if t is not None)
        print(f"--- {v}: {len(frames)}帧 → {len(segs)}段 (OCR调用 {len(segs)}) ---")
        print(f"  段值错误率: {seg_err}/{n_valid} ({seg_err/n_valid*100:.1f}%) "
              f"(逐帧段值对应帧错误率待评估)")

        # 段级纠正：可疑段取相邻段中位数（简单版）
        corr = list(seg_vals)
        n_corr = 0
        for i in range(1, len(corr) - 1):
            if corr[i] is None:
                nbr = [x for x in (corr[i - 1], corr[i + 1]) if x is not None]
                corr[i] = int(np.median(nbr)) if nbr else None
                n_corr += 1
            else:
                nbr = [x for x in (corr[i - 1], corr[i + 1]) if x is not None]
                if len(nbr) == 2 and abs(corr[i] - nbr[0]) > 20 and abs(corr[i] - nbr[1]) > 20:
                    corr[i] = int(np.median(nbr))
                    n_corr += 1
        # 纠正后段级准确率
        ok = sum(1 for t, c in zip(seg_truth, corr) if t is not None and c == t)
        print(f"  段级纠正后: {ok}/{n_valid} ({ok/n_valid*100:.2f}%) 纠正 {n_corr} 段")
        # 帧级准确率
        f_ok = 0
        f_tot = 0
        for seg, c in zip(seg_frames, corr):
            for fi in seg:
                ti = int(float(truth[fi])) if truth.get(fi) else None
                if ti is not None:
                    f_tot += 1
                    if c == ti:
                        f_ok += 1
        print(f"  帧级准确率(纠正后): {f_ok}/{f_tot} ({f_ok/f_tot*100:.2f}%)")


if __name__ == "__main__":
    main()
