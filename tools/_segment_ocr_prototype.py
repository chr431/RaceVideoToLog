"""实验 2 原型：diff 分段 → 每段少量 OCR → 分段纠错。

流程：
1. 用二值化 diff 把相同帧分段（holding 拉伸 = 显示未变段）
2. 每段挑「最清晰」代表帧（高对比度），OCR 一次（正常输入，二值化伤 OCR）
   —— 变体：挑 K 个代表帧投票
3. 段内全部帧取该值
4. 评估准确率 + OCR 调用数（应 << 逐帧）

用法：python tools/_segment_ocr_prototype.py [videos...] [--model tiny|small]
      [--k 1|3] [--max-frames N]
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
    # max_width 从 truth CSV 头（test5/test6=72 扁字体，0 会大幅降准确率）
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
    """清晰度代理：灰度标准差（高 = 数字边缘锐利）。"""
    return float(gray(crop).std())


def calibrate_thresh(crops: dict, frames: list) -> int:
    from tools._roi_binary_probe import otsu_thresh
    step = max(1, len(frames) // 50)
    ths = []
    for fi in frames[::step][:50]:
        ths.append(otsu_thresh(gray(crops[fi])))
    return int(np.median(ths))


def cluster_max(diff: np.ndarray) -> float:
    """diff 掩码的最大连通分量大小（8-连通）。"""
    from scipy import ndimage
    n = int(diff.sum())
    if n == 0:
        return 0.0
    labels, nlab = ndimage.label(diff, structure=np.ones((3, 3)))
    sizes = ndimage.sum(diff, labels, range(1, nlab + 1))
    return float(sizes.max()) if nlab > 0 else 0.0


def segment(crops: dict, frames: list, thresh: int, T: float,
            mode: str = "frac", gamma: float = 0.0) -> list[list]:
    """按二值化 diff 分段。mode='frac'：像素翻转占比<T；mode='cluster'：
    最大连通分量<C（数字变化聚集、噪声分散，聚类判别更强）。"""
    edges = []
    for i in range(len(frames) - 1):
        a, b = crops[frames[i]], crops[frames[i + 1]]
        ga, gb = gray(a), gray(b)
        if gamma > 0:
            ga = (255.0 * np.power(ga.astype(np.float32) / 255.0, gamma)).astype(np.uint8)
            gb = (255.0 * np.power(gb.astype(np.float32) / 255.0, gamma)).astype(np.uint8)
        diff = (ga > thresh) != (gb > thresh)
        if mode == "cluster":
            edges.append(cluster_max(diff) < T)
        else:
            edges.append(float(diff.mean()) < T)
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
    ap.add_argument("--k", type=int, default=1, help="每段 OCR 代表帧数（1 或 3 投票）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--mode", default="frac", choices=["frac", "cluster"],
                    help="分段 diff 判别：占比 / 最大连通分量")
    ap.add_argument("--gamma", type=float, default=0.0, help="diff 计算 gamma（0=不启用）")
    ap.add_argument("--T", type=float, default=0.002,
                    help="frac 模式阈值；cluster 模式 = max 分量上限")
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
        thresh = calibrate_thresh(crops, frames)
        segs = segment(crops, frames, thresh, args.T, mode=args.mode,
                       gamma=args.gamma)

        # 逐帧基线（正常输入，生产 max_width）
        base_ok = base_err = 0
        for fi in frames:
            proc = _preprocess_standard(crops[fi], 48, 0, max_width=max_width)
            sv, _rt, _c = extract_speed_value(eng([proc])[0])
            ti = int(float(truth[fi])) if truth[fi] else None
            pi = int(sv) if sv is not None and sv >= 0 else None
            if ti is not None and pi == ti:
                base_ok += 1
            else:
                base_err += 1
        base_tot = base_ok + base_err

        # 分段 OCR
        ok = err = 0
        n_ocr = 0
        t0 = time.perf_counter()
        for seg in segs:
            n_ocr += 1
            if args.k == 1:
                rep = max(seg, key=lambda fi: sharpness(crops[fi]))
                proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
                sv, _rt, _c = extract_speed_value(eng([proc])[0])
                val = int(sv) if sv is not None and sv >= 0 else None
            else:
                # K 个最清晰代表投票
                reps = sorted(seg, key=lambda fi: sharpness(crops[fi]))[-args.k:]
                votes = {}
                for rep in reps:
                    proc = _preprocess_standard(crops[rep], 48, 0, max_width=max_width)
                    sv, _rt, _c = extract_speed_value(eng([proc])[0])
                    if sv is not None and sv >= 0:
                        votes[int(sv)] = votes.get(int(sv), 0) + 1
                val = max(votes, key=votes.get) if votes else None
            for fi in seg:
                ti = int(float(truth[fi])) if truth[fi] else None
                if ti is not None and val == ti:
                    ok += 1
                else:
                    err += 1
        dt = time.perf_counter() - t0
        tot = ok + err
        red = n_ocr / base_tot
        print(f"{v}: {len(frames)}帧 → {len(segs)}段 ({red*100:.0f}% OCR调用, {n_ocr}次) "
              f"[{args.mode}{'+g'+str(args.gamma) if args.gamma else ''} T={args.T}]")
        print(f"  逐帧基线 {args.model}: {base_ok} ({base_ok/base_tot*100:.2f}%) err {base_err}")
        print(f"  分段 k={args.k}     : {ok} ({ok/tot*100:.2f}%) err {err} | "
              f"OCR {dt/n_ocr*1000:.1f}ms/段")


if __name__ == "__main__":
    main()
