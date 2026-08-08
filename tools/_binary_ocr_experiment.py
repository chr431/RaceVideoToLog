"""实验 1：二值化图像喂 OCR —— 识别率与性能分析（tiny/small × 初次/重OCR）。

用 ground_truth_roi 的 ROI PNG（2x，resize 到 target_h 后模型输入等效 1x）。
对比 正常 RGB 输入 vs Otsu 二值化输入 的：
- 原始识别准确率（速度值 vs truth）
- 推理耗时（ms/帧）
- 初次 OCR（每模型全帧）+ 重 OCR（small 修 tiny 误读帧）分别分析

用法：python tools/_binary_ocr_experiment.py [videos...] [--max-frames N]
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


def load_truth(video: str) -> tuple[dict, Path, int]:
    """(frame->speed, ROI 目录, max_width)。"""
    vdir = PROJECT / "ground_truth_roi" / video
    if not vdir.exists():
        raise SystemExit(f"!! no ROI dir for {video} (run tools/_extract_roi.py)")
    truth = {}
    with open(vdir / "truth.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].isdigit():
                truth[int(row[0])] = row[1]
    # max_width 从 truth CSV 头
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


def calibrate_thresh(crops: list, n=50) -> int:
    """采样 crops 的 Otsu 阈值中位数。"""
    from tools._roi_binary_probe import otsu_thresh
    step = max(1, len(crops) // n)
    ths = []
    for c in crops[::step][:n]:
        g = (c.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        ths.append(otsu_thresh(g))
    return int(np.median(ths))


def binarize(crop: np.ndarray, thresh: int, invert: bool = False) -> np.ndarray:
    """Otsu 二值化 → RGB。invert=False：数字暗/背景亮；True：反相。"""
    g = (crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    b = np.where(g > thresh, 255, 0).astype(np.uint8)
    if invert:
        b = 255 - b
    return np.stack([b] * 3, axis=-1)


def preprocess(crop: np.ndarray, mode: str, thresh: int, max_width: int):
    from video_utils import _preprocess_standard
    if mode == "binarized":
        src = binarize(crop, thresh)
    elif mode == "binarized_inv":
        src = binarize(crop, thresh, invert=True)
    else:
        src = crop
    return _preprocess_standard(src, 48, 0, max_width=max_width)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from PySide6.QtGui import QImage

    engines = {
        "tiny": OcrEngine("v6_tiny", "onnxruntime"),
        "small": OcrEngine("v6_small", "onnxruntime"),
    }

    for v in args.videos:
        truth, vdir, max_width = load_truth(v)
        frames = sorted(truth)
        if args.max_frames:
            frames = frames[: args.max_frames]
        # 读 crops
        crops = []
        for fi in frames:
            img = QImage(str(vdir / f"frame_{fi:05d}.png")).convertToFormat(
                QImage.Format.Format_RGB888)
            data = img.constBits().tobytes()
            bpl = img.bytesPerLine()
            rgb = np.empty((img.height(), img.width(), 3), dtype=np.uint8)
            for r in range(img.height()):
                row = np.frombuffer(data, np.uint8, count=bpl, offset=r * bpl)
                rgb[r] = row[: img.width() * 3].reshape(img.width(), 3)
            crops.append(rgb)
        thresh = calibrate_thresh(crops)
        crop_by_frame = dict(zip(frames, crops))
        B = 32
        print(f"\n=== {v} ({len(frames)} 帧, max_width={max_width}, bin_thresh={thresh}) ===")

        # ── 初次 OCR：tiny/small × 正常/二值化/反相二值化 ──
        for model, eng in engines.items():
            for mode in ("normal", "binarized", "binarized_inv"):
                ok = err = 0
                t0 = time.perf_counter()
                for s in range(0, len(frames), B):
                    chunk = frames[s:s + B]
                    procs = [preprocess(crop_by_frame[fi], mode, thresh, max_width)
                             for fi in chunk]
                    res = eng(procs)
                    for fi, r in zip(chunk, res):
                        sv, _rt, _c = extract_speed_value(r)
                        t = truth[fi]
                        ti = int(float(t)) if t else None
                        pi = int(sv) if sv is not None and sv >= 0 else None
                        if ti is not None and pi == ti:
                            ok += 1
                        else:
                            err += 1
                dt = time.perf_counter() - t0
                tot = ok + err
                print(f"  初次 {model:5s} {mode:9s}: ok={ok} ({ok/tot*100:6.2f}%) "
                      f"err={err} | {dt/tot*1000:5.1f}ms/帧")
        # ── 重 OCR：tiny 误读帧 → small 修（正常 vs 二值化）──
        # 用 tiny 正常模式找误读帧
        tiny_err = []
        eng_t = engines["tiny"]
        for s in range(0, len(frames), B):
            chunk = frames[s:s + B]
            procs = [preprocess(crop_by_frame[fi], "normal", thresh, max_width)
                     for fi in chunk]
            for fi, r in zip(chunk, eng_t(procs)):
                sv, _rt, _c = extract_speed_value(r)
                ti = int(float(truth[fi])) if truth[fi] else None
                pi = int(sv) if sv is not None and sv >= 0 else None
                if ti is None or pi != ti:
                    tiny_err.append(fi)
        if tiny_err:
            eng_s = engines["small"]
            for mode in ("normal", "binarized", "binarized_inv"):
                fixed = 0
                t0 = time.perf_counter()
                for s in range(0, len(tiny_err), B):
                    chunk = tiny_err[s:s + B]
                    procs = [preprocess(crop_by_frame[fi], mode, thresh, max_width)
                             for fi in chunk]
                    for fi, r in zip(chunk, eng_s(procs)):
                        sv, _rt, _c = extract_speed_value(r)
                        tf = truth[fi]
                        if (sv is not None and sv >= 0 and tf
                                and int(sv) == int(float(tf))):
                            fixed += 1
                dt = time.perf_counter() - t0
                print(f"  重OCR {mode:9s}: tiny误读 {len(tiny_err)} 帧, small修对 {fixed} "
                      f"({fixed/len(tiny_err)*100:.1f}%) | {dt/len(tiny_err)*1000:.1f}ms/帧")


if __name__ == "__main__":
    main()
