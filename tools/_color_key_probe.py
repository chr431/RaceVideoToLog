"""自动扣色原型：时间变化定位数字 → 背景模型扣色 → 二值化喂 OCR。

背景恒定、数字变化。逐像素背景 = 采样帧的中位颜色（数字在各段只有部分帧
被点亮 → 中位数收敛到背景）。任意帧的数字掩码 = |颜色 - 背景| > 阈值
（颜色距离，白字黄底靠 B 通道分离，灰度阈值做不到）。

用法：python tools/_color_key_probe.py [videos...] [--max-frames N]
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


def calibrate_background(crops: dict, frames: list, n=60) -> np.ndarray:
    """逐像素背景 = 采样帧中位颜色（数字随时间变 → 中位数收敛到背景）。"""
    step = max(1, len(frames) // n)
    sample = [crops[fi] for fi in frames[::step][:n]]
    return np.median(np.stack(sample), axis=0).astype(np.uint8)


def color_key(crop: np.ndarray, bg: np.ndarray, dist: float) -> np.ndarray:
    """颜色距离 > dist 的像素 = 数字（前景），返回二值图 [H,W] bool。"""
    d = np.sqrt(((crop.astype(float) - bg.astype(float)) ** 2).sum(axis=-1))
    return d > dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test6"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--dist", type=float, default=40.0, help="颜色距离阈值")
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine("v6_tiny", "onnxruntime")
    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        if args.max_frames:
            frames = frames[: args.max_frames]
        crops = read_crops(vdir, frames)
        bg = calibrate_background(crops, frames)
        print(f"=== {v} ({len(frames)} 帧, dist={args.dist}) ===")

        ok_n = ok_bin = ok_key = 0
        err_n = err_bin = err_key = 0
        for fi in frames:
            src = crops[fi]
            proc_n = _preprocess_standard(src, 48, 0, max_width=max_width)
            # 灰度阈值二值化（旧法，Otsu 校准）
            g = (src.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
            th = int(np.median([np.percentile(g, 85)]))  # 简单高阈值
            b_gray = np.where(g > 190, 255, 0).astype(np.uint8)
            # 颜色扣色二值化
            mask = color_key(src, bg, args.dist)
            b_key = np.where(mask, 255, 0).astype(np.uint8)
            for tag, proc in (("normal", proc_n),
                              ("graybin", _preprocess_standard(
                                  np.stack([b_gray] * 3, axis=-1), 48, 0, max_width=max_width)),
                              ("colorkey", _preprocess_standard(
                                  np.stack([b_key] * 3, axis=-1), 48, 0, max_width=max_width))):
                sv, _rt, _ = extract_speed_value(eng([proc])[0])
                ti = int(float(truth[fi])) if truth[fi] else None
                pi = int(sv) if sv is not None and sv >= 0 else None
                ok = pi == ti
                if tag == "normal":
                    ok_n += ok; err_n += not ok
                elif tag == "graybin":
                    ok_bin += ok; err_bin += not ok
                else:
                    ok_key += ok; err_key += not ok
        for tag, ok, err in (("normal", ok_n, err_n), ("graybin", ok_bin, err_bin),
                             ("colorkey", ok_key, err_key)):
            tot = ok + err
            print(f"  {tag:9s}: ok={ok} ({ok/tot*100:6.2f}%) err={err}")


if __name__ == "__main__":
    main()
