"""二值化 OCR 失败模式分析：原图 OCR 对但二值化 OCR 错的帧。

对每个这样的帧，保存：
- 原始 ROI crop（ground_truth_roi 的 2x PNG）
- 实际喂给 OCR 的二值化预处理图（_preprocess_standard(binarized, 48, 0, max_width) 的 uint8）
附 OCR 原始文本对照（原图文本 vs 二值化文本 vs truth）。

用法：python tools/_binary_ocr_failures.py [videos...] [--max-frames N]
"""
from __future__ import annotations
import argparse
import csv
import sys
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


def read_crop(vdir: Path, fi: int) -> np.ndarray:
    from PySide6.QtGui import QImage
    img = QImage(str(vdir / f"frame_{fi:05d}.png")).convertToFormat(
        QImage.Format.Format_RGB888)
    data = img.constBits().tobytes()
    bpl = img.bytesPerLine()
    rgb = np.empty((img.height(), img.width(), 3), dtype=np.uint8)
    for r in range(img.height()):
        row = np.frombuffer(data, np.uint8, count=bpl, offset=r * bpl)
        rgb[r] = row[: img.width() * 3].reshape(img.width(), 3)
    return rgb


def gray(crop) -> np.ndarray:
    return (crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def calibrate(crops_dict: dict, frames: list) -> int:
    from tools._roi_binary_probe import otsu_thresh
    step = max(1, len(frames) // 50)
    ths = []
    for fi in frames[::step][:50]:
        ths.append(otsu_thresh(gray(crops_dict[fi])))
    return int(np.median(ths))


def save_png(arr: np.ndarray, path: Path) -> None:
    from PySide6.QtGui import QImage
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    img.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test6"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-failures", type=int, default=20, help="每视频最多存几个失败帧")
    args = ap.parse_args()

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine("v6_tiny", "onnxruntime")
    out_root = PROJECT / "outputs" / "binary_failures"
    out_root.mkdir(parents=True, exist_ok=True)

    for v in args.videos:
        truth, vdir, max_width = load(v)
        frames = sorted(truth)
        if args.max_frames:
            frames = frames[: args.max_frames]
        crops = {fi: read_crop(vdir, fi) for fi in frames}
        thresh = calibrate(crops, frames)
        vout = out_root / v
        vout.mkdir(exist_ok=True)

        print(f"=== {v} ({len(frames)} 帧, max_width={max_width}, bin_thresh={thresh}) ===")
        failures = []
        for fi in frames:
            src = crops[fi]
            proc_n = _preprocess_standard(src, 48, 0, max_width=max_width)
            # 二值化
            g = gray(src)
            b = np.where(g > thresh, 255, 0).astype(np.uint8)
            b_rgb = np.stack([b] * 3, axis=-1)
            proc_b = _preprocess_standard(b_rgb, 48, 0, max_width=max_width)
            sv_n, rt_n, _ = extract_speed_value(eng([proc_n])[0])
            sv_b, rt_b, _ = extract_speed_value(eng([proc_b])[0])
            ti = int(float(truth[fi])) if truth[fi] else None
            n_ok = (sv_n is not None and sv_n >= 0 and ti is not None and int(sv_n) == ti)
            b_ok = (sv_b is not None and sv_b >= 0 and ti is not None and int(sv_b) == ti)
            if n_ok and not b_ok:
                failures.append((fi, ti, sv_n, rt_n, sv_b, rt_b, proc_n, proc_b))

        print(f"  原图对但二值化错: {len(failures)} 帧")
        for fi, ti, svn, rtn, svb, rtb, pn, pb in failures[:args.max_failures]:
            print(f"  {fi}: truth={ti} 原图='{rtn}'({svn}) 二值化='{rtb}'({svb})")
            save_png(pn.astype(np.uint8).copy(), vout / f"f{fi:05d}_orig_OCR_{rtn}.png")
            save_png(pb.astype(np.uint8).copy(), vout / f"f{fi:05d}_bin_OCR_{rtb}.png")


if __name__ == "__main__":
    main()
