"""灰度对比度增强 + 二值化：解决白字黄底淹没。

原灰度阈值把黄(~210)和白(~250)都判为亮 → 数字淹没。线性拉伸
[lo,hi]→[0,255] 放大高段对比：黄→102（黑）、白→238（白），阈值分离。

输出：失败帧 4 图对照（原图 | 灰度 | 增强 | 增强二值化）+ OCR 准确率对比。

用法：python tools/_enhance_binarize.py [videos...] [--lo 180] [--hi 255]
"""
from __future__ import annotations
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


def load_truth(video: str):
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


def gray_of(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def enhance(gray: np.ndarray, lo: int, hi: int) -> np.ndarray:
    span = max(hi - lo, 1)
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / span, 0, 255).astype(np.uint8)


def binarize_rgb(gray: np.ndarray, lo: int, hi: int, th: int) -> np.ndarray:
    enh = enhance(gray, lo, hi)
    b = np.where(enh > th, 255, 0).astype(np.uint8)
    return np.stack([b] * 3, axis=-1), enh


def np_to_qimage(rgb: np.ndarray) -> "QImage":
    """numpy RGB → QImage，按 Qt 4 字节对齐 stride 填充（否则错读行数据产生竖条纹）。"""
    from PySide6.QtGui import QImage
    h, w = rgb.shape[:2]
    nbytes = 3 * w
    aligned = ((nbytes + 3) // 4) * 4
    if aligned == nbytes:
        return QImage(rgb.data, w, h, nbytes, QImage.Format.Format_RGB888).copy()
    buf = np.zeros((h, aligned), dtype=np.uint8)
    for r in range(h):
        buf[r, :nbytes] = rgb[r].reshape(-1)
    return QImage(buf.data, w, h, aligned, QImage.Format.Format_RGB888).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test"])
    ap.add_argument("--lo", type=int, default=180)
    ap.add_argument("--hi", type=int, default=255)
    ap.add_argument("--th", type=int, default=128, help="增强后二值化阈值")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication([])

    from ocr_native import OcrEngine
    from ocr_engine import extract_speed_value
    from video_utils import _preprocess_standard

    eng = OcrEngine("v6_tiny", "onnxruntime")
    for v in args.videos:
        truth, vdir, max_width = load_truth(v)
        frames = sorted(truth)
        if args.max_frames:
            frames = frames[: args.max_frames]
        fail_set = {int(Path(p).name[1:6]) for p in glob.glob(
            str(PROJECT / "outputs/binary_failures" / v / "f*_orig_*.png"))}
        out_dir = PROJECT / "outputs" / "enhance_binarize" / v
        out_dir.mkdir(parents=True, exist_ok=True)

        ok_n = ok_e = err_n = err_e = 0
        for fi in frames:
            src = read_crop(vdir, fi)
            proc_n = _preprocess_standard(src, 48, 0, max_width=max_width)
            sv_n, _rn, _ = extract_speed_value(eng([proc_n])[0])
            ti = int(float(truth[fi])) if truth[fi] else None
            n_ok = sv_n is not None and sv_n >= 0 and ti is not None and int(sv_n) == ti

            # 增强二值化：从原始 crop 灰度 → 增强 → 二值化 → 预处理喂 OCR
            g = gray_of(src)
            b_e, enh = binarize_rgb(g, args.lo, args.hi, args.th)
            proc_e = _preprocess_standard(b_e, 48, 0, max_width=max_width)
            sv_e, _re, _ = extract_speed_value(eng([proc_e])[0])
            e_ok = sv_e is not None and sv_e >= 0 and ti is not None and int(sv_e) == ti
            ok_n += n_ok; err_n += not n_ok
            ok_e += e_ok; err_e += not e_ok

            if fi in fail_set:
                # 4 图对照：全部 numpy 拼图 + 单张 QImage 保存（避开 drawImage 的
                # RGB888→RGB32 转换 bug —— 会错读行数据产生竖条纹）
                io = QImage(sorted(glob.glob(str(PROJECT / "outputs/binary_failures"
                                                 / v / f"f{fi:05d}_orig_*.png")))[0])
                io = io.convertToFormat(QImage.Format.Format_RGB888)  # PNG 默认 ARGB32 → 必须转 RGB888
                w0, h0 = io.width(), io.height()
                data = io.constBits().tobytes()
                bpl = io.bytesPerLine()
                rgb = np.empty((h0, w0, 3), dtype=np.uint8)
                for r in range(h0):
                    row = np.frombuffer(data, np.uint8, count=bpl, offset=r * bpl)
                    rgb[r] = row[: w0 * 3].reshape(w0, 3)
                gio = gray_of(rgb)
                enh_io = enhance(gio, args.lo, args.hi)
                b_io = np.where(enh_io > args.th, 255, 0).astype(np.uint8)
                # numpy 拼 4 面板（白缝 240）
                panels = [rgb, np.stack([gio] * 3, axis=-1),
                          np.stack([enh_io] * 3, axis=-1),
                          np.stack([b_io] * 3, axis=-1)]
                gap = np.full((h0, 24, 3), 240, dtype=np.uint8)
                canvas_np = panels[0]
                for k in range(1, 4):
                    canvas_np = np.concatenate([canvas_np, gap, panels[k]], axis=1)
                # 3x nearest（numpy repeat）
                canvas_np = np.repeat(np.repeat(canvas_np, 3, axis=0), 3, axis=1)
                cimg = np_to_qimage(canvas_np)
                # 文本标注（drawText 不涉及 drawImage，安全）
                p = QPainter(cimg)
                p.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
                p.setPen(QColor(0, 0, 0))
                p.drawText(10, 24, f"f{fi} t={int(float(truth[fi]))} "
                                   f"origOCR={sv_n} enhOCR={sv_e}")
                p.end()
                cimg.save(str(out_dir / f"f{fi:05d}_enh4.png"))

        for tag, ok, err in (("normal", ok_n, err_n), ("enhanced", ok_e, err_e)):
            tot = ok + err
            print(f"{v} {tag:9s}: ok={ok} ({ok/tot*100:6.2f}%) err={err}")


if __name__ == "__main__":
    main()
