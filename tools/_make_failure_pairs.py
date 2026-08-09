"""为二值化失败帧生成「原图 | 灰度 | 二值化」三图对照（3x 放大 + OCR 标注）。

读取 outputs/binary_failures/<video>/ 下已保存的 orig/bin 单图，
合成 *_pair3.png：左原图、中灰度、右二值化，顶部标帧号与 OCR 文本。
另存独立灰度图 *_gray.png。

用法：python tools/_make_failure_pairs.py [videos...]
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", default=["test", "test6"])
    args = ap.parse_args()

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import (QGuiApplication, QImage, QPainter, QColor, QFont)
    app = QGuiApplication([])  # QPainter 无 QApplication 会原生崩溃

    root = PROJECT / "outputs" / "binary_failures"
    for v in args.videos:
        vdir = root / v
        if not vdir.exists():
            print(f"!! no failures dir for {v}")
            continue
        pdir = vdir / "pairs"
        pdir.mkdir(exist_ok=True)
        frames = sorted({Path(p).name[1:6] for p in glob.glob(str(vdir / "f*_orig_*.png"))})
        print(f"{v}: {len(frames)} 失败帧 → {pdir}")
        for fi in frames:
            o = glob.glob(str(vdir / f"f{fi}_orig_*.png"))
            b = glob.glob(str(vdir / f"f{fi}_bin_*.png"))
            if not o or not b:
                continue
            otext = Path(o[0]).name.split("OCR_")[1].replace(".png", "")
            btext = Path(b[0]).name.split("OCR_")[1].replace(".png", "")
            io = QImage(o[0])
            ib = QImage(b[0])
            # 灰度：Qt 原生转换（避免手写 numpy→QImage 的 stride/拉伸问题）
            gimg = io.convertToFormat(QImage.Format.Format_Grayscale8)
            w0, h0 = io.width(), io.height()
            # 独立灰度图
            gimg.save(str(pdir / f"f{fi}_gray.png"))
            # 三图对照
            s = 3
            w = (w0 + 24 + w0 + 24 + ib.width()) * s
            h = max(h0, ib.height()) * s + 60
            canvas = QImage(w, h, QImage.Format.Format_RGB32)
            canvas.fill(0xF0F0F0)
            p = QPainter(canvas)
            p.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
            p.setPen(QColor(0, 0, 0))
            p.drawText(10, 24, f"frame {fi}  (orig={otext} | bin={btext})")
            p.drawImage(10, 40, io.scaled(w0 * s, h0 * s))
            p.drawImage((w0 + 14) * s + 10, 40, gimg.scaled(w0 * s, h0 * s))
            p.drawImage((2 * w0 + 28) * s + 10, 40, ib.scaled(ib.width() * s, ib.height() * s))
            p.end()
            canvas.save(str(pdir / f"f{fi}_pair3.png"))
        print(f"  {len(frames)} 三图对照已生成")


if __name__ == "__main__":
    main()
