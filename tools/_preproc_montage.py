"""OCR 误读 crop 蒙太奇：提取误读段/正确段的代表帧，供人工检查预处理方向。

输出 outputs/_preproc_misreads.png（网格，标注 帧/OCR值/真值）。
同时输出每张的数值特征（sharpness/对比度/尺寸），供分析。

用法：python tools/_preproc_montage.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0
UPSCALE = 4


def to_pil(crop: np.ndarray) -> Image.Image:
    img = Image.fromarray(crop, "RGB")
    return img.resize((img.width * UPSCALE, img.height * UPSCALE),
                      Image.Resampling.NEAREST)


def main() -> None:
    videos = ["test", "test2"]
    misreads = []   # (frame, ocr, truth, crop, sharpness)
    corrects = []   # 对照
    for v in videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_prep_{v}.csv"))
        sv = pipe._ocr_vals
        for i, seg in enumerate(pipe._segs):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or sv[i] is None:
                continue
            crop = pipe.crops.get(rep)
            if crop is None:
                continue
            g = crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
            sharp = float(g.std())
            item = (f"{v}#{rep}", sv[i], int(t), crop, sharp)
            if abs(sv[i] - t) > TOL:
                misreads.append(item)
            elif len(corrects) < 20:
                corrects.append(item)
        print(f"{v}: 误读 {len([1 for i in range(len(sv)) if sv[i] is not None and truth.get(pipe.segments[i]['rep_frame']) is not None and abs(sv[i]-truth[pipe.segments[i]['rep_frame']])>TOL])}")
    print(f"误读 crops {len(misreads)}, 正确对照 {len(corrects)}")

    # 数值特征
    print(f"\n{'标签':<10} {'OCR':>4} {'真':>4} {'尺寸':>10} {'sharp':>6}")
    for label, ocr, t, crop, sharp in misreads + corrects:
        print(f"{label:<10} {ocr:>4} {t:>4} {crop.shape[1]}x{crop.shape[0]:<4} {sharp:>6.0f}")

    # 蒙太奇：误读 + 正确对照，每行 4 个
    items = misreads + [("--- 正确对照 ---", 0, 0, None, 0)] + corrects
    cols = 4
    cell_w = 85 * UPSCALE
    cell_h = 52 * UPSCALE
    header_h = 20
    rows = (len(items) + cols - 1) // cols
    W = cols * cell_w
    H = rows * (cell_h + header_h)
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (label, ocr, t, crop, sharp) in enumerate(items):
        r, c = divmod(idx, cols)
        x = c * cell_w
        y = r * (cell_h + header_h)
        if crop is None:  # 分隔标题
            draw.text((x + 4, y + 2), label, fill=(255, 200, 0), font=font)
            continue
        canvas.paste(to_pil(crop), (x, y + header_h))
        text = f"{label} ocr={ocr} t={t}"
        color = (255, 120, 120) if abs(ocr - t) > TOL else (150, 255, 150)
        draw.text((x + 4, y + 2), text, fill=color, font=font)

    out = PROJECT / "outputs" / "_preproc_misreads.png"
    canvas.save(out)
    print(f"\n蒙太奇已保存: {out}")


if __name__ == "__main__":
    main()
