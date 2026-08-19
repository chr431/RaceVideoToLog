"""实验：Tesseract（本机 D:\\Software\\Tesseract-OCR）在生产段级管线的表现。

对比口径：同 3000 帧窗口、同分段（生产 win3 + Otsu），对每段代表帧：
  - 生产 OCR（PP-OCRv6 via onnxruntime）原始读数（seg_vals）
  - Tesseract v5.5.3 读数（psm 7 单行 + 数字白名单，及 psm 8 变体）
  - truth（TOL±1 判定）

输出：各视频各配置的 段数 / 正确数 / 误读数 / 误读率（%），
  Tesseract 吞吐（段/s），及与前生产读数的差异明细示例。
只覆盖生产管线读出的段（不改变分段本身）。
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np  # noqa: E402
import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from video_utils import nv12_to_rgb  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

TESS = r"D:\Software\Tesseract-OCR\tesseract.exe"


def tess_recognize(bmp_bytes: bytes, psm: int = 7,
                   whitelist: str = "0123456789.") -> str:
    """tesseract 识别（临时文件模式，最稳；psm 7 单行 + 数字白名单）。"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tf:
        tf.write(bmp_bytes)
        tmp = tf.name
    try:
        r = subprocess.run(
            [TESS, tmp, "stdout", "--psm", str(psm), "-l", "eng",
             "-c", f"tessedit_char_whitelist={whitelist}"],
            capture_output=True)
        return r.stdout.decode("utf-8", errors="replace").strip()
    finally:
        os.unlink(tmp)


def crop_to_bmp_bytes(crop) -> bytes:
    """rep_crop（YUV420 packed 或 RGB）→ 放大 2x 的 24-bit BMP（纯 numpy）。"""
    if crop.ndim == 2:  # YUV420 packed (h*3/2, w)
        rgb = nv12_to_rgb(crop)
    else:
        rgb = crop[..., :3]
    # 放大 2x 提高小数字识别率（48 高 → 96 高）
    big = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    bh, bw = big.shape[:2]
    row_size = (bw * 3 + 3) & ~3  # 4 字节对齐
    data = np.zeros((bh, row_size), dtype=np.uint8)
    data[:, :bw * 3] = big[:, :, ::-1].reshape(bh, bw * 3)  # RGB→BGR
    data = data[::-1]  # 自底向上
    import struct
    file_size = 14 + 40 + row_size * bh
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, bw, bh, 1, 24, 0,
                       row_size * bh, 2835, 2835, 0, 0)
    return header + info + data.tobytes()


def parse_speed(txt: str):
    """tesseract 文本 → 速度值（容错：去空格/多余小数点）。"""
    txt = txt.replace(" ", "").replace("|", "1").replace("O", "0")
    if not txt:
        return None
    # 最多保留一个小数点
    if txt.count(".") > 1:
        parts = txt.split(".")
        txt = parts[0] + "." + "".join(parts[1:])
    try:
        return float(txt)
    except ValueError:
        return None


def run_video(v: str, frames_cap=3000, psm=7):
    roi, f_start, f_end, fps, ms, ma, mw, truth_map = load_meta(v)
    f_end = min(f_end, f_start + frames_cap)
    out_csv = PROJECT / "outputs" / f"_tess_{v}_psm{psm}.csv"
    p = SegmentPipeline(
        video_path=f"D:/Videos/racelog_test/{v}.mp4",
        roi=roi, max_speed_kmh=ms, max_accel_mps2=ma,
        buffer_size=config.DEFAULT_BUFFER_SIZE,
        decode_backend="auto", ocr_backend="cpu",
        fill_width=config.DEFAULT_FILL_WIDTH,
        speed_format="km/h", frame_start=f_start, frame_end=f_end,
        force_aspect=mw, fps=None, yuv_output=True)
    p.run(str(out_csv))
    segs = p.segments
    print(f"\n== {v}（{len(segs)} 段，帧 {f_start}-{f_end}）==")

    n_tess = n_correct = n_mis = 0
    n_pp = n_pp_correct = n_pp_mis = 0
    t0 = time.perf_counter()
    diffs = []  # (seg_idx, start, pp_value, truth, tess_value)
    for i, seg in enumerate(segs):
        crop = seg.get("rep_crop")
        if crop is None:
            continue
        mid_frame = seg["frames"][len(seg["frames"]) // 2]
        tv = truth_map.get(mid_frame)
        pp = seg.get("ocr_value")
        if tv is None:
            continue
        png = crop_to_bmp_bytes(crop)
        txt = tess_recognize(png, psm=psm)
        tv_t = parse_speed(txt)
        n_tess += 1
        if tv_t is not None and abs(tv_t - tv) <= 1:
            n_correct += 1
        else:
            n_mis += 1
            diffs.append((i, seg["start"], pp, tv, tv_t, txt))
        if pp is not None:
            n_pp += 1
            if abs(pp - tv) <= 1:
                n_pp_correct += 1
            else:
                n_pp_mis += 1
    wall = time.perf_counter() - t0
    print(f"  Tesseract psm={psm}: {n_tess} 段, 正确 {n_correct} "
          f"({n_correct/max(n_tess,1)*100:.1f}%), 误读 {n_mis}")
    print(f"  PP-OCRv6（生产）: {n_pp} 段, 正确 {n_pp_correct} "
          f"({n_pp_correct/max(n_pp,1)*100:.1f}%), 误读 {n_pp_mis}")
    print(f"  Tesseract 吞吐: {n_tess/wall:.0f} 段/s（{wall:.1f}s，单进程）")
    for d in diffs[:8]:
        print(f"    seg#{d[0]} start={d[1]} truth={d[3]} pro={d[2]} "
              f"tess={d[4]} '{d[5]}'")
    return n_correct, n_mis, n_pp_correct, n_pp_mis


if __name__ == "__main__":
    for v in ("test", "test2", "test3", "test5", "test6"):
        run_video(v, frames_cap=3000, psm=7)