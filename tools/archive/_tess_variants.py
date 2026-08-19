"""Tesseract 变体对比：psm 7/8/11 + 左缘裁剪（首位补 1 疑似表盘边框）。

同一批代表帧（解码一次），多配置识别后与 truth 对比：
  各配置 正确率 / 误读模式（首位补1、尾部补0、漏读、其他）。
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np  # noqa: E402
import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from video_utils import nv12_to_rgb  # noqa: E402
from tools.archive._tess_probe import parse_speed, TESS  # noqa: E402

VIDEOS = [("test", 1536), ("test2", 3000), ("test3", 3000),
          ("test5", 3000), ("test6", 3000)]


def crop_to_bmp(crop, scale=2, crop_left=0) -> bytes:
    """YUV420 packed 或 RGB crop → 放大 scale 的 24-bit BMP（纯 numpy）。"""
    if crop.ndim == 2:
        rgb = nv12_to_rgb(crop)
    else:
        rgb = crop[..., :3]
    if crop_left:
        rgb = rgb[:, crop_left:]
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    bh, bw = big.shape[:2]
    row = (bw * 3 + 3) & ~3
    data = np.zeros((bh, row), dtype=np.uint8)
    data[:, :bw * 3] = big[:, :, ::-1].reshape(bh, bw * 3)
    data = data[::-1]
    import struct
    file_size = 14 + 40 + row * bh
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, bw, bh, 1, 24, 0,
                       row * bh, 2835, 2835, 0, 0)
    return header + info + data.tobytes()


def tess(bmp: bytes, psm: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tf:
        tf.write(bmp)
        tmp = tf.name
    try:
        r = subprocess.run(
            [TESS, tmp, "stdout", "--psm", str(psm), "-l", "eng",
             "-c", "tessedit_char_whitelist=0123456789."],
            capture_output=True)
        return r.stdout.decode("utf-8", errors="replace").strip()
    finally:
        os.unlink(tmp)


def main():
    for v, cap in VIDEOS:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        f_end = min(f_end, f_start + cap)
        p = SegmentPipeline(
            video_path=f"D:/Videos/racelog_test/{v}.mp4", roi=roi,
            max_speed_kmh=ms, max_accel_mps2=ma,
            buffer_size=config.DEFAULT_BUFFER_SIZE,
            decode_backend="auto", ocr_backend="cpu",
            fill_width=config.DEFAULT_FILL_WIDTH,
            speed_format="km/h", frame_start=f_start, frame_end=f_end,
            force_aspect=mw, fps=None, yuv_output=True)
        p.run(str(PROJECT / "outputs" / f"_tessv_{v}.csv"))
        segs = p.segments
        print(f"\n== {v}（{len(segs)} 段）==")
        items = []
        for seg in segs:
            crop = seg.get("rep_crop")
            if crop is None:
                continue
            mid = seg["frames"][len(seg["frames"]) // 2]
            tv = truth.get(mid)
            if tv is None:
                continue
            items.append((crop, tv, seg.get("ocr_value")))

        for psm, cl in ((7, 0), (8, 0), (11, 0), (7, 3), (7, 6)):
            ok = mis = 0
            pat = {"lead1": 0, "trail0": 0, "empty": 0, "other": 0}
            t0 = time.perf_counter()
            for crop, tv, _pp in items:
                bmp = crop_to_bmp(crop, crop_left=cl)
                txt = tess(bmp, psm)
                val = parse_speed(txt)
                if val is not None and abs(val - tv) <= 1:
                    ok += 1
                else:
                    mis += 1
                    if not txt:
                        pat["empty"] += 1
                    elif (txt.startswith("1") and len(txt) > 1 and val is not None
                          and tv > 0 and abs(val / 100 - tv) <= 1):
                        pat["lead1"] += 1
                    elif (val is not None and tv > 0
                          and abs(val * 10 - tv) <= 1):
                        pat["trail0"] += 1
                    else:
                        pat["other"] += 1
            wall = time.perf_counter() - t0
            tot = len(items)
            print(f"  psm={psm:>2} cropL={cl}: {ok}/{tot} "
                  f"({ok/tot*100:.1f}%) {wall:.0f}s "
                  f"lead1={pat['lead1']} trail0={pat['trail0']} "
                  f"empty={pat['empty']} other={pat['other']}")


if __name__ == "__main__":
    main()