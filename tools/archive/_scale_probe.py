"""核心数扩展曲线（3000 帧快速测量）：
1. 解码（按编码分开）：test(HEVC) / test5(h264) / test6(AV1) × dcd
2. OCR（ONNX 纯推理）：ocrT 线程数 × 段/秒
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from tools.detect_eval import load_meta  # noqa: E402

VIDEOS = {"test": "HEVC", "test5": "h264", "test6": "AV1"}
N = 3000


def decode_scale(v: str, dcd: int):
    import psutil
    from decord import VideoReader, cpu
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0),
                     output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=dcd)
    frames = list(range(f_start, f_start + N))
    proc = psutil.Process()
    t0 = time.perf_counter()
    c0 = proc.cpu_times()
    vr.get_batch(frames, roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1))
    wall = time.perf_counter() - t0
    c1 = proc.cpu_times()
    user = (c1.user - c0.user) + (c1.system - c0.system)
    return N / wall, user / wall


def ocr_scale(threads: int, batch=100):
    """ONNX 纯推理吞吐：用 test5 前 batch 段的代表帧反复推理。"""
    import numpy as np
    from ocr_native import OcrEngine
    from video_utils import _preprocess_standard
    from decord import VideoReader, cpu
    import config
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta("test5")
    vr = VideoReader("D:/Videos/racelog_test/test5.mp4", ctx=cpu(0),
                     output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=4)
    frames = list(range(f_start, f_start + 600))
    g = vr.get_batch(frames, roi=(roi[0], roi[1], roi[2] + 1,
                                  roi[3] + 1)).asnumpy()[..., 0]
    sharp = g.std(axis=(1, 2))
    reps = [frames[k] for k in np.argsort(sharp)[-batch:]]
    procs = [_preprocess_standard(g[k - f_start][..., None],
                                  force_aspect=mw) for k in reps]
    eng = OcrEngine(config.DEFAULT_OCR_MODEL, "onnxruntime",
                    fill_width=config.DEFAULT_FILL_WIDTH, num_threads=threads)
    # warm + 计时：30 轮 batch 推理
    for _ in range(3):
        eng(procs)
    t0 = time.perf_counter()
    rounds = 30
    for _ in range(rounds):
        eng(procs)
    wall = time.perf_counter() - t0
    return rounds * batch / wall


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "decode"):
        print("══ 解码核心数扩展（3000 帧，纯解码 ROI gray） ══")
        for v, codec in VIDEOS.items():
            print(f"── {v}（{codec}）──")
            for dcd in (1, 2, 4, 8, 12, 16):
                fps_dec, cw = decode_scale(v, dcd)
                print(f"  dcd={dcd:>2}: {fps_dec:>6.0f} fps  cpu/wall={cw:.1f}")
    if mode in ("all", "ocr"):
        print("\n══ ONNX OCR 核心数扩展（test5 代表帧，段/s） ══")
        for t in (1, 2, 4, 8, 12, 16):
            print(f"  ocrT={t:>2}: {ocr_scale(t):>6.0f} 段/s")


if __name__ == "__main__":
    main()
