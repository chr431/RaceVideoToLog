"""实验：CPU+ONNX 串行 vs 并行（解码与 OCR 各自独占全部核心）。

串行模式 = 解码全帧（decord num_threads=cores）→ 分段 → 代表帧 ONNX OCR
（ocrT=cores），每阶段独占全部核心，无争抢。对比并行流水线（现状）。

用法：python tools/archive/_bench_serial.py [--cores 4,8,16] [--video test5]
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT))  # noqa: E402
import config  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from ocr_native import OcrEngine, auto_ocr_thread_count  # noqa: E402
from segmentation import _cluster_win3, _otsu  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402

VIDEO_DIR = "D:/Videos/racelog_test"
B = config.OCR_BATCH_SIZE


def serial_run(v: str, cores: int, video="test5"):
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    if cores:
        import psutil
        psutil.Process().cpu_affinity(list(range(cores)))
        os.environ["RVTOL_OCR_THREADS"] = str(cores)
    from decord import VideoReader, cpu
    t0 = time.perf_counter()
    vr = VideoReader(f"{VIDEO_DIR}/{v}.mp4", ctx=cpu(0), output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=cores)
    frames = list(range(f_start, f_end))
    crops = []
    CH = 512
    for k in range(0, len(frames), CH):
        b = vr.get_batch(frames[k:k + CH],
                         roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)
                         ).asnumpy()
        crops.append(b[..., 0] if b.ndim == 4 else b)
    g = np.concatenate(crops, axis=0)
    del vr, crops
    decode_s = time.perf_counter() - t0
    # 分段（Otsu + 二值 XOR + 聚类）
    t1 = time.perf_counter()
    N = len(g)
    step = max(1, N // config.SEG_CALIB_FRAMES)
    ths = [_otsu(g[k]) for k in range(0, min(N, config.SEG_CALIB_FRAMES * step),
                                    step)]
    th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
    prev_b = g[0] > th
    edges = []
    for k in range(1, N):
        b = g[k] > th
        edges.append(_cluster_win3(prev_b != b) < config.SEG_C)
        prev_b = b
    segs = []
    s = 0
    for k in range(N - 1):
        if not edges[k]:
            segs.append(list(range(s, k + 1)))
            s = k + 1
    segs.append(list(range(s, N)))
    seg_s = time.perf_counter() - t1
    # OCR（全核 ONNX）
    t2 = time.perf_counter()
    eng = OcrEngine(config.DEFAULT_OCR_MODEL, "onnxruntime",
                    fill_width=config.DEFAULT_FILL_WIDTH, num_threads=cores)
    sharp = g.std(axis=(1, 2))
    reps = [max(seg, key=lambda k: sharp[k]) for seg in segs]
    n_seg = 0
    for k in range(0, len(segs), B):
        procs = [_preprocess_standard(g[r][..., None], force_aspect=mw)
                 for r in reps[k:k + B]]
        for r, res in zip(reps[k:k + B], eng(procs)):
            sv, _rt, _c = extract_speed_value(res)
            n_seg += 1 if (sv is not None and sv >= 0) else 0
    ocr_s = time.perf_counter() - t2
    total = decode_s + seg_s + ocr_s
    print(f"  {v} {cores}核 串行: decode={decode_s:>6.2f}s "
          f"segment={seg_s:>5.2f}s ocr={ocr_s:>6.2f}s "
          f"total={total:>6.2f}s（段 {len(segs)}）")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", default="4,8,16")
    ap.add_argument("--video", default="test5")
    args = ap.parse_args()
    print(f"══ CPU+ONNX 串行（全核独占）vs 并行（现状）══\n"
          f"并行参考（test5，此前实测）：16核 9.6s / 8核分核 17.8s / "
          f"4核分核 28.0s / 4核不分核 33.1s")
    for c in (int(x) for x in args.cores.split(",")):
        serial_run(args.video, c, args.video)


if __name__ == "__main__":
    main()
