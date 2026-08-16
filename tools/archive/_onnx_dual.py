"""实验：ONNX OCR 多实例并行 vs 单实例多线程。

单实例 intra-op 线程扩展亚线性（16 线程仅 4.2×，内存带宽/同步开销）。
假设：2 个独立 ONNX 实例（各 8 线程）并发处理不同批，避免单实例内部
线程池同步 → 更高吞吐。生产管线 HYBRID_OCR 已有双引擎架构（TRT+ONNX），
可扩展为 ONNX+ONNX。
"""
from __future__ import annotations
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np  # noqa: E402
import config  # noqa: E402
from ocr_native import OcrEngine  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402


def make_batch(v="test5", n=100):
    from decord import VideoReader, cpu
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0),
                     output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=4)
    g = vr.get_batch(list(range(f_start, f_start + 600)),
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)
                     ).asnumpy()[..., 0]
    sharp = g.std(axis=(1, 2))
    reps = np.argsort(sharp)[-n:]
    return [_preprocess_standard(g[k][..., None], force_aspect=mw) for k in reps]


def single(procs, threads, rounds=30):
    eng = OcrEngine(config.DEFAULT_OCR_MODEL, "onnxruntime",
                    fill_width=config.DEFAULT_FILL_WIDTH, num_threads=threads)
    for _ in range(3):
        eng(procs)
    t0 = time.perf_counter()
    for _ in range(rounds):
        eng(procs)
    return rounds * len(procs) / (time.perf_counter() - t0)


def dual(procs, threads_each, rounds=30):
    """两个独立引擎实例，各处理一半批（生产 HYBRID_OCR 双引擎模式）。"""
    engs = [OcrEngine(config.DEFAULT_OCR_MODEL, "onnxruntime",
                      fill_width=config.DEFAULT_FILL_WIDTH,
                      num_threads=threads_each) for _ in range(2)]
    half = len(procs) // 2
    halves = [procs[:half], procs[half:]]

    def worker(eng, chunk):
        for _ in range(rounds):
            eng(chunk)

    for eng, chunk in zip(engs, halves):
        worker(eng, chunk)  # warm
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(e, c))
          for e, c in zip(engs, halves)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return rounds * len(procs) / (time.perf_counter() - t0)


if __name__ == "__main__":
    procs = make_batch()
    print(f"批 {len(procs)} 段，30 轮推理，段/s：")
    print(f"  单实例 16 线程 : {single(procs, 16):>6.0f}")
    print(f"  单实例 8 线程  : {single(procs, 8):>6.0f}")
    print(f"  双实例 8+8 线程 : {dual(procs, 8):>6.0f}")
    print(f"  双实例 4+4 线程 : {dual(procs, 4):>6.0f}")
