"""生产者（解码+分段）微剖析：GPU/CPU 解码下 frame_stream 各子步骤耗时。

背景：gpu_ocr16 总 9.2s 中 decode 阶段 8.5s —— 生产者是否成为瓶颈？
本探针复刻 _run_pipelined 的 frame_stream（无 OCR 消费端），对
get_batch/asnumpy/gray/std/二值化/逐帧循环分别计时。

用法：python tools/archive/_producer_probe.py [--backend nvdec|cpu]
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline, _gray_seg_batch, _cluster_win3  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402


def probe(decode_backend: str) -> None:
    roi, f_start, f_end, fps, ms, ma, mw, _truth = load_meta("test5")
    p = SegmentPipeline("D:/Videos/racelog_test/test5.mp4", roi, ms, ma,
                        fps, f_start, f_end, force_aspect=mw,
                        decode_backend=decode_backend,
                        gray_output=(decode_backend != "nvdec"))
    vr = p._open_vr()
    x1, y1, x2, y2 = roi
    total = len(vr)
    end = min(f_end or total, total)
    vr.seek_accurate(f_start)
    frames = list(range(f_start, end))
    print(f"backend={p._backend} frames={len(frames)}")

    # ── 阈值校准（复刻）──
    calib_n = min(50, len(frames))
    calib = []
    for k in range(calib_n):
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
            c = c[y1:y2 + 1, x1:x2 + 1]
        g = _gray_seg_batch(c[np.newaxis])[0]
        calib.append((frames[k], c, g, float(g.std())))
    from segment_flow import _otsu
    ths = [_otsu(g) for _fi, _c, g, _s in calib]
    th = int(np.median(ths)) if ths else 127

    DECODE_BATCH = 16
    acc = {"get_batch": 0.0, "asnumpy": 0.0, "gray": 0.0, "std": 0.0,
           "bin": 0.0, "loop": 0.0}
    n_batches = 0
    n_edges = 0
    segs = 0
    t_all = time.perf_counter()
    s = 0
    rep_sharp = -1.0
    prev_b = None
    for bstart in range(calib_n, len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        t0 = time.perf_counter()
        crops = vr.get_batch(frames[bstart:bend],
                             roi=(x1, y1, x2 + 1, y2 + 1))
        t1 = time.perf_counter()
        arr = crops.asnumpy()
        t2 = time.perf_counter()
        g = _gray_seg_batch(arr)
        t3 = time.perf_counter()
        sharp = g.std(axis=(1, 2))
        t4 = time.perf_counter()
        bs = g > th
        t5 = time.perf_counter()
        for k, gi in enumerate(range(bstart, bend)):
            b = bs[k]
            if prev_b is not None:
                d = prev_b != b
                if _cluster_win3(d) >= p._C:
                    segs += 1
                    s = k
                    rep_sharp = float(sharp[k])
                elif float(sharp[k]) > rep_sharp:
                    rep_sharp = float(sharp[k])
            else:
                rep_sharp = float(sharp[k])
            prev_b = b
        t6 = time.perf_counter()
        acc["get_batch"] += t1 - t0
        acc["asnumpy"] += t2 - t1
        acc["gray"] += t3 - t2
        acc["std"] += t4 - t3
        acc["bin"] += t5 - t4
        acc["loop"] += t6 - t5
        n_batches += 1
        n_edges += 0
    t_all = time.perf_counter() - t_all
    print(f"batches={n_batches} segments={segs} total={t_all:.2f}s")
    for k, v in acc.items():
        print(f"  {k:10s} {v:7.2f}s  ({v/t_all*100:5.1f}%)")
    fps_eff = len(frames) / t_all
    print(f"  effective {fps_eff:7.0f} fps")
    del vr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nvdec", choices=["nvdec", "cpu"])
    args = ap.parse_args()
    if args.backend == "cpu":
        os.environ.setdefault("DECORD_FFMPEG_THREAD_COUNT", "8")
        os.environ.setdefault("DECORD_FILTER_THREADS", "1")
    probe(args.backend)
