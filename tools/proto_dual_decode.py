"""双解码器分片并行吞吐验证原型。

两个 VideoReader（CPU）各解码一半帧，测吞吐是否翻倍。
只测解码，不做 OCR/分段。
"""
from __future__ import annotations
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from decord import VideoReader, cpu  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402
from segment_flow import _gray, _cluster_win3, _otsu  # noqa: E402


def main() -> None:
    v = "test5"
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    x1, y1, x2, y2 = roi
    video = f"D:/Videos/racelog_test/{v}.mp4"
    total = 7223

    def decode_range(vr, s, e, n_threads):
        vr.seek_accurate(s)
        # 阈值校准（与流水线一致）
        ths = []
        for _ in range(50):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            ths.append(_otsu(_gray(c)))
        th = int(np.median(ths))
        prev_b = None
        n = 0
        for fi in range(s, e):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            g = _gray(c)
            b = g > th
            if prev_b is not None:
                _cluster_win3(prev_b != b)
            prev_b = b
            n += 1
        return n

    # ── 单解码器（基线）──
    vr = VideoReader(video, ctx=cpu(0))
    t0 = time.perf_counter()
    n = decode_range(vr, 0, total, 4)
    t1 = time.perf_counter()
    print(f"单解码器: {n} 帧 {t1-t0:.2f}s ({n/(t1-t0):.0f}fps)")
    del vr

    # ── 双解码器分片 ──
    mid = total // 2
    vr1 = VideoReader(video, ctx=cpu(0))
    vr2 = VideoReader(video, ctx=cpu(0))
    res = {}

    def worker(tag, vr_, s, e):
        res[tag] = decode_range(vr_, s, e, 4)

    t0 = time.perf_counter()
    th1 = threading.Thread(target=worker, args=("a", vr1, 0, mid))
    th2 = threading.Thread(target=worker, args=("b", vr2, mid, total))
    th1.start(); th2.start(); th1.join(); th2.join()
    t1 = time.perf_counter()
    print(f"双解码器: {res['a']+res['b']} 帧 {t1-t0:.2f}s "
          f"({(res['a']+res['b'])/(t1-t0):.0f}fps)")
    del vr1, vr2


if __name__ == "__main__":
    main()
