"""CPU 解码路径耗时分解 + 批量解码收益验证。

分解逐帧 next_roi 的耗时构成（CAPI/解码 / asnumpy / 灰度 / std / 阈值+聚类），
并对比 get_batch 批量解码的每帧耗时。只跑前 N 帧，不做 OCR。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from decord import VideoReader, cpu  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from segment_flow import _gray, _cluster_win3, _otsu  # noqa: E402


def profile_next_roi(video: str, roi, n: int = 2000) -> None:
    x1, y1, x2, y2 = roi
    vr = VideoReader(video, ctx=cpu(0))
    # 阈值校准（与流水线一致）
    ths = []
    for _ in range(50):
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        g = _gray(c)
        ths.append(_otsu(g))
    th = int(np.median(ths))
    t_next = t_np = t_gray = t_std = t_bin = t_clu = 0.0
    prev_b = None
    t0 = time.perf_counter()
    for k in range(n):
        t = time.perf_counter()
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
        t_next += time.perf_counter() - t
        t = time.perf_counter()
        a = c.asnumpy()
        t_np += time.perf_counter() - t
        t = time.perf_counter()
        g = _gray(a)
        t_gray += time.perf_counter() - t
        t = time.perf_counter()
        s = float(g.std())
        t_std += time.perf_counter() - t
        t = time.perf_counter()
        b = g > th
        if prev_b is not None:
            d = prev_b != b
            w = _cluster_win3(d)
        prev_b = b
        t_clu += time.perf_counter() - t
    wall = time.perf_counter() - t0
    print(f"next_roi 逐帧 {n} 帧: 总 {wall:.2f}s ({n/wall:.0f}fps)")
    print(f"  next_roi(CAPI): {t_next:.2f}s ({t_next/wall*100:.0f}%)")
    print(f"  asnumpy:        {t_np:.2f}s ({t_np/wall*100:.0f}%)")
    print(f"  _gray:          {t_gray:.2f}s ({t_gray/wall*100:.0f}%)")
    print(f"  g.std():        {t_std:.2f}s ({t_std/wall*100:.0f}%)")
    print(f"  阈值+聚类:      {t_clu:.2f}s ({t_clu/wall*100:.0f}%)")
    print(f"  其余(Python/GIL/拷贝): {wall-t_next-t_np-t_gray-t_std-t_clu:.2f}s")
    del vr


def profile_get_batch(video: str, roi, n: int = 2000, bsz: int = 16) -> None:
    x1, y1, x2, y2 = roi
    vr = VideoReader(video, ctx=cpu(0))
    t0 = time.perf_counter()
    total = 0
    for s in range(0, n, bsz):
        batch = vr.get_batch(list(range(s, min(s + bsz, n))))
        for f in batch.asnumpy():
            total += 1
            _ = f[y1:y2 + 1, x1:x2 + 1].copy()  # ROI 裁剪
    wall = time.perf_counter() - t0
    print(f"get_batch(bsz={bsz}) {total} 帧: 总 {wall:.2f}s ({total/wall:.0f}fps)"
          f" —— 含全帧 numpy 拷贝+ROI 裁剪")
    del vr


def main() -> None:
    v = "test5"
    roi, _fs, _fe, _fps, _ms, _ma, _mw, _t = load_meta(v)
    video = f"D:/Videos/racelog_test/{v}.mp4"
    print(f"== {v} roi={roi} ==")
    profile_next_roi(video, roi)
    print()
    for bsz in (8, 16, 32):
        profile_get_batch(video, roi, bsz=bsz)


if __name__ == "__main__":
    main()
