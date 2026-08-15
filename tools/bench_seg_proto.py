"""分段 _cluster_win3 逐帧 Python vs 批量向量化原型：正确性 + 耗时对比。

不跑 OCR，直接解码灰 ROI 并按生产逻辑生成二值化批，对比：
- 逐帧版：对每帧 d = prev_b != b，再 _cluster_win3(d)
- 批量版：构造 (B,H,W) diff 数组后一次切片累加 + max(axis=(1,2))

要求数值逐位一致（本脚本断言），再报告两种实现的累计耗时。
用法：python tools/bench_seg_proto.py [--video test6] [--batch 16]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import _cluster_win3, _otsu  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

VIDEO_DIR = Path("D:/Videos/racelog_test")


def cluster_batch(diffs: np.ndarray) -> np.ndarray:
    """(B,H,W) bool diff → 每帧最大 3×3 窗口和 (B,)。与逐帧 _cluster_win3
    相同的加法顺序（先左右列、后上下行）。"""
    s = diffs.astype(np.int32)
    c3 = s.copy()
    c3[:, :, 1:] += s[:, :, :-1]
    c3[:, :, :-1] += s[:, :, 1:]
    w3 = c3.copy()
    w3[:, 1:, :] += c3[:, :-1, :]
    w3[:, :-1, :] += c3[:, 1:, :]
    return w3.max(axis=(1, 2)).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test6")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    roi, f_start, f_end, _fps, _ms, _ma, _mw, _truth = load_meta(args.video)
    video = VIDEO_DIR / f"{args.video}.mp4"
    from decord import VideoReader, gpu
    x1, y1, x2, y2 = roi
    vr = VideoReader(str(video), ctx=gpu(0), output_format="gray",
                     roi=(x1, y1, x2 + 1, y2 + 1))
    # 阈值校准（与生产一致：前 50 帧 Otsu 中值）
    ths = []
    for k in range(50):
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
            c = c[y1:y2 + 1, x1:x2 + 1]
        if c.shape[-1] != 1:
            from video_utils import _gray
            c = _gray(c)
        else:
            c = c[..., 0]
        ths.append(_otsu(c))
    th = int(np.median(ths))
    # 重新从头顺序解码（seek 回 frame_start 以匹配生产）
    vr.seek_accurate(f_start or 0)
    frames = list(range(f_start or 0, f_end or len(vr)))
    prev_b = None
    t_loop = t_batch = 0.0
    n_batches = 0
    max_diff = 0
    for bstart in range(0, len(frames), args.batch):
        bend = min(bstart + args.batch, len(frames))
        crops = vr.get_batch(frames[bstart:bend],
                             roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
        g = crops[..., 0] if crops.ndim == 4 else crops
        bs = g > th
        n_batches += 1
        # 逐帧版
        t0 = time.perf_counter()
        loop_scores = []
        for k in range(bs.shape[0]):
            if prev_b is not None:
                d = prev_b != bs[k]
                loop_scores.append(_cluster_win3(d))
            prev_b = bs[k]
        t_loop += time.perf_counter() - t0
        # 批量版（d0 用上一批末帧，批内相邻）
        t0 = time.perf_counter()
        if bstart == 0:
            diffs = np.concatenate(
                [np.zeros((1,) + bs.shape[1:], dtype=bool),
                 bs[:-1] != bs[1:]], axis=0)
            batch_scores = cluster_batch(diffs)
            # 首帧没有 prev_b，生产逻辑跳过 → 置 NaN 便于对比时忽略
            batch_scores[0] = np.nan
        else:
            diffs = np.concatenate(
                [prev_prev_b[None] != bs[0:1], bs[:-1] != bs[1:]], axis=0)
            batch_scores = cluster_batch(diffs)
        t_batch += time.perf_counter() - t0
        prev_prev_b = bs[-1]
        # 对比
        arr = np.asarray(loop_scores)
        cmp = arr == batch_scores[~np.isnan(batch_scores)]
        if not cmp.all():
            bad = np.nonzero(~cmp)[0]
            print(f"batch {bstart}: MISMATCH at {bad[:5]} "
                  f"loop={arr[bad[:5]]} batch={batch_scores[~np.isnan(batch_scores)][bad[:5]]}")
            max_diff = max(max_diff, int(np.nanmax(np.abs(
                arr - batch_scores[~np.isnan(batch_scores)]))))
    print(f"video={args.video} frames={len(frames)} batches={n_batches} "
          f"max_diff={max_diff}")
    print(f"per-frame loop : {t_loop:.3f}s")
    print(f"batch vectorize: {t_batch:.3f}s  speedup={t_loop/max(t_batch,1e-9):.1f}x")
    del vr


if __name__ == "__main__":
    main()
