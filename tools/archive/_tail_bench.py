"""尾部串行段微基准：_store_run_state + _build_rows + _write_csv。

目的：量化 run() 中解码/OCR 并行结束后不可隐藏的串行开销随帧数的
缩放（3000 帧剖面显示 total−decode−correction≈0.3s，全量 23k 帧线性
放大后可能达 1.5-2.5s）。构造假段数据（不触发解码）直接测三件套。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np  # noqa: E402
import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402


def synth(n_seg: int, seg_len: int = 3, start=0) -> tuple:
    """伪造 n_seg 个段（每段 3 帧）的输入形态（不动视频/解码）。"""
    frames = list(range(start, start + n_seg * seg_len))
    segs = [frames[i * seg_len:(i + 1) * seg_len] for i in range(n_seg)]
    seg_vals = [float(50 + (i % 200)) for i in range(n_seg)]
    corr = [v if i % 7 != 0 else v + 1 for i, v in enumerate(seg_vals)]
    rep_frames = [s[1] for s in segs]
    crops = {fp: np.zeros((int(99 * 1.5), 100), dtype=np.uint8)  # 伪 YUV420 (h*1.5,w)
             for fp in rep_frames}
    return frames, segs, seg_vals, corr, rep_frames, crops


def bench(n_seg: int):
    p = SegmentPipeline.__new__(SegmentPipeline)  # 不做 __init__（不动解码路径）
    p._fps = 59.767
    p._frame_start = 100
    p._frame_end = 100 + n_seg * 3
    p._speed_format = "km/h"
    p._video_path = Path("fake.mp4")
    p._backend = "decord/CPU"
    p._ocr_model = "v6_small"
    p._ocr_backend_used = "onnxruntime"
    p._n_segments = n_seg
    p._n_corr = 17
    p._max_speed = 400.0
    p._max_accel = 40.0
    p._force_aspect = 1.5
    p._fill_width = 224
    p._roi = (843, 993, 948, 1025)
    p.timing = {"decode": 1.0, "ocr": 1.0}
    p._dp_anchor_conf = 0.9
    p._conf_vals = None

    frames, segs, seg_vals, corr, rep_frames, crops = synth(n_seg)

    t0 = time.perf_counter()
    p._store_run_state(frames, crops, segs, seg_vals, rep_frames, corr)
    t1 = time.perf_counter()
    rows = p._build_rows(frames, segs, corr, raw=seg_vals, conf=None)
    t2 = time.perf_counter()
    p._write_csv(rows, PROJECT / "outputs" / "_tail_bench.csv")
    t3 = time.perf_counter()
    print(f"seg={n_seg:>6} 帧={len(frames):>7}  "
          f"store={t1-t0:.3f}s  build_rows={t2-t1:.3f}s  write_csv={t3-t2:.3f}s  "
          f"合计={t3-t0:.3f}s")


if __name__ == "__main__":
    for n in (1000, 3000, 8000, 20000):
        bench(n)