"""实验：AV1 纯解码的线程扩展性（wall vs CPU 时间）。

假说：decord 的 drain-then-send 节奏让 dav1d 只有 1-2 帧在途 →
dav1d 内部线程池无法并行 → 无论 thread_count 多少都接近单线程。
验证：解码同样帧数，dcd=1 vs dcd=12 的 wall 与 user CPU 时间
（user/wall ≈ 1 → 单线程；user/wall ≈ N → N 核在跑）。
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from tools.detect_eval import load_meta  # noqa: E402

VIDEO = "D:/Videos/racelog_test/test6.mp4"
N_FRAMES = 3000


def bench(dcd: int):
    import psutil
    from decord import VideoReader, cpu
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta("test6")
    vr = VideoReader(VIDEO, ctx=cpu(0), output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=dcd)
    codec = vr.get_codec()
    frames = list(range(f_start, f_start + N_FRAMES))
    proc = psutil.Process()
    t0 = time.perf_counter()
    c0 = proc.cpu_times()
    out = vr.get_batch(frames, roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1))
    wall = time.perf_counter() - t0
    c1 = proc.cpu_times()
    user = (c1.user - c0.user) + (c1.system - c0.system)
    fps_dec = N_FRAMES / wall
    print(f"  dcd={dcd:>2} codec={codec}: wall={wall:>6.2f}s "
          f"cpu={user:>6.2f}s cpu/wall={user / wall:>4.1f} "
          f"→ {fps_dec:>6.0f} fps（shape={out.shape}）")


if __name__ == "__main__":
    print(f"test6 AV1 纯解码 {N_FRAMES} 帧（ROI gray，get_batch）:")
    for dcd in (1, 2, 4, 8, 12):
        bench(dcd)
