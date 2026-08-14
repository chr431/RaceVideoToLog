"""异步批解码 A/B 验证（一次性实验，v0.7.9 攻关用）。

同一批帧位置，分别用旧 DLL（逐帧 display sync）与新 DLL（异步批解码）
的 GPU 解码器读取（gray + ROI-first，生产路径），输出帧哈希。两轮哈希
逐位一致 → 异步管线输出与同步管线完全一致（无撕裂/无错帧/无顺序错乱）。

用法:
    .venv\\Scripts\\python tools/archive/_verify_async_ab.py
    .venv\\Scripts\\python tools/archive/_verify_async_ab.py --worker   # 子进程模式
"""
from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import sys

VIDEOS = {
    "test":  (r"D:\Videos\racelog_test\test.mp4",  (843, 993, 949, 1026)),
    "test2": (r"D:\Videos\racelog_test\test2.mp4", (843, 993, 949, 1026)),
    "test3": (r"D:\Videos\racelog_test\test3.mp4", (843, 993, 949, 1026)),
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (841, 994, 949, 1026)),
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
}
OLD_DIR = r"D:\Repo\RaceVideoToLog\_decord_old"


def worker() -> None:
    """打印所有视频采样帧的哈希（每行: video:idx:hash）。"""
    import numpy as np
    from decord import VideoReader, gpu
    for name, (path, roi) in VIDEOS.items():
        x1, y1, x2, y2 = roi
        vr = VideoReader(path, ctx=gpu(0), output_format="gray",
                         roi=(x1, y1, x2 + 1, y2 + 1))
        n = len(vr)
        # 覆盖三种访问模式：顺序 get_batch、跨批、seek 后 get_batch
        batches = [
            list(range(0, min(64, n))),                       # 顺序批
            list(range(n // 2, n // 2 + 64)),                 # 中段批（先 seek）
            list(range(max(0, n - 64), n)),                   # 尾部批
        ]
        idx = 0
        for b in batches:
            f = vr.get_batch(b).asnumpy()
            for k, fi in enumerate(b):
                h = hashlib.sha256(f[k].tobytes()).hexdigest()[:16]
                print(f"{name}:{idx}:{h}")
                idx += 1
        # seek 抖动 + 单帧
        for pos in (0, 137, n // 3, n - 100):
            vr.seek_accurate(max(0, pos))
            f = vr.next().asnumpy()
            h = hashlib.sha256(f.tobytes()).hexdigest()[:16]
            print(f"{name}:{idx}:{h}")
            idx += 1
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--old-dir", default=OLD_DIR)
    args = ap.parse_args()
    if args.worker:
        worker()
        return
    env = dict(os.environ)
    env["DECORD_LIBRARY_PATH"] = args.old_dir
    r = subprocess.run([sys.executable, __file__, "--worker",
                        "--old-dir", args.old_dir],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        print("OLD worker failed:\n", r.stderr[-800:])
        sys.exit(1)
    old: dict = {}
    for line in r.stdout.splitlines():
        v, i, h = line.rsplit(":", 2)
        old[f"{v}:{i}"] = h
    new = {}
    out = subprocess.run([sys.executable, __file__, "--worker",
                          "--old-dir", args.old_dir],
                         capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        print("NEW worker failed:\n", out.stderr[-800:])
        sys.exit(1)
    for line in out.stdout.splitlines():
        v, i, h = line.rsplit(":", 2)
        new[f"{v}:{i}"] = h
    bad = 0
    for key in sorted(old):
        if new.get(key) != old[key]:
            print(f"MISMATCH {key}: old={old[key]} new={new.get(key)}")
            bad += 1
    print(f"compared {len(old)} frames: {'ALL IDENTICAL' if bad == 0 else str(bad) + ' MISMATCHES'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
