"""decord DLL smoke test — run after every decord rebuild.

Verifies the modified DLL still decodes correctly:
1. Sequential next() over N frames: shapes sane, no exceptions, EOF OK
2. seek_accurate() round-trips (exercises cached_frame_ replacement)
3. Frame content hash: compares the first K frames against a reference
   DLL run in a subprocess (--hash-compare <dll>) — catches refcount /
   buffer-reuse bugs that produce visually plausible but wrong frames.
4. --roi-check: dual reader, next_roi() vs next()+crop must be byte-identical
   (needs the next_roi API; skipped automatically if absent).

Usage:
    python tools/decord_smoke.py [--frames 500] [--hash-compare _decord_build/decord.dll.bak]
"""
from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

VIDEO = "D:/Videos/racelog_test/test4.mp4"
ROI = (843, 993, 949, 1026)  # from test4 truth header


def _frame_hash(f) -> str:
    return hashlib.sha256(f.tobytes()).hexdigest()[:16]


def seq_read(vr, n_frames: int) -> list:
    """Read n_frames sequentially via next(), return (sha, shape) list."""
    out = []
    for _ in range(n_frames):
        f = vr.next().asnumpy()
        out.append((_frame_hash(f), f.shape))
    return out


def run_smoke(use_roi: bool) -> None:
    import decord
    from decord import VideoReader
    import numpy as np

    print(f"decord DLL: {decord.__file__}")
    vr = VideoReader(VIDEO, ctx=decord.gpu(0))
    n = len(vr)
    print(f"frames: {n}")

    # 1. sequential read
    N = min(args.frames, n)
    print(f"sequential next() x{N} ...", flush=True)
    seq = seq_read(vr, N)
    shapes = {s for _, s in seq}
    print(f"  shapes: {shapes}")
    assert len(shapes) == 1, f"shape changed mid-read: {shapes}"
    assert all(len(h) == 16 for h, _ in seq), "hash format wrong"
    print(f"  first/last hash: {seq[0][0]} / {seq[-1][0]}")
    assert seq[0][0] != seq[10][0], "frames identical (buffer reuse bug?)"

    # 2. seek round-trips
    print("seek_accurate round-trips x50 ...", flush=True)
    for i in range(50):
        target = i * (n // 50)
        vr.seek_accurate(target)
        f = vr.next().asnumpy()
        assert f.ndim == 3 and f.shape[2] == 3, f"bad frame at {target}"
    print("  OK")

    # 3. sequential read to EOF (drain path)
    print("read to EOF ...", flush=True)
    vr.seek_accurate(0)
    cnt = 0
    while True:
        try:
            vr.next().asnumpy()
            cnt += 1
        except StopIteration:
            break
    print(f"  read {cnt} frames to EOF")
    assert cnt == n, f"EOF at {cnt}, expected {n}"

    # 4. ROI byte-compare (dual reader) — needs next_roi API
    if use_roi:
        print("next_roi vs next()+crop byte compare x50 ...", flush=True)
        a = VideoReader(VIDEO, ctx=decord.gpu(0))
        b = VideoReader(VIDEO, ctx=decord.gpu(0))
        x1, y1, x2, y2 = ROI
        for _ in range(50):
            ra = a.next_roi(x1, y1, x2, y2).asnumpy()
            fb = b.next().asnumpy()
            rb = fb[y1:y2, x1:x2]
            np.testing.assert_array_equal(ra, rb)
        print("  OK")
        # seek + continue with next_roi
        a.seek_accurate(1000)
        fa = a.next_roi(x1, y1, x2, y2).asnumpy()
        assert fa.shape == (y2 - y1, x2 - x1, 3), fa.shape
        print(f"  post-seek ROI shape: {fa.shape}")

    # 5. CPU path ROI: NextFrameRoi on a CPU ctx does a row-stride ROI copy
    #    (must be byte-identical to full-frame next() + slice)
    if use_roi:
        print("CPU ctx next_roi vs next()+crop byte compare x50 ...", flush=True)
        vr_cpu_a = VideoReader(VIDEO, ctx=decord.cpu(0))
        vr_cpu_b = VideoReader(VIDEO, ctx=decord.cpu(0))
        for _ in range(50):
            ra = vr_cpu_a.next_roi(x1, y1, x2, y2).asnumpy()
            fb = vr_cpu_b.next().asnumpy()
            rb = fb[y1:y2, x1:x2]
            np.testing.assert_array_equal(ra, rb)
        print("  OK")
        vr_cpu_a.seek_accurate(1000)
        fa = vr_cpu_a.next_roi(x1, y1, x2, y2).asnumpy()
        assert fa.shape == (y2 - y1, x2 - x1, 3), fa.shape
        print(f"  post-seek CPU ROI shape: {fa.shape}")


def hash_worker(frames: int) -> None:
    """Subprocess: print hashes of first K frames (uses DECORD_LIBRARY_PATH)."""
    from decord import VideoReader
    import decord
    print(f"worker DLL: {decord.__file__}", file=sys.stderr)
    vr = VideoReader(VIDEO, ctx=decord.gpu(0))
    for h, _ in seq_read(vr, frames):
        print(h)
    sys.stdout.flush()


def hash_compare(old_dll: Path, frames: int) -> None:
    import decord
    cur_dll = Path(decord.__file__)
    env = dict(os.environ)
    env["DECORD_LIBRARY_PATH"] = str(old_dll)
    r = subprocess.run([sys.executable, __file__, "--hash-worker", str(frames)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=str(Path(__file__).parent))
    if r.returncode != 0:
        print(f"worker failed: {r.stderr[-500:]}")
        sys.exit(1)
    old_hashes = r.stdout.splitlines()
    print(f"reference DLL: {old_dll} ({len(old_hashes)} frames)")
    vr = VideoReader(VIDEO, ctx=decord.gpu(0))
    new_hashes = [h for h, _ in seq_read(vr, frames)]
    for i, (oh, nh) in enumerate(zip(old_hashes, new_hashes)):
        if oh != nh:
            print(f"  MISMATCH frame {i}: old={oh} new={nh}")
            sys.exit(1)
    print(f"  {len(new_hashes)} frame hashes identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--hash-compare", type=str, default="",
                    help="reference DLL path for frame-hash comparison")
    ap.add_argument("--hash-worker", type=int, default=0,
                    help="(internal) print hashes of first N frames")
    ap.add_argument("--no-roi", action="store_true", help="skip next_roi check")
    args = ap.parse_args()
    if args.hash_worker:
        hash_worker(args.hash_worker)
        sys.exit(0)
    has_roi = not args.no_roi
    if has_roi:
        # next_roi missing on old DLLs -> auto-skip
        try:
            from decord import VideoReader
            if not hasattr(VideoReader, "next_roi"):
                has_roi = False
        except Exception:
            has_roi = False
    run_smoke(has_roi)
    if args.hash_compare:
        print("hash compare:")
        hash_compare(args.hash_compare, min(args.frames, 200))
    print("SMOKE OK")
