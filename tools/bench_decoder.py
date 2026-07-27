"""Decoder benchmark: cv2 vs decord on test.mp4.

Usage: python tools/bench_decoder.py
"""
from __future__ import annotations
import subprocess, sys, time, re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
VIDEO = str(PROJECT / "test.mp4")
TRUTH = str(PROJECT / "ground_truth_csv/test_truth.csv")
RUNS = 2  # run twice per decoder, use second run (warm cache)


def run(video_backend: str) -> dict:
    """Run headless pipeline and parse timing from stdout."""
    out_csv = str(PROJECT / f"test_bench_{video_backend}.csv")
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(PROJECT / "RaceVideoToLog.py"),
         VIDEO, "--from-csv", TRUTH,
         "--video-backend", video_backend,
         "--backend", "tensorrt",
         "--log-level", "detailed",
         "-o", out_csv],
        cwd=str(PROJECT), capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
    )
    wall = time.perf_counter() - t0
    timing = {"wall_s": wall, "backend": video_backend}
    # Parse timing line: N frames (cv2/decord), total X.Xs | decode X.Xs (Y fps) | ...
    for line in r.stdout.splitlines():
        m = re.search(r"(\d+) .*\((\w+)\).* ([\d.]+)s.*解码 ([\d.]+)s \(([\d.]+) fps\).*推理 ([\d.]+)s \(([\d.]+) fps\).* ([\d.]+) fps", line)
        if m:
            timing["frames"] = int(m.group(1))
            timing["actual_backend"] = m.group(2)
            timing["total_s"] = float(m.group(3))
            timing["decode_s"] = float(m.group(4))
            timing["decode_fps"] = float(m.group(5))
            timing["inference_s"] = float(m.group(6))
            timing["inference_fps"] = float(m.group(7))
            timing["total_fps"] = float(m.group(8))
    # Also parse correction/total timing
    for line in r.stdout.splitlines():
        m = re.search(r"correction=([\d.]+)s", line)
        if m: timing["correction_s"] = float(m.group(1))
        m = re.search(r"total=([\d.]+)s", line)
        if m: timing["total_pipeline_s"] = float(m.group(1))
    if r.stderr:
        timing["stderr"] = r.stderr[-500:]
    return timing


def print_row(label: str, t: dict) -> None:
    if "frames" not in t:
        print(f"  {label}: ERROR — {t.get('stderr', 'no output')[:200]}")
        return
    print(f"  {label:8s} | {t.get('actual_backend','?'):6s} | {t['frames']:5d} fr | "
          f"wall {t['wall_s']:5.1f}s | "
          f"decode {t['decode_s']:5.1f}s ({t['decode_fps']:6.0f} fps) | "
          f"infer {t['inference_s']:5.1f}s ({t['inference_fps']:6.0f} fps) | "
          f"total {t['total_fps']:6.0f} fps")


if __name__ == "__main__":
    print(f"Video: {VIDEO}")
    print(f"Truth: {TRUTH}")
    print(f"Runs per decoder: {RUNS} (last used for stats)")
    print(f"{'':-^80}")

    for backend in ("cv2", "decord"):
        for run_i in range(RUNS):
            label = f"{backend} r{run_i+1}"
            print(f"  Running {label}...", end=" ", flush=True)
            t = run(backend)
            print_row(label, t)

    print(f"\n{'':-^80}")
    print("Done. Output CSVs: test_bench_cv2.csv, test_bench_decord.csv")
