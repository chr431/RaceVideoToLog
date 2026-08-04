"""Unified padding benchmark — external (replicate) padding sweep.

Compares:
    - External pad values: 0, 2, 4, 8, 16
    - use_preprocess_img: True (default) vs False
    - use_vertical_padding: True (default) vs False

Test parameters:
    test4.mp4, roi=862,945,957,1003, frame_start=114, frame_end=6317, div=1
    target_h=48, backend=tensorrt, model=v6_tiny, no correction/reOCR

Outputs raw OCR text per frame for direct comparison.
"""
from __future__ import annotations

import csv
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2  # type: ignore[import-not-found]  # 仅基准工具需要，需另装 opencv
import numpy as np

from ocr_engine import extract_speed_value
from gpu_setup import select_backend, get_engine_type
from pipeline import _preprocess_standard

VIDEO = r"D:\Repo\RaceVideoToLog\test4.mp4"
ROI = (862, 945, 957, 1003)
FRAME_START = 114
FRAME_END = 6317
DIV = 1
TARGET_H = 48
BACKEND = "tensorrt"
OCR_MODEL = "v6_tiny"

# Configs to test
PAD_VALUES = [0, 2, 4, 8, 16]


def _patch_ocr(ocr, use_preprocess_img: bool, use_vertical_padding: bool):
    """OcrEngine 预处理参数固定（不再支持运行时补丁），保留签名兼容。"""
    return ocr


def _create_ocr():
    """Create a fresh OcrEngine for the given backend and model."""
    from ocr_native import OcrEngine
    et = get_engine_type()
    return OcrEngine(OCR_MODEL, et)


def run_bench(ocr, cap, pad: int, label: str):
    """Run OCR over all frames with given external pad, return raw texts."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
    results = []
    f = FRAME_START
    t0 = time.perf_counter()
    while f <= FRAME_END:
        ret, frame = cap.read()
        if not ret:
            break
        if (f - FRAME_START) % DIV != 0:
            f += 1
            continue
        crop = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        proc = _preprocess_standard(crop, TARGET_H, pad)
        ocr_res = ocr([proc])[0]
        sv, rt, _ = extract_speed_value(ocr_res)
        results.append(sv if sv is not None else "")
        f += 1
    elapsed = time.perf_counter() - t0
    n = len(results)
    valid = sum(1 for r in results if r != "")
    fps = n / elapsed if elapsed > 0 else 0
    print(f"  {label:35s}  {fps:6.1f} fps  valid={valid}/{n}  ({elapsed:.1f}s)")
    return results


def main():
    print("=" * 80)
    print("Padding Benchmark: external pad (OcrEngine)")
    print(f"Video: {VIDEO}, frames: {FRAME_START}-{FRAME_END}, target_h={TARGET_H}")
    print("=" * 80)

    print("\n[1/2] Selecting backend + loading video...")
    select_backend(BACKEND)

    cap = cv2.VideoCapture(VIDEO)
    total = (FRAME_END - FRAME_START) // DIV + 1
    print(f"  Total frames: {total}")

    print("\n[2/2] Running benchmarks...\n")

    all_results = {}

    # ── 1. External padding sweep ──
    print("── External padding sweep (use_preprocess_img=True, use_vertical_padding=True) ──")
    for pad in PAD_VALUES:
        ocr = _create_ocr()
        _patch_ocr(ocr, True, True)
        results = run_bench(ocr, cap, pad, f"pad={pad}")
        all_results[f"pad={pad}_ppi=T_vp=T"] = results

    # ── 2. use_preprocess_img toggle ──
    print("\n── use_preprocess_img=False (use_vertical_padding=True) ──")
    for pad in [0, 4, 8]:
        ocr = _create_ocr()
        _patch_ocr(ocr, False, True)
        results = run_bench(ocr, cap, pad, f"pad={pad}_ppi=F_vp=T")
        all_results[f"pad={pad}_ppi=F_vp=T"] = results

    # ── 3. use_vertical_padding toggle ──
    print("\n── use_vertical_padding=False (use_preprocess_img=True) ──")
    for pad in [0, 4]:
        ocr = _create_ocr()
        _patch_ocr(ocr, True, False)
        results = run_bench(ocr, cap, pad, f"pad={pad}_ppi=T_vp=F")
        all_results[f"pad={pad}_ppi=T_vp=F"] = results

    # ── 4. Both internal flags off ──
    print("\n── Both internal flags off ──")
    for pad in [0, 4]:
        ocr = _create_ocr()
        _patch_ocr(ocr, False, False)
        results = run_bench(ocr, cap, pad, f"pad={pad}_ppi=F_vp=F")
        all_results[f"pad={pad}_ppi=F_vp=F"] = results

    cap.release()

    # ── Comparison ──
    print("\n" + "=" * 80)
    print("Text difference matrix (vs pad=0, ppi=T, vp=T baseline)")
    print("=" * 80)
    baseline_key = "pad=0_ppi=T_vp=T"
    baseline = all_results[baseline_key]
    print(f"  {'config':40s}  {'diffs':>6s}  {'pct':>7s}")
    print(f"  {'-'*40}  {'-'*6}  {'-'*7}")
    for name in all_results:
        diffs = sum(1 for a, b in zip(baseline, all_results[name]) if a != b)
        pct = diffs / len(baseline) * 100
        marker = " ← baseline" if name == baseline_key else ""
        print(f"  {name:40s}  {diffs:6d}  {pct:6.2f}%{marker}")

    # ── Cross-compare: all pairs diff ──
    print("\n── All-pairs difference summary ──")
    names = list(all_results.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            diffs = sum(1 for a, b in zip(all_results[n1], all_results[n2]) if a != b)
            if diffs > 0:
                print(f"  {n1} vs {n2}: {diffs} diffs ({diffs/len(baseline)*100:.2f}%)")

    # ── Save CSVs ──
    print("\n── Writing CSVs ──")
    for name, results in all_results.items():
        safe = name.replace("=", "_").replace(" ", "")
        path = f"d:/Repo/RaceVideoToLog/bench_padding_{safe}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "raw_value"])
            for i, v in enumerate(results):
                writer.writerow([FRAME_START + i * DIV, v])
        print(f"  {path}  ({len(results)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
