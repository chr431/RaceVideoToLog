"""
Accuracy benchmark: test OCR configs against teacher ground truth.

Compares OCR pipeline output against teacher CSVs (from teacher branch).
Tests different models, preprocessing methods, and correction parameters.

Usage:
    python benchmark_accuracy.py                          # all configs on all videos
    python benchmark_accuracy.py --video test2           # only test2
    python benchmark_accuracy.py --model v5_mobile        # only v5_mobile
"""

import argparse, csv, os, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ocr_engine import (
    _reset_backend, _select_backend, _get_model_kwargs,
    extract_speed_value, ocr_digital_fallback,
    correct_speed_series, correct_speed_series_v2,
    SOURCE_TO_KMH, SpeedObservation,
    clamp_region, compute_video_hash,
)

from rapidocr_onnxruntime import RapidOCR

# ═══════════════════════════════════════════════════════
# Config grid
# ═══════════════════════════════════════════════════════

VIDEO_CONFIGS = {
    # Each test uses truth from teacher CSV AND flag=2 rows from original truth
    # The VIDEO is determined by the truth file header, not the test name
    "test":  {"truth": "teacher_test.csv",  "truth_orig": "test_truth.csv"},
    "test2": {"truth": "teacher_test2.csv", "truth_orig": None},
    "test3": {"truth": "teacher_test3.csv", "truth_orig": "test3_truth.csv"},
    "test4": {"truth": "teacher_test4.csv", "truth_orig": "test4_truth.csv"},
}

# OCR models to test
MODELS = ["v3", "v5_mobile"]

# Preprocessing variants
PREPROCESS = ["raw_resize", "otsu_resize", "clahe_otsu"]

# target_h values to test
TARGET_H_VALS = [24, 32, 48]

# pad values
PAD_VALS = [0, 10, 20]

# max_accel values (for correction)
MAX_ACCEL_VALS = [20, 50, 100]

# Fixed parameters
MAX_SPEED = 400.0
BACKEND = "auto"
NUM_WORKERS = 4


# ═══════════════════════════════════════════════════════
# Truth loading
# ═══════════════════════════════════════════════════════

def parse_teacher_header(csv_path: str) -> dict:
    """Parse teacher CSV header rows."""
    params = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i >= 4: break
            line = line.strip().lstrip("#").strip()
            roi_match = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if roi_match:
                params["roi"] = tuple(int(g) for g in roi_match.groups())
                line = re.sub(r"roi=\d+,\d+,\d+,\d+,\s*", "", line)
            for part in line.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if not k or not v: continue
                    try:
                        params[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        params[k] = v
    return params


def load_teacher_data(csv_path: str) -> tuple[list[float], list[float], dict]:
    """Load teacher CSV. Returns (timestamps, speeds, params)."""
    params = parse_teacher_header(csv_path)
    timestamps, speeds = [], []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i < 4: continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    timestamps.append(float(parts[0]))
                    speeds.append(float(parts[2]))
                except ValueError:
                    continue
    return timestamps, speeds, params


def load_flag2_truth(truth_path: str) -> tuple[list[float], list[float], dict]:
    """Load flag=2 (manually corrected) rows from original truth CSV.

    flag=2 rows are absolutely accurate human corrections.
    Returns (timestamps, speeds, params).
    """
    params = {}
    timestamps, speeds = [], []

    with open(truth_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Parse header if present
    for line in lines[:4]:
        line = line.strip()
        if not line.startswith("#"):
            break
        line = line.lstrip("#").strip()
        roi_match = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
        if roi_match:
            params["roi"] = tuple(int(g) for g in roi_match.groups())
            line = re.sub(r"roi=\d+,\d+,\d+,\d+,\s*", "", line)
        for part in line.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip(), v.strip()
                if not k or not v: continue
                try:
                    params[k] = float(v) if "." in v else int(v)
                except ValueError:
                    params[k] = v

    # Parse data rows, filter flag=2
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                flag = int(parts[3])
                if flag == 2:
                    timestamps.append(float(parts[0]))
                    speeds.append(float(parts[2]))
            except (ValueError, IndexError):
                continue

    return timestamps, speeds, params


# ═══════════════════════════════════════════════════════
# OCR pipeline
# ═══════════════════════════════════════════════════════

def preprocess(crop_bgr, target_h, pad, method="otsu_resize"):
    """Preprocess crop for OCR."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    th = max(8.0, float(target_h))
    pad = max(0.0, float(pad))

    if method == "raw_resize":
        # Raw grayscale → resize
        pass
    elif method == "otsu_resize":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "clahe_otsu":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale = th / h if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)), interpolation=cv2.INTER_LINEAR)

    pad_int = int(pad)
    if pad_int > 0:
        gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_ocr_pipeline(video_path, roi, div, frame_start, frame_end,
                     model_name, preproc_method, target_h, pad,
                     max_speed, max_accel, fmt="km/h", use_v2=True):
    """Run full OCR pipeline and return (timestamps, corrected_speeds, raw_observations)."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Init OCR
    _reset_backend()
    _select_backend(BACKEND)
    model_kwargs = _get_model_kwargs(model_name)
    ocr = RapidOCR(**(model_kwargs or {}))

    # Read frames
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    x1, y1, x2, y2 = clamp_region(*roi, width, height)
    frame_step = max(1, div)

    raw_frames = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None: break
        if frame_end is not None and fi >= int(frame_end): break
        if frame_start is not None and fi < int(frame_start):
            fi += 1; continue
        if fi % frame_step != 0:
            fi += 1; continue
        ts = fi / fps if fps > 0 else 0
        crop = frame[y1:y2+1, x1:x2+1].copy()
        raw_frames.append((ts, crop))
        fi += 1
    cap.release()

    # OCR
    observations = []
    for ts, crop in raw_frames:
        proc = preprocess(crop, target_h, pad, preproc_method)
        ocr_result, _ = ocr(proc)
        sv, rt = extract_speed_value(ocr_result)

        if sv is None:
            # Fallback
            proc2 = preprocess(crop, target_h, pad, "clahe_otsu")
            ocr_result, _ = ocr(proc2)
            sv, rt = extract_speed_value(ocr_result)

        if sv is None:
            sv, rt = ocr_digital_fallback(ocr, crop, max_speed)

        if sv is not None and rt is not None:
            observations.append(SpeedObservation(
                timestamp=ts,
                raw_speed_kmh=sv * SOURCE_TO_KMH[fmt],
                raw_text=rt,
            ))

    if not observations:
        return [], [], []

    # Correction
    if use_v2:
        corrected = correct_speed_series_v2(observations, max_speed, max_accel, fps, div)
    else:
        corrected = correct_speed_series(observations, max_speed, max_accel)
    return [o.timestamp for o in observations], corrected, observations


# ═══════════════════════════════════════════════════════
# Accuracy calculation
# ═══════════════════════════════════════════════════════

def compute_accuracy(ocr_ts, ocr_speeds, truth_ts, truth_speeds, tolerance=0.05):
    """Match OCR output to teacher data by timestamp and compute accuracy.

    Returns dict with metrics.
    """
    if not ocr_ts or not truth_ts:
        return {"exact": 0, "within_1": 0, "within_5": 0, "total": 0, "matched": 0}

    # Build truth lookup by timestamp
    truth_dict = {}
    for t, s in zip(truth_ts, truth_speeds):
        truth_dict[t] = s

    exact = within_1 = within_5 = 0
    matched = 0

    for t, s in zip(ocr_ts, ocr_speeds):
        # Find closest teacher timestamp
        closest_t = min(truth_dict.keys(), key=lambda x: abs(x - t))
        if abs(closest_t - t) < tolerance:
            matched += 1
            truth_s = truth_dict[closest_t]
            diff = abs(s - truth_s)
            if diff < 0.5: exact += 1
            if diff <= 1.0: within_1 += 1
            if diff <= 5.0: within_5 += 1

    total = len(ocr_ts)
    # Accuracy based on MATCHED frames (those with corresponding truth data)
    denom = max(matched, 1)
    return {
        "exact": exact, "within_1": within_1, "within_5": within_5,
        "total": total, "matched": matched,
        "exact_pct": 100 * exact / denom,
        "within_1_pct": 100 * within_1 / denom,
        "within_5_pct": 100 * within_5 / denom,
        "match_pct": 100 * matched / max(total, 1),
    }


# ═══════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════

def run_benchmark(video_filter=None, model_filter=None):
    """Run accuracy benchmark across config grid."""
    results = []

    videos = list(VIDEO_CONFIGS.keys())
    if video_filter:
        videos = [v for v in videos if video_filter in v]

    models = MODELS
    if model_filter:
        models = [m for m in models if model_filter in m]

    for vname in videos:
        cfg = VIDEO_CONFIGS[vname]
        video_path = cfg["video"]
        truth_path = cfg["truth"]
        fmt = cfg["format"]

        if not Path(video_path).exists():
            print(f"SKIP {vname}: video not found")
            continue
        if not Path(truth_path).exists():
            print(f"SKIP {vname}: teacher CSV not found")
            continue

        truth_ts, truth_speeds, tparams = load_teacher_data(truth_path)
        roi = tparams.get("roi")
        div = int(tparams.get("div", 4))
        frame_start = tparams.get("frame_start")
        frame_end = tparams.get("frame_end")

        if roi is None:
            print(f"SKIP {vname}: no ROI in teacher CSV")
            continue

        max_speed = MAX_SPEED

        for model in models:
            for preproc in PREPROCESS:
                for target_h in TARGET_H_VALS:
                    for pad in PAD_VALS:
                        for max_accel in MAX_ACCEL_VALS:
                            config_name = f"{vname}|{model}|{preproc}|h={target_h}|pad={pad}|accel={max_accel}"
                            print(f"  {config_name} ...", end=" ", flush=True)

                            try:
                                t0 = time.perf_counter()
                                ocr_ts, ocr_speeds, _ = run_ocr_pipeline(
                                    video_path, roi, div, frame_start, frame_end,
                                    model, preproc, target_h, pad,
                                    max_speed, max_accel, fmt
                                )
                                t_elapsed = time.perf_counter() - t0

                                metrics = compute_accuracy(ocr_ts, ocr_speeds, truth_ts, truth_speeds)
                                metrics["config"] = config_name
                                metrics["video"] = vname
                                metrics["model"] = model
                                metrics["preproc"] = preproc
                                metrics["target_h"] = target_h
                                metrics["pad"] = pad
                                metrics["max_accel"] = max_accel
                                metrics["time"] = t_elapsed
                                metrics["frames"] = len(ocr_ts)
                                metrics["fps"] = len(ocr_ts) / t_elapsed if t_elapsed > 0 else 0

                                results.append(metrics)
                                print(f"acc={metrics['within_1_pct']:.1f}% (≤1km/h) "
                                      f"exact={metrics['exact_pct']:.1f}% "
                                      f"fps={metrics['fps']:.1f}", flush=True)

                            except Exception as e:
                                print(f"ERROR: {e}", flush=True)

    # Rank results
    results.sort(key=lambda r: (r["within_1_pct"], r["exact_pct"]), reverse=True)

    print(f"\n{'='*80}")
    print("TOP 20 CONFIGURATIONS (by ≤1 km/h accuracy)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Config':<55} {'≤1km/h':<8} {'Exact':<8} {'FPS':<8} {'Time':<8}")
    print("-" * 80)

    for i, r in enumerate(results[:20]):
        config_short = f"{r['video']}|{r['model']}|{r['preproc']}|h={r['target_h']}|pad={r['pad']}|acc={r['max_accel']}"
        print(f"{i+1:<5} {config_short:<55} "
              f"{r['within_1_pct']:>6.1f}% {r['exact_pct']:>6.1f}% "
              f"{r['fps']:>7.1f} {r['time']:>7.1f}s")

    # Per-video best
    print(f"\n{'='*80}")
    print("BEST PER VIDEO")
    print(f"{'='*80}")
    for vname in videos:
        video_results = [r for r in results if r["video"] == vname]
        if video_results:
            best = video_results[0]
            print(f"  {vname}: {best['model']} | {best['preproc']} | "
                  f"h={best['target_h']} pad={best['pad']} acc={best['max_accel']} | "
                  f"≤1km/h={best['within_1_pct']:.1f}% exact={best['exact_pct']:.1f}% "
                  f"fps={best['fps']:.1f}")

    # Per-model summary
    print(f"\n{'='*80}")
    print("PER MODEL SUMMARY")
    print(f"{'='*80}")
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        if model_results:
            avg_acc = sum(r["within_1_pct"] for r in model_results) / len(model_results)
            avg_fps = sum(r["fps"] for r in model_results) / len(model_results)
            best = max(model_results, key=lambda r: r["within_1_pct"])
            print(f"  {model}: avg ≤1km/h={avg_acc:.1f}% avg fps={avg_fps:.1f} "
                  f"best={best['within_1_pct']:.1f}% ({best['video']}|{best['preproc']}|h={best['target_h']}|pad={best['pad']})")

    # Per-preproc summary
    print(f"\n{'='*80}")
    print("PER PREPROCESSING SUMMARY")
    print(f"{'='*80}")
    for preproc in PREPROCESS:
        pp_results = [r for r in results if r["preproc"] == preproc]
        if pp_results:
            avg_acc = sum(r["within_1_pct"] for r in pp_results) / len(pp_results)
            best = max(pp_results, key=lambda r: r["within_1_pct"])
            print(f"  {preproc}: avg ≤1km/h={avg_acc:.1f}% best={best['within_1_pct']:.1f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Accuracy Benchmark")
    parser.add_argument("--video", default=None, help="Filter by video name (test, test2, etc.)")
    parser.add_argument("--model", default=None, help="Filter by model (v3, v5_mobile)")
    args = parser.parse_args()

    print("=" * 80)
    print("OCR ACCURACY BENCHMARK")
    print(f"Models: {MODELS}")
    print(f"Preprocessing: {PREPROCESS}")
    print(f"target_h: {TARGET_H_VALS}, pad: {PAD_VALS}, max_accel: {MAX_ACCEL_VALS}")
    print("=" * 80)

    run_benchmark(video_filter=args.video, model_filter=args.model)
