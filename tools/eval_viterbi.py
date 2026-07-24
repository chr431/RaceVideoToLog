"""Viterbi parameter evaluation — in-process pipeline + ground truth comparison.

Usage: python tools/eval_viterbi.py [--baseline] [--sweep]
"""
from __future__ import annotations
import sys, time, copy, json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from ocr_engine import parse_csv_header, Flag
from pipeline import ProcessingPipeline
import config


def load_truth(path: str) -> dict[int, float]:
    truth: dict[int, float] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                truth[int(float(parts[0]))] = float(parts[2])
    return truth


def load_result(path: str) -> dict[int, tuple[float, int]]:
    result: dict[int, tuple[float, int]] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                result[int(float(parts[0]))] = (float(parts[2]), int(parts[3]))
    return result


def evaluate(truth: dict[int, float], result: dict[int, tuple[float, int]]) -> dict:
    """Compare pipeline output against ground truth."""
    common = sorted(set(truth) & set(result))
    n = len(common)
    if n == 0:
        return {"error": "no common frames"}

    diffs = [abs(truth[fi] - result[fi][0]) for fi in common]
    diffs_arr = np.array(diffs)

    correct_2 = int(np.sum(diffs_arr <= 2))       # diff ≤ 2
    severe_5 = int(np.sum(diffs_arr >= 5))          # diff ≥ 5
    total_errors = int(np.sum(diffs_arr > 0.5))     # old metric for comparison

    # Error breakdown
    diff_1 = int(np.sum((diffs_arr > 0.5) & (diffs_arr <= 1)))
    diff_2 = int(np.sum((diffs_arr > 1) & (diffs_arr <= 2)))
    diff_3_5 = int(np.sum((diffs_arr > 2) & (diffs_arr < 5)))
    diff_5plus = severe_5

    # Top-5 deviations (robust against truth=-1 sentinels)
    top5 = sorted(diffs_arr, reverse=True)[:5]

    # False trusted: flag >= 21 but wrong
    false_trusted = 0
    for fi in common:
        r_speed, r_flag = result[fi]
        if Flag.is_trusted(r_flag) and abs(truth[fi] - r_speed) > 0.5:
            false_trusted += 1

    # HIGH_TRUST coverage
    n_trusted = sum(1 for fi in common if Flag.is_trusted(result[fi][1]))

    return {
        "n": n,
        "correct_2": correct_2, "correct_2_pct": 100 * correct_2 / n,
        "severe_5": severe_5, "severe_5_pct": 100 * severe_5 / n,
        "total_errors": total_errors, "error_rate": 100 * total_errors / n,
        "diff_1": diff_1, "diff_2": diff_2, "diff_3_5": diff_3_5, "diff_5plus": diff_5plus,
        "max_diff": float(np.max(diffs_arr)),
        "top5_diffs": [float(d) for d in top5],
        "median_diff": float(np.median(diffs_arr)),
        "mean_diff": float(np.mean(diffs_arr)),
        "false_trusted": false_trusted,
        "n_trusted": n_trusted,
    }


def run_pipeline(video: str, truth_csv: str,
                 max_speed: float, max_accel: float,
                 reocr_only: bool = False) -> tuple[list, str]:
    """Run pipeline in-process, return rows and output path."""
    import tempfile, os
    settings = parse_csv_header(truth_csv)
    roi_str = settings.get("roi", "")
    roi = tuple(int(x) for x in roi_str.split(","))
    div = int(settings.get("div", "1"))
    target_h = int(float(settings.get("target_h", "48")))
    pad = int(float(settings.get("pad", "0")))
    buffer_size = int(settings.get("buffer", "16"))
    backend = settings.get("backend", "auto")
    ocr_model = settings.get("model", "v6_tiny")
    speed_format = settings.get("format", "km/h")
    frame_start = settings.get("frame_start", "")
    frame_end = settings.get("frame_end", "")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output = f.name

    def progress(msg: str, pct: float) -> None:
        pass  # silent

    pipeline = ProcessingPipeline(
        video_path=video, roi=roi, max_speed=max_speed, max_accel=max_accel,
        frame_div=div, target_h=target_h, pad=pad, buffer_size=buffer_size,
        backend=backend, ocr_model=ocr_model, reocr_model="v6_small",
        speed_format=speed_format,
        frame_start=frame_start, frame_end=frame_end,
        progress_cb=progress, log_level="normal",
    )
    pipeline.run_auto(output, reocr_only=reocr_only)

    return pipeline._rows, output


def print_result(label: str, r: dict) -> None:
    if "error" in r:
        print(f"  {label}: ERROR — {r['error']}")
        return
    print(f"  --- {label} ({r['n']} frames) ---")
    print(f"  Correct (diff≤2): {r['correct_2']} ({r['correct_2_pct']:.2f}%)")
    print(f"  Severe  (diff≥5): {r['severe_5']} ({r['severe_5_pct']:.2f}%)")
    print(f"  Error breakdown:  Δ1={r['diff_1']}  Δ2={r['diff_2']}  Δ3-5={r['diff_3_5']}  Δ5+={r['diff_5plus']}")
    print(f"  Total errors (>0.5): {r['total_errors']} ({r['error_rate']:.2f}%)")
    print(f"  Top-5 diffs: {[f'{d:.0f}' for d in r.get('top5_diffs', [r['max_diff']])]}  Median: {r['median_diff']:.2f}  Mean: {r['mean_diff']:.2f}")
    print(f"  False trusted: {r['false_trusted']}  HIGH_TRUST frames: {r['n_trusted']}")


def baseline(video: str, truth_csv: str, max_speed: float, max_accel: float,
             reocr_only: bool = True) -> None:
    """Run baseline evaluation with current config."""
    t0 = time.perf_counter()
    rows, output = run_pipeline(video, truth_csv, max_speed, max_accel, reocr_only=reocr_only)
    elapsed = time.perf_counter() - t0

    truth = load_truth(truth_csv)
    result = load_result(output)
    r = evaluate(truth, result)

    print(f"\n{'='*60}")
    print(f"Video: {Path(video).name}  ({elapsed:.1f}s)")
    print(f"Config: obs={config.VITERBI_OBS_WEIGHT} profile={config.VITERBI_PROFILE_WEIGHT} "
          f"accel={config.VITERBI_ACCEL_WEIGHT} conf_bonus={config.VITERBI_CONF_BONUS}")
    print_result("Baseline", r)

    # Cleanup
    import os
    if os.path.exists(output):
        os.unlink(output)

    return r


def set_config(**kwargs) -> dict:
    """Temporarily update config module attributes. Returns original values."""
    original = {}
    for key, val in kwargs.items():
        original[key] = getattr(config, key, None)
        setattr(config, key, val)
    return original


def restore_config(original: dict) -> None:
    for key, val in original.items():
        if val is not None:
            setattr(config, key, val)


def sweep(video: str, truth_csv: str, max_speed: float, max_accel: float,
          param_name: str, values: list, reocr_only: bool = True) -> list:
    """Sweep a single parameter over multiple values."""
    results = []
    truth = load_truth(truth_csv)  # load once

    for val in values:
        orig = set_config(**{param_name: val})
        t0 = time.perf_counter()
        rows, output = run_pipeline(video, truth_csv, max_speed, max_accel, reocr_only=reocr_only)
        elapsed = time.perf_counter() - t0

        result = load_result(output)
        r = evaluate(truth, result)
        r["param"] = f"{param_name}={val}"
        r["time"] = elapsed
        results.append(r)

        print(f"  {param_name}={val}: correct_2={r['correct_2_pct']:.2f}% "
              f"severe_5={r['severe_5']} max={r['max_diff']:.0f} "
              f"median={r['median_diff']:.2f} false_trusted={r['false_trusted']} "
              f"({elapsed:.0f}s)")

        restore_config(orig)

        # Cleanup
        import os
        if os.path.exists(output):
            os.unlink(output)

    return results


def main() -> None:
    truth_csv = str(PROJECT / "ground_truth_csv/test4_truth.csv")
    video = str(PROJECT / "test4.mp4")
    if not Path(video).exists():
        print(f"ERROR: {video} not found")
        return

    max_speed = 400.0
    max_accel = 70.0

    mode = sys.argv[1] if len(sys.argv) > 1 else "--baseline"

    if mode == "--baseline":
        baseline(video, truth_csv, max_speed, max_accel)

    elif mode == "--sweep-obs":
        print(f"\n{'='*60}")
        print("Sweeping VITERBI_OBS_WEIGHT")
        print(f"{'='*60}")
        results = sweep(video, truth_csv, max_speed, max_accel,
                        "VITERBI_OBS_WEIGHT", [0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

    elif mode == "--sweep-profile":
        print(f"\n{'='*60}")
        print("Sweeping VITERBI_PROFILE_WEIGHT")
        print(f"{'='*60}")
        results = sweep(video, truth_csv, max_speed, max_accel,
                        "VITERBI_PROFILE_WEIGHT", [0.0, 0.15, 0.5, 1.0, 2.0, 5.0])

    elif mode == "--sweep-accel":
        print(f"\n{'='*60}")
        print("Sweeping VITERBI_ACCEL_WEIGHT")
        print(f"{'='*60}")
        results = sweep(video, truth_csv, max_speed, max_accel,
                        "VITERBI_ACCEL_WEIGHT", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

    elif mode == "--sweep-conf":
        print(f"\n{'='*60}")
        print("Sweeping VITERBI_CONF_BONUS")
        print(f"{'='*60}")
        results = sweep(video, truth_csv, max_speed, max_accel,
                        "VITERBI_CONF_BONUS", [0.0, 0.03, 0.05, 0.08, 0.1])

    elif mode == "--sweep-trust":
        print(f"\n{'='*60}")
        print("Sweeping VITERBI_TRUST_THRESHOLD")
        print(f"{'='*60}")
        results = sweep(video, truth_csv, max_speed, max_accel,
                        "VITERBI_TRUST_THRESHOLD", [0.6, 0.7, 0.8, 0.85, 0.9, 0.95])

    elif mode == "--sweep-all":
        # Full grid sweep of key params
        print(f"\n{'='*60}")
        print("Full parameter sweep")
        print(f"{'='*60}")
        all_results = []
        for obs in [0.2, 0.3, 0.5]:
            for profile in [0.05, 0.1, 0.15]:
                for accel in [0.5, 1.0, 2.0]:
                    orig = set_config(
                        VITERBI_OBS_WEIGHT=obs,
                        VITERBI_PROFILE_WEIGHT=profile,
                        VITERBI_ACCEL_WEIGHT=accel,
                    )
                    t0 = time.perf_counter()
                    rows, output = run_pipeline(video, truth_csv, max_speed, max_accel)
                    elapsed = time.perf_counter() - t0
                    restore_config(orig)

                    truth = load_truth(truth_csv)
                    result = load_result(output)
                    r = evaluate(truth, result)
                    r["param"] = f"obs={obs} profile={profile} accel={accel}"
                    r["time"] = elapsed
                    all_results.append(r)

                    print(f"  obs={obs} profile={profile} accel={accel}: "
                          f"correct_2={r['correct_2_pct']:.2f}% "
                          f"severe_5={r['severe_5']} "
                          f"max={r['max_diff']:.0f} "
                          f"median={r['median_diff']:.2f} "
                          f"false_trusted={r['false_trusted']} "
                          f"({elapsed:.0f}s)")

                    import os
                    if os.path.exists(output):
                        os.unlink(output)

        # Best by correct_2
        best = max(all_results, key=lambda r: r['correct_2'])
        print(f"\n  Best by correct_2: {best['param']} → {best['correct_2_pct']:.2f}% "
              f"(severe_5={best['severe_5']}, max={best['max_diff']:.0f}, median={best['median_diff']:.2f})")

        # Best by severe_5
        best_sev = min(all_results, key=lambda r: r['severe_5'])
        print(f"  Best by severe_5: {best_sev['param']} → severe_5={best_sev['severe_5']} "
              f"(correct_2={best_sev['correct_2_pct']:.2f}%, max={best_sev['max_diff']:.0f})")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python tools/eval_viterbi.py [--baseline|--sweep-obs|--sweep-profile|--sweep-accel|--sweep-conf|--sweep-trust|--sweep-all]")


if __name__ == "__main__":
    main()
