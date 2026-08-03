"""Accuracy verification: run pipeline on test video and compare against ground truth.

Usage: python tools/verify_accuracy.py [--headless]
"""
from __future__ import annotations
from typing import cast
import sys
from pathlib import Path

# ── Load truth CSV ──
def load_truth(path: str) -> dict[int, float]:
    """Parse truth CSV, return {frame_index: speed_kmh}. Only loads data rows, skips # headers."""
    truth: dict[int, float] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                fi = int(float(parts[0]))
                speed = float(parts[2])
                truth[fi] = speed
    return truth


def compare(truth: dict[int, float], result_path: str) -> dict:
    """Compare pipeline output CSV against truth.

    Returns stats dict with:
        - total, matched, wrong, missing, extra
        - errors: list of (frame, pipeline_speed, truth_speed)
        - false_trusted: frames with flag>=21 but speed wrong
    """
    from ocr_engine import Flag
    result: dict[int, tuple[float, int]] = {}
    with open(result_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                fi = int(float(parts[0]))
                speed = float(parts[2])
                flag = int(parts[3])
                result[fi] = (speed, flag)

    all_frames = sorted(set(truth.keys()) | set(result.keys()))
    matched = 0
    wrong = 0
    missing = 0
    extra = 0
    errors: list[tuple[int, float, float]] = []
    false_trusted: list[tuple[int, float, float, int]] = []

    for fi in all_frames:
        t = truth.get(fi)
        r = result.get(fi)
        if t is None:
            extra += 1
        elif r is None:
            missing += 1
        else:
            r_speed, r_flag = r
            if abs(r_speed - t) < 0.5:
                matched += 1
            else:
                wrong += 1
                errors.append((fi, r_speed, t))
                if Flag.is_trusted(r_flag):
                    false_trusted.append((fi, r_speed, t, r_flag))

    return {
        "total": len(all_frames),
        "matched": matched,
        "wrong": wrong,
        "missing": missing,
        "extra": extra,
        "error_rate": wrong / max(matched + wrong, 1) * 100,
        "errors": errors,
        "false_trusted": false_trusted,
        "false_trusted_count": len(false_trusted),
    }


def run_pipeline(video_path: str, truth_csv: str, output_path: str) -> None:
    """Run the full pipeline with parameters from truth CSV header."""
    from ocr_engine import parse_csv_header
    from pipeline import ProcessingPipeline

    settings = parse_csv_header(truth_csv)
    roi_str = settings.get("roi", "")
    if not roi_str:
        raise ValueError("Truth CSV missing roi parameter")
    roi = cast("tuple[int, int, int, int]", tuple(int(x) for x in roi_str.split(",")))
    max_speed = float(settings.get("max_speed", "400"))
    max_accel = float(settings.get("max_accel", "70"))
    div = int(settings.get("div", "1"))
    target_h = int(float(settings.get("target_h", "48")))
    pad = int(float(settings.get("pad", "0")))
    buffer_size = int(settings.get("buffer", "16"))
    backend = settings.get("backend", "auto")
    ocr_model = settings.get("model", "v6_tiny")
    reocr_model = settings.get("reocr_model", ocr_model)
    speed_format = settings.get("format", "km/h")
    frame_start = settings.get("frame_start", "")
    frame_end = settings.get("frame_end", "")

    print(f"Video: {video_path}")
    print(f"ROI: {roi}, div={div}, target_h={target_h}, pad={pad}")
    print(f"Model: {ocr_model}, reocr: {reocr_model}, backend: {backend}")
    print(f"max_speed={max_speed}, max_accel={max_accel}")
    print(f"frame_start={frame_start}, frame_end={frame_end}")

    def progress(msg: str, pct: float) -> None:
        print(f"  [{pct:.0f}%] {msg}")

    pipeline = ProcessingPipeline(
        video_path=video_path,
        roi=roi,
        max_speed=max_speed,
        max_accel=max_accel,
        frame_div=div,
        target_h=target_h,
        pad=pad,
        buffer_size=buffer_size,
        backend=backend,
        ocr_model=ocr_model,
        reocr_model=reocr_model,
        speed_format=speed_format,
        frame_start=frame_start,
        frame_end=frame_end,
        progress_cb=progress,
        log_level="normal",
    )
    pipeline.run_auto(output_path)
    print(f"Output: {output_path}")


def main() -> None:

    truth_csv = "ground_truth_csv/test4_truth.csv"
    video = "test4.mp4"  # adjust path as needed

    if not Path(video).exists():
        # Try common locations
        for p in [Path("d:/Video/test4.mp4"), Path.home() / "Videos/test4.mp4"]:
            if p.exists():
                video = str(p)
                break
        else:
            print(f"ERROR: Cannot find {video}. Please specify path.")
            print(f"Usage: python tools/verify_accuracy.py [video_path]")
            return

    if len(sys.argv) > 1 and not sys.argv[-1].startswith("--"):
        video = sys.argv[-1]

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output = f.name

    try:
        run_pipeline(video, truth_csv, output)
        truth = load_truth(truth_csv)
        stats = compare(truth, output)

        print(f"\n{'='*60}")
        print(f"Accuracy Report")
        print(f"{'='*60}")
        print(f"Total frames in truth:  {len(truth)}")
        print(f"Matched (diff < 0.5):  {stats['matched']}")
        print(f"Wrong (diff >= 0.5):   {stats['wrong']}")
        print(f"Missing:               {stats['missing']}")
        print(f"Extra:                 {stats['extra']}")
        print(f"Error rate:            {stats['error_rate']:.2f}%")
        print(f"False trusted (flag>=21 but wrong): {stats['false_trusted_count']}")

        if stats["false_trusted"]:
            print(f"\n--- False Trusted Frames (top 20) ---")
            print(f"{'Frame':>8} {'Pipeline':>10} {'Truth':>10} {'Flag':>6}")
            for fi, r_speed, t, flag in stats["false_trusted"][:20]:
                print(f"{fi:>8} {r_speed:>10.0f} {t:>10.0f} {flag:>6}")

        if stats["errors"]:
            print(f"\n--- All Errors (top 30) ---")
            print(f"{'Frame':>8} {'Pipeline':>10} {'Truth':>10} {'Diff':>8}")
            for fi, r_speed, t in stats["errors"][:30]:
                print(f"{fi:>8} {r_speed:>10.0f} {t:>10.0f} {abs(r_speed-t):>8.0f}")

    finally:
        if os.path.exists(output):
            os.unlink(output)


if __name__ == "__main__":
    main()
