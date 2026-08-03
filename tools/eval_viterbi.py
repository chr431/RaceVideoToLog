"""Viterbi v2 evaluation tool — in-process pipeline + ground truth comparison.

Usage: python tools/eval_viterbi.py [--baseline] [--video NAME] [--manual]
"""
from __future__ import annotations
import sys, time, os, shutil
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from ocr_engine import parse_csv_header, Flag
from pipeline import ProcessingPipeline
from analysis import parse_csv
import config


def load_truth(path: str) -> dict[int, float]:
	truth: dict[int, float] = {}
	with open(path, "r", encoding="utf-8-sig") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"): continue
			parts = line.split(",")
			if len(parts) >= 3: truth[int(float(parts[0]))] = float(parts[2])
	return truth


def load_result(path: str) -> dict[int, tuple[float, int]]:
	result: dict[int, tuple[float, int]] = {}
	with open(path, "r", encoding="utf-8-sig") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"): continue
			parts = line.split(",")
			if len(parts) >= 4: result[int(float(parts[0]))] = (float(parts[2]), int(parts[3]))
	return result


def evaluate(truth: dict[int, float], result: dict[int, tuple[float, int]]) -> dict:
	common = sorted(set(truth) & set(result))
	n = len(common)
	if n == 0: return {"error": "no common frames"}
	diffs = [abs(truth[fi] - result[fi][0]) for fi in common]
	diffs_arr = np.array(diffs)
	correct_2 = int(np.sum(diffs_arr <= 2))
	severe_5 = int(np.sum(diffs_arr >= 5))
	total_errors = int(np.sum(diffs_arr > 0.5))
	top5 = sorted(diffs_arr, reverse=True)[:5]
	real_diffs = [abs(truth[fi] - result[fi][0]) for fi in common if truth[fi] >= 0]
	real_max = max(real_diffs) if real_diffs else 0.0
	false_trusted = sum(1 for fi in common if Flag.is_trusted(result[fi][1]) and abs(truth[fi] - result[fi][0]) > 0.5)
	n_trusted = sum(1 for fi in common if Flag.is_trusted(result[fi][1]))
	return {
		"n": n, "correct_2": correct_2, "correct_2_pct": 100 * correct_2 / n,
		"severe_5": severe_5, "severe_5_pct": 100 * severe_5 / n,
		"total_errors": total_errors, "error_rate": 100 * total_errors / n,
		"max_diff": float(np.max(diffs_arr)), "real_max": float(real_max),
		"top5_diffs": [float(d) for d in top5],
		"median_diff": float(np.median(diffs_arr)), "mean_diff": float(np.mean(diffs_arr)),
		"false_trusted": false_trusted, "n_trusted": n_trusted,
	}


def run_pipeline(video: str, truth_csv: str, max_speed: float, max_accel: float,
                 reocr_only: bool = True, mode: str = "auto") -> tuple[list, str]:
	settings = parse_csv_header(truth_csv)
	roi = tuple(int(x) for x in settings["roi"].split(","))
	div = int(settings.get("div", "1"))
	target_h = int(float(settings.get("target_h", "48")))
	pad = int(float(settings.get("pad", "0")))
	buffer_size = int(settings.get("buffer", "16"))
	backend = settings.get("backend", "auto")
	ocr_model = settings.get("model", "v6_tiny")
	speed_format = settings.get("format", "km/h")
	frame_start = settings.get("frame_start", "")
	frame_end = settings.get("frame_end", "")

	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	video_stem = Path(video).stem
	out_dir = Path("outputs") / f"{ts}_{video_stem}_sp{int(max_speed)}_acc{int(max_accel)}"
	out_dir.mkdir(parents=True, exist_ok=True)
	output = str(out_dir / "result.csv")

	def progress(msg: str, pct: float) -> None: pass

	pipeline = ProcessingPipeline(
		video_path=video, roi=roi, max_speed=max_speed, max_accel=max_accel,
		frame_div=div, target_h=target_h, pad=pad, buffer_size=buffer_size,
		backend=backend, ocr_model=ocr_model, reocr_model="v6_small",
		speed_format=speed_format, frame_start=frame_start, frame_end=frame_end,
		progress_cb=progress, log_level="normal",
	)
	pipeline.run_auto(output, reocr_only=reocr_only, mode=mode)
	actual_output = str(pipeline.last_output_path) if pipeline.last_output_path else output
	# Use public parse_csv API instead of private _rows attribute
	_, _, speeds, flags = parse_csv(actual_output)
	rows = [[float(i), 0.0, speeds[i], flags[i]] for i in range(len(speeds))]
	return rows, actual_output


def print_result(label: str, r: dict) -> None:
	if "error" in r:
		print(f"  {label}: ERROR — {r['error']}"); return
	print(f"  --- {label} ({r['n']} frames) ---")
	print(f"  Correct (d<=2): {r['correct_2']} ({r['correct_2_pct']:.2f}%)")
	print(f"  Severe  (d>=5): {r['severe_5']} ({r['severe_5_pct']:.2f}%)")
	print(f"  Total errors (>0.5): {r['total_errors']} ({r['error_rate']:.2f}%)")
	print(f"  Top-5 diffs: {[f'{d:.0f}' for d in r.get('top5_diffs', [r['max_diff']])]}  RealMax: {r.get('real_max', r['max_diff']):.0f}  Median: {r['median_diff']:.2f}")
	print(f"  False trusted: {r['false_trusted']}  HIGH_TRUST frames: {r['n_trusted']}")


def main() -> None:
	video_name = "test4.mp4"
	for arg in sys.argv[1:]:
		if arg.startswith("--video="): video_name = arg.split("=", 1)[1]

	truth_csv = str(PROJECT / f"ground_truth_csv/{Path(video_name).stem}_truth.csv")
	video = str(PROJECT / video_name)
	if not Path(video).exists():
		print(f"ERROR: {video} not found"); return

	if video_name == "test4.mp4": max_speed, max_accel = 400.0, 70.0
	elif video_name == "test.mp4": max_speed, max_accel = 260.0, 50.0
	else: max_speed, max_accel = 400.0, 50.0

	mode = "--baseline" if len(sys.argv) == 1 or "--baseline" in sys.argv else sys.argv[1]
	is_manual = "--manual" in sys.argv
	run_mode = "manual" if is_manual else "auto"

	if mode == "--baseline":
		t0 = time.perf_counter()
		rows, output = run_pipeline(video, truth_csv, max_speed, max_accel, mode=run_mode)
		elapsed = time.perf_counter() - t0
		truth = load_truth(truth_csv); result = load_result(output)
		r = evaluate(truth, result)
		mode_str = run_mode
		print(f"\n{'='*60}\nVideo: {Path(video).name}  Mode: {mode_str}  ({elapsed:.1f}s)")
		print_result("Baseline", r)
		if os.path.exists(output):
			eval_dir = os.path.dirname(output)
			if "outputs" in eval_dir: shutil.rmtree(eval_dir, ignore_errors=True)


if __name__ == "__main__":
	main()
