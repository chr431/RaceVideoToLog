"""Pipeline benchmark + accuracy check (headless).

Runs the full pipeline on a test video and reports stage timing
(decode/inference/correction/total) plus accuracy vs ground truth.
This is the single measurement harness used for all perf work:
it resolves the video from D:\\Videos\\racelog_test, reuses the truth
CSV header (roi/fps/div/model...) via --from-csv, runs twice (second
run is warm-cache), and saves a JSON record for comparison.

Usage:
    python tools/bench_decoder.py [--video test4] [--backend tensorrt]
                                  [--runs 2] [--json outputs/bench.json]
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs"


def resolve(video_name: str) -> tuple[str, str]:
    """Return (video_path, truth_csv_path) for a video name."""
    video = VIDEO_DIR / f"{video_name}.mp4"
    if not video.exists():
        raise SystemExit(f"ERROR: video not found: {video}")
    truth = PROJECT / f"ground_truth_csv/{video_name}_truth.csv"
    if not truth.exists():
        truth = PROJECT / f"ground_truth_csv/{video_name}_ref.csv"
    if not truth.exists():
        raise SystemExit(f"ERROR: truth CSV not found for {video_name}")
    return str(video), str(truth)


def run(video: str, truth: str, backend: str, out_csv: str) -> dict:
    """Run headless pipeline, parse timing + actual backend from output CSV/stdout."""
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(PROJECT / "RaceVideoToLog.py"),
         video, "--from-csv", truth,
         "--backend", backend,
         "--log-level", "detailed",
         "-o", out_csv],
        cwd=str(PROJECT), capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace",
    )
    timing: dict = {"wall_s": round(time.perf_counter() - t0, 2)}
    # logger goes to stderr (Python logging default), prints go to stdout —
    # scan both. "OCR 完成: N 帧 (decord/gpu), ..." gives frames + backend.
    text = (r.stdout or "") + "\n" + (r.stderr or "")
    for line in text.splitlines():
        m = re.search(r"(\d+) 帧 \((\S+)\),", line)
        if m:
            timing["frames"] = int(m.group(1))
            timing["actual_backend"] = m.group(2)
            break
    # Total pipeline time: "流水线完成: 总计 X.Xs (ocr=.., ...)" (CSV header
    # has no total= field).
    m = re.search(r"流水线完成: 总计 ([\d.]+)s", text)
    if m:
        timing["total_pipeline_s"] = float(m.group(1))
    # Stage timing from the CSV header (written by _write_csv):
    # "# timing: ocr=..s, decode=..s, inference=..s, correction=..s, total=..s"
    with open(out_csv, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.startswith("# timing:"):
                for part in line.split(":", 1)[1].split(","):
                    k, _, v = part.strip().partition("=")
                    try:
                        timing[k.strip() + "_s"] = float(v.rstrip("s"))
                    except ValueError:
                        pass
                break
    if r.returncode != 0:
        timing["error"] = (r.stderr or r.stdout)[-400:]
    elif "Error" in text or "Traceback" in text:
        timing["stderr"] = (r.stderr or text)[-300:]
    return timing


def accuracy(out_csv: str, truth: str) -> dict:
    """Compare pipeline output against truth (verify_accuracy module)."""
    sys.path.insert(0, str(PROJECT))
    from tools.verify_accuracy import compare, load_truth
    stats = compare(load_truth(truth), out_csv)
    return {
        "matched": stats["matched"],
        "wrong": stats["wrong"],
        "error_rate": round(stats["error_rate"], 3),
        "false_trusted": stats["false_trusted_count"],
    }


def print_row(label: str, t: dict, acc: dict | None = None) -> None:
    if "frames" not in t:
        print(f"  {label:12s} ERROR: {t.get('error', t.get('stderr', 'no timing'))[:160]}")
        return
    extra = ""
    if acc:
        extra = (f" | acc: matched {acc['matched']}/{acc['matched'] + acc['wrong']} "
                 f"err {acc['error_rate']:.2f}% falseT {acc['false_trusted']}")
    print(f"  {label:12s} | {t.get('actual_backend', '?'):9s} | {t['frames']:6d} fr | "
          f"wall {t['wall_s']:5.1f}s | decode {t.get('decode_s', 0):5.1f}s | "
          f"infer {t.get('inference_s', 0):5.1f}s | corr {t.get('correction_s', 0):4.1f}s | "
          f"total {t.get('total_pipeline_s', t.get('total_s', 0)):5.1f}s{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="test4", help="video name (resolved under D:/Videos/racelog_test)")
    ap.add_argument("--backend", default="tensorrt", choices=["tensorrt", "onnx", "auto"])
    ap.add_argument("--runs", type=int, default=2, help="runs (last one used for stats)")
    ap.add_argument("--json", type=str, default="", help="save record to JSON (default outputs/bench_<video>.json)")
    args = ap.parse_args()

    video, truth = resolve(args.video)
    OUT_DIR.mkdir(exist_ok=True)
    json_path = args.json or str(OUT_DIR / f"bench_{args.video}.json")

    print(f"Video: {video}")
    print(f"Truth: {truth}")
    print(f"Backend: {args.backend}, runs: {args.runs}")

    record: dict = {"video": args.video, "backend": args.backend, "runs": []}
    for run_i in range(args.runs):
        out_csv = str(OUT_DIR / f"bench_{args.video}_r{run_i + 1}.csv")
        label = f"run {run_i + 1}"
        print(f"  Running {label}...", end=" ", flush=True)
        t = run(video, truth, args.backend, out_csv)
        acc = accuracy(out_csv, truth) if "frames" in t else None
        if run_i == args.runs - 1:  # warm run -> report + record
            print_row(label, t, acc)
            record["timing"] = {k: t.get(k) for k in
                                ("frames", "actual_backend", "wall_s",
                                 "ocr_s", "decode_s", "inference_s",
                                 "correction_s", "total_pipeline_s")}
            record["accuracy"] = acc
        else:
            print_row(label, t)
        if "error" in t:
            print("  " + t["error"][-300:])
        record["runs"].append(t)
        # keep last output CSV for inspection
        if run_i < args.runs - 1:
            Path(out_csv).unlink(missing_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"Record saved: {json_path}")


if __name__ == "__main__":
    main()
