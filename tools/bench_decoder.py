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
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs"

import config  # noqa: E402  (DECODE_BACKEND_KEYS)


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


def run(video: str, truth: str, backend: str, out_csv: str,
        decode_backend: str = "auto") -> dict:
    """Run headless pipeline, parse timing + actual backend from output CSV/stdout.

    OCR 模型固定 v6_small（v2.14 起移除模型选择）。
    """
    Path(out_csv).unlink(missing_ok=True)  # stale CSV from a prior run must not be parsed
    t0 = time.perf_counter()
    env = dict(os.environ)
    # 子进程 RSS 采样（教训：速度测试必须同时监测内存 —— CPU 解码的
    # 视图引用泄漏曾让 3000 帧吃掉 18GB）
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None  # type: ignore[assignment]
    # stdout/stderr 重定向到文件而不是 PIPE：CLI 每帧 progress print +
    # flush，7223 帧输出远超 64KB 管道缓冲 —— 父进程采样循环不读管道时
    # 子进程会阻塞在写 stdout 而挂死（实测 bench 超时 10 分钟）。
    stdout_log = Path(out_csv).with_suffix(".stdout.txt")
    stderr_log = Path(out_csv).with_suffix(".stderr.txt")
    fo = open(stdout_log, "w", encoding="utf-8", errors="replace")
    fe = open(stderr_log, "w", encoding="utf-8", errors="replace")
    cli_args = [sys.executable, str(PROJECT / "RaceVideoToLog.py"),
                video, "--from-csv", truth,
                "--log-level", "detailed"]
    if backend != "auto":
        cli_args += ["--ocr-backend", backend]
    if decode_backend != "auto":
        cli_args += ["--decode-backend", decode_backend]
    cli_args += ["-o", out_csv]
    child = subprocess.Popen(
        cli_args,
        cwd=str(PROJECT), stdout=fo, stderr=fe, env=env,
    )
    peak_rss, cur_rss = 0.0, 0.0
    if psutil is not None:
        # Windows venv 的 python.exe 是 launcher（有子进程）：真正干活的是
        # 子进程，必须对 proc + 全部后代采样，否则 peak RSS 恒为 ~5MB。
        pchild = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                procs = [pchild] + pchild.children(recursive=True)
                cur_rss = 0.0
                for pr in procs:
                    try:
                        cur_rss = max(cur_rss, pr.memory_info().rss / 1e6)
                    except psutil.Error:
                        pass
                peak_rss = max(peak_rss, cur_rss)
            except psutil.Error:
                pass
            time.sleep(0.2)
    try:
        r = child.wait(timeout=900)
    except subprocess.TimeoutExpired:
        child.kill()
        fo.close(); fe.close()
        raise
    fo.close(); fe.close()
    r_stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
    r_stderr = stderr_log.read_text(encoding="utf-8", errors="replace")
    r = subprocess.CompletedProcess(child.args, r, r_stdout, r_stderr)
    timing: dict = {"wall_s": round(time.perf_counter() - t0, 2)}
    if psutil is not None:
        timing["peak_rss_mb"] = round(peak_rss)
        timing["end_rss_mb"] = round(cur_rss)
    # logger goes to stderr (Python logging default), prints go to stdout —
    # scan both. 段管线（v2.13）输出 "[decord/GPU] 解码+分段: 0/7223" 与
    # "总耗时: X.Xs"；backend 从 CSV 头 "# backend=decord/GPU" 读。
    text = (r.stdout or "") + "\n" + (r.stderr or "")
    m = re.search(r"\[decord/([\w+]+)\]", text)
    if m:
        timing["actual_backend"] = m.group(1)
    # 帧数 = CSV 数据行数（比解析 stdout 更稳）
    n_frames = sum(1 for _l in open(out_csv, encoding="utf-8-sig",
                                    errors="replace")
                   if _l.strip() and not _l.startswith("#"))
    if n_frames:
        timing["frames"] = n_frames
    # Total pipeline time: 段管线 stdout "总耗时: X.Xs"（CSV 头无 total=）。
    m = re.search(r"总耗时: ([\d.]+)s", text)
    if m:
        timing["total_pipeline_s"] = float(m.group(1))
    # Stage timing from the CSV header (written by _write_csv):
    # "# timing: ocr=..s, decode=..s, inference=..s, correction=..s, total=..s"
    with open(out_csv, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.startswith("# timing:"):
                for part in line.split(":", 1)[1].split(","):
                    k, _, v = part.strip().partition("=")
                    key = k.strip()
                    # 外部 RSS 采样已提供 peak_rss_mb（int，进程树口径），不覆盖
                    if key in timing:
                        continue
                    try:
                        # peak_*/gpu_* 前缀不带 _s 后缀（与 JSON record 键一致）；
                        # gpu_name 字符串 float() 失败自然跳过。
                        suffix = "" if (key.startswith("peak_") or key.startswith("gpu_")) else "_s"
                        timing[key + suffix] = float(v.rstrip("s"))
                    except ValueError:
                        pass
                break
    if r.returncode != 0:
        timing["error"] = (r.stderr or r.stdout)[-400:]
    elif "Error" in text or "Traceback" in text:
        timing["stderr"] = (r.stderr or text)[-300:]
    return timing


def accuracy(out_csv: str, truth: str) -> dict | None:
    """Compare pipeline output against truth (best-effort; module may be gone)."""
    try:
        sys.path.insert(0, str(PROJECT))
        from tools.verify_accuracy import compare, load_truth
        stats = compare(load_truth(truth), out_csv)
        return {
            "matched": stats["matched"],
            "wrong": stats["wrong"],
            "error_rate": round(stats["error_rate"], 3),
            "false_trusted": stats["false_trusted_count"],
        }
    except (ImportError, Exception):
        return None


def print_row(label: str, t: dict, acc: dict | None = None) -> None:
    if "frames" not in t:
        print(f"  {label:12s} ERROR: {t.get('error', t.get('stderr', 'no timing'))[:160]}")
        return
    extra = ""
    if acc:
        extra = (f" | acc: matched {acc['matched']}/{acc['matched'] + acc['wrong']} "
                 f"err {acc['error_rate']:.2f}% falseT {acc['false_trusted']}")
    mem = ""
    if t.get("peak_rss_mb"):
        mem = f" | peak RSS {t['peak_rss_mb']:5d}MB"
    print(f"  {label:12s} | {t.get('actual_backend', '?'):9s} | {t['frames']:6d} fr | "
          f"wall {t['wall_s']:5.1f}s | decode {t.get('decode_s', 0):5.1f}s | "
          f"infer {t.get('inference_s', 0):5.1f}s | corr {t.get('correction_s', 0):4.1f}s | "
          f"total {t.get('total_pipeline_s', t.get('total_s', 0)):5.1f}s{mem}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="test4", help="video name (resolved under D:/Videos/racelog_test)")
    ap.add_argument("--backend", default="tensorrt", choices=["tensorrt", "cpu", "auto"])
    ap.add_argument("--decode-backend", default="auto",
                    choices=config.DECODE_BACKEND_KEYS,
                    help="decord 解码后端 (auto/cpu/nvdec)")
    ap.add_argument("--runs", type=int, default=2, help="runs (last one used for stats)")
    ap.add_argument("--json", type=str, default="", help="save record to JSON (default outputs/bench_<video>.json)")
    args = ap.parse_args()

    video, truth = resolve(args.video)
    OUT_DIR.mkdir(exist_ok=True)
    json_path = args.json or str(OUT_DIR / f"bench_{args.video}.json")

    print(f"Video: {video}")
    print(f"Truth: {truth}")
    print(f"Backend: {args.backend}, decord: {args.decode_backend}, runs: {args.runs}")

    record: dict = {"video": args.video, "backend": args.backend,
                    "decode_backend": args.decode_backend,
                    "runs": []}
    for run_i in range(args.runs):
        out_csv = str(OUT_DIR / f"bench_{args.video}_r{run_i + 1}.csv")
        label = f"run {run_i + 1}"
        print(f"  Running {label}...", end=" ", flush=True)
        t = run(video, truth, args.backend, out_csv,
                decode_backend=args.decode_backend)
        acc = accuracy(out_csv, truth) if "frames" in t else None
        if run_i == args.runs - 1:  # warm run -> report + record
            print_row(label, t, acc)
            record["timing"] = {k: t.get(k) for k in
                                ("frames", "actual_backend", "wall_s",
                                 "ocr_s", "decode_s", "inference_s",
                                 "correction_s", "total_pipeline_s",
                                 "engine_load_s", "video_open_s", "phase1_s",
                                 "prewarm_s", "finalize_integrate_s",
                                 "finalize_csv_s", "finalize_report_s",
                                 "peak_rss_mb", "peak_cpu_pct",
                                 "peak_gpu_util_pct", "peak_vram_mb",
                                 "end_rss_mb")}
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
