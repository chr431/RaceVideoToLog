"""CPU 抢核问题对照实验：线程预算 × 后端 × 自旋控制的墙钟/峰值 CPU 占用。

背景：16C32T 下 OCR 8 + FFmpeg 8 = 16 物理核"满配"；再加线程（超线程/
超订）反而变慢。本实验摸清：(a) GPU 解码卸载后 OCR 能否吃到更多核；
(b) ORT 自旋等待（RVTOL_ORT_SPIN=0）是否消除抢核；(c) 峰值 CPU 占用。

用法：python tools/archive/_cpu_landscape.py [--video test5]
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from tools.bench_decoder import resolve, run, print_row  # noqa: E402


def _env(base: dict, overrides: dict, deletes: tuple = ()) -> dict:
    env = dict(base)
    # 调优基线（CLAUDE.md 锁定）：FFmpeg 8 帧线程 + filter 1 线程 + prefetch 8
    env.setdefault("DECORD_FFMPEG_THREAD_COUNT", "8")
    env.setdefault("DECORD_FILTER_THREADS", "1")
    env.setdefault("DECORD_PREFETCH_DEPTH", "8")
    for k in overrides:
        env[k] = str(overrides[k])
    for k in deletes:
        env.pop(k, None)
    # 影响测量的环境一律显式化
    env["RVTOL_MONITOR_INTERVAL"] = "0.5"
    # 父进程残留的实验 env 一律清除（每次测量独立）
    for k in ("RVTOL_OCR_THREADS", "RVTOL_ORT_SPIN", "RVTOL_ORT_SPIN_MS"):
        if k not in overrides:
            env.pop(k, None)
    return env


def _peak_cpu(out_csv: str) -> str:
    log = Path(out_csv).with_suffix(".stdout.txt")
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"资源: (.*)", text)
    return m.group(1)[:120] if m else ""


CONFIGS = [
    # (label, overrides, deletes, decode_backend)  调优基线 env 见 _env()
    ("cpu_ocr8_dcd8_基线", {"RVTOL_OCR_THREADS": "8"}, (), "cpu"),
    ("cpu_ocr12_dcd4", {"RVTOL_OCR_THREADS": "12",
                        "DECORD_FFMPEG_THREAD_COUNT": "4"}, (), "cpu"),
    ("cpu_ocr16_dcd4", {"RVTOL_OCR_THREADS": "16",
                        "DECORD_FFMPEG_THREAD_COUNT": "4"}, (), "cpu"),
    ("cpu_ocr16_dcd8", {"RVTOL_OCR_THREADS": "16"}, (), "cpu"),
    ("gpu_ocr8", {"RVTOL_OCR_THREADS": "8"}, (), "nvdec"),
    ("gpu_ocr12", {"RVTOL_OCR_THREADS": "12"}, (), "nvdec"),
    ("gpu_ocr16", {"RVTOL_OCR_THREADS": "16"}, (), "nvdec"),
    ("gpu_ocr16_pf16", {"RVTOL_OCR_THREADS": "16",
                        "DECORD_PREFETCH_DEPTH": "16"}, (), "nvdec"),
    ("gpu_ocr20", {"RVTOL_OCR_THREADS": "20"}, (), "nvdec"),
    ("gpu_ocr24", {"RVTOL_OCR_THREADS": "24"}, (), "nvdec"),
    ("gpu_ocr32", {"RVTOL_OCR_THREADS": "32"}, (), "nvdec"),
    ("gpu_ocr14", {"RVTOL_OCR_THREADS": "14"}, (), "nvdec"),
    # 基线复测（方差参照）
    ("cpu_ocr8_dcd8_基线2", {"RVTOL_OCR_THREADS": "8"}, (), "cpu"),
    ("gpu_ocr16_2", {"RVTOL_OCR_THREADS": "16"}, (), "nvdec"),
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    args = ap.parse_args()
    video, truth = resolve(args.video)
    print(f"Video: {video}\n")
    base_env = dict(os.environ)
    rows = []
    for label, ov, deletes, dcd in CONFIGS:
        out_csv = str(PROJECT / "outputs" / f"land_{args.video}_{label}.csv")
        old = os.environ.copy()
        os.environ.clear()
        os.environ.update(_env(base_env, ov, deletes))
        try:
            print(f"  [{label}] ", end="", flush=True)
            t = run(video, truth, "cpu", out_csv, decode_backend=dcd)
            t["peak_cpu_line"] = _peak_cpu(out_csv)
            print_row(label, t)
            rows.append((label, t))
        finally:
            os.environ.clear()
            os.environ.update(old)
    print("\n汇总:")
    for label, t in rows:
        cpu = t.get("peak_cpu_line", "")
        print(f"  {label:22s} total={t.get('total_pipeline_s', 0):5.1f}s "
              f"decode={t.get('decode_s', 0):4.1f}s ocr={t.get('ocr_s', 0):4.1f}s "
              f"| {cpu}")


if __name__ == "__main__":
    main()
