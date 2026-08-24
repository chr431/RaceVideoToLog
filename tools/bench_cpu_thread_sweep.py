"""CPU 解码 + CPU 推理线程数扫描：总墙钟找最优 (OCR线程, 解码线程)。

假设：16C32T 下 OCR 8 线程 + FFmpeg 4 线程 = 12/16 物理核利用率不足；
OCR 12 线程（12 物理核）+ 解码 4 线程 = 16 物理核满配无 HT 争抢。
DECORD_FFMPEG_THREAD_COUNT 在 DLL 静态初始化读取 → 每配置必须子进程。

用法：python tools/bench_cpu_thread_sweep.py [--video test5]
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from tools.bench_decoder import resolve, run, print_row  # noqa: E402

# (label, env_overrides)：None 值 = 删除该 env（回到默认）
CONFIGS = [
    ("ocr8_dcd4(现状)",  {"OCR_THREADS": "8"}),          # 8 OCR + 4 FFmpeg
    ("ocr10_dcd4",      {"OCR_THREADS": "10"}),
    ("ocr12_dcd4",      {"OCR_THREADS": "12"}),
    ("ocr16_dcd4",      {"OCR_THREADS": "16"}),
    ("ocr20_dcd4",      {"OCR_THREADS": "20"}),
    ("ocr24_dcd4",      {"OCR_THREADS": "24"}),
    ("ocr32_dcd4",      {"OCR_THREADS": "32"}),
    ("ocr8_dcd0(auto)", {"OCR_THREADS": "8", "DECORD_FFMPEG_THREAD_COUNT": "0"}),
    ("ocr8_dcd8",       {"OCR_THREADS": "8", "DECORD_FFMPEG_THREAD_COUNT": "8"}),
    ("ocr12_dcd8",      {"OCR_THREADS": "12", "DECORD_FFMPEG_THREAD_COUNT": "8"}),
    ("ocr16_dcd8",      {"OCR_THREADS": "16", "DECORD_FFMPEG_THREAD_COUNT": "8"}),
    ("ocr12_dcd12",     {"OCR_THREADS": "12", "DECORD_FFMPEG_THREAD_COUNT": "12"}),
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    args = ap.parse_args()
    video, truth = resolve(args.video)
    print(f"Video: {video}  (CPU 解码 + CPU OCR, 每配置 1 次运行)\n")
    results = {}
    for label, env in CONFIGS:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        out_csv = str(PROJECT / "outputs" / f"sweep_{args.video}_{label.split('(')[0]}.csv")
        print(f"  [{label}] ", end="", flush=True)
        t = run(video, truth, "cpu", out_csv, decode_backend="cpu")
        print_row(label, t)
        results[label] = t
    print("\n汇总 (total_pipeline_s):")
    for label, t in sorted(results.items(),
                           key=lambda kv: kv[1].get("total_pipeline_s", 999)):
        print(f"  {label:18s} total={t.get('total_pipeline_s', 0):5.1f}s "
              f"decode={t.get('decode_s', 0):4.1f}s ocr={t.get('ocr_s', 0):4.1f}s "
              f"peakRSS={t.get('peak_rss_mb', 0):5d}MB")


if __name__ == "__main__":
    main()
