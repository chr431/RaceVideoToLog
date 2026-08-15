"""CPU 解码线程组合矩阵：FFmpeg 帧线程 × filter 线程 × OCR 线程。

发现：dcd2(默认)+filter0(默认)+ocr16 = 9.7s vs dcd8+filter1+ocr8 = 13.3s。
本矩阵隔离三变量，确定 CPU 解码的最优线程预算规则。

用法：python tools/archive/_cpu_matrix.py
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from tools.bench_decoder import resolve, run, print_row  # noqa: E402

# (label, FFMPEG, FILTER, OCR) —— None = 不设（fork 默认：FFMPEG 2 / FILTER auto）
CONFIGS = [
    ("dcd2_ft0_ocr16", None, None, "16"),
    ("dcd2_ft0_ocr12", None, None, "12"),
    ("dcd2_ft0_ocr8",  None, None, "8"),
    ("dcd4_ft0_ocr16", "4",  None, "16"),
    ("dcd8_ft0_ocr16", "8",  None, "16"),
    ("dcd2_ft1_ocr16", None, "1",  "16"),
    ("dcd2_ft4_ocr16", None, "4",  "16"),
    ("dcd2_ft0_ocr16_2", None, None, "16"),
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    args = ap.parse_args()
    video, truth = resolve(args.video)
    base = dict(os.environ)
    base.pop("DECORD_FFMPEG_THREAD_COUNT", None)
    base.pop("DECORD_FILTER_THREADS", None)
    base.pop("RVTOL_OCR_THREADS", None)
    print(f"Video: {video}\n")
    for label, dcd, ft, ocr in CONFIGS:
        env = dict(base)
        if dcd:
            env["DECORD_FFMPEG_THREAD_COUNT"] = dcd
        if ft:
            env["DECORD_FILTER_THREADS"] = ft
        env["RVTOL_OCR_THREADS"] = ocr
        env["RVTOL_MONITOR_INTERVAL"] = "0.5"
        out_csv = str(PROJECT / "outputs" / f"matrix_{label}.csv")
        old = os.environ.copy()
        os.environ.clear()
        os.environ.update(env)
        try:
            print(f"  [{label}] ", end="", flush=True)
            t = run(video, truth, "cpu", out_csv, decode_backend="cpu")
            log = Path(out_csv).with_suffix(".stdout.txt")
            cpu_line = ""
            try:
                m = re.search(r"资源: (.*)", log.read_text(
                    encoding="utf-8", errors="replace"))
                cpu_line = m.group(1)[:80] if m else ""
            except OSError:
                pass
            print_row(label, t)
            if cpu_line:
                print(f"         {cpu_line}")
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    main()
