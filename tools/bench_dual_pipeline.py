"""Race 引擎级双流水线 benchmark：不同编码 × 不同切片数，看 CPU/GPU 路径是否闲置。

用法：
    python tools/bench_dual_pipeline.py --videos test test2 test3 test5 test6 \
        --frames 3000 --chunks 2 4 8

输出每个配置的墙钟、decode/ocr 时间，以及双流水线每条路径完成的片数和
墙钟（parallel_pipe1/pipe2）。所有测量单跑、串行，避免 CPU/GPU 互抢。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs" / "dual_pipeline_bench"

sys.path.insert(0, str(PROJECT))

# tools/ 无 __init__.py：用 importlib 从文件加载 load_meta
_spec = importlib.util.spec_from_file_location(
    "detect_eval", PROJECT / "tools" / "detect_eval.py")
_detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detect)
load_meta = _detect.load_meta

from segment_flow import SegmentPipeline  # noqa: E402


def run_one(video: str, dual: bool, chunks: int,
            backends: list | None, frames: int,
            decode_backend: str = "auto", ocr_backend: str = "auto",
            label: str = "") -> dict:
    roi, f_start, f_end, fps, ms, ma, mw, _truth = load_meta(video)
    end = f_start + frames if frames else f_end
    if not label:
        label = f"dual{chunks}" if dual else "single"
    pipe = SegmentPipeline(
        str(VIDEO_DIR / f"{video}.mp4"), roi, ms, ma, fps, f_start, end,
        force_aspect=mw,
        decode_backend=decode_backend,
        ocr_backend=ocr_backend,
        yuv_output=True,
        dual_pipeline=dual,
        dual_pipeline_chunks=chunks,
        dual_backends=backends,
    )
    out = OUT_DIR / f"{video}_{label}.csv"
    t0 = time.perf_counter()
    rows = pipe.run(str(out))
    wall = time.perf_counter() - t0
    timing = pipe.timing_flat()
    raw_timing = getattr(pipe, "timing", {})
    return {
        "video": video,
        "label": label,
        "dual": dual,
        "chunks": chunks,
        "backends": backends,
        "wall_s": round(wall, 3),
        "decode_s": round(float(timing.get("decode", 0.0)), 3),
        "ocr_s": round(float(timing.get("ocr", 0.0)), 3),
        "segments": pipe.n_segments,
        "pipe1_chunks": raw_timing.get("parallel_pipe1_chunks", 0),
        "pipe1_s": round(float(raw_timing.get("parallel_pipe1_s", 0.0)), 3),
        "pipe1_backend": raw_timing.get("parallel_pipe1_backend", ""),
        "pipe1_ocr": raw_timing.get("parallel_pipe1_ocr", ""),
        "pipe2_chunks": raw_timing.get("parallel_pipe2_chunks", 0),
        "pipe2_s": round(float(raw_timing.get("parallel_pipe2_s", 0.0)), 3),
        "pipe2_backend": raw_timing.get("parallel_pipe2_backend", ""),
        "pipe2_ocr": raw_timing.get("parallel_pipe2_ocr", ""),
        "backend": pipe._backend,
        "ocr_backend": pipe._ocr_backend_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*",
                    default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--frames", type=int, default=3000,
                    help="截取帧数（相对 truth frame_start；0=全量）")
    ap.add_argument("--chunks", nargs="*", type=int, default=[2, 4, 8])
    ap.add_argument("--decode-backend", default="auto")
    ap.add_argument("--ocr-backend", default="auto")
    ap.add_argument("--custom-cpu-trt", action="store_true",
                    help="额外跑两条 cpu+auto（CPU 解码+TRT OCR）双流水线")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'video':<6} {'label':<16} {'seg':>5} {'wall':>6} "
          f"{'decode':>6} {'ocr':>6} | {'pipe1(c/s/backend)':>22} "
          f"{'pipe2(c/s/backend)':>22}")

    configs = [("single", False, 0, None)]
    for c in args.chunks:
        configs.append((f"dual{c}", True, c, None))
    if args.custom_cpu_trt:
        for c in args.chunks:
            configs.append(
                (f"cpuTRT{c}", True, c, [("cpu", "auto"), ("cpu", "auto")]))

    for video in args.videos:
        for label, dual, chunks, backends in configs:
            r = run_one(video, dual, chunks, backends, args.frames,
                        args.decode_backend, args.ocr_backend, label)
            p1 = (f"{r['pipe1_chunks']}/{r['pipe1_s']:.2f}/"
                  f"{r['pipe1_backend'].replace('decord/', '')} "
                  f"{r['pipe1_ocr']}")
            p2 = (f"{r['pipe2_chunks']}/{r['pipe2_s']:.2f}/"
                  f"{r['pipe2_backend'].replace('decord/', '')} "
                  f"{r['pipe2_ocr']}")
            print(f"{video:<6} {label:<16} {r['segments']:>5} {r['wall_s']:>6.2f} "
                  f"{r['decode_s']:>6.2f} {r['ocr_s']:>6.2f} | "
                  f"{p1:<22} {p2:<22}")


if __name__ == "__main__":
    main()
