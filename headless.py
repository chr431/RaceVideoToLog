"""CLI / headless mode for RaceVideoToLog."""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

from pipeline import ProcessingPipeline


def run_headless(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    """命令行无头模式：不启动 GUI，直接分析并输出 CSV。"""
    if not args.roi:
        print("错误: 命令行模式需要 --roi X1 Y1 X2 Y2")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 找不到文件 {video_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else video_path.with_suffix(".csv")
    region = (args.roi[0], args.roi[1], args.roi[2], args.roi[3])

    print(f"视频: {video_path}")
    print(f"识别范围: {region}")
    print(f"采样间隔: 1/{args.div}")
    print(f"最大速度: {args.max_speed} km/h, 最大加速度: {args.max_accel} m/s^2")
    print(f"OCR 后端选择: {args.backend}")

    t_total_start = time.perf_counter()

    if getattr(args, 'progress', False):
        def _progress(msg: str, pct: float) -> None:
            print(f"  [{pct:5.1f}%] {msg}")
    else:
        def _progress(msg: str, pct: float) -> None:
            print(f"\r  {msg}", end="", flush=True)
            if pct >= 100.0:
                print()

    pipeline = ProcessingPipeline(
        video_path=video_path,
        roi=region,
        max_speed=args.max_speed,
        max_accel=args.max_accel,
        frame_div=args.div,
        target_h=args.target_h,
        pad=args.pad,
        buffer_size=args.buffer,
        backend=args.backend,
        ocr_model=args.ocr_model,
        reocr_model=getattr(args, 'reocr_model', None),
        speed_format=args.format,
        frame_start=str(args.frame_start or ""),
        frame_end=str(args.frame_end or ""),
        progress_cb=_progress,
        log_level=getattr(args, 'log_level', 'normal'),
    )

    t0 = time.perf_counter()
    try:
        pipeline.run_auto(output_path)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

    t_total = time.perf_counter() - t0
    print(f"总耗时: {t_total:.1f}s")
    # 输出详细的阶段计时
    if pipeline._timing:
        for stage, elapsed in pipeline._timing.items():
            print(f"  {stage}: {elapsed:.1f}s")
    print(f"导出: {output_path}")
