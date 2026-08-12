"""CLI / headless mode for RaceVideoToLog."""
from __future__ import annotations
import argparse
import logging
import os as _os
import sys
import time
from pathlib import Path

import config
import monitor as _monitor
from segment_flow import SegmentPipeline


def _monitor_settings(args: argparse.Namespace) -> tuple[bool, float]:
    """解析资源监测开关与间隔。

    优先级：--no-monitor > RVTOL_MONITOR env > config.MONITOR_ENABLED；
    间隔：--monitor-interval > RVTOL_MONITOR_INTERVAL env > config.MONITOR_INTERVAL_S。
    """
    enabled = config.MONITOR_ENABLED
    _env = _os.environ.get("RVTOL_MONITOR", "").strip().lower()
    if _env in ("0", "off", "false", "no"):
        enabled = False
    elif _env in ("1", "on", "true", "yes"):
        enabled = True
    if getattr(args, "no_monitor", False):
        enabled = False
    interval = config.MONITOR_INTERVAL_S
    if getattr(args, "monitor_interval", None):
        interval = float(args.monitor_interval)
    elif _os.environ.get("RVTOL_MONITOR_INTERVAL"):
        try:
            interval = float(_os.environ["RVTOL_MONITOR_INTERVAL"])
        except ValueError:
            pass
    return enabled, interval


def run_headless(args: argparse.Namespace) -> None:
    """命令行无头模式：不启动 GUI，直接分析并输出 CSV。"""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
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
    print(f"最大速度: {args.max_speed} km/h, 最大加速度: {args.max_accel} m/s^2")
    print(f"分段流水线: diff分段 → 段值OCR → 段级纠错")

    t_total_start = time.perf_counter()

    if getattr(args, 'progress', False):
        def _progress(msg: str, pct: float) -> None:
            print(f"  [{pct:5.1f}%] {msg}")
    else:
        def _progress(msg: str, pct: float) -> None:
            print(f"\r  {msg}", end="", flush=True)
            if pct >= 100.0:
                print()

    pipeline = SegmentPipeline(
        video_path=str(video_path),
        roi=region,
        max_speed_kmh=args.max_speed,
        max_accel_mps2=args.max_accel,
        buffer_size=args.buffer,
        decode_backend=args.decode_backend,
        ocr_backend=args.ocr_backend,
        fill_width=args.fill_width,
        speed_format=args.format,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        progress_cb=_progress,
        force_aspect=getattr(args, 'force_aspect', 0.0),
        fps=None,
    )

    t0 = time.perf_counter()
    _mon_enabled, _mon_interval = _monitor_settings(args)
    if _mon_enabled:
        _monitor.start(interval_s=_mon_interval, with_gpu=config.MONITOR_GPU)
    try:
        pipeline.run(output_path)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
    finally:
        _stats = _monitor.stop()

    t_total = time.perf_counter() - t0
    print(f"总耗时: {t_total:.1f}s")
    # 输出详细的阶段计时（标量键）
    for stage, elapsed in pipeline.timing_flat().items():
        print(f"  {stage}: {elapsed:.1f}s")
    if _stats:
        _monitor.log_run(video_path.name, _stats, pipeline.timing_flat())
        print("资源: " + _monitor.format_stats(_stats))
    print(f"导出: {output_path}")
