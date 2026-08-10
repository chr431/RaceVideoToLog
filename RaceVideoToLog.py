"""RaceVideoToLog v2.11.0 — 赛车视频速度 OCR 提取工具。

从车载视频中实时 OCR 识别速度数字，支持 TensorRT / CPU 两种后端（自动选择），
输出时间-速度-距离 CSV 文件。

用法:
    python RaceVideoToLog.py                          # GUI 模式
    python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2  # CLI 模式
"""
from __future__ import annotations

import sys
import io

# ── 强制 UTF-8 输出，解决 Windows 终端中文乱码 ──
# PyInstaller 打包后 console=False 时 stdout/stderr 为 None
for _stream_name in ('stdout', 'stderr'):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and getattr(_stream, 'encoding', 'utf-8') != 'utf-8':
        try:
            setattr(sys, _stream_name,
                    io.TextIOWrapper(_stream.buffer, encoding='utf-8', errors='replace'))
        except (AttributeError, ValueError):
            pass

import argparse


def main() -> None:
    import config
    parser = argparse.ArgumentParser(description="RaceVideoToLog - 视频速度提取工具")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="识别范围")
    parser.add_argument("--format", choices=["m/s","km/h","mile/h"], default=config.DEFAULT_SPEED_FORMAT)
    parser.add_argument("--buffer", type=int, default=config.DEFAULT_BUFFER_SIZE,
        help="解码∥OCR 流水线队列缓冲（段数）")
    parser.add_argument("--max-speed", type=float, default=config.DEFAULT_MAX_SPEED)
    parser.add_argument("--max-accel", type=float, default=config.DEFAULT_MAX_ACCEL)
    parser.add_argument("--target-h", type=int, default=config.DEFAULT_TARGET_H)
    parser.add_argument("--pad", type=int, default=config.DEFAULT_PAD)
    parser.add_argument("--max-width", type=int, default=None,
        help="预处理最大宽度 px（0=不限）。扁宽字体设为 96 可改善识别")
    parser.add_argument("--ocr-model", choices=["v6_small"], default=config.DEFAULT_OCR_MODEL,
        help="OCR 模型（v2.13 起唯一 v6_small，无重 OCR）")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--frame-start", type=int, metavar="N")
    parser.add_argument("--frame-end", type=int, metavar="N")
    parser.add_argument("--log-level", choices=["normal","detailed","debug"],
        default=config.DEFAULT_LOG_LEVEL, help="日志级别 (默认 normal)")
    parser.add_argument("--no-monitor", action="store_true",
        help="禁用资源监控（内存/CPU/GPU 采样；默认启用，RVTOL_MONITOR=0 等效）")
    parser.add_argument("--monitor-interval", type=float, default=None,
        metavar="SEC", help="资源采样间隔秒（默认 1.0，RVTOL_MONITOR_INTERVAL 等效）")
    parser.add_argument("--from-csv", type=str, metavar="PATH",
        help="从已有 CSV 文件头导入设置（可被显式参数覆盖）")
    args = parser.parse_args()

    # ── 从 CSV 导入设置 ──
    if args.from_csv:
        from ocr_engine import parse_csv_header, parse_csv_setting, csv_field_dest
        # 命令行显式写出的参数（即使等于默认值）优先于 CSV。
        # 例：--ocr-model v6_tiny 等于默认值，旧逻辑误判"未指定"被 CSV 的
        # model=v6_small 覆盖，导致引擎静默换成 small。
        _explicit = {
            a[2:].split("=", 1)[0].replace("-", "_")
            for a in sys.argv[1:] if a.startswith("--")
        }
        csv_settings = parse_csv_header(args.from_csv)
        _defaults = {a.dest: a.default
                        for a in parser._actions if a.dest != "help"}
        for key, val in csv_settings.items():
            dest = csv_field_dest(key)
            if dest is None or not hasattr(args, dest):
                continue  # read-only fields (fps/codec) or unknown — skip
            if dest in _explicit:
                continue  # 命令行显式指定 — 不被 CSV 覆盖
            cur = getattr(args, dest)
            if cur != _defaults.get(dest):
                continue  # user explicitly overrode — skip
            parsed = parse_csv_setting(key, val)
            if parsed is not None:
                setattr(args, dest, parsed)

    # None = 未指定：归一化为配置默认值（显式传 0 必须保留，不能被 CSV 覆盖）
    if args.max_width is None:
        args.max_width = config.DEFAULT_MAX_WIDTH

    if args.video:
        from headless import run_headless
        run_headless(args)
    else:
        from PySide6.QtWidgets import QApplication
        import io, sys as _sys
        _saved = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            from qfluentwidgets import setTheme, Theme
        finally:
            _sys.stdout = _saved
        from gui import RaceVideoToLogApp
        app = QApplication(sys.argv)
        setTheme(Theme.AUTO)
        window = RaceVideoToLogApp()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
