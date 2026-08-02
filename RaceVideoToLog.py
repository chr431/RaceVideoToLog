"""RaceVideoToLog v2.6.0 — 赛车视频速度 OCR 提取工具。

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
    parser.add_argument("--div", type=int, default=config.DEFAULT_FRAME_DIV, choices=list(range(1, 11)))
    parser.add_argument("--max-speed", type=float, default=config.DEFAULT_MAX_SPEED)
    parser.add_argument("--max-accel", type=float, default=config.DEFAULT_MAX_ACCEL)
    parser.add_argument("--target-h", type=int, default=config.DEFAULT_TARGET_H)
    parser.add_argument("--pad", type=int, default=config.DEFAULT_PAD)
    parser.add_argument("--max-width", type=int, default=config.DEFAULT_MAX_WIDTH,
        help="预处理最大宽度 px（0=不限）。扁宽字体设为 96 可改善识别")
    parser.add_argument("--buffer", type=int, default=config.DEFAULT_BUFFER_SIZE)
    parser.add_argument("--backend", choices=["auto","tensorrt","cpu"], default=config.DEFAULT_BACKEND)
    parser.add_argument("--video-backend", choices=["cv2","decord"], default="decord",
        help=argparse.SUPPRESS)  # deprecated; only decord is supported
    parser.add_argument("--ocr-model", choices=["v6_tiny", "v6_small"], default=config.DEFAULT_OCR_MODEL,
        help="主 OCR 模型 (默认 tiny)")
    parser.add_argument("--reocr-model", choices=["v6_tiny", "v6_small"], default=config.DEFAULT_REOCR_MODEL,
        help="重 OCR 模型 (默认 small，推荐 tiny+small 组合)")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--analysis", nargs=2, metavar=("CSV1","CSV2"))
    parser.add_argument("--analysis-out", type=str)
    parser.add_argument("--frame-start", type=int, metavar="N")
    parser.add_argument("--frame-end", type=int, metavar="N")
    parser.add_argument("--log-level", choices=["normal","detailed","debug"],
        default=config.DEFAULT_LOG_LEVEL, help="日志级别 (默认 normal)")
    parser.add_argument("--mode", choices=["auto","manual"], default="auto",
        help="纠错模式 (默认 auto)")
    parser.add_argument("--from-csv", type=str, metavar="PATH",
        help="从已有 CSV 文件头导入设置（可被显式参数覆盖）")
    args = parser.parse_args()

    # ── 从 CSV 导入设置 ──
    if args.from_csv:
        from ocr_engine import parse_csv_header, parse_csv_setting, csv_field_dest
        csv_settings = parse_csv_header(args.from_csv)
        _defaults = {a.dest: a.default
                        for a in parser._actions if a.dest != "help"}
        for key, val in csv_settings.items():
            dest = csv_field_dest(key)
            if dest is None or not hasattr(args, dest):
                continue  # read-only fields (fps/codec) or unknown — skip
            cur = getattr(args, dest)
            if cur != _defaults.get(dest):
                continue  # user explicitly overrode — skip
            parsed = parse_csv_setting(key, val)
            if parsed is not None:
                setattr(args, dest, parsed)

    if args.video:
        from headless import run_headless
        run_headless(args)
    elif args.analysis:
        from analysis import run_analysis_headless
        run_analysis_headless(args)
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
