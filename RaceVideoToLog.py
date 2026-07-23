"""RaceVideoToLog — 赛车视频速度 OCR 提取工具。

从车载视频中实时 OCR 识别速度数字，支持 GPU (TensorRT) / CPU 两种后端，
输出时间-速度-距离 CSV 文件。

用法:
  python RaceVideoToLog.py                          # GUI 模式
  python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2  # CLI 模式
"""
from __future__ import annotations

import argparse
import sys


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
    parser.add_argument("--buffer", type=int, default=config.DEFAULT_BUFFER_SIZE)
    parser.add_argument("--backend", choices=["auto","tensorrt","cpu"], default=config.DEFAULT_BACKEND)
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
    parser.add_argument("--from-csv", type=str, metavar="PATH",
        help="从已有 CSV 文件头导入设置（可被显式参数覆盖）")
    args = parser.parse_args()

    # ── 从 CSV 导入设置 ──
    if args.from_csv:
        from ocr_engine import parse_csv_header
        csv_settings = parse_csv_header(args.from_csv)
        # 仅填充用户未显式指定的参数
        # argparse defaults (action.default) vs user-specified
        _defaults = {a.dest: a.default
		             for a in parser._actions if a.dest != "help"}
        for key, val in csv_settings.items():
            # Map CSV keys to argparse dest names
            _dest = {
                "roi": "roi", "format": "format", "max_speed": "max_speed",
                "max_accel": "max_accel", "div": "div", "target_h": "target_h",
                "pad": "pad", "backend": "backend", "buffer": "buffer",
                "frame_start": "frame_start", "frame_end": "frame_end",
                "model": "ocr_model",
                "reocr_model": "reocr_model",
            }.get(key)
            if _dest is None:
                continue
            # Only apply if arg still has its default value
            cur = getattr(args, _dest)
            if cur == _defaults.get(_dest):
                if _dest == "roi":
                    try:
                        parts = [int(x.strip()) for x in val.split(",")]
                        if len(parts) == 4:
                            setattr(args, _dest, parts)
                    except ValueError:
                        pass
                elif _dest in ("frame_start", "frame_end", "div", "target_h",
				               "pad", "buffer"):
                    try:
                        setattr(args, _dest, int(val))
                    except ValueError:
                        pass
                elif _dest in ("max_speed", "max_accel"):
                    try:
                        setattr(args, _dest, float(val))
                    except ValueError:
                        pass
                else:
                    setattr(args, _dest, val)

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
