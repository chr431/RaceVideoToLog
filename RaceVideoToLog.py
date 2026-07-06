"""RaceVideoToLog — 赛车视频速度 OCR 提取工具。

从车载视频中实时 OCR 识别速度数字，支持 GPU (CUDA) / CPU 两种后端，
输出时间-速度-距离 CSV 文件。

用法:
  python RaceVideoToLog.py                          # GUI 模式
  python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2  # CLI 模式
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
	parser = argparse.ArgumentParser(description="RaceVideoToLog - 视频速度提取工具")
	parser.add_argument("video", nargs="?", help="视频文件路径")
	parser.add_argument("--roi", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="识别范围")
	parser.add_argument("--format", choices=["m/s","km/h","mile/h"], default="km/h")
	parser.add_argument("--div", type=int, default=2, choices=list(range(1, 11)))
	parser.add_argument("--max-speed", type=float, default=400)
	parser.add_argument("--max-accel", type=float, default=50)
	parser.add_argument("--target-h", type=int, default=24)
	parser.add_argument("--pad", type=int, default=0)
	parser.add_argument("--buffer", type=int, default=4)
	parser.add_argument("--backend", choices=["auto","cuda","cpu"], default="auto")
	parser.add_argument("--ocr-model", choices=["v6_small"], default="v6_small")
	parser.add_argument("-o", "--output", type=str)
	parser.add_argument("--analysis", nargs=2, metavar=("CSV1","CSV2"))
	parser.add_argument("--analysis-out", type=str)
	parser.add_argument("--frame-start", type=int, metavar="N")
	parser.add_argument("--frame-end", type=int, metavar="N")
	args = parser.parse_args()

	if args.video:
		from headless import run_headless
		run_headless(args)
	elif args.analysis:
		from analysis import run_analysis_headless
		run_analysis_headless(args)
	else:
		from PySide6.QtWidgets import QApplication
		from qfluentwidgets import setTheme, Theme
		from gui import RaceVideoToLogApp
		app = QApplication(sys.argv)
		setTheme(Theme.AUTO)
		window = RaceVideoToLogApp()
		window.show()
		sys.exit(app.exec())


if __name__ == "__main__":
	main()
