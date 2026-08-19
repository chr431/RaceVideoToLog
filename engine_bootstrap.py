"""引擎子模块路径引导：把 third_party/video_ocr_engine 加入 sys.path。

RaceVideoToLog 自拆仓（v2.16）起通过 git submodule 调用 video_ocr_engine：
引擎仓库根目录（engine_config.py / segmentation.py / ocr_native.py /
video_ocr_engine/ ...）即 Python 源码根目录，必须加入 sys.path 才能
import 引擎顶层模块。

必须在任何 import 引擎模块（含 config 的 `from engine_config import *`
聚合）之前调用。覆盖入口：
  - RaceVideoToLog.py 模块顶部（CLI/GUI）
  - pytest 根 conftest.py（本地/CI）
  - setup_venv.bat 写入 site-packages 的 video_ocr_engine.pth（任意 venv 进程，
    包括 tools/ 脚本）
"""
from __future__ import annotations

import sys
from pathlib import Path

_ENGINE_DIRNAME = "video_ocr_engine"


def engine_path() -> Path:
    """引擎子模块源码根目录（third_party/video_ocr_engine）。"""
    return Path(__file__).resolve().parent / "third_party" / _ENGINE_DIRNAME


def ensure_engine_path() -> Path:
    """把引擎源码根目录加入 sys.path（幂等），返回该路径。

    frozen（PyInstaller EXE）：引擎模块已由 spec 通过 pathex/hiddenimports
    打包进 _internal，无需 sys.path 引导。源码模式：子模块缺失（未 clone /
    未 init）时抛 RuntimeError，给出修复指引。
    """
    p = engine_path()
    if getattr(sys, "frozen", False):
        return p
    if not (p / "video_ocr_engine").is_dir():
        raise RuntimeError(
            f"引擎子模块缺失: {p}\n"
            "请执行 `git submodule update --init --recursive` 获取 "
            "chr431/video_ocr_engine")
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p


if __name__ == "__main__":
    p = ensure_engine_path()
    print("engine path:", p)
