"""pytest 根配置：把引擎子模块源码根加入 sys.path（见 engine_bootstrap）。

测试大量直接 import 引擎顶层模块（engine_config / segmentation /
video_utils / ocr_native / ...）。在收集任何测试前把 submodule 路径放上
sys.path，保证无论从何处运行 pytest 都能解析到引擎。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent / "third_party" / "video_ocr_engine"
if not (_ENGINE / "video_ocr_engine").is_dir():
    raise RuntimeError(
        f"引擎子模块缺失: {_ENGINE}\n"
        "请执行 `git submodule update --init --recursive`")
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
