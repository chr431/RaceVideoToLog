"""建独立仓库评估：引擎依赖的 8 个模块各自的 import（找应用层泄漏）。

目标：确认 video_ocr_engine 独立仓库需要哪些文件、哪些模块需净化。
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CANDIDATES = ["engine_config", "segmentation", "hybrid_decode", "ocr_native",
              "ocr_trt", "ocr_engine", "video_utils", "constants"]
# 应用层（不随引擎走）
APP_MODULES = {"config", "segment_flow", "seg_correction", "ocr_text",
               "csv_io", "gui", "signals", "monitor", "headless",
               "export_controller", "analysis"}

report = {}
for name in CANDIDATES:
    p = ROOT / f"{name}.py"
    if not p.exists():
        report[name] = {"error": "missing"}
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            imports.append(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            imports.append("")
    leaks = sorted(m for m in imports if m in APP_MODULES)
    report[name] = {
        "top_imports": sorted(m for m in imports if m),
        "app_leaks": leaks,
        "size_kb": round(p.stat().st_size / 1024, 1),
    }

for name, r in report.items():
    print(f"\n=== {name} ({r.get('size_kb')}KB) ===")
    print(f"  顶层 import: {r.get('top_imports')}")
    leaks = r.get("app_leaks")
    print(f"  应用泄漏: {leaks if leaks else '无'}")
