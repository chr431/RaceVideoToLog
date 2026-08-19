"""审计视频引擎顶层 import：确认无应用层（速度/GUI）依赖。"""
import ast
from pathlib import Path

src = Path(__file__).resolve().parent.parent.parent / "video_ocr_engine" / "extractor.py"
t = ast.parse(src.read_text(encoding="utf-8"))

imports = []
for n in ast.walk(t):
    if isinstance(n, ast.ImportFrom):
        imports.append((n.module or "", [a.name for a in n.names]))
    elif isinstance(n, ast.Import):
        imports.append(("", [a.name for a in n.names]))

print("引擎顶层 import:")
for m, names in imports:
    print(f"  {m if m else '(internal/3rd)'}: {names}")

app = {"segment_flow", "seg_correction", "ocr_text", "ocr_engine",
       "csv_io", "gui", "config", "signals", "monitor"}
hits = sorted(m for m, _ in imports if m in app)
print("应用层（速度/GUI）模块 import:", hits if hits else "无（引擎纯净）")
print("engine_config (引擎域):", any(m == "engine_config" for m, _ in imports))
print("ocr_engine引用:", any(m == "ocr_engine" for m, _ in imports))
