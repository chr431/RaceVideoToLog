"""引擎净化一：从 extractor.py 移除 _ocr_segments（速度语义，移回 SegmentPipeline）
与顶部 extract_speed_value import。"""
from pathlib import Path

path = Path(__file__).resolve().parent.parent.parent / "video_ocr_engine" / "extractor.py"
src = path.read_text(encoding="utf-8")
lines = src.splitlines()

# 1) 删顶部 import 行
new_lines = [l for l in lines
             if "from ocr_engine import extract_speed_value" not in l]

# 2) 删 _ocr_segments 方法（行号会因第1步变化，用 ast 重新定位）
import ast
src2 = "\n".join(new_lines)
t = ast.parse(src2)
cls = next(n for n in t.body if isinstance(n, ast.ClassDef)
           and n.name == "FieldExtractor")
target = [n for n in cls.body if isinstance(n, ast.FunctionDef)
          and n.name == "_ocr_segments"]
if target:
    n = target[0]
    lo, hi = n.lineno, n.end_lineno
    del new_lines[lo - 1:hi]
    print(f"删除 _ocr_segments（{lo}-{hi}）")
else:
    print("警告：未找到 _ocr_segments")

out = "\n".join(new_lines) + "\n"
path.write_text(out, encoding="utf-8")
# 语法检查
ast.parse(out)
print("OK")
