"""列出引擎 FieldExtractor 方法、__init__ 参数、extract 状态。"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
src = (root / "video_ocr_engine" / "extractor.py").read_text(encoding="utf-8")
t = ast.parse(src)
cls = next(n for n in t.body if isinstance(n, ast.ClassDef)
           and n.name == "FieldExtractor")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]

print("=== FieldExtractor 方法 ===")
for m in methods:
    decs = " ".join(ast.unparse(d) for d in m.decorator_list)
    print(f"  {decs or '-':28} def {m.name}")
    doc = ast.get_docstring(m)
    if doc:
        first = doc.strip().splitlines()[0][:70]
        print(f"      {first}")

init = next(m for m in methods if m.name == "__init__")
pos = [a.arg for a in init.args.args if a.arg != "self"]
kw = [a.arg for a in init.args.kwonlyargs]
print("\n=== __init__ 参数 ===")
print("  位置参数:", ", ".join(pos))
print("  仅关键字:", ", ".join(kw))

print("\n=== extract() 是否已实现 ===")
ext = next((m for m in methods if m.name == "extract"), None)
print("  存在:", ext is not None)
if ext:
    body = ast.unparse(ext.body[0]) if ext.body else "(空)"
    print("  首个语句:", body)

# SegmentPipeline 怎么调它
print("\n=== SegmentPipeline.__init__ 调 super() 的参数 ===")
sf_src = (root / "segment_flow.py").read_text(encoding="utf-8")
if "super().__init__(" in sf_src:
    i = sf_src.index("super().__init__(")
    j = sf_src.index(")", i)
    print(sf_src[i:j + 1])
else:
    print("  未发现 super().__init__ 调用")
