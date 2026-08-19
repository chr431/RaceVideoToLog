"""重建 extractor：追加引擎方法体并整体缩进 4 空格（类内成员）。"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent / "video_ocr_engine"
ext = root / "extractor.py"
body = (root / "_methods_body.py").read_text(encoding="utf-8")

lines = body.splitlines()
start = 1 if lines and lines[0].lstrip().startswith('"""') else 0
# 顶层函数缩进 4 空格 → 类内成员需 8 空格（整体前加 4 空格）
indented = "\n".join("    " + l if l.strip() else l
                     for l in lines[start:]).strip()

t = ext.read_text(encoding="utf-8")
# 去 marker
for marker in ["# ── 识别链方法体", "# 以下成员在各次回归门禁通过前"]:
    t = "\n".join(l for l in t.splitlines() if not l.startswith(marker))

# 去上一次 append 的方法体（若存在）：移除从第一个 8 空格 @property 之后到
# 文件尾的重复体。简单方式：保留到 extract() 的 raise 为止。
anchor = '        raise NotImplementedError\n'
assert anchor in t, "extract anchor missing"
# 若已有方法体追加（timing_flat 等 4 空格 def 残留在尾部），移除它们：
cut = t.find("\ndef timing_flat")
if cut != -1:
    t = t[:cut].rstrip() + "\n"

t = t.replace(anchor, anchor + "\n" + indented + "\n", 1)
ext.write_text(t, encoding="utf-8")
print("appended", len(indented.splitlines()), "lines (indented 4)")
