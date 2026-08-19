"""从 segment_flow.py 删除引擎方法（已由 video_ocr_engine.FieldExtractor 提供），
让 SegmentPipeline 继承真正生效（去重）。只删识别链方法，保留速度后处理。"""
import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
path = PROJECT / "segment_flow.py"
src = path.read_text(encoding="utf-8")
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == "SegmentPipeline")

ENGINE = {
    "frames", "segment_frames", "ocr_values", "ocr_texts",
    "ocr_confidences", "corrected_values", "confidence_values",
    "n_segments", "n_corrected", "_prof_end", "_open_vr",
    "_hybrid_env_enabled", "_is_hybrid", "_hybrid_split",
    "_decord_format", "_decode_num_threads", "_open_decord_reader",
    "_remember_color_range", "_crop_luma", "_batch_luma",
    "_crop_is_expected", "_open_hybrid_vrs", "_ocr_engine_type",
    "_ocr_num_threads", "_decode_all", "_segment", "_ocr_segments",
    "_run_pipelined", "prepare_review_rgb", "timing_flat",
}

# 收集要删的 (start_line, end_line)，含装饰器行（线上 -1）
ranges = []
for n in cls.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ENGINE:
        start = n.lineno
        # 装饰器（@property/@xxx.setter）在 node.lineno 之前
        for dec in n.decorator_list:
            if dec.lineno < start:
                start = dec.lineno
        ranges.append((start, n.end_lineno))

ranges.sort()
# 合并重叠区间
merged = []
for r in ranges:
    if merged and r[0] <= merged[-1][1] + 2:
        merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
    else:
        merged.append(list(r))

lines = src.splitlines()
# 从后往前删，保持行号有效
for start, end in reversed(merged):
    del lines[start - 1:end]
    # 删除后残留的空行：删除区间可能留下多余连续空行，交给末尾统一清理

# 清理连续 3+ 空行为 2
out = []
blank = 0
for l in lines:
    if not l.strip():
        blank += 1
        if blank <= 2:
            out.append(l)
    else:
        blank = 0
        out.append(l)

new_src = "\n".join(out) + "\n"
path.write_text(new_src, encoding="utf-8")
print(f"删除 {len(merged)} 个区间（{sum(e-s+1 for s,e in merged)} 行）")
for s, e in merged:
    print(f"  {s}-{e}")
