"""AST 分析 segment_flow.py：提取识别链方法依赖的 self 字段与小驼峰符号。"""
import ast
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
src = Path(PROJECT) / "segment_flow.py"
tree = ast.parse(src.read_text(encoding="utf-8"))

# 识别链方法（引擎侧）：解码/灰度/OCR 编排；不含 conf/DP/spike/build/rows
ENGINE_METHODS = {
    "_prof_end", "_open_vr", "_hybrid_env_enabled", "_is_hybrid",
    "_hybrid_split", "_decord_format", "_decode_num_threads",
    "_open_decord_reader", "_remember_color_range", "_crop_luma",
    "_batch_luma", "_crop_is_expected", "_open_hybrid_vrs",
    "_ocr_engine_type", "_ocr_num_threads", "_decode_all", "_segment",
    "_ocr_segments", "_run_pipelined", "_store_run_state",
    "prepare_review_rgb", "timing_flat", "run",
}
# 后处理方法（速度侧，留在 SegmentPipeline）
POST_METHODS = {
    "_local_bandwidth", "_detect", "_correct", "_confidence",
    "_fill_values", "_dense_correct", "_spike_second_pass", "_dp_run",
    "finalize", "_build_rows", "_write_csv",
}


def method_body_nodes(cls):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def collect(node, out):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == "self":
        out.add(node.attr)
    for child in ast.iter_child_nodes(node):
        collect(child, out)


cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == "SegmentPipeline")

# 引擎方法集合引用
engine_refs = {}
for node in method_body_nodes(cls):
    if node.name not in ENGINE_METHODS:
        continue
    refs = set()
    collect(node, refs)
    engine_refs[node.name] = refs

all_engine_refs = set().union(*engine_refs.values())
# 过滤掉引擎方法名本身和已知 public（不搬的）
ignored = ENGINE_METHODS | {
    "profile", "segments", "crops", "rows", "timing",
    "frames", "segment_frames", "ocr_values", "ocr_texts",
    "ocr_confidences", "corrected_values", "confidence_values",
    "n_segments", "n_corrected", "_ocr_vals", "_ocr_texts",
    "_ocr_confs", "_corr_vals", "_conf_vals", "_segs", "_frames",
    "_pinned", "run",
}
engine_state = sorted(all_engine_refs - ignored)
print("=== 识别链（引擎方法）依赖的 self 状态字段 ===")
for s in engine_state:
    print("  ", s)

print("\n=== 各引擎方法依赖个数 ===")
for name, refs in sorted(engine_refs.items()):
    own = sorted(refs - ignored)
    print(f"  {name}: {len(own)} 状态字段")