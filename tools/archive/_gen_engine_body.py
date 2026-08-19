"""生成 video_ocr_engine/extractor.py：从 segment_flow.py 精确剪切识别链方法。

引擎方法（FieldExtractor）：280-739 + 841-1242 行区间内属于识别链的方法
（解码/灰度/OCR 编排），不含 _local_bandwidth/_detect/_correct/_confidence/
_fill_values/_dense_correct/_spike_second_pass/_dp_run/finalize/_build_rows/
_write_csv/run（速度后处理，留 SegmentPipeline）。

生成后请人工 review：import 改为引擎域（engine_config 替换 config 的速度后
处理参数引用在方法体内不存在——引擎方法只用引擎常量），字段初始化移到引擎
__init__。
"""
import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
src_path = PROJECT / "segment_flow.py"
src = src_path.read_text(encoding="utf-8")
lines = src.splitlines()

tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == "SegmentPipeline")

# 识别链方法（含排序后的定义顺序）
ENGINE = [
    "_prof_end", "_open_vr", "_hybrid_env_enabled", "_is_hybrid",
    "_hybrid_split", "_decord_format", "_decode_num_threads",
    "_open_decord_reader", "_remember_color_range", "_crop_luma",
    "_batch_luma", "_crop_is_expected", "_open_hybrid_vrs",
    "_ocr_engine_type", "_ocr_num_threads", "_decode_all", "_segment",
    "_ocr_segments", "_run_pipelined", "prepare_review_rgb", "timing_flat",
    "frames", "segment_frames", "ocr_values", "ocr_texts",
    "ocr_confidences", "corrected_values", "confidence_values",
    "n_segments", "n_corrected",
]

methods = []
for node in cls.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
            and node.name in ENGINE:
        methods.append((node.name, node.lineno, node.end_lineno))

# 按源码顺序收集方法文本
methods.sort(key=lambda m: m[1])
out = []
for name, start, end in methods:
    block = "\n".join(lines[start - 1:end])
    out.append(block)

out_text = "\n\n".join(out)
out_path = PROJECT / "video_ocr_engine" / "_engine_methods_body.txt"
out_path.write_text(out_text, encoding="utf-8")
print(f"抽取 {len(methods)} 个引擎方法，{sum(e-s+1 for _,s,e in methods)} 行 → {out_path.name}")
for name, s, e in methods:
    print(f"  {name}: {s}-{e}")
