"""用 ast.unparse 生成 video_ocr_engine/_methods_body.py（引擎方法体）。

从 segment_flow.SegmentPipeline 抽取引擎方法（含 property/setter 装饰器），
输出为纯方法体模块（不含类骨架/import），供人工拼进 extractor.py。
"""
import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
src_path = PROJECT / "segment_flow.py"
src = src_path.read_text(encoding="utf-8")
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == "SegmentPipeline")

ENGINE = {
    "_prof_end", "_open_vr", "_hybrid_env_enabled", "_is_hybrid",
    "_hybrid_split", "_decord_format", "_decode_num_threads",
    "_open_decord_reader", "_remember_color_range", "_crop_luma",
    "_batch_luma", "_crop_is_expected", "_open_hybrid_vrs",
    "_ocr_engine_type", "_ocr_num_threads", "_decode_all", "_segment",
    "_ocr_segments", "_run_pipelined", "prepare_review_rgb", "timing_flat",
    "frames", "segment_frames", "ocr_values", "ocr_texts",
    "ocr_confidences", "corrected_values", "confidence_values",
    "n_segments", "n_corrected",
}

methods = [n for n in cls.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name in ENGINE]
methods.sort(key=lambda n: n.lineno)

body = "\n\n".join(ast.unparse(m) for m in methods)
out = PROJECT / "video_ocr_engine" / "_methods_body.py"
out.write_text(f'"""引擎识别链方法体（由 {Path(__file__).name} 生成，勿手改）。"""\n\n'
               + body, encoding="utf-8")
print(f"写 {out}：{len(methods)} 方法，{len(body.splitlines())} 行")
