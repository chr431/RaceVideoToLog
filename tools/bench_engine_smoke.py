"""引擎冒烟：video_ocr_engine.FieldExtractor 识别链独立跑通（test 视频）。

用净化后的识别链 _run_pipelined（返回原始文本 texts/confs，无速度语义），
验证引擎不依赖 SegmentPipeline（速度后处理）与 extract_speed_value。
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
# tools/ 无 __init__.py：用 importlib 从文件加载 load_meta（避免命名空间包问题）
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "detect_eval", PROJECT / "tools" / "detect_eval.py")
_detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detect)
load_meta = _detect.load_meta  # noqa: E402
from video_ocr_engine import FieldExtractor  # noqa: E402

roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta("test")
ex = FieldExtractor(
    video_path="D:/Videos/racelog_test/test.mp4", roi=roi,
    frame_start=f_start, frame_end=f_start + 600,
    decode_backend="auto", ocr_backend="cpu",
    rep_crop_format="yuv",
    # 冒烟只验证识别链，不需要保留代表帧预览图
    keep_crops=False)
frames, segs, texts, confs, rep_frames = ex._run_pipelined()
print(f"解码+分段+OCR: {len(frames)} 帧 → {len(segs)} 段 → {len(texts)} 段文本")
# 引擎文本保全验证（texts 已随段保存）
print("ocr_texts 前 6:")
for i, t in enumerate(texts[:6]):
    print(f"  seg{i}: text={t!r} conf={confs[i]}")
# 验证引擎无速度语义：不 import extract_speed_value
import video_ocr_engine.extractor as ex_mod
assert not hasattr(ex_mod, "extract_speed_value"), \
    "引擎不应含速度解析符号！"
print("PASS: 引擎识别链独立跑通，纯文本输出（无速度语义）")
