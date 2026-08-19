"""引擎冒烟：video_ocr_engine.FieldExtractor 识别链独立跑通（test 视频）。

用串行识别链（_decode_all + _segment + _ocr_segments）验证引擎类自带
解码/分段/OCR 能力；不依赖 SegmentPipeline（速度后处理）与 extract()。
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
    yuv_output=True)
frames, crops, grays, sharp = ex._decode_all()
print(f"解码完成: {len(frames)} 帧, backend={ex._backend}")
segs = ex._segment(frames, grays)
print(f"分段: {len(segs)} 段")
seg_vals, rep_frames = ex._ocr_segments(segs, crops, sharp)
print(f"OCR: {len(seg_vals)} 段文本")
# 引擎文本保全验证（texts 已随段保存）
print("ocr_texts 前 6:")
for i, t in enumerate(ex.ocr_texts[:6]):
    print(f"  seg{i}: text={t!r}")
print("PASS: 引擎识别链独立跑通")
