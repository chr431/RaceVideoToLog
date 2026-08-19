"""验证：OCR 文本保全（_ocr_texts/_ocr_confs/segments[].text）。

通用引擎目标：识别层必须保留每段原始文本与置信度（应用层才做数值/语义
解析）。本工具跑 test 视频 600 帧窗口，断言：
  - len(ocr_texts)==len(ocr_values)==len(segments)
  - segments[].text 与 ocr_texts 逐段一致
输出每段 value/text/conf 对应。用法：
  python tools/bench_text_preserve.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta("test")
p = SegmentPipeline(
    video_path="D:/Videos/racelog_test/test.mp4", roi=roi,
    max_speed_kmh=ms, max_accel_mps2=ma,
    buffer_size=config.DEFAULT_BUFFER_SIZE,
    decode_backend="auto", ocr_backend="cpu",
    fill_width=config.DEFAULT_FILL_WIDTH,
    speed_format="km/h", frame_start=f_start, frame_end=f_start + 600,
    force_aspect=mw, fps=None, yuv_output=True)
p.run(str(PROJECT / "outputs" / "_text_preserve.csv"))
print("segments:", len(p.segments),
      "ocr_values:", len(p.ocr_values),
      "ocr_texts:", len(p.ocr_texts),
      "ocr_confs:", len(p.ocr_confidences))
for i in range(min(12, len(p.segments))):
    seg = p.segments[i]
    print(f"  seg{i}: value={seg['value']} ocr_value={seg['ocr_value']} "
          f"text={seg['text']!r} conf={seg['ocr_conf']}")
# 一致性：有 text 但 value 为 None 的段数（数字文本必能解析；非数字文本
# 会解析失败——对速度应用属正常（引擎保留文本、解析失败回 None））
diff = sum(1 for v, t in zip(p.ocr_values, p.ocr_texts)
           if v is None and t is not None)
print("value=None 且 text 非空 的段数:", diff)
# segments[].text 与 ocr_texts 对齐
mismatch = sum(1 for i, seg in enumerate(p.segments)
               if seg["text"] != p.ocr_texts[i])
print("segments[].text 与 ocr_texts 不一致段数:", mismatch)
# 长度对齐断言（引擎完整性：段/数值/文本/置信度必须等长）
assert len(p.segments) == len(p.ocr_values) == len(p.ocr_texts) \
    == len(p.ocr_confidences), "段/数值/文本/置信度长度不一致！"
assert mismatch == 0, "segments[].text 与 ocr_texts 不一致！"
print("PASS: 文本保全完整，长度与内容一致。")