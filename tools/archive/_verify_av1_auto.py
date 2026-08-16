"""验证 test6 AV1 CPU+ONNX 自动分配（不设 env，代码自探测）的耗时与准确率。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_matrix as M

os.environ.pop("RVTOL_OCR_THREADS", None)
os.environ.pop("DECORD_FFMPEG_THREAD_COUNT", None)
d = M.run_bench("test6", "cpu", "cpu", None, 0, tag="_av1auto")
t = d.get("timing") or {}
print("test6 CPU+ONNX 自动分配（16核，AV1 规则 dcd=12/ocrT=4）：")
print(f"  total={t.get('total_pipeline_s')}s decode={t.get('decode_s')}s "
      f"ocr={t.get('ocr_s')}s rss={t.get('peak_rss_mb')}MB")
print("  准确率:", d.get("accuracy"))
