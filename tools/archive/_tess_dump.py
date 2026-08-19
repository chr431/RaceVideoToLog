"""调试：dump 真实 rep_crop → BMP，验证 tesseract 是否能识别。"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from tools.archive._tess_probe import crop_to_bmp_bytes  # noqa: E402

roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta("test")
p = SegmentPipeline(
    video_path="D:/Videos/racelog_test/test.mp4", roi=roi,
    max_speed_kmh=ms, max_accel_mps2=ma,
    buffer_size=config.DEFAULT_BUFFER_SIZE,
    decode_backend="auto", ocr_backend="cpu",
    fill_width=config.DEFAULT_FILL_WIDTH,
    speed_format="km/h", frame_start=f_start, frame_end=f_start + 300,
    force_aspect=mw, fps=None, yuv_output=True)
p.run("outputs/_tess_smoke.csv")
for i in (0, 5, 50, 200):
    seg = p.segments[i]
    crop = seg["rep_crop"]
    print(f"seg#{i} crop shape={crop.shape} dtype={crop.dtype} "
          f"pp_value={seg['ocr_value']}")
    b = crop_to_bmp_bytes(crop)
    Path(f"outputs/_tess_seg{i}.bmp").write_bytes(b)
    print(f"  BMP {len(b)} bytes -> outputs/_tess_seg{i}.bmp")