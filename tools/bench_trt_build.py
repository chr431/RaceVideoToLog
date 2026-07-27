"""Compare TRT engine build time: FP16 vs FP32.

Deletes cached .engine files before each test, then times the
first RapidOCR instantiation (which triggers engine building).
"""
from __future__ import annotations
import time, glob, os, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from ocr_engine import _init_rapidocr, _get_model_params
from gpu_setup import select_backend, get_engine_params, reset_backend

_init_rapidocr()

MODEL_DIR = Path(PROJECT, ".venv", "Lib", "site-packages", "rapidocr", "models", "models")
MODEL = "v6_small"  # the re-OCR model, likely larger


def clear_engines():
    for f in glob.glob(str(MODEL_DIR / "*.engine")):
        os.remove(f)
    print(f"  Cleared engine cache ({len(glob.glob(str(MODEL_DIR / '*.engine')))} remaining)")


def build(label: str, use_fp16: bool):
    clear_engines()
    reset_backend()
    select_backend("tensorrt")

    et = "tensorrt"
    engine_params = get_engine_params()
    engine_params["EngineConfig.tensorrt.use_fp16"] = use_fp16  # override after copy
    model_params = _get_model_params(MODEL, et)
    all_params = {**(model_params or {}), **engine_params}

    from rapidocr import RapidOCR
    t0 = time.perf_counter()
    ocr = RapidOCR(params=all_params)
    elapsed = time.perf_counter() - t0

    print(f"  {label}: {elapsed:.0f}s (fp16={use_fp16})")

    del ocr


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Engine dir: {MODEL_DIR}")

    build("FP32", use_fp16=False)
    build("FP16", use_fp16=True)
