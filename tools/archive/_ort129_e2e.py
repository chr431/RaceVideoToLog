"""onnxruntime 1.29.0 vs 1.28.0 端到端 A/B（3000 帧，快速迭代）。

只测 OCR 相关组合（dec 决定是否含解码干扰）：
  - test5 dec=auto ocr=cpu（GPU 硬解 + CPU ONNX OCR = OCR 独立瓶颈）
  - test5 dec=cpu ocr=cpu（CPU 软解 + CPU OCR）
  - test6 dec=auto ocr=cpu（AV1 GPU 硬解 + CPU OCR）
runs=3 取中位数（我取 run 内 avg 由 bench_decoder --runs 控制）。

用法：python tools/archive/_ort129_e2e.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from tools.archive._bench_matrix import run_bench, row  # noqa: E402

COMBOS = [
    ("test5", "cpu", "auto"),
    ("test5", "cpu", "cpu"),
    ("test6", "cpu", "auto"),
]

if __name__ == "__main__":
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__} · 3000 帧 · runs=3\n")
    for v, ocr, dec in COMBOS:
        d = run_bench(v, ocr, dec, runs=3, frames=3000)
        row(f"{v} dec={dec} ocr={ocr}", d)