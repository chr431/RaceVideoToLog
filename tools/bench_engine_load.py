"""OCR 引擎初始化/加载微基准（TRT engine 与 ONNX session）。

每个引擎类型：首次创建（冷，含 CUDA/ORT 上下文初始化）后，再连续创建
4 次（热，仅反序列化/session 创建）。生产 CLI 每次运行是冷路径；GUI
若复用引擎则后续导出走热路径。

用法：python tools/bench_engine_load.py [--runs 5]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


def measure(kind: str, runs: int) -> list[float]:
    from ocr_native import OcrEngine
    out = []
    for i in range(runs):
        t0 = time.perf_counter()
        eng = OcrEngine("v6_small", kind, num_threads=4)
        out.append(time.perf_counter() - t0)
        del eng
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--kinds", default="tensorrt,onnxruntime")
    args = ap.parse_args()
    record = {}
    for kind in args.kinds.split(","):
        kind = kind.strip()
        if not kind:
            continue
        vals = measure(kind, args.runs)
        print(f"{kind}: cold={vals[0]:.3f}s warm_min={min(vals[1:]):.3f}s "
              f"all=" + " ".join(f"{v:.3f}" for v in vals), flush=True)
        record[kind] = {"cold_s": round(vals[0], 3),
                        "warm_min_s": round(min(vals[1:]), 3),
                        "runs": [round(v, 3) for v in vals]}
    out = PROJECT / "outputs" / "engine_load.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
