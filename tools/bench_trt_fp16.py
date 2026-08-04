"""Benchmark TRT engine precision: FP32 (current default) vs FP16.

Builds an FP16 engine to a separate cache path (never touches the FP32
cache) and measures per-batch inference time for both engines on the
same (6,3,48,320) inputs.

Usage: python tools/bench_trt_fp16.py [--variant v6_tiny] [--batches 300]
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

# 项目根的 tensorrt.py shim（tensorrt → tensorrt_bindings）只在项目根
# 在 sys.path 时可见；从 tools/ 直接执行脚本时需显式加入
PROJECT = Path(__file__).parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MODELS = PROJECT / "assets" / "ocr_models"


def build_engine(variant: str, out_path: Path, fp16: bool) -> None:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
    builder = trt.Builder(logger)  # type: ignore[attr-defined]
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[attr-defined]
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)  # type: ignore[attr-defined]
    size = variant.replace("v6_", "")
    onnx_path = MODELS / f"PP-OCRv6_rec_{size}.onnx"
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"ONNX 解析失败: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # type: ignore[attr-defined]
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name,
                      min=(1, 3, 48, 32), opt=(6, 3, 48, 320),
                      max=(6, 3, 48, 2048))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine 构建失败")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialized)


def bench(engine_path: Path, batches: int) -> tuple[float, float]:
    """Load engine, run `batches` of (6,3,48,320), return (total_s, fps)."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
    logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:  # type: ignore[attr-defined]
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    prof_in = engine.get_tensor_profile_shape(in_name, 0)
    prof_out = engine.get_tensor_profile_shape(out_name, 0)
    max_batch = int(prof_in[2][0])
    shape = (min(6, max_batch), 3, 48, 320)
    ctx.set_input_shape(in_name, shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    in_nbytes = int(np.prod(shape)) * 4
    out_nbytes = int(np.prod(out_shape)) * 4
    _, dev_in = cudart.cudaMalloc(in_nbytes)
    _, dev_out = cudart.cudaMalloc(out_nbytes)
    host_in = np.zeros(shape, dtype=np.float32)
    host_out = np.empty(out_shape, dtype=np.float32)
    cudart.cudaMemcpy(dev_in, host_in.ctypes.data, in_nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    # warmup
    for _ in range(5):
        ctx.execute_v2([dev_in, dev_out])
    t0 = time.perf_counter()
    for _ in range(batches):
        ctx.execute_v2([dev_in, dev_out])
    elapsed = time.perf_counter() - t0
    cudart.cudaMemcpy(host_out.ctypes.data, dev_out, out_nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    per_batch = elapsed / batches
    return elapsed, batches * 6 / elapsed  # frames per second


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="v6_tiny")
    ap.add_argument("--batches", type=int, default=300)
    args = ap.parse_args()

    size = args.variant.replace("v6_", "")
    cache = Path(__import__("os").environ.get("LOCALAPPDATA",
                                              str(Path.home()))) / "RaceVideoToLog" / "ocr_engines"
    fp32_path = cache / f"multi_PP-OCRv6_rec_{size}_sm89_fp32_tf32unset.engine"
    fp16_path = cache / f"multi_PP-OCRv6_rec_{size}_sm89_fp16.engine"

    print(f"variant: {args.variant}, batches: {args.batches}")
    if fp16_path.exists():
        print(f"FP16 engine exists: {fp16_path}")
    else:
        print("Building FP16 engine (2-3 min)...", flush=True)
        build_engine(args.variant, fp16_path, fp16=True)
        print("FP16 engine built")
    if not fp32_path.exists():
        print(f"ERROR: FP32 engine not found: {fp32_path}")
        return
    t32, fps32 = bench(fp32_path, args.batches)
    t16, fps16 = bench(fp16_path, args.batches)
    print(f"FP32: {t32:.2f}s total, {fps32:.0f} fps, {t32 / args.batches * 1000:.2f} ms/batch")
    print(f"FP16: {t16:.2f}s total, {fps16:.0f} fps, {t16 / args.batches * 1000:.2f} ms/batch")
    print(f"speedup: {fps16 / fps32:.2f}x")


if __name__ == "__main__":
    main()
