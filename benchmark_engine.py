"""
Benchmark: 多引擎 vs 单引擎 并行方案对比。
用法: python benchmark_engine.py

测试参数:
  视频: test.mp4, div=2, ROI=[880,935,960,985]
  方案A (多引擎): N 个独立 RapidOCR 实例，各加载模型
  方案B (单引擎): 1 个 RapidOCR 实例，多线程共享

测量: GPU 显存峰值, 总时间, FPS
"""
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── 注册 GPU DLL ──
def _register_gpu_dlls():
    try:
        import ctypes as _ct
        _cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        _cudnn_base = r"C:\Program Files\NVIDIA\CUDNN"
        for _ver in ["v12.9","v12.8","v12.6","v12.4"]:
            _cb = os.path.join(_cuda_base, _ver, "bin")
            if os.path.isdir(_cb): os.add_dll_directory(_cb)
            if os.path.isdir(_cudnn_base):
                for _dv in os.listdir(_cudnn_base):
                    for _cv in [_ver, _ver.lstrip("v")]:
                        _db = os.path.join(_cudnn_base, _dv, "bin", _cv, "x64")
                        if os.path.isdir(_db):
                            os.add_dll_directory(_db)
                            for _f in os.listdir(_db):
                                if _f.endswith(".dll"):
                                    try: _ct.CDLL(os.path.join(_db, _f))
                                    except: pass
            break
    except: pass

_register_gpu_dlls()

import pynvml
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils import OrtInferSession

# ────────────── onnxruntime CUDA monkey-patch ──────────────
def _patch_ort_for_cuda():
    import onnxruntime as ort
    ep = ("CUDAExecutionProvider", {"device_id":0,"arena_extend_strategy":"kNextPowerOfTwo",
          "cudnn_conv_algo_search":"EXHAUSTIVE","do_copy_in_default_stream":True})
    cpu_ep = ("CPUExecutionProvider", {"arena_extend_strategy":"kSameAsRequested"})
    def _patched_init(self, config):
        from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        EP_list = [ep] if ep[0] != cpu_ep[0] else []
        EP_list.append(cpu_ep)
        self._verify_model(config['model_path'])
        self.session = InferenceSession(config['model_path'], sess_options=sess_opt, providers=EP_list)
    OrtInferSession.__init__ = _patched_init

_patch_ort_for_cuda()

# ────────────── 预处理 ──────────────
def preprocess_crop(crop, target_h=48, pad_px=20):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape[:2]
    scale = target_h / float(h) if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)
    pad_int = int(pad_px)
    if pad_int > 0:
        gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ────────────── GPU 内存测量 ──────────────
pynvml.nvmlInit()
GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_memory_mb():
    """返回当前进程 GPU 显存用量 (MB)。"""
    info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
    return info.used / 1024 / 1024

# ────────────── 加载视频 ──────────────
VIDEO_PATH = Path(__file__).parent / "test.mp4"
ROI = (880, 935, 960, 985)  # x1,y1,x2,y2
FRAME_DIV = 2
TARGET_H = 48
PAD_PX = 20
WORKERS_LIST = [2, 4, 6, 8, 10, 12, 16]

def load_frames() -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    x1, y1, x2, y2 = ROI
    frames = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fi % FRAME_DIV != 0:
            fi += 1
            continue
        crop = frame[y1:y2+1, x1:x2+1].copy()
        frames.append(crop)
        fi += 1
    cap.release()
    return frames

# ────────────── 单引擎并行推理 ──────────────
def run_single_engine(frames, num_workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    engine = RapidOCR()

    preprocessed = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        preprocessed = list(pool.map(
            lambda c: preprocess_crop(c, TARGET_H, PAD_PX), frames))

    def _ocr_one(idx, proc):
        ocr_result, _ = engine(proc)
        return idx

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_ocr_one, i, p) for i, p in enumerate(preprocessed)]
        for f in as_completed(futures):
            f.result()

    del engine
    gc.collect()

# ────────────── 主测试 ──────────────
def run_benchmark():
    print("=" * 70)
    print("Benchmark: 单引擎方案 — 不同 worker 数量性能对比")
    print(f"  视频: {VIDEO_PATH}")
    print(f"  ROI: {ROI}, div={FRAME_DIV}, target_h={TARGET_H}, pad={PAD_PX}")
    print("=" * 70)

    frames = load_frames()
    total = len(frames)
    print(f"\n总帧数: {total}\n")

    baseline_mem = get_gpu_memory_mb()
    print(f"基线 GPU 显存: {baseline_mem:.1f} MB\n")

    results = []

    for num_w in WORKERS_LIST:
        gc.collect()
        time.sleep(1)

        mem_before = get_gpu_memory_mb()
        t0 = time.perf_counter()

        run_single_engine(frames, num_w)

        t_elapsed = time.perf_counter() - t0
        mem_after = get_gpu_memory_mb()
        mem_peak = max(mem_before, mem_after) - baseline_mem
        fps = total / t_elapsed if t_elapsed > 0 else 0

        results.append((num_w, t_elapsed, fps, mem_peak))

        print(f"  workers={num_w:<3}: {t_elapsed:.1f}s, {fps:.1f} fps, GPU 显存峰值 +{mem_peak:.0f} MB")

        gc.collect()
        time.sleep(2)

    # ── 汇总表格 ──
    print("\n" + "=" * 60)
    print(f"{'Workers':<9} {'时间':<10} {'FPS':<9} {'GPU峰值(MB)':<14} {'vs W=2':<10}")
    print("-" * 60)
    baseline_fps = results[0][2] if results else 1
    for w, t, fps, mem in results:
        ratio = f"{fps / baseline_fps:.2f}x"
        print(f"{w:<9} {t:<10.1f} {fps:<9.1f} {mem:<14.0f} {ratio:<10}")
    print("=" * 60)

    # 最优 workers
    best = max(results, key=lambda r: r[2])
    print(f"\n最优: workers={best[0]}, {best[2]:.1f} fps, {best[1]:.1f}s")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    run_benchmark()
