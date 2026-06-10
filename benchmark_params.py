"""
Benchmark: OCR 高度 & 边缘填充 参数调优。
关注指标: 识别数量（准确率代理）, 处理速度 (fps), 首尾识别稳定性。

固定参数: test.mp4, ROI=[880,935,960,985], div=4, workers=8
"""
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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

import re
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils import OrtInferSession

# ── CUDA monkey-patch ──
def _patch_ort():
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

_patch_ort()

VIDEO_PATH = Path(__file__).parent / "test.mp4"
ROI = (880, 935, 960, 985)
FRAME_DIV = 4
NUM_WORKERS = 8

OCR_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")

# ── OCR 文本提取 ──
def normalize_ocr_text(text: str) -> str:
    translation = str.maketrans(
        {"O":"0","o":"0","Q":"0","D":"0","I":"1","l":"1","|":"1","!":"1",
         "Z":"2","z":"2","S":"5","s":"5","B":"8","G":"6","g":"6","T":"7","t":"7",",":"."})
    return text.translate(translation)

def extract_speed_value(ocr_result):
    if not ocr_result: return None, None
    candidates = []
    for item in ocr_result:
        if not item or len(item) < 2: continue
        text = str(item[1]).strip()
        if text: candidates.append(text)
    if not candidates: return None, None
    joined = normalize_ocr_text(" ".join(candidates)).replace(" ", "")
    match = OCR_NUMBER_RE.search(joined)
    if not match: return None, None
    raw_text = re.sub(r"\D", "", match.group(0))
    if not raw_text: return None, None
    try: return float(raw_text), raw_text
    except ValueError: return None, None

def convert_speed_to_kmh(speed_value: float, source_unit: str = "km/h") -> float:
    return float(speed_value) * {"m/s": 3.6, "km/h": 1.0, "mile/h": 1.609344}[source_unit]

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

# ── 加载帧 ──
def load_frames():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    x1, y1, x2, y2 = ROI
    frames = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None: break
        if fi % FRAME_DIV != 0:
            fi += 1; continue
        crop = frame[y1:y2+1, x1:x2+1].copy()
        frames.append(crop)
        fi += 1
    cap.release()
    return frames

# ── 运行测试 ──
def run_test(frames, target_h, pad_px):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    engine = RapidOCR()

    preprocessed = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        preprocessed = list(pool.map(
            lambda c: preprocess_crop(c, target_h, pad_px), frames))

    results: list[tuple[int, float | None]] = [None] * len(frames)

    def _ocr_one(idx, proc):
        ocr_result, _ = engine(proc)
        sv, rt = extract_speed_value(ocr_result)
        if sv is not None and rt is not None:
            return idx, convert_speed_to_kmh(sv)
        return idx, None

    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    try:
        futures = [pool.submit(_ocr_one, i, p) for i, p in enumerate(preprocessed)]
        for f in as_completed(futures):
            idx, val = f.result()
            results[idx] = (idx, val)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    t_elapsed = time.perf_counter() - t0

    speeds = [v for _, v in results if v is not None]
    fps = len(frames) / t_elapsed if t_elapsed > 0 else 0

    del engine; gc.collect()
    time.sleep(1)
    return len(speeds), t_elapsed, fps, speeds

# ── 主测试 ──
def main():
    frames = load_frames()
    total = len(frames)
    print(f"帧数: {total} | div={FRAME_DIV} | workers={NUM_WORKERS} | test.mp4 ROI={ROI}")
    print()

    TARGET_H_VALS = [20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96]
    PAD_VALS = [0, 5, 10, 15, 20, 25, 30, 40, 50]

    # 先测试典型值找到最优 target_h（pad=20）
    print("=== Phase 1: 固定 pad=20，扫描 target_h ===")
    print(f"{'h':>4} {'识别数':>7} {'用时(s)':>8} {'FPS':>7}")
    print("-" * 32)
    best_h = 48
    best_count = 0
    for h in TARGET_H_VALS:
        count, t, fps, _ = run_test(frames, h, 20)
        print(f"{h:>4} {count:>7} {t:>8.1f} {fps:>7.1f}")
        if count > best_count:
            best_count = count
            best_h = h

    print(f"\n→ target_h 最优: {best_h} (识别 {best_count} 条)\n")

    # 固定 best_h，扫描 pad
    print(f"=== Phase 2: 固定 h={best_h}，扫描 pad ===")
    print(f"{'pad':>4} {'识别数':>7} {'用时(s)':>8} {'FPS':>7}")
    print("-" * 32)
    best_pad = 20
    best_count2 = 0
    for p in PAD_VALS:
        count, t, fps, _ = run_test(frames, best_h, p)
        print(f"{p:>4} {count:>7} {t:>8.1f} {fps:>7.1f}")
        if count > best_count2:
            best_count2 = count
            best_pad = p

    print(f"\n→ pad 最优: {best_pad} (识别 {best_count2} 条)\n")

    # 在最优值附近交叉验证
    print(f"=== Phase 3: 交叉验证 (h={best_h-4}~{best_h+8}, pad={max(0,best_pad-10)}~{best_pad+10}) ===")
    h_range = range(max(16, best_h-4), best_h+9, 4)
    p_range = range(max(0, best_pad-10), best_pad+15, 5)

    print(f"{'h':>4} {'pad':>4} {'识别数':>7} {'用时(s)':>8} {'FPS':>7} {'评分':>8}")
    print("-" * 48)

    all_results = []
    for h in h_range:
        for p in p_range:
            count, t, fps, speeds = run_test(frames, h, p)
            # 综合评分: 识别数优先, FPS 加权
            score = count * 10 + fps
            all_results.append((h, p, count, t, fps, score))
            marker = " ← 最优" if score == max(r[5] for r in all_results) else ""
            print(f"{h:>4} {p:>4} {count:>7} {t:>8.1f} {fps:>7.1f} {score:>8.0f}{marker}")

    # 最终结果
    best = max(all_results, key=lambda r: r[5])
    print(f"\n{'='*48}")
    print(f"最优组合: target_h={best[0]}, pad={best[2]}")
    print(f"  识别数: {best[2]}/{total} ({best[2]/total*100:.1f}%)")
    print(f"  用时: {best[3]:.1f}s, FPS: {best[4]:.1f}")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
