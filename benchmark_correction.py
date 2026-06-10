"""
Benchmark: 新旧纠错机制对比。
imports from RaceVideoToLog.py, div=2, workers=8, max_accel=30
"""
import gc, os, re, sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2, numpy as np
from dataclasses import dataclass

# ── GPU setup ──
def _register_gpu_dlls():
    try:
        import ctypes as _ct
        _cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        for _ver in ["v12.9","v12.8","v12.6","v12.4"]:
            _cb = os.path.join(_cuda_base, _ver, "bin")
            if not os.path.isdir(_cb): continue
            os.add_dll_directory(_cb)
            for _f in os.listdir(_cb):
                _fl = _f.lower()
                if _fl.endswith(".dll") and any(_fl.startswith(p) for p in
                    ("cudart","cublas","cufft","curand","cusparse","nvjitlink")):
                    try: _ct.CDLL(os.path.join(_cb, _f))
                    except: pass
            _cudnn_base = r"C:\Program Files\NVIDIA\CUDNN"
            if os.path.isdir(_cudnn_base):
                for _root, _dirs, _files in os.walk(_cudnn_base):
                    for _f in _files:
                        if _f.lower().startswith("cudnn") and _f.endswith(".dll"):
                            if _ver.replace("v","") in _root.replace("\\","/"):
                                try: _ct.CDLL(os.path.join(_root, _f))
                                except: pass
            _existing = os.environ.get("PATH", "")
            if _cb not in _existing:
                os.environ["PATH"] = _cb + ";" + _existing
            break
    except: pass

_register_gpu_dlls()

from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils import OrtInferSession

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

# ── import the correction functions from main script ──
from RaceVideoToLog import (
    SpeedObservation, build_speed_candidates, correct_speed_series,
    normalize_ocr_text, extract_speed_value, convert_speed_to_kmh,
)

# ── Old correction (for comparison) ──
def correct_speed_series_old(samples, max_speed_kmh, max_accel_mps2):
    if not samples: return []
    if max_speed_kmh <= 0 or max_accel_mps2 <= 0:
        return [s.raw_speed_kmh for s in samples]

    candidate_lists = []
    for sample in samples:
        candidates = build_speed_candidates_old(sample.raw_text, max_speed_kmh)
        if sample.raw_speed_kmh <= max_speed_kmh:
            candidates.append(float(sample.raw_speed_kmh))
        if not candidates:
            candidates = [min(max(sample.raw_speed_kmh, 0.0), max_speed_kmh)]
        candidate_lists.append(sorted(set(candidates)))

    states = [(abs(c - samples[0].raw_speed_kmh), c, None) for c in candidate_lists[0]]
    bp = [[-1]*len(candidate_lists[0])]
    for i in range(1, len(samples)):
        cc, pc = candidate_lists[i], candidate_lists[i-1]
        dt = max(samples[i].timestamp - samples[i-1].timestamp, 1e-6)
        md = max_accel_mps2 * dt * 3.6
        cs, cb = [], []
        for cv in cc:
            best_c, best_p = float("inf"), 0
            for pi, pv in enumerate(pc):
                td = abs(cv - pv)
                tc = td * 0.05 + (td - md) * 50.0 if td > md else td * 0.05
                cost = states[pi][0] + tc + abs(cv - samples[i].raw_speed_kmh) * 0.5
                if cost < best_c:
                    best_c, best_p = cost, pi
            cs.append((best_c, cv, best_p)); cb.append(best_p)
        states = cs; bp.append(cb)

    bi = min(range(len(states)), key=lambda i: states[i][0])
    corrected = [0.0]*len(samples)
    corrected[-1] = states[bi][1]
    for i in range(len(samples)-1, 0, -1):
        bi = bp[i][bi]
        corrected[i-1] = candidate_lists[i-1][bi]
    return corrected

def build_speed_candidates_old(raw_text, max_speed_kmh):
    if max_speed_kmh <= 0: return []
    text = re.sub(r"\D","",raw_text)
    if not text: return []
    ms = int(math.floor(max_speed_kmh))
    c = set()
    mn = 1 if len(text)==1 else max(1,len(text)-2)
    for sl in range(mn, len(text)+1):
        st = text[-sl:]
        try: sv = int(st)
        except: continue
        step = 10**sl
        for v in range(sv, ms+1, step): c.add(float(v))
    return sorted(c)

# ── Parameters ──
VIDEO_PATH = Path(__file__).parent / "test.mp4"
ROI = (880, 935, 960, 985)
FRAME_DIV = 2; WORKERS = 8; MAX_ACCEL = 30.0; MAX_SPEED = 400.0
_OCR_RE = re.compile(r"\d+(?:[\.,]\d+)?")

def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w = gray.shape[:2]; scale = 24.0/h if h>0 else 1.0
    if abs(scale-1)>0.02: gray = cv2.resize(gray,(max(1,int(w*scale)),24))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def load_frames():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    x1,y1,x2,y2 = ROI; frames=[]; fi=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if fi%FRAME_DIV!=0: fi+=1; continue
        frames.append(f[y1:y2+1,x1:x2+1].copy()); fi+=1
    cap.release(); return frames

# ── Run ──
def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames = load_frames(); total = len(frames)
    fps_v = 60.0; timestamps = [i*FRAME_DIV/fps_v for i in range(total)]
    engine = RapidOCR()
    pre = [preprocess(f) for f in frames]
    results = [None]*total
    def _ocr(i,p):
        r,_ = engine(p); sv,rt = extract_speed_value(r)
        return i, SpeedObservation(timestamp=timestamps[i], raw_speed_kmh=convert_speed_to_kmh(sv or 0,"km/h"), raw_text=rt or "")
    with ThreadPoolExecutor(WORKERS) as pool:
        fs = [pool.submit(_ocr,i,p) for i,p in enumerate(pre)]
        for f in as_completed(fs): i,obs = f.result(); results[i]=obs

    samples = [s for s in results if s is not None and s.raw_speed_kmh>0]
    print(f"OCR 识别: {len(samples)}/{total}")

    t0=time.perf_counter(); new = correct_speed_series(samples, MAX_SPEED, MAX_ACCEL)
    tnew = time.perf_counter()-t0
    t0=time.perf_counter(); old = correct_speed_series_old(samples, MAX_SPEED, MAX_ACCEL)
    told = time.perf_counter()-t0

    nc = sum(1 for o,c in zip(samples,new) if abs(o.raw_speed_kmh-c)>0.01)
    oc = sum(1 for o,c in zip(samples,old) if abs(o.raw_speed_kmh-c)>0.01)

    # Find worst segments
    diffs_new = [abs(o.raw_speed_kmh-c) for o,c in zip(samples,new)]
    diffs_old = [abs(o.raw_speed_kmh-c) for o,c in zip(samples,old)]
    max_diff_new = max(diffs_new); max_diff_old = max(diffs_old)
    avg_diff_new = sum(diffs_new)/len(diffs_new); avg_diff_old = sum(diffs_old)/len(diffs_old)

    print(f"\n{'指标':<20} {'旧机制':<15} {'新机制':<15} {'改善':<10}")
    print("-"*60)
    print(f"{'纠正数量':<20} {oc:<15} {nc:<15} {'-' if nc>oc else '✓':<10}")
    print(f"{'准确率':<20} {(1-oc/len(samples))*100:<14.1f}% {(1-nc/len(samples))*100:<14.1f}% {((nc-oc)/oc*100) if oc else 0:<9.1f}%")
    print(f"{'最大偏差 (km/h)':<20} {max_diff_old:<15.1f} {max_diff_new:<15.1f} {(1-max_diff_new/max_diff_old)*100 if max_diff_old else 0:<9.1f}%")
    print(f"{'平均偏差 (km/h)':<20} {avg_diff_old:<15.2f} {avg_diff_new:<15.2f} {(1-avg_diff_new/avg_diff_old)*100 if avg_diff_old else 0:<9.1f}%")
    print(f"{'耗时':<20} {told:<15.3f}s {tnew:<15.3f}s")
    print(f"\n偏差<2km/h: 新 {sum(1 for d in diffs_new if d<2)}/{len(diffs_new)} vs 旧 {sum(1 for d in diffs_old if d<2)}/{len(diffs_old)}")
    print(f"偏差<5km/h: 新 {sum(1 for d in diffs_new if d<5)}/{len(diffs_new)} vs 旧 {sum(1 for d in diffs_old if d<5)}/{len(diffs_old)}")

    del engine; gc.collect()

if __name__ == "__main__":
    main()
