"""
Benchmark: CPU / DirectML / CUDA 三种后端性能对比。
用法:
  python benchmark.py                          # 默认：test.mp4 + test2.mp4，三种模式全部测试
  python benchmark.py --mode cuda              # 仅 CUDA
  python benchmark.py --mode cpu,dml           # CPU + DirectML
  python benchmark.py --video test.mp4          # 仅指定视频
"""

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════ DLL 注册（CUDA / cuDNN）═══════════════════
def _register_gpu_dlls() -> None:
    """将 CUDA 12.x 和 cuDNN 9 的 DLL 目录注册到当前进程搜索路径。
    使用 ctypes 预加载 cuDNN DLL，确保 onnxruntime 能找到它们。"""
    try:
        import ctypes as _ct
        _cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        _cudnn_base = r"C:\Program Files\NVIDIA\CUDNN"
        for _ver in ["v12.9", "v12.8", "v12.6", "v12.4"]:
            _cb = os.path.join(_cuda_base, _ver, "bin")
            if os.path.isdir(_cb):
                os.add_dll_directory(_cb)
            # cuDNN 子目录可能带 v 前缀（v12.9）或不带（12.9）
            _cudnn_vers = [_ver, _ver.lstrip("v")]
            if os.path.isdir(_cudnn_base):
                for _dv in os.listdir(_cudnn_base):
                    for _cv in _cudnn_vers:
                        _db = os.path.join(_cudnn_base, _dv, "bin", _cv, "x64")
                        if os.path.isdir(_db):
                            os.add_dll_directory(_db)
                            # 预加载 cuDNN DLL，确保 onnxruntime 能找到
                            for _dll in os.listdir(_db):
                                if _dll.endswith(".dll"):
                                    try:
                                        _ct.CDLL(os.path.join(_db, _dll))
                                    except Exception:
                                        pass
            break
    except Exception:
        pass


_register_gpu_dlls()

# ═══════════════════ 常量 / 正则 ═══════════════════
SOURCE_TO_KMH = {"m/s": 3.6, "km/h": 1.0, "mile/h": 1.609344}
OCR_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")
SPEED_FORMAT = "km/h"

# 默认参数（与 GUI 默认值一致）
DEFAULT_REGION = (881, 932, 963, 985)  # 速度显示区域 ROI
DEFAULT_TARGET_H = 48.0
DEFAULT_PAD_PX = 20.0
DEFAULT_MAX_SPEED = 400.0
DEFAULT_MAX_ACCEL = 50.0
DEFAULT_FRAME_DIV = 2  # 每 2 帧取 1 帧


@dataclass
class SpeedObservation:
    timestamp: float
    raw_speed_kmh: float
    raw_text: str


# ═══════════════════ OCR 预处理 & 解析 ═══════════════════
def normalize_ocr_text(text: str) -> str:
    trans = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1",
        "|": "1", "!": "1", "Z": "2", "z": "2", "S": "5", "s": "5",
        "B": "8", "G": "6", "g": "6", "T": "7", "t": "7", ",": ".",
    })
    return text.translate(trans)


def extract_speed_value(ocr_result):
    if not ocr_result:
        return None, None
    candidates = []
    for item in ocr_result:
        if not item or len(item) < 2:
            continue
        candidates.append(str(item[1]).strip())
    if not candidates:
        return None, None
    joined = normalize_ocr_text(" ".join(candidates)).replace(" ", "")
    m = OCR_NUMBER_RE.search(joined)
    if not m:
        return None, None
    raw = re.sub(r"\D", "", m.group(0))
    if not raw:
        return None, None
    try:
        return float(raw), raw
    except ValueError:
        return None, None


def preprocess_crop(crop: np.ndarray, target_h: float, pad_px: float) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    target_h = max(8.0, float(target_h))
    pad_px = max(0.0, float(pad_px))
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scale = target_h / float(h) if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(target_h)), interpolation=cv2.INTER_LINEAR)
    pad_int = int(pad_px)
    if pad_int > 0:
        gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_speed_candidates(raw_text: str, max_speed_kmh: float) -> list[float]:
    if max_speed_kmh <= 0:
        return []
    text = re.sub(r"\D", "", raw_text)
    if not text:
        return []
    max_speed_int = int(math.floor(max_speed_kmh))
    if max_speed_int < 0:
        return []
    min_len = 1 if len(text) == 1 else max(1, len(text) - 2)
    candidates: set[float] = set()
    for sl in range(min_len, len(text) + 1):
        try:
            sv = int(text[-sl:])
        except ValueError:
            continue
        step = 10 ** sl
        for c in range(sv, max_speed_int + 1, step):
            candidates.add(float(c))
    return sorted(candidates)


def correct_speed_series(samples, max_speed_kmh, max_accel_mps2):
    if not samples or max_speed_kmh <= 0 or max_accel_mps2 <= 0:
        return [s.raw_speed_kmh for s in samples]
    cand_lists = []
    for s in samples:
        cands = build_speed_candidates(s.raw_text, max_speed_kmh)
        if s.raw_speed_kmh <= max_speed_kmh:
            cands.append(float(s.raw_speed_kmh))
        if not cands:
            cands = [min(max(s.raw_speed_kmh, 0.0), max_speed_kmh)]
        cand_lists.append(sorted(set(cands)))
    states = [(abs(c - samples[0].raw_speed_kmh), c, None) for c in cand_lists[0]]
    bp = [[-1] * len(cand_lists[0])]
    for si in range(1, len(samples)):
        cc = cand_lists[si]
        pc = cand_lists[si - 1]
        dt = max(samples[si].timestamp - samples[si - 1].timestamp, 1e-6)
        max_delta = max_accel_mps2 * dt * 3.6
        ns, nb = [], []
        for ci_idx, ci in enumerate(cc):
            best_cost, best_prev = float("inf"), 0
            for pi, pv in enumerate(pc):
                prev_cost = states[pi][0]
                td = abs(ci - pv)
                tc = td * 0.05
                if td > max_delta:
                    tc += (td - max_delta) * 50.0
                cost = prev_cost + tc + abs(ci - samples[si].raw_speed_kmh) * 0.5
                if cost < best_cost:
                    best_cost, best_prev = cost, pi
            ns.append((best_cost, ci, best_prev))
            nb.append(best_prev)
        states = ns
        bp.append(nb)
    best_idx = min(range(len(states)), key=lambda i: states[i][0])
    corrected = [0.0] * len(samples)
    corrected[-1] = states[best_idx][1]
    cur = best_idx
    for si in range(len(samples) - 1, 0, -1):
        cur = bp[si][cur]
        corrected[si - 1] = cand_lists[si - 1][cur]
    return corrected


def count_corrections(observations, corrected):
    count = 0
    for obs, corr in zip(observations, corrected):
        if abs(obs.raw_speed_kmh - corr) > 0.01:
            count += 1
    return count


# ═══════════════════ 帧读取 ═══════════════════
def read_sampled_frames(video_path: str, region: tuple, fps: float, frame_div: int):
    x1, y1, x2, y2 = region
    frame_step = max(1, frame_div)
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fi % frame_step != 0:
            fi += 1
            continue
        ts = fi / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            cap.release()
            raise RuntimeError("识别范围超出画面")
        raw_frames.append((ts, crop))
        fi += 1
    cap.release()
    return raw_frames


# ═══════════════════ OCR 引擎工厂 ═══════════════════
def create_ocr_for_mode(mode: str):
    """
    根据 mode 创建 RapidOCR 实例。
    mode: 'cpu' | 'dml' | 'cuda'
    返回 (ocr_engine, actual_backend_name)
    """
    import onnxruntime as ort
    from rapidocr_onnxruntime.utils import OrtInferSession
    from rapidocr_onnxruntime import RapidOCR

    available = ort.get_available_providers()

    cpu_ep = 'CPUExecutionProvider'
    cpu_opts = {'arena_extend_strategy': 'kSameAsRequested'}
    cuda_ep = 'CUDAExecutionProvider'
    cuda_opts = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }
    dml_ep = 'DmlExecutionProvider'

    if mode == 'cpu':
        # 强制 CPU
        _orig = OrtInferSession.__init__

        def _cpu_init(self, config):
            from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
            so = SessionOptions()
            so.log_severity_level = 4
            so.enable_cpu_mem_arena = False
            so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            self._verify_model(config['model_path'])
            self.session = InferenceSession(
                config['model_path'], sess_options=so,
                providers=[(cpu_ep, cpu_opts)]
            )

        OrtInferSession.__init__ = _cpu_init
        ocr = RapidOCR()
        OrtInferSession.__init__ = _orig
        return ocr, "CPU"

    elif mode == 'cuda':
        if cuda_ep not in available:
            raise RuntimeError(f"CUDA 不可用。当前可用 providers: {available}")
        _orig = OrtInferSession.__init__

        def _cuda_init(self, config):
            from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
            so = SessionOptions()
            so.log_severity_level = 4
            so.enable_cpu_mem_arena = False
            so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            self._verify_model(config['model_path'])
            self.session = InferenceSession(
                config['model_path'], sess_options=so,
                providers=[(cuda_ep, cuda_opts)]
            )

        OrtInferSession.__init__ = _cuda_init
        ocr = RapidOCR()
        OrtInferSession.__init__ = _orig
        return ocr, "CUDA"

    elif mode == 'dml':
        if dml_ep not in available:
            raise RuntimeError(
                f"DirectML 不可用。当前可用 providers: {available}\n"
                f"请安装 onnxruntime-directml: pip install onnxruntime-directml"
            )
        _orig = OrtInferSession.__init__

        def _dml_init(self, config):
            from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
            so = SessionOptions()
            so.log_severity_level = 4
            so.enable_cpu_mem_arena = False
            so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            self._verify_model(config['model_path'])
            self.session = InferenceSession(
                config['model_path'], sess_options=so,
                providers=[dml_ep]
            )

        OrtInferSession.__init__ = _dml_init
        ocr = RapidOCR()
        OrtInferSession.__init__ = _orig
        return ocr, "DirectML"

    else:
        raise ValueError(f"未知模式: {mode}")


# ═══════════════════ 顺序 OCR 处理 ═══════════════════
def run_sequential(raw_frames, ocr, target_h, pad_px, speed_format, max_speed, max_accel, label=""):
    observations = []
    total = len(raw_frames)
    for i, (ts, crop) in enumerate(raw_frames):
        proc = preprocess_crop(crop, target_h, pad_px)
        ocr_result, _ = ocr(proc)
        sv, rt = extract_speed_value(ocr_result)
        if sv is not None and rt is not None:
            observations.append(SpeedObservation(
                timestamp=ts,
                raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
                raw_text=rt,
            ))
        if (i + 1) % 100 == 0:
            print(f"  [{label}] OCR 进度: {i+1}/{total}", flush=True)
    if not observations:
        return [], [], 0
    print(f"  [{label}] 纠错中...", flush=True)
    corrected = correct_speed_series(observations, max_speed, max_accel)
    return observations, corrected, count_corrections(observations, corrected)


# ═══════════════════ 并行 OCR 处理 ═══════════════════
def run_parallel(raw_frames, ocr_engines, target_h, pad_px, speed_format, max_speed, max_accel, label=""):
    """多 OCR 引擎并行推理，利用 CUDA 多流加速。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = len(ocr_engines)
    total = len(raw_frames)

    # 阶段 1：并行预处理
    print(f"  [{label}] 预处理 ({num_workers} workers)...", flush=True)
    t_prep = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        preprocessed = list(pool.map(
            lambda item: (item[0], preprocess_crop(item[1], target_h, pad_px)),
            raw_frames,
        ))
    t_prep = time.perf_counter() - t_prep
    print(f"  [{label}] 预处理完成: {t_prep:.1f}s", flush=True)

    # 阶段 2：并行 OCR 推理
    print(f"  [{label}] OCR 推理中...", flush=True)
    observations: list = [None] * total

    def _ocr_one(idx: int, ts: float, proc):
        engine = ocr_engines[idx % num_workers]
        ocr_result, _ = engine(proc)
        sv, rt = extract_speed_value(ocr_result)
        if sv is not None and rt is not None:
            return idx, SpeedObservation(
                timestamp=ts,
                raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
                raw_text=rt,
            )
        return idx, None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_ocr_one, i, ts, proc)
            for i, (ts, proc) in enumerate(preprocessed)
        ]
        done = 0
        for f in as_completed(futures):
            idx, obs = f.result()
            observations[idx] = obs
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  [{label}] OCR 进度: {done}/{total}", flush=True)

    observations = [o for o in observations if o is not None]
    if not observations:
        return [], [], 0
    print(f"  [{label}] 纠错中...", flush=True)
    corrected = correct_speed_series(observations, max_speed, max_accel)
    return observations, corrected, count_corrections(observations, corrected)


# ═══════════════════ 单组 benchmark ═══════════════════
def benchmark_one(video_path: str, mode: str, region: tuple,
                  target_h: float, pad_px: float,
                  max_speed: float, max_accel: float,
                  frame_div: int, num_workers: int = 1) -> dict:
    """返回 benchmark 结果字典。num_workers>1 时启用多引擎并行推理。"""
    num_workers = max(1, int(num_workers))
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    # 读取所有帧（frame_div=1），再子采样
    print(f"  读取帧中...", flush=True)
    all_raw = read_sampled_frames(video_path, region, fps, frame_div=1)
    raw = [all_raw[i] for i in range(0, len(all_raw), frame_div)]
    sample_fps = fps / frame_div
    print(f"  采样帧: {len(raw)} (原始 {len(all_raw)}, div={frame_div})", flush=True)

    t0 = time.perf_counter()
    if num_workers > 1:
        print(f"  初始化 {num_workers} 个 OCR 引擎 ({mode.upper()})...", flush=True)
        ocr_engines = [create_ocr_for_mode(mode)[0] for _ in range(num_workers)]
        backend = create_ocr_for_mode(mode)[1]  # just for display name
    else:
        print(f"  初始化 OCR 引擎 ({mode.upper()})...", flush=True)
        ocr, backend = create_ocr_for_mode(mode)
    t_init = time.perf_counter() - t0

    label = f"{mode.upper()} x{num_workers}" if num_workers > 1 else mode.upper()
    t1 = time.perf_counter()
    if num_workers > 1:
        obs, corr, cc = run_parallel(raw, ocr_engines, target_h, pad_px, SPEED_FORMAT, max_speed, max_accel, label=label)
    else:
        obs, corr, cc = run_sequential(raw, ocr, target_h, pad_px, SPEED_FORMAT, max_speed, max_accel, label=label)
    t_ocr = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    acc = (1 - cc / len(obs)) * 100 if obs else 100.0

    mode_name = f"{backend} x{num_workers}" if num_workers > 1 else backend
    return {
        "video": Path(video_path).name,
        "mode": mode_name,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "sample_fps": sample_fps,
        "frames": len(raw),
        "total_frames": fcount,
        "detected": len(obs),
        "corrections": cc,
        "accuracy": acc,
        "t_init": t_init,
        "t_ocr": t_ocr,
        "t_total": t_total,
        "fps_processing": len(raw) / t_ocr if t_ocr > 0 else 0,
    }


def print_results(results: list[dict]) -> None:
    """格式化打印 benchmark 结果表格。"""
    if not results:
        return

    # 表头
    header = (
        f"{'视频':<12} {'模式':<10} {'分辨率':<14} {'帧率':>8} {'采样帧':>7} "
        f"{'识别':>6} {'纠错':>5} {'准确率':>7} {'初始化':>8} {'OCR耗时':>9} {'总耗时':>9} {'处理速度':>9}"
    )
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("Benchmark 结果汇总")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['video']:<12} {r['mode']:<10} {r['resolution']:<14} "
            f"{r['fps']:>7.1f} "
            f"{r['frames']:>6}/{r['total_frames']:.0f} "
            f"{r['detected']:>5}  {r['corrections']:>4}  "
            f"{r['accuracy']:>6.1f}% "
            f"{r['t_init']:>7.2f}s "
            f"{r['t_ocr']:>8.2f}s "
            f"{r['t_total']:>8.2f}s "
            f"{r['fps_processing']:>8.1f}fps"
        )
        print(line)

    print("-" * len(header))

    # 按模式汇总
    print(f"\n{'='*60}")
    print("按模式汇总")
    print(f"{'='*60}")
    modes = {}
    for r in results:
        m = r['mode']
        if m not in modes:
            modes[m] = []
        modes[m].append(r)

    for mode_name in ["CPU", "CUDA", "DirectML"]:
        if mode_name in modes:
            items = modes[mode_name]
            avg_acc = sum(r['accuracy'] for r in items) / len(items)
            avg_fps = sum(r['fps_processing'] for r in items) / len(items)
            avg_total = sum(r['t_total'] for r in items) / len(items)
            print(f"  {mode_name:<10}: 平均准确率 {avg_acc:.1f}%, "
                  f"平均处理速度 {avg_fps:.1f} fps, "
                  f"平均总耗时 {avg_total:.2f}s")


# ═══════════════════ 主入口 ═══════════════════
def main():
    parser = argparse.ArgumentParser(
        description="RaceVideoToLog Benchmark - CPU / DirectML / CUDA 性能对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python benchmark.py                                          # 全部模式，两个视频
  python benchmark.py --mode cuda                              # 仅 CUDA
  python benchmark.py --mode cpu,dml                           # CPU + DirectML
  python benchmark.py --video test.mp4 --mode all              # 仅 test.mp4
        """,
    )
    parser.add_argument("--video", nargs="+", default=None,
                        help="视频文件路径 (默认: test.mp4 test2.mp4)")
    parser.add_argument("--mode", type=str, default="all",
                        help="测试模式: cpu, dml, cuda, all (默认: all)")
    parser.add_argument("--roi", nargs=4, type=int,
                        default=DEFAULT_REGION,
                        help=f"ROI 坐标 x1 y1 x2 y2 (默认: {DEFAULT_REGION[0]} {DEFAULT_REGION[1]} {DEFAULT_REGION[2]} {DEFAULT_REGION[3]})")
    parser.add_argument("--target-h", type=float, default=DEFAULT_TARGET_H,
                        help=f"OCR 目标高度 px (默认: {DEFAULT_TARGET_H})")
    parser.add_argument("--pad", type=float, default=DEFAULT_PAD_PX,
                        help=f"边缘填充 px (默认: {DEFAULT_PAD_PX})")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        help=f"最大速度 km/h (默认: {DEFAULT_MAX_SPEED})")
    parser.add_argument("--max-accel", type=float, default=DEFAULT_MAX_ACCEL,
                        help=f"最大加速度 m/s² (默认: {DEFAULT_MAX_ACCEL})")
    parser.add_argument("--div", type=int, default=DEFAULT_FRAME_DIV,
                        help=f"采样间隔 1/N (默认: {DEFAULT_FRAME_DIV})")
    parser.add_argument("--workers", type=int, nargs="+", default=[1],
                        help="并行线程数，可多个值 (默认: 1)。如 --workers 1 2 4 8")

    args = parser.parse_args()

    # 视频列表
    if args.video:
        video_paths = [str(Path(v).resolve()) for v in args.video]
    else:
        default_videos = ["test.mp4", "test2.mp4"]
        video_paths = []
        for v in default_videos:
            p = Path(v)
            if p.exists():
                video_paths.append(str(p.resolve()))
            else:
                print(f"警告: 默认视频 {v} 不存在，跳过。")

    if not video_paths:
        print("错误: 未找到任何视频文件。请使用 --video 指定。")
        sys.exit(1)

    # 模式列表
    if args.mode.lower() == "all":
        modes = ["cpu", "cuda", "dml"]
    else:
        modes = [m.strip().lower() for m in args.mode.split(",")]
        valid_modes = {"cpu", "dml", "cuda"}
        for m in modes:
            if m not in valid_modes:
                print(f"错误: 未知模式 '{m}'，可选: cpu, dml, cuda, all")
                sys.exit(1)

    worker_counts = [max(1, int(w)) for w in args.workers]
    region = tuple(args.roi)
    print(f"ROI: {region}", flush=True)
    print(f"参数: target_h={args.target_h}, pad={args.pad}, "
          f"max_speed={args.max_speed}, max_accel={args.max_accel}, div={args.div}", flush=True)
    print(f"视频: {video_paths}", flush=True)
    print(f"模式: {modes}, 线程数: {worker_counts}", flush=True)

    all_results = []
    errors = []

    for mode in modes:
        for nw in worker_counts:
            for vp in video_paths:
                wtag = f" x{nw}" if nw > 1 else ""
                label = f"{Path(vp).name} [{mode.upper()}{wtag}]"
                print(f"\n{'─'*50}")
                print(f"  测试: {label}")
                print(f"{'─'*50}")
                try:
                    result = benchmark_one(
                        vp, mode, region,
                        args.target_h, args.pad,
                        args.max_speed, args.max_accel,
                        args.div, num_workers=nw,
                    )
                    all_results.append(result)
                    print(f"  OK 完成: {result['detected']} 条识别, "
                          f"准确率 {result['accuracy']:.1f}%, "
                          f"耗时 {result['t_total']:.2f}s")
                except Exception as e:
                    err_msg = f"FAIL {label}: {e}"
                    print(f"  {err_msg}")
                    errors.append(err_msg)

    # 汇总
    print_results(all_results)

    if errors:
        print(f"\n以下测试失败 ({len(errors)} 项):")
        for e in errors:
            print(f"  {e}")

    print("\n完成。")


if __name__ == "__main__":
    main()
