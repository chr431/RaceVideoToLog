"""分段流水线（生产）：解码+分段+段值OCR 流水线化 → 段级纠错 → CSV。

分段 OCR 大幅减少调用（36-64%），解码（I/O 瓶颈）与段值 OCR 线程重叠
摊薄墙钟。算法（experiment-binary-ocr 分支验证）：
- diff 分段：聚类判别（max 3×3 窗口和 < C ⇒ 显示未变），解码循环内增量计算
- 段值：每段最清晰代表帧 OCR（sharpness=灰度std），OCR 线程批处理闭合段
- 段级检测：中值滤波（跟随弯曲，误读=尖峰被中值剔除）
- 段级纠正：可信锚点插值（锚点距离上界，跳过曲线正确段）
- 解码：实验开关 HYBRID_DECODE_ENV（RVTOL_HYBRID_DECODE，默认关，不暴露
  GUI/CLI 参数）开启时，GPU 模式（auto/nvdec）改走 CPU+NVDEC 双解码器
  并行（CPU 前段 + GPU 后段，见 _open_hybrid_vrs；AV1 特判按纯 GPU）；
  显式传 decode_backend='cpu+nvdec' 仍为混合（旧版程序化用法）

实现已按职责拆分：segmentation.py（灰度/Otsu/聚类）、hybrid_decode.py
（混合解码 worker/队列）、seg_correction.py（检测/置信度/DP）。
本文件保留 SegmentPipeline 编排、串行参考路径与 CSV 输出。

注意：_decode_all/_segment/_ocr_segments/_detect/_correct 是串行参考路径
（仅 tools/ 与测试使用），生产 run() 走 _run_pipelined + _dense_correct。
"""
from __future__ import annotations
import csv
import logging
import os as _os
import time
from pathlib import Path

import numpy as np

import config
from constants import Flag
from ocr_engine import extract_speed_value
from segmentation import (  # noqa: F401 — 兼容 tools/tests 的历史导入路径
    _apply_gamma, _cluster_win3, _gray, _gray_batch, _gray_seg,
    _gray_seg_batch, _otsu, _seg_gamma,
)
from hybrid_decode import (  # noqa: F401 — 兼容 tests 的历史导入路径
    HYBRID_BACKEND_ALIASES, _decode_range_worker, _drain_queue, _hybrid_ranges,
)
from seg_correction import (
    confidence_scores, correct_segments, dense_correct, detect_segments,
    dp_run, fill_values, local_bandwidth,
)

logger = logging.getLogger("RaceVideoToLog.segment_flow")


def _ocr_batch_size() -> int:
    """OCR 批大小（段数）：RVTOL_OCR_BATCH 实验钩子 > config.OCR_BATCH_SIZE。"""
    _env = _os.environ.get("RVTOL_OCR_BATCH")
    if _env and _env.isdigit():
        return max(1, int(_env))
    return config.OCR_BATCH_SIZE


class SegmentPipeline:
    """生产分段流水线。"""

    def __init__(self, video_path: str, roi: tuple, max_speed_kmh: float,
                 max_accel_mps2: float, fps: float | None, frame_start: int | None,
                 frame_end: int | None, force_aspect: float = 0.0,
                 speed_format: str = "km/h",
                 decode_backend: str = "auto",
                 ocr_backend: str = "auto",
                 buffer_size: int = config.DEFAULT_BUFFER_SIZE,
                 fill_width: int = config.DEFAULT_FILL_WIDTH,
                 C: float = config.SEG_C, win: int = config.SEG_WIN,
                 mult: float = config.SEG_MULT,
                 min_dev: float = config.SEG_MIN_DEV,
                 med_k: int = config.SEG_MED_K,
                 detect_floor: float = config.SEG_DETECT_FLOOR,
                 single_floor: float = config.SEG_SINGLE_FLOOR,
                 anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES,
                 conf_w_med: float = config.SEG_CONF_W_MED,
                 conf_w_jerk: float = config.SEG_CONF_W_JERK,
                 conf_jerk_scale: float = config.SEG_CONF_JERK_SCALE,
                 dp_obs_weight: float = config.SEG_DP_OBS_WEIGHT,
                 dp_accel_weight: float = config.SEG_DP_ACCEL_WEIGHT,
                 dp_max_dv_cap: float = config.SEG_DP_MAX_DV_CAP,
                 dp_anchor_cost: float = config.SEG_DP_ANCHOR_COST,
                 dp_change_threshold: float = config.SEG_DP_CHANGE_THRESHOLD,
                 dp_anchor_conf: float = config.SEG_DP_ANCHOR_CONF,
                 dp_deanchor_jerk_min: float = config.SEG_DP_DEANCHOR_JERK_MIN,
                 dp_deanchor_jerk_max: float = config.SEG_DP_DEANCHOR_JERK_MAX,
                 progress_cb=None, cancel_check=None,
                 gray_output: bool = False):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        self._max_speed = max_speed_kmh
        self._max_accel = max_accel_mps2
        self._fps = fps  # None → _decode_all 里从 decord 推导
        self._frame_start = frame_start or 0
        self._frame_end = frame_end
        self._force_aspect = force_aspect      # 强制宽高比（0=不启用）
        self._ocr_model = config.DEFAULT_OCR_MODEL  # 唯一模型 v6_small（v2.13 移除选择）
        self._decode_backend = decode_backend
        self._ocr_backend = ocr_backend
        self._speed_format = speed_format
        self._buffer_size = buffer_size
        self._fill_width = fill_width
        self._C = C
        self._win = win
        self._mult = mult
        self._min_dev = min_dev
        self._med_k = med_k
        self._detect_floor = detect_floor
        self._single_floor = single_floor
        self._anchor_max = anchor_max
        self._conf_w_med = conf_w_med
        self._conf_w_jerk = conf_w_jerk
        self._conf_jerk_scale = conf_jerk_scale
        self._dp_obs_weight = dp_obs_weight
        self._dp_accel_weight = dp_accel_weight
        self._dp_max_dv_cap = dp_max_dv_cap
        self._dp_anchor_cost = dp_anchor_cost
        self._dp_change_threshold = dp_change_threshold
        self._dp_anchor_conf = dp_anchor_conf
        self._dp_deanchor_jerk_min = dp_deanchor_jerk_min
        self._dp_deanchor_jerk_max = dp_deanchor_jerk_max
        # 灰度输出（decord output_format='gray'，CPU/GPU 解码路径都生效，
        # ≥0.7.9）：直出 Y 平面（与 CPU swscale GRAY8 逐位一致），分段灰度
        # 跨后端统一；跳过 RGB→灰转换与 Python 侧 matmul
        self._gray_output = gray_output
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.rows: list = []
        self.timing: dict = {}
        # 段级 review / finalize 支持（run 后填充）
        self.segments: list[dict] = []   # [{start,end,value,rep_frame,rep_crop}]
        self.crops: dict = {}            # 每段代表帧 ROI（review 懒加载预览用）
        self._segs: list = []
        self._frames: list = []
        self._ocr_vals: list = []
        self._conf_vals: list = []        # 每段置信度（run 后填充，flag 判定用）
        self._pinned: set = set()          # 用户手动修正的段索引（finalize 时设）
        # review 预览用：生产管线走 gray 输出（无色彩信息），预览需要原始
        # RGB ROI 时惰性打开一个 CPU RGB reader（见 load_rgb_crop）
        self._rgb_vr = None
        # 细粒度性能剖面（实验专用，RVTOL_PROFILE=1 才启用；默认零开销）
        self._profile_enabled = _os.environ.get(
            "RVTOL_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
        self.profile: dict = {}
        self._prof_lock = None
        if self._profile_enabled:
            import threading as _threading
            self._prof_lock = _threading.Lock()

    def _prof_end(self, group: str, key: str, t0: float) -> None:
        """累加一段耗时到 profile（线程安全；关闭时仅一次属性判断）。"""
        if not self._profile_enabled:
            return
        elapsed = time.perf_counter() - t0
        with self._prof_lock:
            d = self.profile.setdefault(group, {})
            d[key] = d.get(key, 0.0) + elapsed

    def _open_vr(self):
        """按 decode_backend 打开 decord 解码器（auto/cpu/nvdec）。

        auto: 尝试 GPU (NVDEC) 失败回退 CPU。cpu: 强制 CPU。
        nvdec: 强制 GPU（失败回退 CPU 并警告）。替代旧 DECORD_FORCE_CPU env。
        混合（显式 cpu+nvdec 或 HYBRID_DECODE_ENV 开启）：走
        _open_hybrid_vrs（双解码器并行），不经过本方法。

        ROI-first（decord ≥0.7.5）：构造时传入固定 ROI（半开区间）——
        解码器只输出该矩形（CPU filter 先 crop 再转换 / GPU 转换 kernel
        只算 ROI 窗口 + 输出池 ROI 尺寸），免全帧转换与逐帧裁剪。
        """
        from decord import VideoReader, cpu as _cpu
        try:
            import decord.video_reader as _vr_mod
            _has_roi_api = hasattr(_vr_mod, "_CAPI_VideoReaderSetRoi")
        except ImportError:
            _has_roi_api = False
        # 本项目 ROI 语义：闭合框 (x1,y1,x2,y2) → decord 半开 +1
        roi = (self._roi[0], self._roi[1], self._roi[2] + 1, self._roi[3] + 1)
        roi_kw = {"roi": roi} if _has_roi_api else {}
        backend = (self._decode_backend or "auto").lower()
        vr = None
        label = "CPU"
        if backend in ("auto", "nvdec"):
            try:
                from decord import gpu as _g
                # GPU 也支持 output_format='gray'（decord ≥0.7.9，直出 Y
                # 平面与 CPU GRAY8 逐位一致）；旧版忽略该参数仍输出 RGB。
                vr = VideoReader(str(self._video_path), ctx=_g(0),
                                 output_format='gray' if self._gray_output
                                 else 'rgb', **roi_kw)
                label = "GPU"
            except Exception:
                vr = None
                if backend == "nvdec":
                    logger.warning("NVDEC 解码不可用，回退 CPU")
        if vr is None:
            vr = VideoReader(str(self._video_path), ctx=_cpu(0),
                             output_format='gray' if self._gray_output
                             else 'rgb', **roi_kw)
            label = "CPU"
        self._backend = f"decord/{label}"
        return vr

    def _hybrid_env_enabled(self) -> bool:
        """实验开关 config.HYBRID_DECODE_ENV（RVTOL_HYBRID_DECODE）。

        1/true/yes/on（大小写不敏感）为开启，默认关闭。开启后 GPU 模式
        （auto / nvdec）内部改走 CPU+NVDEC 双解码器并行；不暴露给 GUI/CLI。
        """
        _v = _os.environ.get(config.HYBRID_DECODE_ENV, "").strip().lower()
        return _v in ("1", "true", "yes", "on")

    def _is_hybrid(self) -> bool:
        """是否启用 CPU+NVDEC 混合并行解码。

        显式传 decode_backend='cpu+nvdec'/'hybrid'（旧版程序化用法）恒为
        混合；否则需 HYBRID_DECODE_ENV 开启 且 后端为 GPU 系（auto /
        nvdec）——即"混合是 GPU 模式的实验变体"，cpu 不受影响。
        """
        _b = (self._decode_backend or "auto").lower()
        if _b in HYBRID_BACKEND_ALIASES:
            return True
        return self._hybrid_env_enabled() and _b in ("auto", "nvdec")

    def _hybrid_split(self) -> float:
        """混合解码的 CPU 段帧数比例（env RVTOL_HYBRID_SPLIT 优先）。

        保守分法（默认 config.HYBRID_CPU_SPLIT）：只把 CPU 软解当"增量"。
        AV1 特判：CPU 软解 AV1 极耗核且与 GPU 段并发竞争反而拖慢 GPU 吞吐
        → 返回 0（CPU 段空，等效纯 GPU；_open_hybrid_vrs 已按纯 GPU 分支走，
        此返回为其他路径的防御性兜底）。
        """
        if getattr(self, "_hybrid_codec", "") == "av1":
            return 0.0
        _env = _os.environ.get("RVTOL_HYBRID_SPLIT")
        if _env:
            try:
                v = float(_env)
                if 0.0 < v < 1.0:
                    return v
            except ValueError:
                pass
        return float(config.HYBRID_CPU_SPLIT)

    def load_rgb_crop(self, frame: int):
        """惰性解码一帧原始 RGB ROI（段 review 预览用）。

        生产解码器是 gray 输出（Y 平面，无法恢复色彩）；此方法单独维护
        一个 CPU RGB reader，按需 seek 到目标帧取 ROI。调用方（GUI review）
        串行使用，失败返回 None 由调用方回退到灰度显示。
        """
        x1, y1, x2, y2 = self._roi
        try:
            if self._rgb_vr is None:
                from decord import VideoReader, cpu as _cpu
                try:
                    import decord.video_reader as _vr_mod
                    _has_roi_api = hasattr(_vr_mod, "_CAPI_VideoReaderSetRoi")
                except ImportError:
                    _has_roi_api = False
                roi_half = (x1, y1, x2 + 1, y2 + 1)
                roi_kw = {"roi": roi_half} if _has_roi_api else {}
                self._rgb_vr = VideoReader(str(self._video_path), ctx=_cpu(0),
                                           output_format="rgb", **roi_kw)
                self._rgb_roi_half = roi_half
            self._rgb_vr.seek_accurate(frame)
            crop = self._rgb_vr.next_roi(*self._rgb_roi_half).asnumpy()
            if crop.shape[0] != y2 - y1 + 1 or crop.shape[1] != x2 - x1 + 1:
                # 旧版非 ROI-first 解码器回退全帧 → 子裁剪
                crop = crop[y1:y2 + 1, x1:x2 + 1]
            if crop.ndim == 2:
                crop = crop[..., None]
            if crop.shape[-1] == 1:
                crop = np.repeat(crop, 3, axis=-1)
            elif crop.shape[-1] == 4:
                crop = crop[..., :3]
            return np.ascontiguousarray(crop, dtype=np.uint8)
        except Exception:
            return None

    def _open_hybrid_vrs(self):
        """CPU+NVDEC 混合解码：打开一对 ROI-first 解码器（CPU 前段 + GPU 后段）。

        与 _open_vr 相同 ROI 语义（闭合框 → 半开 +1）。CPU reader 灰度
        输出（sws 转换量 1/3，_gray_seg_batch 直接取通道）；GPU reader
        灰度（decord ≥0.7.9 直出 Y，与 CPU GRAY8 逐位一致）。GPU 不可用
        → 回退单 CPU reader（vr_gpu=None，调用方按纯 CPU 走）。
        AV1 特判：CPU 软解 AV1 极耗核（~330fps）且与 GPU 段并发竞争拖慢
        GPU 吞吐 → 不再打开 CPU reader，直接返回 (vr_gpu, vr_gpu)；调用方
        见 vr_gpu is vr → 置 hybrid=False 走纯 GPU 分支（无队列/线程开销，
        与纯 GPU 完全一致）。_hybrid_split 同步返回 0（防御性，其他路径兜底）。
        返回 (vr_cpu, vr_gpu)。
        """
        from decord import VideoReader, cpu as _cpu
        try:
            import decord.video_reader as _vr_mod
            _has_roi_api = hasattr(_vr_mod, "_CAPI_VideoReaderSetRoi")
        except ImportError:
            _has_roi_api = False
        roi = (self._roi[0], self._roi[1], self._roi[2] + 1, self._roi[3] + 1)
        roi_kw = {"roi": roi} if _has_roi_api else {}
        # 先开 GPU reader：codec 探测用它（AV1 时免开 CPU reader）；
        # GPU 不可用 → 回退单 CPU reader。
        try:
            from decord import gpu as _g
            # GPU reader 与 CPU 同语义：_gray_output 时直出 GRAY8（= Y 平面，
            # 与 CPU swscale GRAY8 逐位一致）→ 分段灰度跨后端统一；
            # 旧 decord（GPU 忽略 output_format）回退 RGB + matmul 灰度。
            vr_gpu = VideoReader(str(self._video_path), ctx=_g(0),
                                 output_format='gray' if self._gray_output
                                 else 'rgb', **roi_kw)
            self._backend = "decord/CPU+NVDEC"
        except Exception:
            logger.warning("NVDEC 解码不可用，CPU+NVDEC 回退纯 CPU")
            self._backend = "decord/CPU"
            return VideoReader(str(self._video_path), ctx=_cpu(0),
                               output_format='gray' if self._gray_output
                               else 'rgb', **roi_kw), None
        try:
            self._hybrid_codec = str(vr_gpu.get_codec() or "").lower()
        except Exception:
            self._hybrid_codec = ""
        if self._hybrid_codec == "av1":
            logger.warning("AV1 视频：CPU 软解与 GPU 并发竞争反而拖慢解码，"
                           "CPU+NVDEC 按纯 GPU 解码（不打开 CPU reader）")
            self._backend = "decord/GPU"
            return vr_gpu, vr_gpu
        return VideoReader(str(self._video_path), ctx=_cpu(0),
                           output_format='gray' if self._gray_output
                           else 'rgb', **roi_kw), vr_gpu

    def _ocr_engine_type(self) -> str:
        """OCR 推理后端：auto/tensorrt → tensorrt（OcrEngine 失败回退 onnx），cpu → onnxruntime。"""
        return "onnxruntime" if (self._ocr_backend or "auto").lower() == "cpu" \
            else "tensorrt"

    def _ocr_num_threads(self) -> int:
        """OCR 推理线程预算：RVTOL_OCR_THREADS env 钩子优先，否则全物理核。

        解码（NVDEC 全卸载 / CPU 下 FFmpeg 帧线程 2 + filter auto 只占
        SMT 份额）不抢物理核，OCR 吃满全部物理核；CPU/GPU 解码后端统一。
        显式参数传入引擎，不污染全局 env。
        """
        from ocr_native import auto_ocr_thread_count
        _env = _os.environ.get("RVTOL_OCR_THREADS")
        if _env:
            return max(1, int(_env))
        return auto_ocr_thread_count()

    # ── 阶段 1：解码 + 特征（diff/清晰度）──
    def _decode_all(self):
        hybrid = self._is_hybrid()
        vr_gpu = None
        if hybrid:
            vr, vr_gpu = self._open_hybrid_vrs()
            if vr_gpu is None:
                hybrid = False  # GPU 不可用 → 已回退纯 CPU
            elif vr_gpu is vr:
                hybrid = False  # AV1 特判：不打开 CPU reader，等效纯 GPU
        else:
            vr = self._open_vr()
        # fps 未指定时从解码器推导（CLI/GUI 无需传 fps）
        if self._fps is None:
            for m in ("get_avg_fps", "get_fps"):
                fn = getattr(vr, m, None)
                if fn is None:
                    continue
                try:
                    self._fps = float(fn())
                    break
                except Exception:
                    self._fps = None
            if not self._fps or self._fps <= 0:
                self._fps = config.DEFAULT_FPS_FALLBACK
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end))
        crops = {}
        grays = {}
        sharp = {}
        t0 = time.perf_counter()
        if hybrid:
            # CPU+NVDEC 混合：两解码线程并行解码各自区间（批量 get_batch），
            # 按序合并（dict 键即帧号，帧序由 frames 列表保证）。
            import threading
            from queue import Queue
            cpu_fis, gpu_fis = _hybrid_ranges(frames, 0, self._hybrid_split())
            q_cpu: Queue = Queue(maxsize=config.HYBRID_QUEUE_SIZE)  # 有界：防某端先解完内存膨胀
            q_gpu = Queue(maxsize=config.HYBRID_QUEUE_SIZE) if gpu_fis else None
            err: list = []
            threads: list = []
            roi_half = (x1, y1, x2 + 1, y2 + 1)
            if cpu_fis:
                t = threading.Thread(target=_decode_range_worker,
                                     args=(vr, cpu_fis, q_cpu, roi_half,
                                           None, err), daemon=True)
                t.start()
                threads.append(t)
            else:
                q_cpu.put(None)
            if q_gpu is not None:
                try:
                    vr_gpu.seek_accurate(gpu_fis[0])
                except Exception as e:  # noqa: BLE001 — 与解码错误同通道回传
                    err.append(e)
                    q_gpu.put(None)
                else:
                    t = threading.Thread(target=_decode_range_worker,
                                         args=(vr_gpu, gpu_fis, q_gpu,
                                               roi_half, None, err),
                                         daemon=True)
                    t.start()
                    threads.append(t)
            for q in (q_cpu, q_gpu):
                if q is None:
                    continue
                for fi, c, g, s, _b in _drain_queue(q):
                    if c.shape[0] != y2 - y1 + 1 \
                            or c.shape[1] != x2 - x1 + 1:
                        # 旧版非 ROI-first 解码器：全帧输出 → 子裁剪
                        c = c[y1:y2 + 1, x1:x2 + 1]
                        g = _gray_seg(c)
                        s = float(g.std())
                    crops[fi] = c
                    grays[fi] = g
                    sharp[fi] = s
            for t in threads:
                t.join()
            if err:
                raise err[0]
        else:
            for k, fi in enumerate(frames):
                c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
                if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                    c = c[y1:y2 + 1, x1:x2 + 1]
                crops[fi] = c
                g = _gray_seg(c)
                grays[fi] = g
                sharp[fi] = float(g.std())
                if k % 500 == 0:
                    self._progress(f"[{self._backend}] 解码: {k}/{len(frames)}",
                                   3 + k / max(len(frames), 1) * 70)
                if k % 100 == 0:
                    self._cancel()
        self.timing["decode"] = time.perf_counter() - t0
        del vr, vr_gpu
        return frames, crops, grays, sharp

    # ── 阶段 2：分段（聚类 diff）──
    def _segment(self, frames, grays):
        t0 = time.perf_counter()
        ths = []
        step = max(1, len(frames) // config.SEG_CALIB_FRAMES)
        for fi in frames[::step][:config.SEG_CALIB_FRAMES]:
            ths.append(_otsu(grays[fi]))
        th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
        self._bin_thresh = th
        prev_b = grays[frames[0]] > th
        edges = []
        for fi in frames[1:]:
            b = grays[fi] > th
            d = prev_b != b
            edges.append(_cluster_win3(d) < self._C)
            prev_b = b
        segs = []
        s = 0
        for i in range(len(frames) - 1):
            if not edges[i]:
                segs.append(frames[s:i + 1])
                s = i + 1
        segs.append(frames[s:])
        self.timing["segment"] = time.perf_counter() - t0
        return segs

    # ── 阶段 3：段值 OCR（每段最清晰代表帧，批量）──
    def _ocr_segments(self, segs, crops, sharp):
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard
        eng = OcrEngine(self._ocr_model, self._ocr_engine_type(),
                        fill_width=self._fill_width,
                        num_threads=self._ocr_num_threads(),
                        progress_cb=lambda msg: self._progress(msg, 2.0))
        seg_vals = []
        rep_frames = []
        t0 = time.perf_counter()
        # 批量：每组 B 个代表帧一次 session.run（TRT 引擎 profile batch 上限
        # 由引擎元数据决定，内部自动分片；B 摊薄预处理/launch 开销）
        B = _ocr_batch_size()
        reps = [max(seg, key=lambda fi: sharp[fi]) for seg in segs]
        for k in range(0, len(segs), B):
            chunk = segs[k:k + B]
            procs = [_preprocess_standard(crops[rep],
                                          force_aspect=self._force_aspect)
                     for rep in reps[k:k + B]]
            results = eng(procs)
            for rep, res in zip(reps[k:k + B], results):
                sv, _rt, _c = extract_speed_value(res)
                seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
                rep_frames.append(rep)
            done = min(k + B, len(segs))
            self._progress(f"[OCR] 段: {done}/{len(segs)}",
                           73 + done / max(len(segs), 1) * 15)
        self.timing["ocr"] = time.perf_counter() - t0
        self._n_segments = len(segs)
        return seg_vals, rep_frames

    # ── 阶段 4：段级检测 + 纠正（薄封装，实现在 seg_correction.py）──
    def _local_bandwidth(self, seg_vals, seg_times):
        return local_bandwidth(seg_vals, seg_times, self._win)

    def _detect(self, seg_vals, seg_times, seg_lens=None):
        return detect_segments(seg_vals, seg_times, seg_lens,
                               med_k=self._med_k, mult=self._mult,
                               detect_floor=self._detect_floor,
                               single_floor=self._single_floor,
                               win=self._win)

    def _correct(self, seg_vals, seg_times, suspect):
        return correct_segments(seg_vals, seg_times, suspect,
                                anchor_max=self._anchor_max,
                                min_dev=self._min_dev)

    def _confidence(self, seg_vals, seg_times, seg_lens=None):
        return confidence_scores(seg_vals, seg_times, seg_lens,
                                 med_k=self._med_k,
                                 detect_floor=self._detect_floor,
                                 conf_jerk_scale=self._conf_jerk_scale,
                                 conf_w_med=self._conf_w_med,
                                 conf_w_jerk=self._conf_w_jerk,
                                 win=self._win)

    def _fill_values(self, seg_vals, seg_times, is_anchor):
        return fill_values(seg_vals, seg_times, is_anchor,
                           anchor_max=self._anchor_max)

    def _dense_correct(self, seg_vals, seg_times, conf):
        return dense_correct(
            seg_vals, seg_times, conf,
            max_speed=self._max_speed, max_accel=self._max_accel,
            fps=self._fps or config.DEFAULT_FPS_FALLBACK,
            dp_anchor_conf=self._dp_anchor_conf,
            dp_deanchor_jerk_min=self._dp_deanchor_jerk_min,
            dp_deanchor_jerk_max=self._dp_deanchor_jerk_max,
            dp_change_threshold=self._dp_change_threshold,
            dp_obs_weight=self._dp_obs_weight,
            dp_accel_weight=self._dp_accel_weight,
            dp_max_dv_cap=self._dp_max_dv_cap,
            dp_anchor_cost=self._dp_anchor_cost,
            anchor_max=self._anchor_max)

    def _dp_run(self, lo, hi, seg_vals, seg_times, is_anchor, fill=None):
        return dp_run(lo, hi, seg_vals, seg_times, is_anchor,
                      self._max_speed, self._max_accel, self._fps or 30.0,
                      dp_obs_weight=self._dp_obs_weight,
                      dp_accel_weight=self._dp_accel_weight,
                      dp_max_dv_cap=self._dp_max_dv_cap,
                      dp_anchor_cost=self._dp_anchor_cost,
                      fill=fill)

    # ── 主入口（流水线：解码∥分段∥段OCR 重叠 → 检测纠正 → CSV）──
    def run(self, output_path):
        t_total = time.perf_counter()
        self._progress("解码+分段+段值OCR...", 2.0)
        frames, segs, seg_vals, rep_frames = self._run_pipelined()
        self._cancel()
        self._progress("检测纠正...", 88.0)
        t_corr = time.perf_counter()
        seg_times = [seg[len(seg) // 2] for seg in segs]
        conf = self._confidence(seg_vals, seg_times,
                                [len(s) for s in segs])
        self._conf_vals = list(conf)       # 供 finalize/flag 判定复用
        corr, self._n_corr = self._dense_correct(seg_vals, seg_times, conf)
        self.timing["correction"] = time.perf_counter() - t_corr
        self.rows = self._build_rows(frames, segs, corr, raw=seg_vals,
                                     conf=conf)
        self._store_run_state(frames, self.crops, segs, seg_vals, rep_frames, corr)
        self._write_csv(self.rows, output_path)
        self.timing["total"] = time.perf_counter() - t_total
        self._progress("完成", 100.0)
        return self.rows

    def _run_pipelined(self):
        """流水线：解码线程增量分段，OCR 线程批处理已闭合段的代表帧。

        解码是 I/O 瓶颈（CPU 占用低），段边界（win3）在解码循环内增量计算，
        段一闭合就把代表帧（最清晰）交给 OCR 工作线程 —— 解码∥OCR 重叠摊薄
        总墙钟。代表帧选择与串行 _segment/_ocr_segments 完全一致（每段 max
        灰度 std），OCR 批 _ocr_batch_size()。cpu+nvdec 时两个解码线程
        （CPU 前段 + GPU 后段）并行填有界队列，消费者按序合并，帧序与单解码器一致。

        返回 (frames, segs, seg_vals, rep_frames)；self.crops = {rep_frame:
        crop}（仅代表帧，供 review 预览，比存全帧省内存）。
        """
        from queue import Queue
        import threading
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard
        from ocr_engine import extract_speed_value

        # ── 打开解码器 + fps ──
        # CPU+NVDEC 混合：CPU reader 前段 + GPU reader 后段并行解码
        # （见 _open_hybrid_vrs）；GPU 不可用已在其中回退纯 CPU。
        _t_open = time.perf_counter()
        hybrid = self._is_hybrid()
        vr_gpu = None
        if hybrid:
            vr, vr_gpu = self._open_hybrid_vrs()
            if vr_gpu is None:
                hybrid = False
        else:
            vr = self._open_vr()
        if self._fps is None:
            for m in ("get_avg_fps", "get_fps"):
                fn = getattr(vr, m, None)
                if fn is None:
                    continue
                try:
                    self._fps = float(fn())
                    break
                except Exception:
                    self._fps = None
            if not self._fps or self._fps <= 0:
                self._fps = config.DEFAULT_FPS_FALLBACK
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end))
        self._prof_end("producer", "open_and_fps", _t_open)

        # ── 阈值校准：缓冲前 N 帧（seek 校准每次 seek_accurate ~30ms，
        # 逐帧 seek 得不偿失；前 N 帧 Otsu 阈值与全片抽样一致）──
        calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
        calib: list = []  # (fi, crop, gray, sharp)
        _t_cal = time.perf_counter()
        for k in range(calib_n):
            _t_p = time.perf_counter()
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            self._prof_end("producer", "calib_decode", _t_p)
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            _t_p = time.perf_counter()
            g = _gray_seg(c)
            self._prof_end("producer", "calib_gray", _t_p)
            calib.append((frames[k], c, g, float(g.std())))
        ths = [_otsu(g) for _fi, _c, g, _s in calib]
        th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
        self._bin_thresh = th
        self._prof_end("producer", "calib_total", _t_cal)

        # ── OCR 工作线程：批处理闭合段代表帧 ──
        q: Queue = Queue(maxsize=max(1, self._buffer_size))  # 有界：OCR 慢时背压解码（防内存膨胀）
        results: dict = {}
        ocr_err: list = []
        ocr_wall = [0.0]

        def ocr_worker() -> None:
            t0 = time.perf_counter()
            try:
                # 实验性 OCR 混合（config.HYBRID_OCR_ENV，默认关）：TRT +
                # ONNX 双引擎并发处理段批。与解码不同，OCR 无状态约束——
                # 结果按段索引聚合，批处理顺序无关 → 实现只需共享一个
                # infer_q、两个推理线程各持一引擎，谁空闲谁取批。
                # 预期收益场景：TRT 可用但 OCR 走 ONNX（OCR=cpu 时
                # ONNX 全物理核线程是瓶颈，混合可压到 TRT 水平）。
                _hybrid_ocr = _os.environ.get(
                    config.HYBRID_OCR_ENV, "").strip().lower() in \
                    ("1", "true", "yes", "on")
                _t_eng = time.perf_counter()
                _engine_progress = lambda msg: self._progress(msg, 2.0)
                if _hybrid_ocr:
                    engines = [
                        OcrEngine(self._ocr_model, "tensorrt",
                                  fill_width=self._fill_width,
                                  num_threads=self._ocr_num_threads(),
                                  progress_cb=_engine_progress),
                        OcrEngine(self._ocr_model, "onnxruntime",
                                  fill_width=self._fill_width,
                                  num_threads=self._ocr_num_threads(),
                                  progress_cb=_engine_progress),
                    ]
                else:
                    engines = [OcrEngine(
                        self._ocr_model, self._ocr_engine_type(),
                        fill_width=self._fill_width,
                        num_threads=self._ocr_num_threads(),
                        progress_cb=_engine_progress)]
                self._prof_end("ocr", "engine_init", _t_eng)
                B = _ocr_batch_size()
                # 预处理（单线程 numpy，持 GIL）与推理（ONNX 多线程 /
                # TRT，session.run 释放 GIL）流水线重叠：主循环攒批预处理，
                # 推理线程消费。原串行 flush（预处理→推理→预处理→…）
                # 让多线程推理空转等单线程预处理；重叠后 OCR 总时长逼近推理本身。
                infer_q: Queue = Queue(maxsize=config.OCR_INFER_QUEUE_SIZE)
                ocr_progress_frac = [0.0]

                def _report_ocr_progress(idx: int, frac: float) -> None:
                    # 按解码进度每 1% 报告一次（段数可达数千，避免刷屏；
                    # 最终段保证报告 100% 位置）
                    if frac - ocr_progress_frac[0] >= 0.01 or frac >= 1.0:
                        ocr_progress_frac[0] = frac
                        # 生产者随段一起携带解码进度 frac（0-1），OCR 完成
                        # 该段后据此推进 58→86 的进度段，反映真实管线位置
                        self._progress(f"[OCR] 段 {idx + 1}",
                                       58.0 + frac * 28.0)

                def infer_worker(eng) -> None:
                    while True:
                        item = infer_q.get()
                        if item is None:  # 哨兵
                            return
                        idxs, reps, procs, fracs = item
                        _t_i = time.perf_counter()
                        res = eng(procs)
                        self._prof_end("ocr", "infer", _t_i)
                        _t_c = time.perf_counter()
                        for idx, rep, r, frac in zip(idxs, reps, res, fracs):
                            sv, _rt, _c = extract_speed_value(r)
                            results[idx] = (
                                int(sv) if sv is not None and sv >= 0
                                else None, rep)
                            _report_ocr_progress(idx, frac)
                        self._prof_end("ocr", "ctc_decode", _t_c)

                infer_threads = [
                    threading.Thread(target=infer_worker, args=(eng,),
                                     daemon=True)
                    for eng in engines]
                for t in infer_threads:
                    t.start()
                b_idx, b_reps, b_crops, b_fracs = [], [], [], []

                def flush() -> None:
                    if not b_idx:
                        return
                    _t_p = time.perf_counter()
                    procs = [_preprocess_standard(c,
                                                  force_aspect=self._force_aspect)
                             for c in b_crops]
                    self._prof_end("ocr", "preprocess", _t_p)
                    infer_q.put((list(b_idx), list(b_reps), procs,
                                 list(b_fracs)))
                    b_idx.clear(); b_reps.clear(); b_crops.clear(); b_fracs.clear()

                while True:
                    _t_w = time.perf_counter()
                    item = q.get()
                    self._prof_end("ocr", "q_get_wait", _t_w)
                    if item is None:  # 哨兵
                        break
                    idx, rep, crop, frac = item
                    b_idx.append(idx); b_reps.append(rep); b_crops.append(crop)
                    b_fracs.append(frac)
                    if len(b_idx) >= B:
                        flush()
                flush()
                for _ in infer_threads:
                    infer_q.put(None)  # 每个推理线程一个哨兵
                for t in infer_threads:
                    t.join()
            except Exception as e:
                ocr_err.append(e)
            finally:
                ocr_wall[0] = time.perf_counter() - t0

        ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
        ocr_thread.start()

        # ── 生产者：批量解码 + 批量特征 + 增量分段 + 发闭合段 ──
        # 批量取帧（get_batch + ROI）摊薄逐帧 CAPI 往返的固定成本，让
        # FFmpeg 帧线程真正并行。帧序由 get_batch 顺序保证
        # （内部 NextFrameImpl 顺序解码，批间无 seek）。特征计算按批
        # 向量化（crops 天然是批量数组，一次大 numpy 自动释放 GIL）。
        # 分段状态机仍逐帧推进（仅索引/比较），行为与逐帧取帧一致。
        DECODE_BATCH = config.DECODE_BATCH_SIZE

        dec_threads: list = []
        dec_err: list = []

        if hybrid:
            # ── CPU+NVDEC 混合：两解码线程并行填两个有界队列，按序消费 ──
            # CPU 解 frames[calib_n:split_pos]（前段），GPU 解
            # frames[split_pos:]（后段，独立 seek 到段首）。跨后端相邻帧
            # 对仅接缝一处（CPU 段末帧 vs GPU 段首帧）：两后端同帧灰度差
            # ±2-3 散布全帧（实测静止帧二值化 XOR ~67、最大窗口和 << C=5），
            # 不产生假边界；完整漏斗门禁（11 错）兜底。
            from queue import Queue as _Queue
            cpu_fis, gpu_fis = _hybrid_ranges(frames, calib_n,
                                              self._hybrid_split())
            cpu_q: _Queue = _Queue(maxsize=config.HYBRID_QUEUE_SIZE)  # 有界：防先解完的一端内存膨胀
            gpu_q = _Queue(maxsize=config.HYBRID_QUEUE_SIZE) if gpu_fis else None
            roi_half = (x1, y1, x2 + 1, y2 + 1)
            if cpu_fis:
                t = threading.Thread(
                    target=_decode_range_worker,
                    args=(vr, cpu_fis, cpu_q, roi_half, th, dec_err,
                          DECODE_BATCH), daemon=True)
                t.start()
                dec_threads.append(t)
            else:
                cpu_q.put(None)
            if gpu_q is not None:
                try:
                    vr_gpu.seek_accurate(gpu_fis[0])
                except Exception as e:  # noqa: BLE001 — 与解码错误同通道回传
                    dec_err.append(e)
                    gpu_q.put(None)
                else:
                    t = threading.Thread(
                        target=_decode_range_worker,
                        args=(vr_gpu, gpu_fis, gpu_q, roi_half, th, dec_err,
                              DECODE_BATCH), daemon=True)
                    t.start()
                    dec_threads.append(t)

            def frame_stream():
                """先产出校准帧（CPU reader），再按序消费 CPU 段队列
                与 GPU 段队列 —— 帧序与单解码器完全一致。"""
                for fi, c, g, s in calib:
                    yield (fi, c, g, s, g > th)
                yield from _drain_queue(cpu_q)
                if gpu_q is not None:
                    yield from _drain_queue(gpu_q)
        else:
            def frame_stream():
                """先产出校准帧，再批量流式解码剩余帧。

                yield (fi, crop, gray, sharp, bin) —— bin 为预计算的二值化。
                """
                for fi, c, g, s in calib:
                    yield (fi, c, g, s, g > th)
                for bstart in range(calib_n, len(frames), DECODE_BATCH):
                    bend = min(bstart + DECODE_BATCH, len(frames))
                    _t_d = time.perf_counter()
                    crops = vr.get_batch(
                        frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1)
                    ).asnumpy()
                    self._prof_end("producer", "decode_batch", _t_d)
                    # 批量特征：一次 (B,H,W,3)@(3,) + std + 比较（大数组，
                    # numpy 释放 GIL，不与 OCR 预处理线程互斥）；gray 输出
                    # (B,H,W,1) 时直接取通道（跳过 matmul）
                    _t_g = time.perf_counter()
                    g = _gray_seg_batch(crops)
                    self._prof_end("producer", "gray_batch", _t_g)
                    _t_s = time.perf_counter()
                    sharp = g.std(axis=(1, 2))
                    self._prof_end("producer", "sharp_batch", _t_s)
                    _t_b = time.perf_counter()
                    bs = g > th
                    self._prof_end("producer", "bin_batch", _t_b)
                    for k, gi in enumerate(range(bstart, bend)):
                        yield (frames[gi], crops[k], g[k], float(sharp[k]),
                               bs[k])

        segs: list = []
        rep_crops: dict = {}
        seg_idx = 0
        s = 0
        rep_frame = frames[0]
        rep_crop = None
        rep_sharp = -1.0
        prev_b = None
        t0 = time.perf_counter()
        consumer_ok = [False]
        try:
            for k, (fi, c, g, sharp, b) in enumerate(frame_stream()):
                if prev_b is not None:
                    d = prev_b != b
                    _t_seg = time.perf_counter()
                    changed = _cluster_win3(d) >= self._C
                    self._prof_end("producer", "segmentation", _t_seg)
                    if changed:
                        # 闭合段 [s..k-1]：发代表帧给 OCR 线程
                        seg = frames[s:k]
                        segs.append(seg)
                        _t_push = time.perf_counter()
                        q.put((seg_idx, rep_frame, rep_crop,
                               k / max(len(frames), 1)))
                        self._prof_end("producer", "q_put_block", _t_push)
                        rep_crops[rep_frame] = rep_crop
                        seg_idx += 1
                        s = k
                        rep_frame = fi; rep_crop = c; rep_sharp = sharp
                    elif sharp > rep_sharp:
                        rep_sharp = sharp; rep_frame = fi; rep_crop = c
                else:
                    rep_frame = fi; rep_crop = c; rep_sharp = sharp
                prev_b = b
                if k % 100 == 0:
                    self._cancel()
                if k % 500 == 0:
                    self._progress(f"[{self._backend}] 解码+分段: {k}/{len(frames)}",
                                   3 + k / max(len(frames), 1) * 55)
            # 闭合最后一段
            seg = frames[s:]
            segs.append(seg)
            _t_push = time.perf_counter()
            q.put((seg_idx, rep_frame, rep_crop, 1.0))
            self._prof_end("producer", "q_put_block", _t_push)
            rep_crops[rep_frame] = rep_crop
            seg_idx += 1
            consumer_ok[0] = True
        finally:
            # decode_s 口径修正：只统计生产者消费流结束（不含 OCR 线程
            # 收尾 join 的尾部时间）；OCR 收尾单列 ocr_tail。
            _t_consume_end = time.perf_counter()
            self.timing["decode"] = _t_consume_end - t0
            self._prof_end("producer", "consumer_total", t0)
            # 解码线程只在正常消费完后 join（消费者异常中断时解码线程
            # 阻塞在有界队列 put 上，daemon 交进程回收，避免 join 挂死）
            if consumer_ok[0]:
                for t in dec_threads:
                    t.join()
            q.put(None)
            ocr_thread.join()
            self.timing["ocr_tail"] = time.perf_counter() - _t_consume_end

        if dec_err:
            raise dec_err[0]
        if ocr_err:
            raise ocr_err[0]
        self.timing["ocr"] = ocr_wall[0]
        self._n_segments = len(segs)
        self.crops = rep_crops
        del vr, vr_gpu
        return frames, segs, [results[i][0] for i in range(seg_idx)], \
            [results[i][1] for i in range(seg_idx)]

    def _store_run_state(self, frames, crops, segs, seg_vals, rep_frames, corr):
        """保存 run() 的中间状态，供 GUI 段级 review / finalize 使用。"""
        self._frames = frames
        self.crops = crops
        self._segs = segs
        self._ocr_vals = list(seg_vals)
        self._corr_vals = list(corr)
        self.segments = [
            {"start": seg[0], "end": seg[-1],
             "frames": list(seg),  # 该段的采样帧列表（review 逐帧绘制用）
             "value": corr[i],
             "ocr_value": seg_vals[i],
             "rep_frame": rep_frames[i],
             "rep_crop": crops.get(rep_frames[i])}
            for i, seg in enumerate(segs)
        ]

    def timing_flat(self) -> dict:
        """展平 timing dict（丢弃嵌套值），兼容 headless/gui_export 调用。"""
        return {k: v for k, v in self.timing.items()
                if isinstance(v, (int, float))}

    def finalize(self, output_path, segment_values=None, pinned_indices=None):
        """从（可能被用户编辑的）段值重建 rows 并重写 CSV。

        segment_values: 与 self.segments 等长的段修正值；None = 用 run() 的
        纠正结果。GUI 段级 review 改值后调用，single-pass 重写输出。
        pinned_indices: 用户手动修正的段索引集合（标 Flag.PINNED），
        覆盖 DP/插值 flag 判定。
        """
        if pinned_indices:
            self._pinned = set(pinned_indices)
        if segment_values is None:
            vals = self._corr_vals
        else:
            vals = list(segment_values)
            if len(vals) != len(self._segs):
                raise ValueError(f"段值数量 {len(vals)} ≠ 段数 {len(self._segs)}")
            # corrected 计数相对原始 OCR 值重算
            self._n_corr = sum(1 for ov, nv in zip(self._ocr_vals, vals)
                               if nv is not None and ov != nv)
            self._corr_vals = vals
        self.rows = self._build_rows(self._frames, self._segs, vals)
        self._write_csv(self.rows, output_path)
        return self.rows

    # ── 阶段 5：构建 rows + 写 CSV ──
    def _build_rows(self, frames, segs, corr, raw=None, conf=None,
                    pinned=None):
        """构建输出行 [frame, dist, speed, flag]。

        flag 按段来源推理（段内帧共享）：
        - pinned（用户手动修正）→ PINNED
        - raw None（OCR 未读出）→ FILL_INTERP
        - 值被改（corr != raw）→ DP_CORRECTED
        - 值未被改但 conf ≥ 锚定阈值（物理验证通过）→ HIGH_TRUST（绝大多数）
        - 其余（未验证的原始值）→ RAW
        """
        if raw is None:
            raw = self._ocr_vals
        if conf is None:
            conf = getattr(self, "_conf_vals", None)
        if pinned is None:
            pinned = getattr(self, "_pinned", set())
        rows = []
        for i, (seg, val) in enumerate(zip(segs, corr)):
            if i in pinned:
                flag = Flag.PINNED
            elif i < len(raw) and raw[i] is None:
                flag = Flag.FILL_INTERP
            elif i < len(raw) and val is not None and val != raw[i]:
                flag = Flag.DP_CORRECTED
            elif conf is not None and i < len(conf) \
                    and conf[i] >= self._dp_anchor_conf:
                flag = Flag.HIGH_TRUST
            else:
                flag = Flag.RAW
            for fi in seg:
                rows.append([fi, 0.0, val if val is not None else -1, flag])
        rows.sort(key=lambda r: r[0])
        # 距离积分
        dist = 0.0
        prev = None
        for row in rows:
            if prev is not None and row[2] >= 0:
                dt = (row[0] - prev[0]) / self._fps
                dist += prev[2] / config.MPS_TO_KMH * dt
            row[1] = round(dist, 2)
            prev = row
        return rows

    def _write_csv(self, rows, output_path):
        with Path(output_path).open("w", newline="", encoding="utf-8-sig") as fh:
            r = self._roi
            # 头行用 fh.write 直接写（csv.writer 会给含逗号的注释加引号，
            # 行首变 " 导致 parse_csv_header 解析失败 —— GUI 导入 CSV 兼容性）
            fh.write(f"# RaceVideoToLog v{config.__version__}\n")
            fh.write(f"# video={self._video_path.name}, fps={self._fps:.3f}\n")
            fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}"
                     f", frame_start={self._frame_start or ''}"
                     f", frame_end={self._frame_end or ''}\n")
            fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}"
                     f", force_aspect={self._force_aspect}, fill_width={self._fill_width}\n")
            fh.write(f"# backend={self._backend}, model={self._ocr_model}\n")
            fh.write(f"# segments={self._n_segments}, corrected={self._n_corr}\n")
            tstr = ", ".join(f"{k}={v:.2f}" for k, v in self.timing.items())
            if tstr:
                fh.write(f"# timing: {tstr}\n")
            # 数据行用 csv.writer（int 帧号/距离/速度/flag，对齐旧格式）
            w = csv.writer(fh)
            for row in rows:
                w.writerow((f"{int(row[0])}", f"{row[1]:.2f}",
                            f"{int(row[2])}", str(row[3])))
