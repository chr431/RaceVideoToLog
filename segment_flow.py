"""分段流水线（生产）：解码+分段+段值OCR 流水线化 → 段级纠错 → CSV。

与 ProcessingPipeline 同接口（run/finalize），CLI 用 --segment 切换。
分段 OCR 大幅减少调用（36-64%），解码（I/O 瓶颈）与段值 OCR 线程重叠
摊薄墙钟（test6 26.1s → 16.8s）。

算法（experiment-binary-ocr 分支验证）：
- diff 分段：聚类判别（max 3×3 窗口和 < C ⇒ 显示未变），解码循环内增量计算
- 段值：每段最清晰代表帧 OCR（sharpness=灰度std），OCR 线程批处理闭合段
- 段级检测：中值滤波（跟随弯曲，误读=尖峰被中值剔除）
- 段级纠正：可信锚点插值（锚点距离上界，跳过曲线正确段）
- 解码：decode_backend='cpu+nvdec' 时 CPU+NVDEC 双解码器并行（CPU 前段 +
  GPU 后段，见 _open_hybrid_vrs），解码阶段 wall 砍半（test5 7.4s→3.7s）

注意：_decode_all/_segment/_ocr_segments/_detect/_correct 是串行参考路径
（仅 tools/ 与测试使用），生产 run() 走 _run_pipelined + _dense_correct。
"""
from __future__ import annotations
import csv
import logging
import os as _os
import re
import time
from pathlib import Path

import numpy as np

import config
from constants import Flag
from monitor import STAGE
from ocr_engine import extract_speed_value
from video_utils import _gray, _GRAY_W

logger = logging.getLogger("RaceVideoToLog.segment_flow")


def _gray_batch(crops: np.ndarray) -> np.ndarray:
    """批量灰度：(B,H,W,3) → (B,H,W)；decord gray 输出 (B,H,W,1) 直接取通道。

    gray_output 模式下（CPU 解码）crops 已是 1 通道，跳过 matmul。
    """
    if crops.shape[-1] == 1:
        return crops[..., 0]
    return (crops.astype(np.float32) @ _GRAY_W).astype(np.uint8)


def _seg_gamma() -> float:
    """分段/代表帧选择的灰度 gamma（实验钩子）。

    RVTOL_SEG_GAMMA env > config.SEG_GAMMA。默认 0 = 现状 raw 灰度
    （锁定基线）；>0 时对灰度做 255*(g/255)^gamma 增强后再分段与选代表帧
    —— 与 OCR 正式预处理（gray + gamma 2.0）对齐的对照实验用。
    """
    _env = _os.environ.get("RVTOL_SEG_GAMMA")
    if _env:
        try:
            return float(_env)
        except ValueError:
            pass
    return float(config.SEG_GAMMA)


def _apply_gamma(g: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return g
    return (255.0 * np.power(g.astype(np.float32) / 255.0, gamma)
            ).astype(np.uint8)


def _gray_seg(crop: np.ndarray) -> np.ndarray:
    """分段用灰度：raw（默认）或 gamma 增强（RVTOL_SEG_GAMMA / SEG_GAMMA）。"""
    return _apply_gamma(_gray(crop), _seg_gamma())


def _gray_seg_batch(crops: np.ndarray) -> np.ndarray:
    """批量分段灰度（_gray_seg 的批量版，含 gamma 钩子）。"""
    return _apply_gamma(_gray_batch(crops), _seg_gamma())


def _otsu(g: np.ndarray) -> int:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = int(g.size)
    st = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = 127
    vmax = -1.0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb = sb / wb
        mf = (st - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def _cluster_win3(diff: np.ndarray) -> float:
    """最大 3×3 窗口变化像素和 —— 聚类判别的廉价代理（纯 numpy，无 scipy）。

    原 scipy.ndimage.label 连通分量对 test6 23k 边贡献 ~2.3s；且 scipy 非
    pyproject 依赖，PyInstaller 打包会连带整个 scipy 增肥 exe。本实现用
    6 次切片错位累加求最大 3×3 窗口和（越界按 0），16µs/边，数值与
    uniform_filter 逐位一致（含边界，500 随机掩码最大差 0）。
    语义：真实数字变化必然产生 ≥5 像素连成 3×3 的密集簇（实测变帧恒=9）；
    噪声孤立像素的最大窗口和 < 5。C=5 下 test/test5/test6 0 漏检且段数更少。
    """
    if not diff.any():
        return 0.0
    s = diff.astype(np.int32)
    # 行向 3 列和（左右越界 0）
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    # 列向 3 行和（上下越界 0）
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


HYBRID_BACKEND_ALIASES: tuple[str, ...] = ("cpu+nvdec", "hybrid")


def _hybrid_ranges(frames: list, calib_n: int,
                   split_ratio: float) -> tuple[list, list]:
    """CPU+NVDEC 混合解码区间切分：(cpu_fis, gpu_fis)，无重叠全覆盖。

    split_pos = max(calib_n, int(len(frames)*split_ratio))：CPU 解
    frames[calib_n:split_pos]（前段，接在校准帧之后），GPU 解
    frames[split_pos:]（后段）。跨后端相邻帧对仅接缝一处，重叠 0 帧。
    """
    split_pos = min(len(frames),
                    max(calib_n, int(len(frames) * split_ratio)))
    return frames[calib_n:split_pos], frames[split_pos:]


def _decode_range_worker(vr, fis, q, roi, th, err, batch: int = 16):
    """批量解码帧区间 [fis] → (fi, crop, gray, sharp, bin) 入队。

    CPU+NVDEC 混合解码的解码线程体：_run_pipelined 传 th（分段阈值，
    bin 参与增量分段）；_decode_all 传 th=None（bin=None 不用，参考
    路径）。批量特征（gray/std/二值化）在解码线程内向量化完成，与
    消费者分段线程重叠。异常记入 err 并放哨兵，保证消费者不被卡死。
    roi 为半开区间。
    """
    try:
        for bstart in range(0, len(fis), batch):
            bend = min(bstart + batch, len(fis))
            crops = vr.get_batch(fis[bstart:bend], roi=roi).asnumpy()
            g = _gray_seg_batch(crops)
            sharp = g.std(axis=(1, 2))
            bs = (g > th) if th is not None else None
            for k, fi in enumerate(fis[bstart:bend]):
                b = None if bs is None else bs[k]
                q.put((fi, crops[k], g[k], float(sharp[k]), b))
    except Exception as e:  # noqa: BLE001 — 经 err 回传主线程 raise
        err.append(e)
    finally:
        q.put(None)


def _drain_queue(q):
    """按序消费解码队列直到哨兵（None）。"""
    while True:
        item = q.get()
        if item is None:
            return
        yield item


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
        self._ocr_model = config.DEFAULT_OCR_MODEL  # 唯一模型 v6_small（v2.14 移除选择）
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
        # ≥0.7.8）：直出 Y 平面（与 CPU swscale GRAY8 逐位一致），分段灰度
        # 跨后端统一；跳过 RGB→灰转换与 Python 侧 matmul
        self._gray_output = gray_output
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.rows: list = []
        self.timing: dict = {}
        # 段级 review / finalize 支持（run 后填充）
        self.segments: list[dict] = []   # [{start,end,value,rep_frame,rep_crop}]
        self.crops: dict = {}            # 解码的 ROI 帧（review 懒加载预览用）
        self._segs: list = []
        self._frames: list = []
        self._ocr_vals: list = []
        self._conf_vals: list = []        # 每段置信度（run 后填充，flag 判定用）
        self._pinned: set = set()          # 用户手动修正的段索引（finalize 时设）

    def _open_vr(self):
        """按 decode_backend 打开 decord 解码器（auto/cpu/nvdec）。

        auto: 尝试 GPU (NVDEC) 失败回退 CPU。cpu: 强制 CPU。
        nvdec: 强制 GPU（失败回退 CPU 并警告）。替代旧 DECORD_FORCE_CPU env。
        cpu+nvdec: 走 _open_hybrid_vrs（双解码器并行），不经过本方法。

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
                # GPU 也支持 output_format='gray'（decord ≥0.7.8，直出 Y
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

    def _is_hybrid(self) -> bool:
        """decode_backend == 'cpu+nvdec'（CPU+NVDEC 混合并行解码）。"""
        return (self._decode_backend or "auto").lower() in HYBRID_BACKEND_ALIASES

    def _hybrid_split(self) -> float:
        """混合解码的 CPU 段帧数比例（env RVTOL_HYBRID_SPLIT 优先）。

        实测 CPU ROI-first 解码 ~1260fps vs GPU ~1000fps → CPU 拿 55%
        时两端同时完成，wall = max(两边) ≈ 单解码器一半。OCR 满负荷时
        CPU 解码与 OCR 抢核减速（GPU 段不受影响），可调低让 GPU 多拿。
        """
        _env = _os.environ.get("RVTOL_HYBRID_SPLIT")
        if _env:
            try:
                v = float(_env)
                if 0.0 < v < 1.0:
                    return v
            except ValueError:
                pass
        return float(config.HYBRID_CPU_SPLIT)

    def _open_hybrid_vrs(self):
        """CPU+NVDEC 混合解码：打开一对 ROI-first 解码器（CPU 前段 + GPU 后段）。

        与 _open_vr 相同 ROI 语义（闭合框 → 半开 +1）。CPU reader 灰度
        输出（sws 转换量 1/3，_gray_seg_batch 直接取通道）；GPU reader
        RGB（GPU kernel 只算 ROI 窗口）。GPU 不可用 → 回退单 CPU reader
        （vr_gpu=None，调用方按纯 CPU 走）。返回 (vr_cpu, vr_gpu)。
        """
        from decord import VideoReader, cpu as _cpu
        try:
            import decord.video_reader as _vr_mod
            _has_roi_api = hasattr(_vr_mod, "_CAPI_VideoReaderSetRoi")
        except ImportError:
            _has_roi_api = False
        roi = (self._roi[0], self._roi[1], self._roi[2] + 1, self._roi[3] + 1)
        roi_kw = {"roi": roi} if _has_roi_api else {}
        vr_cpu = VideoReader(str(self._video_path), ctx=_cpu(0),
                             output_format='gray' if self._gray_output
                             else 'rgb', **roi_kw)
        vr_gpu = None
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
        return vr_cpu, vr_gpu

    def _ocr_engine_type(self) -> str:
        """OCR 推理后端：auto/tensorrt → tensorrt（OcrEngine 失败回退 onnx），cpu → onnxruntime。"""
        return "onnxruntime" if (self._ocr_backend or "auto").lower() == "cpu" \
            else "tensorrt"

    def _ocr_num_threads(self) -> int:
        """OCR 推理线程预算：RVTOL_OCR_THREADS env 钩子优先，否则全物理核。

        根本性解决"CPU 满负荷抢核变慢"：解码（NVDEC 全卸载 / CPU 下
        FFmpeg 帧线程 2 + filter auto 只占 SMT 份额）不抢物理核，OCR
        吃满全部物理核——实测 16C32T：GPU 解码 8→16 线程 11.3s→9.0s、
        CPU 解码 8→16 线程 12.8s→9.5s，满负荷正收益；超过物理核
        （超线程）不再提升，故封顶。显式参数传入引擎，不污染全局 env。
        """
        from ocr_native import auto_ocr_thread_count
        _env = _os.environ.get("RVTOL_OCR_THREADS")
        if _env:
            return max(1, int(_env))
        gpu = getattr(self, "_backend", "") == "decord/GPU"
        return auto_ocr_thread_count(gpu)

    # ── 阶段 1：解码 + 特征（diff/清晰度）──
    def _decode_all(self):
        hybrid = self._is_hybrid()
        vr_gpu = None
        if hybrid:
            vr, vr_gpu = self._open_hybrid_vrs()
            if vr_gpu is None:
                hybrid = False  # GPU 不可用 → 已回退纯 CPU
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
                self._fps = 30.0
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
            cpu_fis, gpu_fis = _hybrid_ranges(frames, 0,
                                              self._hybrid_split())
            q_cpu: Queue = Queue(maxsize=8)  # 有界：防某端先解完内存膨胀
            q_gpu = Queue(maxsize=8) if gpu_fis else None
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
        step = max(1, len(frames) // 50)
        for fi in frames[::step][:50]:
            ths.append(_otsu(grays[fi]))
        th = int(np.median(ths)) if ths else 127
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
                        num_threads=self._ocr_num_threads())
        seg_vals = []
        rep_frames = []
        t0 = time.perf_counter()
        # 批量：每组 B 个代表帧一次 session.run（TRT 引擎 profile batch 上限 6，
        # 内部自动分片；B=16 摊薄预处理/launch 开销）
        B = 16
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
            if k % 1000 == 0:
                self._progress(f"[OCR] 段: {k}/{len(segs)}",
                               73 + k / max(len(segs), 1) * 15)
        self.timing["ocr"] = time.perf_counter() - t0
        self._n_segments = len(segs)
        return seg_vals, rep_frames

    # ── 阶段 4：段级检测 + 纠正 ──
    def _local_bandwidth(self, seg_vals, seg_times):
        """每段局部带宽：帧窗口内相邻差绝对值的中位数（未加 floor）。

        _detect 与 _confidence 共用（两者原为逐行相同的重复循环）。
        """
        n = len(seg_vals)
        if n >= 2:
            gaps = np.diff(seg_times)
            med_gap = float(np.median(gaps)) if len(gaps) else 1.0
        else:
            med_gap = 1.0
        win_frames = min(self._win * max(med_gap, 1.0), 120.0)
        st = np.asarray(seg_times, dtype=np.float64)
        bw_raw = [0.0] * n
        for i in range(n):
            ti = seg_times[i]
            lo = int(np.searchsorted(st, ti - win_frames, side="left"))
            hi = int(np.searchsorted(st, ti + win_frames, side="right"))
            dvs = [abs(seg_vals[j] - seg_vals[j - 1])
                   for j in range(lo + 1, hi)
                   if seg_vals[j] is not None and seg_vals[j - 1] is not None]
            bw_raw[i] = float(np.median(dvs)) if dvs else 0.0
        return bw_raw

    def _detect(self, seg_vals, seg_times, seg_lens=None):
        """中值滤波检测：平滑值曲线（跟随弯曲），误读=尖峰被中值剔除。

        对每段 i，smoothed = 局部非 None 值的中位数（段索引窗口 ±med_k）。
        正确段贴合中值（偏差 ≤ 局部带宽），误读尖峰偏差大。门限 =
        max(局部相邻差中位数, floor) × mult。边缘段（左右一侧无上下文）
        不 flag —— 中值在单调上升/下降区滞后，视频起止的低/高速段会被窗口
        拉偏误判（test2 起始 5→8→12 回归源）。None 段恒 suspect。

        单帧段（seg_lens[i]==1）用更紧 floor：误读率 4.2% vs 多帧 0.3%
        （12.6×，80% 误读是单帧段——过渡/模糊帧难 OCR 又易成单帧段），
        平缓区门限降到 4 抓小偏差误读，弯曲区仍按实际带宽放宽。
        """
        n = len(seg_vals)
        bw_raw = self._local_bandwidth(seg_vals, seg_times)
        suspect = [False] * n
        for i in range(n):
            if seg_vals[i] is None:
                suspect[i] = True
                continue
            lo = max(0, i - self._med_k)
            hi = min(n, i + self._med_k + 1)
            nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
            if len(nbrs) < 3:
                suspect[i] = True
                continue
            lefts = any(seg_vals[j] is not None for j in range(lo, i))
            rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
            if not (lefts and rights):
                continue
            med = float(np.median(nbrs))
            floor = self._single_floor if (seg_lens and seg_lens[i] == 1) \
                else self._detect_floor
            if abs(seg_vals[i] - med) > max(bw_raw[i], floor) * self._mult:
                suspect[i] = True
        return suspect

    def _correct(self, seg_vals, seg_times, suspect):
        """锚点插值纠正：suspect/None 段取最近可信锚点线性插值。

        anchor_max 限锚点帧距离：近锚点（≤anchor_max 帧）才插值，防远锚点
        跨弯曲区误插值（低 min_dev 下过度纠正的回归源）。None 段恒插值。
        """
        out = list(seg_vals)
        n_corr = 0
        for i in range(len(seg_vals)):
            if seg_vals[i] is None:
                # None 段（OCR 未读出）→ 必须插值，否则帧输出 -1
                pass
            elif not suspect[i]:
                continue
            ti = seg_times[i]
            la = None
            for j in range(i - 1, -1, -1):
                if not suspect[j] and seg_vals[j] is not None:
                    if ti - seg_times[j] <= self._anchor_max:
                        la = j
                    break
            ra = None
            for j in range(i + 1, len(seg_vals)):
                if not suspect[j] and seg_vals[j] is not None:
                    if seg_times[j] - ti <= self._anchor_max:
                        ra = j
                    break
            interp = None
            if la is not None and ra is not None:
                span = seg_times[ra] - seg_times[la]
                frac = (ti - seg_times[la]) / span if span > 1e-3 else 0.5
                interp = seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac
            elif la is not None:
                interp = seg_vals[la]
            elif ra is not None:
                interp = seg_vals[ra]
            if interp is not None:
                if seg_vals[i] is None or abs(interp - seg_vals[i]) > self._min_dev:
                    out[i] = round(interp)
                    n_corr += 1
        return out, n_corr

    def _confidence(self, seg_vals, seg_times, seg_lens=None):
        """中值偏差 + 急动度加权置信度 [0,100]（门控急动度）。

        med_score = 100·exp(-dev/bw)：贴合曲线程度。**门控**：贴合曲线
        （med_score ≥ 50）的段 conf 直接取中值分——急动度会被邻居误读
        污染，贴合曲线的正确段不应被拉低（这是 conf 准确度的关键修复，
        实测 145 vs 中值 144 的正确段此前因污染 conf=50 不锚定被 DP 拖走）。
        偏离曲线的段才让急动度参与：刹车（平滑）高分、误读（尖锐）低分。
        边缘段保守 100；None 段 0（必纠正）。
        """
        n = len(seg_vals)
        bw_raw = self._local_bandwidth(seg_vals, seg_times)
        conf = [0.0] * n
        for i in range(n):
            if seg_vals[i] is None:
                conf[i] = 0.0
                continue
            lo = max(0, i - self._med_k)
            hi = min(n, i + self._med_k + 1)
            nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
            if len(nbrs) < 3:
                conf[i] = 30.0
                continue
            lefts = any(seg_vals[j] is not None for j in range(lo, i))
            rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
            if not (lefts and rights):
                conf[i] = 100.0
                continue
            med = float(np.median(nbrs))
            dev = abs(seg_vals[i] - med)
            bw = max(bw_raw[i], self._detect_floor)
            med_score = 100.0 * np.exp(-dev / bw)
            # 贴合曲线 → 直接中值分（忽略被污染的急动度）
            if med_score >= 50.0:
                conf[i] = med_score
                continue
            # 偏离曲线 → 急动度分辨刹车 vs 误读
            jl = seg_vals[i - 1] if i - 1 >= 0 else None
            jr = seg_vals[i + 1] if i + 1 < n else None
            if jl is not None and jr is not None:
                jerk = abs(jr - 2 * seg_vals[i] + jl)
                jerk_score = 100.0 * np.exp(-jerk / self._conf_jerk_scale)
                conf[i] = (self._conf_w_med * med_score
                           + self._conf_w_jerk * jerk_score)
            else:
                conf[i] = med_score
        return conf

    def _fill_values(self, seg_vals, seg_times, is_anchor):
        """每段的局部锚点插值（最近左右锚点的时间线性插值）。

        ±10 窗口的中值在加速斜坡区被斜坡污染（实测段48 中值 84.5 vs 局部
        87）。非锚点段（suspect，可能是误读）的观测目标用锚点插值——贴合
        局部曲线，而非错误 raw 或污染的窗口。
        """
        n = len(seg_vals)
        fill: list = [None] * n
        for i in range(n):
            la = None
            for j in range(i - 1, -1, -1):
                if is_anchor[j] and seg_vals[j] is not None:
                    if seg_times[i] - seg_times[j] <= self._anchor_max:
                        la = j
                    break
            ra = None
            for j in range(i + 1, n):
                if is_anchor[j] and seg_vals[j] is not None:
                    if seg_times[j] - seg_times[i] <= self._anchor_max:
                        ra = j
                    break
            if la is not None and ra is not None:
                span = seg_times[ra] - seg_times[la]
                frac = (seg_times[i] - seg_times[la]) / span if span > 1e-3 else 0.5
                fill[i] = (seg_vals[la]
                           + (seg_vals[ra] - seg_vals[la]) * frac)
            elif la is not None:
                fill[i] = seg_vals[la]
            elif ra is not None:
                fill[i] = seg_vals[ra]
        return fill

    def _dense_correct(self, seg_vals, seg_times, conf):
        """段级稠密格点 DP 纠正（对齐旧 viterbi_dense，无 ref）。

        锚点 = conf ≥ SEG_DP_ANCHOR_CONF 的段（门控 conf 后正确段高分
        可靠锚定）。其余段跑 DP：观测 = 纯惩罚偏离 raw（重 OCR 已删 → ref
        删除；观测的意义是惩罚改动，防把正确的改错），转移 = 加速度约束。
        """
        n = len(seg_vals)
        out = list(seg_vals)
        n_corr = 0
        # 无效 raw（None/0）段的填充目标：局部锚点插值（非 ±10 中值，
        # 后者在加速斜坡区被污染）
        is_anchor = [c >= self._dp_anchor_conf and v is not None
                     for c, v in zip(conf, seg_vals)]
        # 孤立尖峰豁免（A4）：conf∈[20,50) 的锚定段若 jerk（二阶差分）中等
        # （孤立尖峰误读特征；真刹车 jerk≈0、丢位邻居污染 jerk≥80）→ 解除
        # 锚定交给 DP，防误读被锚定保留（实测 13→12 零误改，参数见 config）
        if self._dp_deanchor_jerk_min > 0:
            for i in range(1, n - 1):
                if not is_anchor[i] or conf[i] >= 50.0 \
                        or seg_vals[i] is None:
                    continue
                jl, jr = seg_vals[i - 1], seg_vals[i + 1]
                if jl is None or jr is None:
                    continue
                jerk = abs(jr - 2 * seg_vals[i] + jl)
                if self._dp_deanchor_jerk_min <= jerk \
                        <= self._dp_deanchor_jerk_max:
                    is_anchor[i] = False
        fill = self._fill_values(seg_vals, seg_times, is_anchor)
        i = 0
        while i < n:
            if is_anchor[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and not is_anchor[j + 1]:
                j += 1
            lo = i - 1 if i > 0 else i
            hi = j + 1 if j + 1 < n else j
            if lo != hi or not is_anchor[lo]:
                path = self._dp_run(lo, hi, seg_vals, seg_times, is_anchor,
                                    fill)
                for k in range(hi - lo + 1):
                    idx = lo + k
                    v = seg_vals[idx]
                    val = float(path[k])
                    if v is None:
                        out[idx] = round(val)
                        n_corr += 1
                    elif abs(val - v) > self._dp_change_threshold:
                        out[idx] = round(val)
                        n_corr += 1
            i = j + 1
        return out, n_corr

    def _dp_run(self, lo, hi, seg_vals, seg_times, is_anchor, fill=None):
        """稠密 DP：观测=罚偏离 raw（无效 raw 填向局部锚点插值），转移=加速度约束。"""
        V = int(self._max_speed) + 1
        grid = np.arange(V, dtype=np.float64)
        n = hi - lo + 1
        obs_list = []
        for k in range(lo, hi + 1):
            v = seg_vals[k]
            if is_anchor[k]:
                o = np.full(V, np.inf)
                if v is not None and 0 <= v < V:
                    o[int(round(v))] = self._dp_anchor_cost
                obs_list.append(o)
            else:
                o = np.full(V, self._dp_obs_weight)
                # 非锚点（suspect）：填向局部锚点插值（fill）——错误 raw 不可信，
                # 观测目标应是曲线；无 fill 时回退到 raw。
                r = fill[k] if fill else None
                if r is not None and r > 0:
                    ratio = np.abs(grid - r) / max(1.0, abs(r))
                    np.minimum(1.0, ratio, out=ratio)
                    o = self._dp_obs_weight * ratio
                elif v is not None and v > 0:
                    ratio = np.abs(grid - v) / max(1.0, abs(v))
                    np.minimum(1.0, ratio, out=ratio)
                    o = self._dp_obs_weight * ratio
                obs_list.append(o)
        dp = obs_list[0].copy()
        back = []
        for k in range(1, n):
            fi = lo + k
            dt = (seg_times[fi] - seg_times[fi - 1]) / max(self._fps, 1.0)
            if dt <= 0:
                dt = 1.0 / max(self._fps, 1.0)
            max_dv = min(self._max_accel * dt * 3.6, self._dp_max_dv_cap)
            # O(V²) 转移：T[v] = min_w (dp[w] + accel*max(0,|v-w|-max_dv)^2)
            w = grid[:, None]
            vv = grid[None, :]
            cost = (self._dp_accel_weight
                    * np.maximum(0.0, np.abs(vv - w) - max_dv) ** 2)
            T = dp[:, None] + cost
            best = T.min(axis=0)
            back.append(T.argmin(axis=0))
            dp = best + obs_list[k]
        path = np.zeros(n)
        cur = int(np.argmin(dp))
        for k in range(n - 1, -1, -1):
            path[k] = grid[cur]
            if k > 0:
                cur = int(back[k - 1][cur])
        return path

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
        灰度 std），OCR 批 B=16。cpu+nvdec 时两个解码线程（CPU 前段 +
        GPU 后段）并行填有界队列，消费者按序合并，帧序与单解码器一致。

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
                self._fps = 30.0
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end))

        # ── 阈值校准：缓冲前 50 帧（seek 校准每次 seek_accurate ~30ms，
        # 50 次加 ~1.5s 得不偿失；前 50 帧 Otsu 阈值与全片抽样一致）──
        calib_n = min(50, len(frames))
        calib: list = []  # (fi, crop, gray, sharp)
        for k in range(calib_n):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = _gray_seg(c)
            calib.append((frames[k], c, g, float(g.std())))
        ths = [_otsu(g) for _fi, _c, g, _s in calib]
        th = int(np.median(ths)) if ths else 127
        self._bin_thresh = th

        # ── OCR 工作线程：批处理闭合段代表帧 ──
        q: Queue = Queue(maxsize=max(1, self._buffer_size))  # 有界：OCR 慢时背压解码（防内存膨胀）
        results: dict = {}
        ocr_err: list = []
        ocr_wall = [0.0]

        def ocr_worker() -> None:
            t0 = time.perf_counter()
            try:
                eng = OcrEngine(self._ocr_model, self._ocr_engine_type(),
                                fill_width=self._fill_width,
                                num_threads=self._ocr_num_threads())
                B = 16
                # 预处理（单线程 numpy，持 GIL）与推理（ONNX 8 线程，
                # session.run 释放 GIL）流水线重叠：主循环攒批预处理，
                # 推理线程消费。原串行 flush（预处理→推理→预处理→…）
                # 让 8 线程推理空转等单线程预处理（OCR 6.8s 中预处理
                # ~2s），重叠后 OCR 总时长逼近推理本身。
                infer_q: Queue = Queue(maxsize=4)

                def infer_worker() -> None:
                    while True:
                        item = infer_q.get()
                        if item is None:  # 哨兵
                            return
                        idxs, reps, procs = item
                        res = eng(procs)
                        for idx, rep, r in zip(idxs, reps, res):
                            sv, _rt, _c = extract_speed_value(r)
                            results[idx] = (
                                int(sv) if sv is not None and sv >= 0
                                else None, rep)

                infer_thread = threading.Thread(target=infer_worker,
                                                daemon=True)
                infer_thread.start()
                b_idx, b_reps, b_crops = [], [], []

                def flush() -> None:
                    if not b_idx:
                        return
                    procs = [_preprocess_standard(c,
                                                  force_aspect=self._force_aspect)
                             for c in b_crops]
                    infer_q.put((list(b_idx), list(b_reps), procs))
                    b_idx.clear(); b_reps.clear(); b_crops.clear()

                while True:
                    item = q.get()
                    if item is None:  # 哨兵
                        break
                    idx, rep, crop = item
                    b_idx.append(idx); b_reps.append(rep); b_crops.append(crop)
                    if len(b_idx) >= B:
                        flush()
                flush()
                infer_q.put(None)  # 推理线程哨兵
                infer_thread.join()
            except Exception as e:
                ocr_err.append(e)
            finally:
                ocr_wall[0] = time.perf_counter() - t0

        ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
        ocr_thread.start()

        # ── 生产者：批量解码 + 批量特征 + 增量分段 + 发闭合段 ──
        # 批量取帧（get_batch + ROI）摊薄逐帧 CAPI 往返的固定成本，让
        # FFmpeg 帧线程真正并行（实测 CPU 软解：逐帧 next_roi 780fps vs
        # 批量16 + FFmpeg 8 线程 1247fps）。帧序由 get_batch 顺序保证
        # （内部 NextFrameImpl 顺序解码，批间无 seek）。特征计算按批
        # 向量化（crops 天然是批量数组，一次大 numpy 自动释放 GIL）：
        # 逐帧小 numpy（gray/std/二值化 ~0.3ms/帧）曾是批量后的新瓶颈。
        # 分段状态机仍逐帧推进（仅索引/比较），行为与逐帧取帧一致。
        DECODE_BATCH = 16

        dec_threads: list = []
        dec_err: list = []

        if hybrid:
            # ── CPU+NVDEC 混合：两解码线程并行填两个有界队列，按序消费 ──
            # CPU 解 frames[calib_n:split_pos]（前段），GPU 解
            # frames[split_pos:]（后段，独立 seek 到段首）。跨后端相邻帧
            # 对仅接缝一处（CPU 段末帧 vs GPU 段首帧）：两后端同帧灰度差
            # ±2-3 散布全帧（实测静止帧二值化 XOR ~67、最大窗口和 << C=5），
            # 不产生假边界；完整漏斗门禁（12 错）兜底。
            from queue import Queue as _Queue
            cpu_fis, gpu_fis = _hybrid_ranges(frames, calib_n,
                                              self._hybrid_split())
            cpu_q: _Queue = _Queue(maxsize=8)  # 有界：防先解完的一端内存膨胀
            gpu_q = _Queue(maxsize=8) if gpu_fis else None
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
                    crops = vr.get_batch(
                        frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1)
                    ).asnumpy()
                    # 批量特征：一次 (B,H,W,3)@(3,) + std + 比较（大数组，
                    # numpy 释放 GIL，不与 OCR 预处理线程互斥）；gray 输出
                    # (B,H,W,1) 时直接取通道（跳过 matmul）
                    g = _gray_seg_batch(crops)
                    sharp = g.std(axis=(1, 2))
                    bs = g > th
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
                    if _cluster_win3(d) >= self._C:
                        # 闭合段 [s..k-1]：发代表帧给 OCR 线程
                        seg = frames[s:k]
                        segs.append(seg)
                        q.put((seg_idx, rep_frame, rep_crop))
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
            q.put((seg_idx, rep_frame, rep_crop))
            rep_crops[rep_frame] = rep_crop
            seg_idx += 1
            consumer_ok[0] = True
        finally:
            # 解码线程只在正常消费完后 join（消费者异常中断时解码线程
            # 阻塞在有界队列 put 上，daemon 交进程回收，避免 join 挂死）
            if consumer_ok[0]:
                for t in dec_threads:
                    t.join()
            q.put(None)
            ocr_thread.join()

        if dec_err:
            raise dec_err[0]
        if ocr_err:
            raise ocr_err[0]
        self.timing["decode"] = time.perf_counter() - t0
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
                dist += prev[2] / 3.6 * dt
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
