"""统一处理流水线 — GUI 和 CLI 共用。
运行在调用者线程中（GUI 应在原生 threading.Thread 中调用以避免 QThread 性能损失）。
"""
from __future__ import annotations
import csv
import gc
import os as _os
import logging
import time as _time
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_native import OcrEngine

import numpy as np



def open_decord_vr(video_path, force_cpu: bool = False):
    """Open video with decord — GPU (NVDEC) preferred, CPU fallback.

    Returns (VideoReader, label) where label is ``'GPU'`` or ``'CPU'``.
    Set ``DECORD_FORCE_CPU=1`` in the environment or pass *force_cpu=True*
    to skip GPU even when available.
    """
    from decord import VideoReader as _VR

    _vr = None
    _label = "CPU"
    _force = force_cpu or _os.environ.get("DECORD_FORCE_CPU", "").strip() == "1"

    if not _force:
        try:
            from decord import gpu as _decord_gpu
            _vr = _VR(str(video_path), ctx=_decord_gpu(0))
            _label = "GPU"
        except Exception:
            pass

    if _vr is None:
        try:
            from decord import cpu as _decord_cpu
            _vr = _VR(str(video_path), ctx=_decord_cpu(0))
        except ModuleNotFoundError:
            raise RuntimeError(
                "decord 未安装（需要自建 fork，PyPI 版不支持）。"
                "请运行 setup_venv.bat 或从 chr431/decord 获取发布产物到 _decord_build\\")
        except Exception as _e:
            raise RuntimeError(f"decord 无法打开视频: {_e}")

    return _vr, _label


from ocr_engine import (
    clamp_region, compute_video_hash,
    extract_speed_value, ocr_rec_batch, SpeedObservation, Flag,
    SOURCE_TO_KMH, _parse_int_or_none,
    _reset_backend, _select_backend,
)
import config
from error_detection import detect_errors
from correction import correct_errors
from gpu_setup import get_gpu_backend, get_engine_type
from monitor import STAGE, peak_fields as _peak_fields, read_snapshot as _read_snapshot

def _rss_mb() -> float:
    import os as _os
    try:
        import psutil
        return psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0

def _sum_nbytes(seq) -> int:
    s = 0
    for x in seq:
        if hasattr(x, 'nbytes'): s += x.nbytes
        elif hasattr(x, '__len__') and len(x) == 2:
            if hasattr(x[1], 'nbytes'): s += x[1].nbytes
    return s

logger = logging.getLogger("RaceVideoToLog.pipeline")


import video_utils
from video_utils import _preprocess_standard

ProgressFn = Callable[[str, float], None]


class ProcessingPipeline:
    """统一处理流水线。

    同时支持自动纠错模式和人工辅助（两段式）模式。
    通过 progress_cb 回调报告进度，与 UI 框架解耦。
    """

    def __init__(self, video_path: str | Path, roi: tuple[int, int, int, int],
                    max_speed: float, max_accel: float,
                    frame_div: int, target_h: int, pad: int, buffer_size: int,
                    backend: str, ocr_model: str, speed_format: str,
                    frame_start: str = "", frame_end: str = "",
                    progress_cb: ProgressFn | None = None,
                    cancel_check: "Callable[[], None] | None" = None,
                    log_level: str = "normal",
                final_check: bool = False,
                max_width: int = 0):
        if target_h < 8:
            raise ValueError(f"target_h 必须 >= 8，当前为 {target_h}")
        if pad < 0:
            raise ValueError(f"pad 必须 >= 0，当前为 {pad}")
        self._video_path = Path(video_path)
        self._roi = roi
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._frame_div = frame_div
        self._target_h = target_h
        self._pad = pad
        self._buffer_size = buffer_size
        self._backend = backend
        self._log_level = log_level  # "normal" | "detailed" | "debug"
        self._ocr_model = ocr_model
        # 重 OCR 自动推导（用户不可选）：主 tiny → small（跨模型第二意见，
        # 实测 test5 0.29%→0.04%、test6 1.87%→0.47% 精度 4x）；主 small →
        # None（同引擎重 OCR 净效果为零，实测 test6 0.16% 与无重 OCR 一致）。
        self._reocr_model: str | None = (
            config.DEFAULT_REOCR_MODEL if self._ocr_model == "v6_tiny" else None)
        self._speed_format = speed_format
        self._frame_start = frame_start
        self._cancel_check = cancel_check
        self._frame_end = frame_end
        self._progress = progress_cb
        self._final_check = final_check
        self._max_width = max_width
        self._video_backend_actual: str = ""  # set by _run_ocr: "decord/GPU" or "decord/CPU"
        self._codec: str = ""  # 视频编码（_run_ocr 打开 reader 时记录）

        # 状态
        self._ocr: "OcrEngine | None" = None
        self._reocr: "OcrEngine | None" = None  # 重 OCR 引擎
        self._raw_frames: list[tuple[float, np.ndarray]] = []
        self._observations: list[SpeedObservation] = []
        self._rows: list[list] = []
        self._split_results: dict[int, str] = {}  # frame_idx → split-OCR combined text
        self._pinned: set[int] = set()
        self._segments: list[dict] = []
        # ── 性能计时 ──
        self._timing: dict[str, float] = {}
        # ── 重 OCR 缓存（绑定到 Pipeline 实例生命周期）──
        self._reocr_cache: dict[int, set[float]] = {}
        # 后台 re-OCR 预热线程（主 OCR 阶段提前启动，correction 前接力补帧）
        self._prewarm_thread: "threading.Thread | None" = None
        self._prewarm_queue: "Queue | None" = None
        # Phase 1 流式检测器（主 OCR 批处理时增量计算局部信号）
        self._detector: "object | None" = None
        self._fps: float = 0.0
        self._error_report: "object | None" = None
        self._detection_confidence: list[dict] = []
        self._confidences: list[dict] = []
        self._prewarm_t0: float = 0.0  # 后台预热线程启动时刻（prewarm 计时）
        self.last_output_path: Path | None = None

    # ── 公开只读属性（消除跨模块私有访问）──
    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    @property
    def rows(self) -> list:
        return self._rows

    @property
    def observations(self) -> list:
        return self._observations

    @property
    def raw_frames(self) -> list:
        return self._raw_frames

    @property
    def confidences(self) -> list:
        return self._confidences

    def timing_flat(self) -> dict[str, float]:
        """_timing 中仅含标量的子集（排除 correction_stages 嵌套 dict）。

        CSV 头 / 控制台 / bench 解析只读标量键；嵌套 dict 只进 _summary.json。
        """
        return {k: v for k, v in self._timing.items()
                if isinstance(v, (int, float))}

    def _merge_stage_timing(self) -> None:
        """把 STAGE 计时合并进 _timing（幂等，可多次调用）。

        corr.* 平铺键重挂为嵌套 correction_stages（仅 _summary.json 序列化）。
        """
        res = STAGE.result()
        corr_stages = {k[5:]: v for k, v in res.items() if k.startswith("corr.")}
        if corr_stages:
            self._timing["correction_stages"] = corr_stages
        for k, v in res.items():
            if k.startswith("corr.") or not isinstance(v, (int, float)):
                continue
            self._timing[k] = v

    # ═══════════════ 公开接口 ═══════════════

    def run_auto(self, output_path: str | Path, mode: str = "auto") -> None:
        """纠错流水线 → 写 CSV。

        mode: \"auto\" → full pipeline + force_smooth
              \"manual\" → full pipeline (no force_smooth)
        """
        t_total = _time.perf_counter()
        STAGE.reset()
        self._emit("加载 OCR 引擎...", 1.0)
        with STAGE.stage("engine_load"):
            self._ensure_ocr()
        self._run_ocr()

        if not self._observations:
            raise RuntimeError("未识别到任何速度数据。")

        if self._max_speed <= 0:
            self._emit("跳过纠错（原始OCR输出）...", 95.0)
            self._rows = []
            for obs in self._observations:
                self._rows.append([obs.timestamp, 0.0, int(obs.raw_speed_kmh), Flag.RAW])
            self._integrate_distance()
            self._write_csv(self._rows, Path(output_path))
            self._write_diagnostics(Path(output_path))
        else:
            self._run_correction(Path(output_path), mode=mode)
            for row in self._rows:
                if row[2] > self._max_speed:
                    row[2] = -1
        self._timing["total"] = _time.perf_counter() - t_total
        self._merge_stage_timing()
        logger.info("流水线完成: 总计 %.1fs (%s)",
                        self._timing["total"],
                        ", ".join(f"{k}={v:.1f}s" for k, v in self.timing_flat().items()))
        self._emit("完成", 100.0)


    # ═══════════════ 内部 ═══════════════

    def _emit(self, msg: str, pct: float) -> None:
        if self._cancel_check:
            self._cancel_check()
        if self._progress:
            self._progress(msg, pct)

    def _build_initial_rows(self) -> list:
        """从 observations 构建初始 RAW 行列表。"""
        rows = []
        for obs in self._observations:
            rows.append([obs.timestamp, 0.0, int(obs.raw_speed_kmh), Flag.RAW])
        return rows

    def _ensure_ocr(self) -> "OcrEngine":
        from ocr_native import OcrEngine
        if self._ocr is None:
            _reset_backend()
            self._backend_actual = _select_backend(self._backend)
            _et = get_engine_type()
            if _et == "tensorrt":
                self._emit("加载 TensorRT 引擎...", 1.5)
            self._ocr = OcrEngine(self._ocr_model, _et,
                                  progress_cb=lambda m: self._emit(m, 2.0))
            # 重 OCR 引擎由 __init__ 自动推导（tiny→small / small→None），
            # 主/重模型必然不同 → 独立引擎（同引擎共享路径已无意义）。
            if self._reocr_model:
                self._reocr = OcrEngine(self._reocr_model, _et,
                                        progress_cb=lambda m: self._emit(m, 3.0))
        return self._ocr

    def _correct(self, progress_base: float, progress_span: float,
                 mode: str = "auto") -> None:
        """Two-phase correction: Phase 1 detect, Phase 2 correct."""
        self._rows = self._build_initial_rows()
        t0 = _time.perf_counter()
        n = len(self._rows)
        times = [r[0] / self._fps for r in self._rows]
        self._emit("Phase 1: error detection...", progress_base + 1.0)
        if self._detector is not None:
            # 流式 Phase 1：信号已随主 OCR 增量算完，只补尾部 WINDOW 帧
            from error_detection import ErrorReport
            confidence = self._detector.finalize(self._observations)
            self._error_report = ErrorReport(confidence=confidence)
            self._detector = None
        else:
            self._error_report = detect_errors(
                self._rows, self._observations, times,
                self._max_accel, self._max_speed, fps=self._fps)
        self._detection_confidence = self._error_report.confidence
        self._confidences = self._detection_confidence
        n_low = sum(1 for c in self._detection_confidence if c['score'] < 30)
        n_med = sum(1 for c in self._detection_confidence if 30 <= c['score'] < 70)
        n_high = sum(1 for c in self._detection_confidence if c['score'] >= 70)
        logger.info("Phase 1: %d frames high=%d medium=%d low=%d", n, n_high, n_med, n_low)
        # 后台 re-OCR 预热接力：把最终 correction_frames 全量提交给预热线程
        # （线程内部按 cache 去重，跳过中部已处理帧），correction 阶段不再
        # 串行推理 —— 预热时间完全移出 correction 墙钟。
        if self._prewarm_thread is not None and self._prewarm_queue is not None:
            _all_frames = [i for i, c in enumerate(self._detection_confidence)
                           if c["score"] < 70 and i < len(self._raw_frames)]
            if _all_frames:
                self._prewarm_queue.put(_all_frames)
            self._prewarm_queue.put(None)
            self._prewarm_thread.join()
            self._prewarm_thread = None
            self._prewarm_queue = None
            if self._prewarm_t0:
                STAGE.accumulate("prewarm", _time.perf_counter() - self._prewarm_t0)
                self._prewarm_t0 = 0.0
        self._emit(f"Phase 2: correction ({mode})...", progress_base + 2.0)
        def _prog(done, total):
            if done % max(1, total // 5) != 0 and done != total: return
            self._emit(f"corr: {done}/{total}", progress_base + 2.0 + (done/max(total,1))*progress_span)
        _reocr = self._reocr  # None = 主模型 small 无重 OCR（correct_errors 处理）
        self._rows, self._detection_confidence = correct_errors(
            self._rows, self._observations, self._raw_frames, _reocr,
            self._detection_confidence, times,
            self._max_speed, self._max_accel, mode=mode,
            pinned=self._pinned if self._pinned else None,
            reocr_cache=self._reocr_cache,
            split_results=self._split_results if self._split_results else None,
            fps=self._fps, progress_fn=_prog,
            notes=self._diag_notes if self._diag else None,
            max_width=self._max_width)
        self._confidences = self._detection_confidence
        self._populate_diag_final()
        self._timing["correction"] = _time.perf_counter() - t0

    def finalize(self, output_path: str | Path) -> Path:
        """Write CSV, stage report, diagnostics."""
        out_path = Path(output_path)
        with STAGE.stage("finalize_integrate"):
            self._integrate_distance()
        with STAGE.stage("finalize_csv"):
            self._write_csv(self._rows, out_path)
        with STAGE.stage("finalize_report"):
            self._write_stage_report(out_path)
        with STAGE.stage("finalize_diag"):
            self._write_diagnostics(out_path)
        # Note: _raw_frames NOT cleared here — GUI review dialog
        # needs them after finalize. Cleanup happens in GUI's
        # _finish_export() or when pipeline goes out of scope.
        if self._diag:
            self._diag.clear()
        import gc; gc.collect()
        self.last_output_path = out_path
        return out_path

    def _run_correction(self, output_path: Path, mode: str = "auto") -> None:
        """纠错 + 写 CSV（调用 _correct 后继续）。"""
        with STAGE.stage("correction"):
            self._correct(91.0, 6.0, mode=mode)
        if not self._final_check:
            self.finalize(output_path)

    def _run_ocr(self) -> None:
        """解码 + OCR：producer 解码(decord)/预处理 → consumer 推理。

        Queue 流水线重叠 I/O 与 GPU 推理。decord GPU 优先（NVDEC），
        不可用时自动回退 CPU。OCR 后端由 gpu_setup 自动选择。
        """
        import threading
        from queue import Queue

        ocr = self._ocr
        assert ocr is not None
        speed_format = self._speed_format
        target_h = self._target_h
        pad = self._pad
        _max_width = self._max_width
        frame_step = max(1, self._frame_div)
        t_start = _time.perf_counter()

        # ── 视频源：decord GPU 优先 → CPU 回退 ──
        with STAGE.stage("video_open"):
            _vr, _gpu_label = open_decord_vr(self._video_path)
            try:
                self._codec = _vr.get_codec() or ""
            except Exception:
                pass
            total_video_frames = len(_vr)
            fps = _vr.get_avg_fps(); self._fps = fps
            _first = _vr[0].asnumpy()
            h, w = _first.shape[:2]
            self._video_backend_actual = f"decord/{_gpu_label}"
            logger.info("Video source: decord (%s)", _gpu_label)

        x1, y1, x2, y2 = clamp_region(*self._roi, w, h)
        f_start = _parse_int_or_none(self._frame_start)
        f_end = _parse_int_or_none(self._frame_end)
        _end_limit = f_end if f_end is not None else total_video_frames

        self._raw_frames = []
        buf_size = max(2, self._buffer_size * 2)
        q: Queue = Queue(maxsize=buf_size)
        errors: list[Exception] = []
        _decode_ms = 0.0  # accumulator: decode + preprocess time (ms)

        def _producer() -> None:
            nonlocal _decode_ms
            try:
                _fi = f_start or 0
                if _fi > 0:
                    _vr.seek_accurate(_fi)
                _limit = min(_end_limit, total_video_frames)
                while _fi < _limit:
                    if self._cancel_check and _fi % 10 == 0:
                        self._cancel_check()
                    with STAGE.stage("decode", accumulate=True) as _st:
                        try:
                            # next_roi：GPU 上只拷 ROI 到主机（免全帧 D2H + crop）；
                            # 老 DLL / CPU 回退 next() + 裁切
                            _crop = video_utils.next_frame_roi(
                                _vr, x1, y1, x2 + 1, y2 + 1)
                        except StopIteration:
                            break
                        self._raw_frames.append((_fi, _crop))
                        _proc = _preprocess_standard(_crop, target_h, pad, max_width=_max_width)
                        _decode_ms += _st.elapsed * 1000.0
                    q.put((_fi, _proc))
                    skip = frame_step - 1
                    if skip > 0:
                        _vr.skip_frames(skip)
                    _fi += frame_step
                q.put(None)
            except Exception as e:
                errors.append(e)
                q.put(None)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        observations: list[SpeedObservation] = []
        # Phase 1 流式检测器：consumer 每批推进时增量计算局部信号（与整段
        # detect_errors 逐位一致），Phase 1 墙钟并入主 OCR 阶段。
        from error_detection import IncrementalDetector
        self._detector = IncrementalDetector(fps, self._max_accel,
                                             self._max_speed)
        _detector = self._detector  # _flush_batch 闭包读取
        _collect_diag = self._log_level in ("detailed", "debug")
        diag: list[dict] = []
        done = 0
        _prewarm_started = False
        _inference_ms = 0.0  # accumulator: OCR inference time (ms)
        est_total = (_end_limit - (f_start or 0)) // frame_step
        # ── 批处理识别：一次 session.run 处理多帧，摊销每帧固定开销 ──
        # batch 上限 6：匹配 TRT rec 引擎 profile max_shape 的 batch 维度；
        # CPU 下 batch 6 已接近最优（实测 0.26 ms/帧 vs 单帧 0.85 ms）。
        _batch_size = max(1, config.OCR_FRAME_BATCH)
        _batch: list = []

        def _flush_batch() -> None:
            """对 _batch 内所有帧做一次批识别，追加到 observations/diag。"""
            nonlocal done, _inference_ms, _prewarm_started
            if not _batch:
                return
            items = _batch[:]
            _batch.clear()
            with STAGE.stage("inference", accumulate=True) as _st:
                results = ocr_rec_batch(ocr, [proc for _, proc in items])
            t_ocr = _st.elapsed * 1000.0
            _inference_ms += t_ocr
            t_ocr_each = t_ocr / len(items)
            for (fi, _proc), ocr_result in zip(items, results):
                sv, rt, conf = extract_speed_value(ocr_result)
                if sv is not None and rt is not None:
                    observations.append(SpeedObservation(
                        timestamp=fi,
                        raw_speed_kmh=int(sv * SOURCE_TO_KMH[speed_format]),
                        raw_text=rt,
                        confidence=conf))
                else:
                    observations.append(SpeedObservation(fi, -1, ""))
                _t_p1 = _time.perf_counter()
                _detector.add(fi / fps, observations[-1].raw_speed_kmh)
                STAGE.accumulate("phase1", _time.perf_counter() - _t_p1)
                if _collect_diag:
                    diag.append({
                        "frame": fi,
                        "raw_text": rt or "",
                        "raw_value": sv,
                        "confidence": round(conf, 4),
                        "ocr_time_ms": round(t_ocr_each, 2),
                    })
                done += 1
                if done % 10 == 0 or done <= 3 or done == est_total:
                    pct = 3.0 + (done / max(est_total, 1)) * 87.0
                    _label = f"{self._video_backend_actual} + {get_gpu_backend()}"
                    self._emit(f"[{_label}] OCR: {done}/{est_total}", pct)
                _t_p1 = _time.perf_counter()
                _detector.advance(observations)
                STAGE.accumulate("phase1", _time.perf_counter() - _t_p1)
                # ── re-OCR 预热提前启动（流式 Phase1 信号就绪区 ~50% 时）──
                # 预热是 correction 的瓶颈（~4s，占 correction ~97%）。Phase 1
                # 的局部信号随主 OCR 批处理增量计算（IncrementalDetector），
                # 主 OCR 进行到 50% 时已就绪的信号是"精确"置信度（与整段
                # detect_errors 逐位一致）→ 直接取中部 correction 帧启动后台
                # 预热；最终 correction_frames 由 correction 前接力入队补上
                # （线程按 cache 去重）。线程全程异常保护：失败时 correction
                # 内串行补预热（cache 机制兜底，功能正确）。
                if (not _prewarm_started and done >= int(est_total * 0.50)
                        and self._reocr is not None):
                    _prewarm_started = True
                    try:
                        _frames = {c["index"] for c in _detector.confidence_so_far()
                                   if c["score"] < 70}
                        if _frames:
                            from queue import Queue as _Queue
                            from correction import _prewarm_reocr

                            def _prewarm_loop(fq: "_Queue",
                                              rf: list, ocr, cache,
                                              ms, mw) -> None:
                                """接力式预热：队列消费完中部帧后继续处理
                                correction 阶段提交的尾部差集，收到 None 退出。"""
                                while True:
                                    batch = fq.get()
                                    if batch is None:
                                        break
                                    try:
                                        _prewarm_reocr(batch, rf, ocr, cache,
                                                       ms, mw)
                                    except Exception as _e:
                                        logger.warning("后台 re-OCR 接力失败: %s", _e)

                            self._prewarm_queue = _Queue()
                            _th = threading.Thread(
                                target=_prewarm_loop,
                                args=(self._prewarm_queue, self._raw_frames,
                                      self._reocr, self._reocr_cache,
                                      self._max_speed, self._max_width),
                                daemon=True)
                            self._prewarm_t0 = _time.perf_counter()
                            _th.start()
                            self._prewarm_queue.put(_frames)
                            self._prewarm_thread = _th
                            logger.info("后台 re-OCR 预热启动: %d 帧", len(_frames))
                    except Exception as _e:
                        logger.warning("后台 re-OCR 预热启动失败: %s", _e)

        while True:
            item = q.get()
            if item is None:
                _flush_batch()
                break
            _batch.append(item)
            if len(_batch) >= _batch_size:
                _flush_batch()
        t.join()
        if errors:
            raise errors[0]
        self._observations = observations
        self._diag = diag
        self._diag_notes: dict[int, str] = {}
        # Release decoder to free internal frame buffers
        if _vr is not None:
            del _vr
        gc.collect()
        self._timing["ocr"] = _time.perf_counter() - t_start
        self._timing["decode"] = _decode_ms / 1000.0
        self._timing["inference"] = _inference_ms / 1000.0
        n_frames = len(observations)
        decode_fps = n_frames / max(self._timing["decode"], 0.001)
        inference_fps = n_frames / max(self._timing["inference"], 0.001)
        total_fps = n_frames / max(self._timing["ocr"], 0.001)
        logger.info("OCR 完成: %d 帧 (%s), 总 %.1fs | 解码 %.1fs (%.0f fps) | 推理 %.1fs (%.0f fps) | 总 %.0f fps",
                        n_frames, self._video_backend_actual,
                        self._timing["ocr"], self._timing["decode"], decode_fps,
                        self._timing["inference"], inference_fps, total_fps)

    def _integrate_distance(self) -> None:
        fps = self._fps if self._fps > 0 else 1.0
        dist = 0.0; prev_fi = prev_v = None
        for r in self._rows:
            v = r[2] / config.MPS_TO_KMH if r[2] >= 0 else 0.0
            fi = r[0]
            if prev_fi is not None and prev_v is not None:
                dt = (fi - prev_fi) / fps
                if dt > 0:
                    dist += (prev_v + v) * 0.5 * dt
            prev_fi, prev_v = fi, v
            r[1] = dist

    def _write_csv(self, rows: list, output_path: Path) -> None:
        self._merge_stage_timing()
        vhash = compute_video_hash(self._video_path)
        r = self._roi
        # ── 统计信息 ──
        n_total = len(rows)
        n_trusted = sum(1 for row in rows if Flag.is_trusted(row[3]))
        n_pinned = sum(1 for row in rows if row[3] == Flag.PINNED)
        n_corrected = sum(1 for row in rows if Flag.is_corrected(row[3]))
        timing_str = ", ".join(f"{k}={v:.1f}s" for k, v in self.timing_flat().items())
        # 资源峰值字段（monitor 开启且有样本时追加；gpu_name 为字符串）
        _peak = _peak_fields()
        if _peak:
            _peak_str = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in _peak.items())
            timing_str = f"{timing_str}, {_peak_str}"
        _codec = self._codec or ""
        with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
            fh.write(f"# RaceVideoToLog v{config.__version__}\n")
            fh.write(f"# video_hash={vhash}, video={self._video_path.name}"
                     f", fps={self._fps:.3f}")
            if _codec:
                fh.write(f", codec={_codec}")
            fh.write("\n")
            fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}"
                        f", frame_start={self._frame_start or ''}"
                        f", frame_end={self._frame_end or ''}\n")
            fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}"
                        f", div={self._frame_div}, target_h={self._target_h}"
                        f", max_width={self._max_width}"
                        f", pad={self._pad}, buffer={self._buffer_size}\n")
            fh.write(f"# backend={self._backend_actual}, model={self._ocr_model}")
            reocr_info = f", reocr_model={self._reocr_model}" if self._reocr_model and self._reocr_model != self._ocr_model else ""
            fh.write(f"{reocr_info}")
            fh.write(f", video_backend={self._video_backend_actual or 'decord'}\n")
            if n_pinned > 0:
                fh.write(f"# pinned={n_pinned}\n")
            fh.write(f"# stats: total={n_total}, trusted={n_trusted},"
                        f" corrected={n_corrected}\n")
            if timing_str:
                fh.write(f"# timing: {timing_str}\n")
            w = csv.writer(fh)
            # 批量写（writerows 与 writerow 逐行输出逐字节一致，长视频省 1-2s）
            w.writerows((f"{int(row[0])}", f"{row[1]:.2f}",
                         f"{int(row[2])}", str(row[3])) for row in rows)

    def _populate_diag_final(self) -> None:
        """Fill final_value, flag, and correction_note into diagnostics."""
        if not self._diag:
            return
        notes = self._diag_notes
        for i, row in enumerate(self._rows):
            if i < len(self._diag):
                self._diag[i]["final_value"] = row[2]
                self._diag[i]["flag"] = int(row[3])
                self._diag[i]["correction_note"] = notes.get(i, "")

    def _write_stage_report(self, output_path: Path) -> None:
        """Write per-frame stage report with signal breakdowns + summary JSON.
        Only on detailed/debug log level."""
        if not self._detection_confidence or not self._rows:
            return
        if self._log_level not in ("detailed", "debug"):
            return
        self._merge_stage_timing()
        import json as _json
        report_path = output_path.with_suffix("")
        report_path = report_path.with_name(report_path.name + "_stage_report.csv")
        with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
            fields = ["frame", "raw_text", "raw_val",
                "sig_ocr_conf", "sig_abs",
                "sig_accel", "sig_freq",
                "combined_conf", "conf_tier",
                "old_flag", "new_flag", "old_val", "new_val", "correction_note"]
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            notes = self._diag_notes
            for i, row in enumerate(self._rows):
                conf = self._detection_confidence[i] if i < len(self._detection_confidence) else {}
                sigs = conf.get("signals", {})
                obs = self._observations[i] if i < len(self._observations) else None
                w.writerow({"frame": int(row[0]),
                    "raw_text": obs.raw_text if obs else "",
                    "raw_val": obs.raw_speed_kmh if obs else -1,
                    "sig_ocr_conf": sigs.get("ocr_conf", ""),
                    "sig_abs": sigs.get("abs", ""),
                    "sig_accel": sigs.get("accel", ""),
                    "sig_freq": sigs.get("freq", ""),
                    "combined_conf": conf.get("score", ""),
                    "conf_tier": conf.get("tier", ""),
                    "old_flag": row[3], "new_flag": row[3],
                    "old_val": row[2], "new_val": row[2],
                    "correction_note": notes.get(i, "")})
        summary_path = output_path.with_suffix("")
        summary_path = summary_path.with_name(summary_path.name + "_summary.json")
        n_low = sum(1 for c in self._detection_confidence if c.get('score', 100) < 30)
        n_med = sum(1 for c in self._detection_confidence if 30 <= c.get('score', 100) < 70)
        n_high = sum(1 for c in self._detection_confidence if c.get('score', 100) >= 70)
        n_corr = sum(1 for row in self._rows if Flag.is_corrected(row[3]))
        n_trust = sum(1 for row in self._rows if Flag.is_trusted(row[3]))
        summary = {"video": self._video_path.name,
            "params": {"max_speed": self._max_speed, "max_accel": self._max_accel,
                "frame_div": self._frame_div, "fps": round(self._fps, 2)},
            "stats": {"total_frames": len(self._rows), "corrected": n_corr, "trusted": n_trust},
            "confidence_distribution": {"low": n_low, "medium": n_med, "high": n_high},
            "timing": self._timing}
        _mon = _read_snapshot()
        if _mon:
            summary["monitor"] = _mon
        with summary_path.open("w", encoding="utf-8") as fh:
            _json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info("Stage report saved: %s", report_path)

    def _write_diagnostics(self, output_path: Path) -> None:
        """Write per-frame diagnostics CSV (log_level=debug only)."""
        if not self._diag or self._log_level != "debug":
            return
        # Ensure final values populated (may not be if correction was skipped)
        if self._diag and self._rows and "final_value" not in self._diag[0]:
            self._populate_diag_final()
        diag_path = output_path.with_suffix("")
        diag_path = diag_path.with_name(diag_path.name + "_diagnostics.csv")
        with diag_path.open("w", newline="", encoding="utf-8-sig") as fh:
            fields = ["frame", "raw_text", "raw_value", "confidence",
                        "ocr_time_ms", "final_value", "flag", "correction_note"]
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for i, d in enumerate(self._diag):
                # Use stored frame number if available, fall back to index
                if "frame" not in d:
                    d["frame"] = i
                w.writerow(d)
        logger.info("诊断日志已保存: %s (%d 帧)", diag_path, len(self._diag))
