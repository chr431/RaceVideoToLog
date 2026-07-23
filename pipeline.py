"""统一处理流水线 — GUI 和 CLI 共用。
运行在调用者线程中（GUI 应在原生 threading.Thread 中调用以避免 QThread 性能损失）。
"""
from __future__ import annotations
import csv
import logging
import time as _time
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from ocr_engine import (
    clamp_region, compute_video_hash,
    extract_speed_value, SpeedObservation, Flag,
    SOURCE_TO_KMH, _parse_int_or_none,
    _reset_backend, _select_backend, _get_model_params,
)
from config import MPS_TO_KMH
from correction import correct_with_trust, compute_confidence, find_problem_segments
from gpu_setup import get_gpu_backend, get_engine_params, get_engine_type

logger = logging.getLogger("RaceVideoToLog.pipeline")


def _preprocess_standard(crop: np.ndarray, target_h: int, pad: int) -> np.ndarray:
    """标准预处理：纯 resize + 填充。保留 BGR 通道以利用颜色信息。"""
    h, w = crop.shape[:2]
    if target_h < 8:
        raise ValueError(f"target_h 必须 >= 8，当前为 {target_h}")
    scale = target_h / h if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        resized = cv2.resize(crop, (max(1, int(w * scale)), target_h))
    else:
        resized = crop
    if pad > 0:
        resized = cv2.copyMakeBorder(resized, pad, pad, pad, pad,
                                        cv2.BORDER_REPLICATE)
    return resized


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
                    reocr_model: str | None = None,
                    cancel_check: "Callable[[], None] | None" = None,
                    log_level: str = "normal",
                final_check: bool = False):
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
        self._reocr_model = reocr_model  # None = 使用 ocr_model
        self._backend = backend
        self._log_level = log_level  # "normal" | "detailed" | "debug"
        self._ocr_model = ocr_model
        self._speed_format = speed_format
        self._frame_start = frame_start
        self._cancel_check = cancel_check
        self._frame_end = frame_end
        self._progress = progress_cb
        self._final_check = final_check

        # 状态
        self._ocr: "RapidOCR | None" = None
        self._reocr: "RapidOCR | None" = None  # 重 OCR 引擎
        self._raw_frames: list[tuple[float, np.ndarray]] = []
        self._observations: list[SpeedObservation] = []
        self._rows: list[list] = []
        self._pinned: set[int] = set()
        self._segments: list[dict] = []
        # ── 性能计时 ──
        self._timing: dict[str, float] = {}
        # ── 重 OCR 缓存（绑定到 Pipeline 实例生命周期）──
        self._reocr_cache: dict[int, set[float]] = {}
        self._fps: float = 0.0  # 视频帧率，用于帧号→高精度时间

    # ═══════════════ 公开接口 ═══════════════

    def run_auto(self, output_path: str | Path, reocr_only: bool = False) -> None:
        """自动纠错模式：完整流水线 → 写 CSV。"""
        t_total = _time.perf_counter()
        self._emit("加载 OCR 引擎...", 1.0)
        self._ensure_ocr()
        self._run_ocr()

        if not self._observations:
            raise RuntimeError("未识别到任何速度数据。")

        if self._max_speed <= 0:
            self._emit("跳过纠错（原始OCR输出）...", 95.0)
            self._rows = []
            for obs in self._observations:
                self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, Flag.RAW])
            self._integrate_distance()
            self._write_csv(self._rows, Path(output_path))
            self._write_diagnostics(Path(output_path))
        else:
            self._run_correction(Path(output_path), skip_fill=False, reocr_only=reocr_only)
            for row in self._rows:
                if row[2] > self._max_speed:
                    row[2] = -1.0
        self._timing["total"] = _time.perf_counter() - t_total
        logger.info("流水线完成: 总计 %.1fs (%s)",
                        self._timing["total"],
                        ", ".join(f"{k}={v:.1f}s" for k, v in self._timing.items()))
        self._emit("完成", 100.0)

    def run_review_pass1(self, output_path: str | Path | None = None) -> tuple | None:
        """人工辅助第 1 轮：OCR → 轻量纠错（仅重OCR）→ 置信度 → 问题段。

        轻量纠错：仅对错误帧重 OCR，在原始值和重 OCR 值之间选择更优者。
        不进行混淆候选、部分数字推断、插值填充，避免产生"物理合理但实际错误"的数据。
        这样问题段反映的是真正需要人工介入的帧。
        """
        self._emit("加载 OCR 引擎...", 1.0)
        self._ensure_ocr()
        self._run_ocr()

        if not self._observations:
            raise RuntimeError("未识别到任何速度数据。")

        # 构建初始 rows（全部 RAW，无预选锚点）
        self._rows = self._build_initial_rows()

        # ── 轻量纠错：仅重 OCR，不混淆/推断/填充 ──
        self._emit("轻量纠错 (仅重OCR)...", 88.0)
        def _prog(done, total):
            if done % max(1, total // 3) != 0 and done != total:
                return
            self._emit(f"轻量纠错: {done}/{total} 帧", 88.0 + (done / max(total, 1)) * 4.0)
        self._rows = correct_with_trust(
            self._rows, self._observations, self._raw_frames, self._reocr,
            self._max_speed, self._max_accel,
            progress_fn=_prog, skip_fill=True, light_mode=True,
            reocr_cache=self._reocr_cache)

        # 验证：两次 OCR 的最佳值若超过最大速度则输出 -1
        for row in self._rows:
            if row[2] > self._max_speed:
                row[2] = -1.0

        self._emit("计算置信度...", 95.0)
        confidences = compute_confidence(self._rows, self._observations,
                                            self._max_speed, self._max_accel)
        self._confidences = confidences
        segments = find_problem_segments(confidences, min_segment_len=1)

        if segments:
            self._emit(f"发现 {len(segments)} 个问题段，等待人工审核...", 98.5)
            self._segments = segments
            return (list(self._rows), list(self._observations),
                    list(self._raw_frames), confidences, segments)
        else:
            self._emit("未发现问题段，无需人工审核。", 98.5)
            self._integrate_distance()
            if output_path is not None:
                self._write_csv(self._rows, Path(output_path))
                self._write_diagnostics(Path(output_path))
            self._emit("完成", 100.0)
            return None

    def run_review_pass2(self, corrections: dict[int, float],
                            confirmed_segments: set[int],
                            output_path: str | Path,
                            partial_corrections: dict[int, str] | None = None) -> None:
        """人工辅助第 2 轮：合并手动修正 → 再纠错 → 写 CSV。

        修正值在 _correct 内部 rows 重建后才应用，避免被覆盖。
        """
        # 构建 pinned 集（用户修正帧 + 确认段首尾）
        self._pinned = set(corrections.keys())
        for seg_start in confirmed_segments:
            for seg in self._segments:
                if seg["start"] == seg_start:
                    self._pinned.add(seg["start"])
                    self._pinned.add(seg["end"])
                    break

        self._correct(91.0, 7.0, skip_fill=False,
                        corrections=corrections,
                        partial_corrections=partial_corrections,
                        confirmed=confirmed_segments)

        self._integrate_distance()
        self._write_csv(self._rows, Path(output_path))
        self._write_diagnostics(Path(output_path))
        self._emit(f"人工审核完成 — {len(corrections)} 帧已修正，结果已保存。", 100.0)

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
            rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, Flag.RAW])
        return rows

    def _ensure_ocr(self) -> "RapidOCR":
        from rapidocr import RapidOCR
        if self._ocr is None:
            _reset_backend()
            self._backend_actual = _select_backend(self._backend)
            engine_params = get_engine_params()
            _et = get_engine_type()
            model_params = _get_model_params(self._ocr_model, _et)
            all_params = {**(model_params or {}), **engine_params}
            if _et == "tensorrt":
                self._emit("加载 TensorRT 引擎（首次使用可能需要几分钟构建引擎）...", 1.5)
            self._ocr = RapidOCR(params=all_params)
            # 若指定了不同的重 OCR 模型，创建独立引擎
            _reocr_model = self._reocr_model or self._ocr_model
            if _reocr_model != self._ocr_model:
                reocr_model_params = _get_model_params(_reocr_model, _et)
                reocr_all_params = {**(reocr_model_params or {}), **engine_params}
                self._reocr = RapidOCR(params=reocr_all_params)
            else:
                self._reocr = self._ocr
        return self._ocr

    def _correct(self, progress_base: float, progress_span: float,
                    skip_fill: bool,
                    partial_corrections: dict[int, str] | None = None,
                    corrections: dict[int, float] | None = None,
                    confirmed: set[int] | None = None,
                    reocr_only: bool = False) -> None:
        """共享纠错逻辑：构建 rows → 应用修正 → LCS 评分 → correct_with_trust。

        先重建全部 RAW rows，再覆盖用户修正（PINNED）和确认段（CONFIRMED_SEG），
        最后交给 correct_with_trust 进行 LCS 错误检测和 5 阶段纠错。
        corrections 和 confirmed 仅由 pass2 传入。
        """
        # 构建初始 rows（全部 RAW）
        self._rows = self._build_initial_rows()

        # 应用用户修正（在 rows 重建之后，避免被覆盖）
        if corrections:
            for fi, v in corrections.items():
                if 0 <= fi < len(self._rows):
                    self._rows[fi][2] = v
                    self._rows[fi][3] = Flag.PINNED
        if confirmed:
            for seg_start in confirmed:
                for seg in (self._segments if hasattr(self, '_segments') else []):
                    if seg["start"] == seg_start:
                        for fi in range(seg["start"], seg["end"] + 1):
                            if fi not in (corrections or {}) and 0 <= fi < len(self._rows):
                                self._rows[fi][3] = Flag.CONFIRMED_SEG
                        break

        t0 = _time.perf_counter()
        self._emit("纠错: LCS 评分 + 检测误差...", progress_base + 1.0)
        corr_timing: dict[str, float] = {}
        def _prog(done: int, total: int) -> None:
            if done % max(1, total // 5) != 0 and done != total:
                return
            pct = done / max(total, 1)
            self._emit(f"纠错: {done}/{total} 帧", progress_base + 1.0 + pct * progress_span)
        self._rows = correct_with_trust(
            self._rows, self._observations, self._raw_frames, self._reocr,
            self._max_speed, self._max_accel,
            progress_fn=_prog, timing=corr_timing, skip_fill=skip_fill,
            partial_corrections=partial_corrections,
            reocr_cache=self._reocr_cache,
            notes=self._diag_notes if self._diag else None,
            pinned=self._pinned if self._pinned else None,
            reocr_only=reocr_only,
            fps=self._fps)
        self._populate_diag_final()
        self._timing["correction"] = _time.perf_counter() - t0
        if corr_timing.get("re_ocr", 0) > 0:
            logger.info("重OCR 耗时: %.2fs", corr_timing["re_ocr"])

    def finalize(self, output_path: str | Path) -> None:
        """最终检查后写入 CSV：积分距离 + 写 CSV + 诊断日志。"""
        self._integrate_distance()
        self._write_csv(self._rows, Path(output_path))
        self._write_diagnostics(Path(output_path))

    def _run_correction(self, output_path: Path, skip_fill: bool, reocr_only: bool = False) -> None:
        """自动纠错模式的完整纠错 + 写 CSV（调用 _correct 后继续）。"""
        self._correct(91.0, 6.0, skip_fill=skip_fill, reocr_only=reocr_only)
        if not self._final_check:
            self.finalize(output_path)

    def _run_ocr(self) -> None:
        """解码 + OCR：producer 解码(decord/cv2)/预处理 → consumer 推理。

        Queue 流水线重叠 I/O 与 GPU 推理。decord (NVDEC) 优先，
        不可用时自动回退 cv2 (CPU)。OCR 后端由 gpu_setup 自动选择。
        """
        import threading
        from queue import Queue

        ocr = self._ocr
        speed_format = self._speed_format
        target_h = self._target_h
        pad = self._pad
        frame_step = max(1, self._frame_div)
        t_start = _time.perf_counter()

        # ── 视频源：decord (GPU/NVDEC) 优先，cv2 (CPU) 回退 ──
        _src_type = "cv2"  # fallback
        _vr = None
        _cap: "cv2.VideoCapture | None" = None

        try:
            from decord import VideoReader as _VR, cpu as _decord_cpu
            _vr = _VR(str(self._video_path), ctx=_decord_cpu(0))
            total_video_frames = len(_vr)
            fps = _vr.get_avg_fps(); self._fps = fps
            _first = _vr[0].asnumpy()
            h, w = _first.shape[:2]
            _src_type = "decord"
            logger.info("Video source: decord (NVDEC)")
        except Exception as _e:
            logger.info("Video source: cv2 (CPU fallback — %s)", _e)
            _cap = cv2.VideoCapture(str(self._video_path))
            total_video_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(_cap.get(cv2.CAP_PROP_FPS) or 0.0); self._fps = fps
            w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        x1, y1, x2, y2 = clamp_region(*self._roi, w, h)
        f_start = _parse_int_or_none(self._frame_start)
        f_end = _parse_int_or_none(self._frame_end)
        _end_limit = f_end if f_end is not None else total_video_frames

        self._raw_frames = []
        buf_size = max(2, self._buffer_size * 2)
        q: Queue = Queue(maxsize=buf_size)
        errors: list[Exception] = []

        def _producer() -> None:
            try:
                if _src_type == "decord":
                    for _fi in range(f_start or 0, min(_end_limit, total_video_frames)):
                        if (_fi - (f_start or 0)) % frame_step != 0:
                            continue
                        if self._cancel_check and _fi % 10 == 0:
                            self._cancel_check()
                        _frame = _vr[_fi].asnumpy()
                        _ts = _fi / fps if fps > 0 else 0.0
                        _crop = _frame[y1:y2 + 1, x1:x2 + 1].copy()
                        self._raw_frames.append((_fi, _crop))
                        _proc = _preprocess_standard(_crop, target_h, pad)
                        q.put((_fi, _proc))
                else:
                    fi = 0
                    while fi < total_video_frames:
                        if self._cancel_check and fi % 10 == 0:
                            self._cancel_check()
                        if fi >= _end_limit:
                            break
                        if f_start is not None and fi < f_start:
                            _cap.grab(); fi += 1; continue
                        if fi % frame_step != 0:
                            _cap.grab(); fi += 1; continue
                        if not _cap.grab():
                            break
                        ok, frame = _cap.retrieve()
                        if not ok or frame is None:
                            break
                        crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
                        self._raw_frames.append((fi, crop))
                        proc = _preprocess_standard(crop, target_h, pad)
                        q.put((fi, proc))
                        fi += 1
                q.put(None)
            except Exception as e:
                errors.append(e)
                q.put(None)
            finally:
                if _cap is not None:
                    _cap.release()

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        observations: list[SpeedObservation] = []
        _collect_diag = self._log_level in ("detailed", "debug")
        diag: list[dict] = []
        done = 0
        est_total = (_end_limit - (f_start or 0)) // frame_step
        while True:
            item = q.get()
            if item is None:
                break
            fi, proc = item
            t_ocr0 = _time.perf_counter()
            ocr_result = ocr(proc)
            t_ocr = (_time.perf_counter() - t_ocr0) * 1000.0
            sv, rt, conf = extract_speed_value(ocr_result)
            if sv is not None and rt is not None:
                observations.append(SpeedObservation(
                    timestamp=fi,
                    raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
                    raw_text=rt))
            else:
                observations.append(SpeedObservation(fi, -1.0, ""))
            if _collect_diag:
                diag.append({
                    "raw_text": rt or "",
                    "raw_value": sv,
                    "confidence": round(conf, 4),
                    "ocr_time_ms": round(t_ocr, 2),
                })
            done += 1
            if done % 10 == 0 or done <= 3 or done == est_total:
                pct = 3.0 + (done / max(est_total, 1)) * 87.0
                self._emit(f"[{get_gpu_backend()}] OCR: {done}/{est_total}", pct)
        t.join()
        if errors:
            raise errors[0]
        self._observations = observations
        self._diag = diag
        self._diag_notes: dict[int, str] = {}
        self._timing["ocr"] = _time.perf_counter() - t_start
        logger.info("OCR 完成: %d 帧, 耗时 %.1fs",
                        len(observations), self._timing["ocr"])

    def _integrate_distance(self) -> None:
        fps = self._fps if self._fps > 0 else 1.0
        dist = 0.0; prev_fi = prev_v = None
        for r in self._rows:
            v = r[2] / MPS_TO_KMH if r[2] >= 0 else 0.0
            fi = r[0]
            if prev_fi is not None and prev_v is not None:
                dt = (fi - prev_fi) / fps
                if dt > 0:
                    dist += (prev_v + v) * 0.5 * dt
            prev_fi, prev_v = fi, v
            r[1] = dist

    def _write_csv(self, rows: list, output_path: Path) -> None:
        vhash = compute_video_hash(self._video_path)
        r = self._roi
        # ── 统计信息 ──
        n_total = len(rows)
        n_trusted = sum(1 for row in rows if Flag.is_trusted(row[3]))
        n_pinned = sum(1 for row in rows if row[3] == Flag.PINNED)
        n_corrected = sum(1 for row in rows if Flag.is_corrected(row[3]))
        timing_str = ", ".join(f"{k}={v:.1f}s" for k, v in self._timing.items())
        with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
            fh.write("# RaceVideoToLog v2.5.0\n")
            fh.write(f"# video_hash={vhash}, video={self._video_path.name}\n")
            fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}"
                        f", frame_start={self._frame_start or ''}"
                        f", frame_end={self._frame_end or ''}\n")
            fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}"
                        f", div={self._frame_div}, target_h={self._target_h}"
                        f", pad={self._pad}, buffer={self._buffer_size}\n")
            fh.write(f"# backend={self._backend_actual}, model={self._ocr_model}")
            reocr_info = f", reocr_model={self._reocr_model}" if self._reocr_model and self._reocr_model != self._ocr_model else ""
            fh.write(f"{reocr_info}\n")
            if n_pinned > 0:
                fh.write(f"# pinned={n_pinned}\n")
            fh.write(f"# stats: total={n_total}, trusted={n_trusted},"
                        f" corrected={n_corrected}\n")
            if timing_str:
                fh.write(f"# timing: {timing_str}\n")
            w = csv.writer(fh)
            for row in rows:
                w.writerow([f"{row[0]:.2f}", f"{row[1]:.2f}",
                            f"{row[2]}", str(row[3])])

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
                d["frame"] = i
                w.writerow(d)
        logger.info("诊断日志已保存: %s (%d 帧)", diag_path, len(self._diag))
