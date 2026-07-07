"""统一处理流水线 — GUI 和 CLI 共用。
运行在调用者线程中（GUI 应在原生 threading.Thread 中调用以避免 QThread 性能损失）。
"""
from __future__ import annotations
import csv, time
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from rapidocr_onnxruntime import RapidOCR
import ocr_engine as _oe
from ocr_engine import (
    auto_select_anchors, clamp_region, compute_video_hash,
    extract_speed_value, SpeedObservation,
    SOURCE_TO_KMH,
    _reset_backend, _select_backend, _get_model_kwargs,
)
from correction import correct_with_anchors, compute_confidence, find_problem_segments


def _parse_int_or_none(s: str) -> int | None:
    try:
        return int(s.strip()) if s.strip() else None
    except Exception:
        return None


def _preprocess_standard(crop: np.ndarray, target_h: float, pad: float) -> np.ndarray:
    """标准预处理：灰度化 + 缩放 + 填充 + 转 BGR。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return _finish_preprocess(gray, target_h, pad)


def _preprocess_otsu(crop: np.ndarray, target_h: float, pad: float) -> np.ndarray:
    """OTSU 备选预处理：灰度化 + OTSU 二值化 + 缩放 + 填充 + 转 BGR。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _finish_preprocess(gray, target_h, pad)


def _finish_preprocess(gray: np.ndarray, target_h: float, pad: float) -> np.ndarray:
    """统一的缩放 + 填充 + 转 BGR。"""
    h, w = gray.shape[:2]
    th = max(8.0, float(target_h))
    scale = th / float(h) if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)))
    pad_int = int(pad)
    if pad_int > 0:
        gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int,
                                  cv2.BORDER_REPLICATE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


ProgressFn = Callable[[str, float], None]


class ProcessingPipeline:
    """统一处理流水线。

    同时支持自动锚点模式和人工辅助（两段式）模式。
    通过 progress_cb 回调报告进度，与 UI 框架解耦。
    """

    def __init__(self, video_path: str | Path, roi: tuple[int, int, int, int],
                 max_speed: float, max_accel: float,
                 frame_div: int, target_h: float, pad: float, buffer_size: int,
                 backend: str, ocr_model: str, speed_format: str,
                 frame_start: str = "", frame_end: str = "",
                 progress_cb: ProgressFn | None = None):
        self._video_path = Path(video_path)
        self._roi = roi
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._frame_div = frame_div
        self._target_h = target_h
        self._pad = pad
        self._buffer_size = buffer_size
        self._backend = backend
        self._ocr_model = ocr_model
        self._speed_format = speed_format
        self._frame_start = frame_start
        self._frame_end = frame_end
        self._progress = progress_cb

        # 状态（run_review_pass1 后填充，供 run_review_pass2 使用）
        self._ocr: RapidOCR | None = None
        self._raw_frames: list[tuple[float, np.ndarray]] = []
        self._observations: list[SpeedObservation] = []
        self._rows: list[list] = []
        self._anchor_indices: set = set()
        self._segments: list[dict] = []
        self._backend_actual: str = "CPU"

    # ═══════════════ 公开接口 ═══════════════

    def run_auto(self, output_path: str | Path) -> None:
        """自动锚点模式：完整流水线 → 写 CSV。"""
        self._emit("加载 OCR 引擎...", 1.0)
        self._ensure_ocr()

        self._emit("解码视频帧...", 3.0)
        self._extract_frames()
        if not self._raw_frames:
            raise RuntimeError("未从视频中读取到任何帧。")

        self._run_ocr()

        if not self._observations:
            raise RuntimeError("未识别到任何速度数据。")

        self._run_correction_pass(Path(output_path), skip_fill=False)
        self._emit("完成", 100.0)

    def run_review_pass1(self) -> tuple | None:
        """人工辅助第 1 轮：OCR → 纠错 → 置信度 → 问题段。

        Returns:
            (rows, observations, raw_frames, confidences, segments) 或 None（无问题段时自动写 CSV）
        """
        self._emit("加载 OCR 引擎...", 1.0)
        self._ensure_ocr()

        self._emit("解码视频帧...", 3.0)
        self._extract_frames()
        if not self._raw_frames:
            raise RuntimeError("未从视频中读取到任何帧。")

        self._run_ocr()
        if not self._observations:
            raise RuntimeError("未识别到任何速度数据。")

        self._emit("锚点选择 + 重OCR纠错...", 91.0)
        self._anchor_indices = auto_select_anchors(
            self._observations, self._max_speed, max_accel_mps2=self._max_accel)
        if len(self._anchor_indices) < 3:
            raise RuntimeError("自动锚点选择失败：未找到足够的可靠帧。")

        self._rows = []
        for i, obs in enumerate(self._observations):
            self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh,
                               2 if i in self._anchor_indices else 0])

        self._emit("重OCR纠错: 检测错误帧...", 92.0)
        from correction import correct_with_anchors, compute_confidence, find_problem_segments
        corr_timing: dict[str, float] = {}
        def _prog(done: int, total: int) -> None:
            if done % max(1, total // 5) != 0 and done != total:
                return
            pct = done / max(total, 1)
            self._emit(f"重OCR纠错: {done}/{total} 帧", 92.0 + pct * 5.0)
        self._rows = correct_with_anchors(
            self._rows, self._observations, self._raw_frames, self._ocr,
            self._max_speed, self._max_accel, self._anchor_indices,
            progress_fn=_prog, timing=corr_timing)
        self._print_reocr_timing(corr_timing)

        self._emit("计算置信度...", 97.5)
        confidences = compute_confidence(self._rows, self._observations,
                                         self._max_speed, self._max_accel)
        segments = find_problem_segments(confidences)

        if segments:
            self._emit(f"发现 {len(segments)} 个问题段，等待人工审核...", 98.5)
            self._segments = segments
            return (list(self._rows), list(self._observations),
                    list(self._raw_frames), confidences, segments)
        else:
            self._emit("未发现问题段，无需人工审核。", 98.5)
            self._integrate_distance()
            self._write_csv(self._rows, Path(""), auto_anchor=False)
            self._emit("完成", 100.0)
            return None

    def run_review_pass2(self, corrections: dict[int, float],
                         confirmed_segments: set[int],
                         output_path: str | Path) -> None:
        """人工辅助第 2 轮：合并手动修正 → 再纠错 → 写 CSV。

        依赖 run_review_pass1 已填充的内部状态。
        """
        # 合并手动修正
        for fi, v in corrections.items():
            if 0 <= fi < len(self._rows):
                self._rows[fi][2] = v
                self._rows[fi][3] = 2
                self._anchor_indices.add(fi)

        # 确认正确的段: 所有帧 flag=2
        for seg_start in confirmed_segments:
            for seg in self._segments:
                if seg["start"] == seg_start:
                    for fi in range(seg["start"], seg["end"] + 1):
                        if fi not in corrections and 0 <= fi < len(self._rows):
                            self._rows[fi][3] = 2
                            self._anchor_indices.add(fi)
                    break

        corr_timing: dict[str, float] = {}
        self._rows = correct_with_anchors(
            self._rows, self._observations, self._raw_frames, self._ocr,
            self._max_speed, self._max_accel, self._anchor_indices,
            skip_fill=True, timing=corr_timing)
        self._print_reocr_timing(corr_timing)

        self._integrate_distance()
        self._write_csv(self._rows, Path(output_path), auto_anchor=False,
                        manual_anchor=True)
        self._emit(f"人工审核完成 — {len(corrections)} 帧已修正，结果已保存。", 100.0)

    # ═══════════════ 内部：阶段实现 ═══════════════

    def _emit(self, msg: str, pct: float) -> None:
        if self._progress:
            self._progress(msg, pct)

    def _ensure_ocr(self) -> RapidOCR:
        if self._ocr is None:
            _reset_backend()
            self._backend_actual = _select_backend(self._backend)
            kw = _get_model_kwargs(self._ocr_model)
            self._ocr = RapidOCR(**(kw or {}))
        return self._ocr

    def _extract_frames(self) -> None:
        cap = cv2.VideoCapture(str(self._video_path))
        x1, y1, x2, y2 = clamp_region(*self._roi, 0, 0)
        frame_step = max(1, self._frame_div)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # 重读分辨率以 clamp
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        x1, y1, x2, y2 = clamp_region(*self._roi, w, h)

        f_start = _parse_int_or_none(self._frame_start)
        f_end = _parse_int_or_none(self._frame_end)
        _end_limit = f_end if f_end is not None else total_video_frames

        self._raw_frames = []
        fi = 0
        while fi < total_video_frames:
            if fi >= _end_limit:
                break
            if f_start is not None and fi < f_start:
                cap.grab(); fi += 1; continue
            if fi % frame_step != 0:
                cap.grab(); fi += 1; continue
            if not cap.grab():
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            if fi % max(1, frame_step * 200) == 0:
                pct = 3.0 + 4.0 * (fi / max(_end_limit, 1))
                self._emit(f"解码视频: {fi}/{_end_limit} 帧", pct)
            ts = fi / fps if fps > 0 else 0.0
            crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
            self._raw_frames.append((ts, crop))
            fi += 1
        cap.release()

    def _run_ocr(self) -> None:
        import threading
        from queue import Queue

        ocr = self._ocr
        max_speed = self._max_speed
        speed_format = self._speed_format
        total = len(self._raw_frames)
        buf_size = max(2, self._buffer_size * 2)
        q: Queue = Queue(maxsize=buf_size)
        errors: list[Exception] = []

        def _producer() -> None:
            try:
                for ts, crop in self._raw_frames:
                    proc = _preprocess_standard(crop, self._target_h, self._pad)
                    q.put((ts, crop, proc))
                q.put(None)
            except Exception as e:
                errors.append(e)
                q.put(None)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        observations: list[SpeedObservation] = []
        done = 0
        while True:
            item = q.get()
            if item is None:
                break
            ts, crop, proc = item
            ocr_result, _ = ocr(proc)
            sv, rt = extract_speed_value(ocr_result)
            if sv is None:
                proc_fb = _preprocess_otsu(crop, self._target_h, self._pad)
                ocr_result, _ = ocr(proc_fb)
                sv, rt = extract_speed_value(ocr_result)
            if sv is not None and rt is not None:
                observations.append(SpeedObservation(
                    timestamp=ts,
                    raw_speed_kmh=sv * SOURCE_TO_KMH[speed_format],
                    raw_text=rt))
            else:
                observations.append(SpeedObservation(ts, -1.0, ""))
            done += 1
            if done % 10 == 0:
                pct = 7.0 + (done / total) * 83.0
                self._emit(f"[{_oe._gpu_backend}] OCR: {done}/{total}", pct)
        t.join()
        if errors:
            raise errors[0]
        self._observations = observations

    def _run_correction_pass(self, output_path: Path, skip_fill: bool,
                             manual_anchor: bool = False) -> None:
        self._emit("锚点选择 + 重OCR纠错...", 91.0)
        self._anchor_indices = auto_select_anchors(
            self._observations, self._max_speed, max_accel_mps2=self._max_accel)
        if len(self._anchor_indices) < 3:
            raise RuntimeError("自动锚点选择失败：未找到足够的可靠帧。")

        self._rows = []
        for i, obs in enumerate(self._observations):
            self._rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh,
                               2 if i in self._anchor_indices else 0])

        self._emit("重OCR纠错: 检测错误帧...", 92.0)
        corr_timing: dict[str, float] = {}
        def _prog(done: int, total: int) -> None:
            if done % max(1, total // 5) != 0 and done != total:
                return
            pct = done / max(total, 1)
            self._emit(f"重OCR纠错: {done}/{total} 帧", 92.0 + pct * 6.0)
        self._rows = correct_with_anchors(
            self._rows, self._observations, self._raw_frames, self._ocr,
            self._max_speed, self._max_accel, self._anchor_indices,
            progress_fn=_prog, timing=corr_timing, skip_fill=skip_fill)
        self._print_reocr_timing(corr_timing)

        self._integrate_distance()
        self._write_csv(self._rows, output_path, auto_anchor=not manual_anchor,
                        manual_anchor=manual_anchor)
        self._emit("完成", 100.0)

    def _integrate_distance(self) -> None:
        dist = 0.0; prev_t = prev_v = None
        for r in self._rows:
            v = r[2] / 3.6
            if prev_t is not None and prev_v is not None:
                dt = r[0] - prev_t
                if dt > 0:
                    dist += (prev_v + v) * 0.5 * dt
            prev_t, prev_v = r[0], v
            r[1] = dist

    def _write_csv(self, rows: list, output_path: Path,
                   auto_anchor: bool = True, manual_anchor: bool = False) -> None:
        vhash = compute_video_hash(self._video_path)
        r = self._roi
        with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
            fh.write("# RaceVideoToLog\n")
            fh.write(f"# video_hash={vhash}, video={self._video_path.name}\n")
            fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}\n")
            tag = "manual_anchor" if manual_anchor else ("auto_anchor" if auto_anchor else "")
            fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}, "
                     f"div={self._frame_div}, target_h={self._target_h}, "
                     f"pad={self._pad}, backend={self._backend_actual}, "
                     f"model={self._ocr_model}, buffer={self._buffer_size}, "
                     f"frame_start={self._frame_start or ''}, "
                     f"frame_end={self._frame_end or ''}"
                     + (f", {tag}=1" if tag else "") + "\n")
            w = csv.writer(fh)
            for row in rows:
                w.writerow([f"{row[0]:.2f}", f"{row[1]:.2f}",
                           f"{row[2]:.2f}", str(row[3])])

    @staticmethod
    def _print_reocr_timing(t: dict) -> None:
        if t.get("re_ocr", 0) > 0:
            print(f"  [重OCR] {t['re_ocr']:.2f}s")
