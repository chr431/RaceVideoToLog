"""RaceVideoToLog export thread — runs Pipeline in a native thread.

Separated from gui.py to avoid coupling _ExportThread to RaceVideoToLogApp.
Takes all parameters explicitly instead of reaching through self.app.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget


def _to_int_or_none(s: str) -> int | None:
    """空串/非法 → None（段管线 frame_start/end 需要 int|None）。"""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None



class _CancelExport(Exception):
    """内部异常：用户取消了导出任务。"""
    pass



class ExportThread(QThread):
    """后台导出线程：在原生 threading.Thread 中运行 Pipeline，通过信号与 GUI 通信。

    使用原生线程而非 QThread 的工作线程，以避免 QThread 导致的
    GPU 推理性能损失。所有参数通过构造函数显式传入，无隐式依赖。
    """

    progress_updated = Signal(str, float)
    finished = Signal(str)
    error_occurred = Signal(str)
    cancelled = Signal()
    pipeline_ready = Signal(object, object)  # (pipeline, output_path)

    def __init__(self,
            video_path: Path,
            roi: tuple,
            max_speed_kmh: float,
            max_accel_mps2: float,
            buffer_size: int,
            target_h: int,
            pad_px: int,
            ocr_model: str,
            speed_format: str,
            frame_start: str,
            frame_end: str,
            max_width: int,
            output_path: Path,
            monitor_enabled: bool = True,
            parent: QWidget | None = None,
        ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._roi = roi
        self._max_speed_kmh = max_speed_kmh
        self._max_accel_mps2 = max_accel_mps2
        self._buffer_size = buffer_size
        self._target_h = target_h
        self._pad_px = pad_px
        self._ocr_model = ocr_model
        self._speed_format = speed_format
        self._frame_start = _to_int_or_none(frame_start)
        self._frame_end = _to_int_or_none(frame_end)
        self._max_width = max_width
        self._monitor_enabled = monitor_enabled
        self._output_path = output_path
        self._cancel_flag = False

    def run(self) -> None:
        """Run Pipeline in a native threading.Thread, wait for completion."""
        from segment_flow import SegmentPipeline

        done = threading.Event()
        error_container: list[Exception] = []
        result_container: dict = {}

        def _worker() -> None:
            import config
            import monitor as _monitor
            try:
                if self._monitor_enabled:
                    _monitor.start(interval_s=config.MONITOR_INTERVAL_S,
                                   with_gpu=config.MONITOR_GPU)
                self._check_cancel()
                pipeline = SegmentPipeline(
                    video_path=str(self._video_path),
                    roi=self._roi,
                    max_speed_kmh=self._max_speed_kmh,
                    max_accel_mps2=self._max_accel_mps2,
                    buffer_size=self._buffer_size,
                    target_h=self._target_h,
                    pad=self._pad_px,
                    ocr_model=self._ocr_model,
                    speed_format=self._speed_format,
                    frame_start=self._frame_start,
                    frame_end=self._frame_end,
                    progress_cb=self._emit_progress,
                    max_width=self._max_width,
                    fps=None,
                    cancel_check=self._check_cancel,
                )
                pipeline.run(self._output_path)
                result_container["mode"] = "auto"
                result_container["timing"] = pipeline.timing_flat()
                self.pipeline_ready.emit(pipeline, self._output_path)
            except _CancelExport:
                result_container["cancelled"] = True
            except Exception as exc:
                import traceback
                traceback.print_exc()
                error_container.append(exc)
            finally:
                # 取消/异常路径也必须停止监测（finally 兜底，幂等）
                if self._monitor_enabled:
                    _stats = _monitor.stop()
                    if _stats:
                        _monitor.log_run(self._video_path.name, _stats,
                                         result_container.get("timing"))
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        while not done.wait(1.0):
            if self._cancel_flag:
                done.set()
                t.join(2.0)
                result_container["cancelled"] = True
                break
        t.join(2.0)

        if error_container:
            self.error_occurred.emit(str(error_container[0]))
        elif result_container.get("cancelled"):
            self.cancelled.emit()
        else:
            self.finished.emit(result_container["mode"])

    def _check_cancel(self) -> None:
        if self._cancel_flag:
            raise _CancelExport()

    def _emit_progress(self, msg: str, pct: float) -> None:
        self.progress_updated.emit(msg, pct)
