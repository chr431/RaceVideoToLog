"""RaceVideoToLog export thread — runs Pipeline in a native thread.

Separated from gui.py to avoid coupling _ExportThread to RaceVideoToLogApp.
Takes all parameters explicitly instead of reaching through self.app.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from ocr_engine import _CancelExport


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
            frame_div: int,
            target_h: int,
            pad_px: int,
            buffer_size: int,
            backend: str,
            ocr_model: str,
            reocr_model: str | None,
            speed_format: str,
            frame_start: str,
            frame_end: str,
            log_level: str,
            video_backend: str,
            correction_mode: str,
            output_path: Path,
            parent: QWidget | None = None,
        ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._roi = roi
        self._max_speed_kmh = max_speed_kmh
        self._max_accel_mps2 = max_accel_mps2
        self._frame_div = frame_div
        self._target_h = target_h
        self._pad_px = pad_px
        self._buffer_size = buffer_size
        self._backend = backend
        self._ocr_model = ocr_model
        self._reocr_model = reocr_model
        self._speed_format = speed_format
        self._frame_start = frame_start
        self._frame_end = frame_end
        self._log_level = log_level
        self._video_backend = video_backend
        self._correction_mode = correction_mode
        self._output_path = output_path
        self._cancel_flag = False

    def run(self) -> None:
        """Run Pipeline in a native threading.Thread, wait for completion."""
        from pipeline import ProcessingPipeline

        done = threading.Event()
        error_container: list[Exception] = []
        result_container: dict = {}

        def _worker() -> None:
            try:
                self._check_cancel()
                pipeline = ProcessingPipeline(
                    video_path=self._video_path,
                    roi=self._roi,
                    max_speed=self._max_speed_kmh,
                    max_accel=self._max_accel_mps2,
                    frame_div=self._frame_div,
                    target_h=self._target_h,
                    pad=self._pad_px,
                    buffer_size=self._buffer_size,
                    backend=self._backend,
                    ocr_model=self._ocr_model,
                    reocr_model=self._reocr_model,
                    speed_format=self._speed_format,
                    frame_start=self._frame_start,
                    frame_end=self._frame_end,
                    progress_cb=self._emit_progress,
                    cancel_check=self._check_cancel,
                    log_level=self._log_level,
                    video_backend=self._video_backend,
                )
                mode = self._correction_mode
                if mode == "auto":
                    pipeline.run_auto(self._output_path, mode="auto")
                    result_container["mode"] = "auto"
                else:
                    pipeline.run_auto(self._output_path, mode="manual")
                    result_container["mode"] = "review"
                self.pipeline_ready.emit(pipeline, self._output_path)
            except _CancelExport:
                result_container["cancelled"] = True
            except Exception as exc:
                import traceback
                traceback.print_exc()
                error_container.append(exc)
            finally:
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
