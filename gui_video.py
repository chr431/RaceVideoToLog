"""视频加载与预览逻辑（RaceVideoToLogApp 的 mixin）。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

from ocr_engine import VideoMetadata, format_duration
from widget_utils import set_value_silent


class VideoLoadMixin:
    """依赖宿主：video_path/_preview_vr/_preview_frame_no/metadata 等状态。"""

    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择需要处理的视频", "",
            "视频文件 (*.mp4 *.mkv *.avi *.mov *.m4v *.wmv *.flv *.webm);;所有文件 (*.*)")
        if not path: return
        try: self._load_video(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            self._status_label.setText("导入失败。")

    def _load_video(self, path: Path) -> None:
        from video_utils import open_decord_vr
        from monitor import gui_mark

        gui_mark("load_video: start")
        vr, label = open_decord_vr(str(path))
        gui_mark("load_video: decord open")
        # 编码信息直接来自 decord（自建版新增 get_codec），无子进程开销
        try:
            codec = vr.get_codec() or "?"
        except Exception:
            codec = "?"
        gui_mark("load_video: codec")
        try:
            fc = len(vr)
            fps = vr.get_avg_fps()
            first = vr[0].asnumpy()  # decord returns RGB
            gui_mark("load_video: first frame")
            h, w = first.shape[:2]
            dur = fc / fps if fps > 0 else 0.0
        except Exception:
            del vr
            raise RuntimeError("无法读取视频第一帧。")

        if self._preview_vr is not None:
            del self._preview_vr
        self._preview_vr = vr
        self._preview_frame_no = 0

        self.video_path = path
        self.metadata = VideoMetadata(path=path, duration_sec=dur, width=w, height=h,
            fps=fps, codec=codec, frame_count=fc)
        hh, ww, ch = first.shape
        self.first_frame_qimg = QImage(first.data, ww, hh, ch * ww,
            QImage.Format.Format_RGB888).copy()

        self._file_label.setText(str(path))
        self._dur_label.setText(format_duration(dur))
        self._res_label.setText(f"{w} x {h}")
        self._fps_label.setText(f"{fps:.3f}" if fps > 0 else "Unknown")
        self._codec_label.setText(codec)
        self._status_label.setText("视频已载入，请输入识别范围并预览。")
        self._slider.setRange(0, fc - 1); self._slider.setValue(0)
        self._frame_label.setText(f"#{0}/{fc}")
        self._preview_widget.set_video_size(w, h)
        self._preview_widget.set_roi(self.roi_x1.value(), self.roi_y1.value(),
                                     self.roi_x2.value(), self.roi_y2.value())
        self._show_frame(0)
        for s, m in [(self.roi_x1, w), (self.roi_y1, h), (self.roi_x2, w), (self.roi_y2, h)]:
            s.setMaximum(m - 1)

    def _on_preview_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """预览拖拽 ROI → 同步 spinbox（静默赋值，不触发联动校验）。"""
        for s, v in [(self.roi_x1, x1), (self.roi_y1, y1),
                     (self.roi_x2, x2), (self.roi_y2, y2)]:
            set_value_silent(s, v)

    def _show_frame(self, frame_no: int) -> None:
        pm = None
        vr = self._preview_vr
        if frame_no > 0 and vr is not None and frame_no < len(vr):
            try:
                frame = vr[frame_no].asnumpy()  # decord returns RGB
                h, w, ch = frame.shape
                qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                pm = QPixmap.fromImage(qimg)
            except Exception:
                pass
        if pm is None and self.first_frame_qimg is not None:
            pm = QPixmap.fromImage(self.first_frame_qimg)
        if pm is not None:
            self._preview_widget.set_frame(pm)

    def _on_slider(self, value: int) -> None:
        if self.metadata:
            self._frame_label.setText(f"#{value}/{self.metadata.frame_count}")
        if self._throttle_timer:
            self._throttle_timer.stop()
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(lambda: self._show_frame(value))
        self._throttle_timer.start(30)

    def _show_throttled_frame(self) -> None:
        self._show_frame(self._slider.value())

    def _step(self, delta: int) -> None:
        if not self.metadata: return
        v = max(0, min(self.metadata.frame_count - 1, self._slider.value() + delta))
        self._slider.setValue(v)

    def _on_roi_spin(self, spin) -> None:
        if spin is self.roi_x1 and self.roi_x1.value() > self.roi_x2.value() - 1:
            spin.blockSignals(True); spin.setValue(self.roi_x2.value() - 1); spin.blockSignals(False)
        elif spin is self.roi_x2 and self.roi_x2.value() < self.roi_x1.value() + 1:
            spin.blockSignals(True); spin.setValue(self.roi_x1.value() + 1); spin.blockSignals(False)
        elif spin is self.roi_y1 and self.roi_y1.value() > self.roi_y2.value() - 1:
            spin.blockSignals(True); spin.setValue(self.roi_y2.value() - 1); spin.blockSignals(False)
        elif spin is self.roi_y2 and self.roi_y2.value() < self.roi_y1.value() + 1:
            spin.blockSignals(True); spin.setValue(self.roi_y1.value() + 1); spin.blockSignals(False)
        self._preview_widget.set_roi(
            self.roi_x1.value(), self.roi_y1.value(),
            self.roi_x2.value(), self.roi_y2.value())
