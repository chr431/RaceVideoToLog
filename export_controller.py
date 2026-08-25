"""导出流程编排（RaceVideoToLogApp 的 mixin）。

负责 ExportThread 生命周期、CSV 设置导入、最终检查 finalize 与内存清理。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

import config
from gui_export import ExportThread


# CSV 头的 backend 写的是实际解码器标签（decord/CPU、decord/GPU、
# decord/GPU+CPU-hybrid），导入 GUI 时归一化到 DECODE_BACKEND_KEYS 下标。
_DECODE_BACKEND_ALIASES: dict[str, str] = {
    "auto": "auto",
    "cpu": "cpu", "decord/cpu": "cpu",
    "nvdec": "nvdec", "gpu": "nvdec", "decord/gpu": "nvdec",
    "hybrid": "hybrid", "gpu+cpu-hybrid": "hybrid",
    "decord/gpu+cpu-hybrid": "hybrid",
}


def _decode_backend_combo_index(raw: str) -> int:
    """CSV backend 标签 → 解码下拉框下标（未知标签回退 auto）。"""
    key = _DECODE_BACKEND_ALIASES.get(str(raw).strip().lower(), "auto")
    if key not in config.DECODE_BACKEND_KEYS:
        return 0
    return config.DECODE_BACKEND_KEYS.index(key)


class ExportControllerMixin:
    """依赖宿主：video_path/metadata/_settings/_export_thread/_pipeline 等状态。"""

    def _gui_mark(self, mark: str) -> None:
        from monitor import gui_mark
        gui_mark(mark)

    def _import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "",
            "CSV 文件 (*.csv);;所有文件 (*.*)")
        if not path:
            return
        from ocr_engine import parse_csv_header, parse_csv_setting
        settings = parse_csv_header(path)
        if not settings:
            QMessageBox.warning(self, "导入失败", "无法解析 CSV 文件头。")
            return
        s = self._settings

        # ── ROI（特殊：4 个 spinbox）──
        if "roi" in settings:
            parts = parse_csv_setting("roi", settings["roi"])
            if isinstance(parts, (list, tuple)) and len(parts) == 4:
                for spin in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
                    spin.blockSignals(True)
                self.roi_x2.setValue(parts[2]); self.roi_y2.setValue(parts[3])
                self.roi_x1.setValue(parts[0]); self.roi_y1.setValue(parts[1])
                for spin in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
                    spin.blockSignals(False)
                self._preview_widget.set_roi(
                    self.roi_x1.value(), self.roi_y1.value(),
                    self.roi_x2.value(), self.roi_y2.value())

        # ── 数值字段（统一使用共享解析器）──
        _num_fields = {
            "max_speed": s["max_speed_edit"], "max_accel": s["max_accel_edit"],
            "force_aspect": s["force_aspect_edit"],
            "frame_start": s["frame_start_edit"], "frame_end": s["frame_end_edit"],
        }
        for key, widget in _num_fields.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                widget.setText(str(val))

        _spin_fields = {
            "buffer": s["buffer_spin"],
            "fill_width": s["fill_width_spin"],
        }
        for key, widget in _spin_fields.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                widget.setValue(val)

        # ── 下拉框字段：CSV backend 写的是 decord/CPU|GPU|GPU+CPU-hybrid 标签，兼容旧头 ──
        _combo_map = {
            "backend": (s["backend_combo"], _decode_backend_combo_index),
            # CSV 的 ocr_backend 记录实际引擎：onnxruntime → CPU；
            # tensorrt+onnxruntime 是历史实验混合，GUI 无对应项，归一到 auto
            "ocr_backend": (s["ocr_backend_combo"], {
                "auto": 0, "cpu": 1, "onnxruntime": 1,
                "tensorrt": 2, "tensorrt+onnxruntime": 0,
            }),
        }
        for key, (combo, mapping) in _combo_map.items():
            val = parse_csv_setting(key, settings.get(key, ""))
            if val is not None:
                if callable(mapping):
                    idx = mapping(val)
                else:
                    idx = mapping.get(str(val).lower(), 0)
                combo.setCurrentIndex(idx)
        if "format" in settings:
            fmt = settings["format"].lower()
            for rb, key in [(s["format_ms"], "m/s"), (s["format_kmh"], "km/h"),
                            (s["format_mph"], "mile/h")]:
                if key == fmt:
                    rb.setChecked(True); break
        self._status_label.setText(f"已导入设置: {Path(path).name}")

    def _export_csv(self) -> None:
        if self.video_path is None or self.metadata is None:
            QMessageBox.warning(self, "未导入视频", "请先导入视频。"); return
        roi = (self.roi_x1.value(), self.roi_y1.value(),
               self.roi_x2.value(), self.roi_y2.value())
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            QMessageBox.warning(self, "识别范围不完整", "请先填写或拖拽选择识别范围。"); return

        out, _ = QFileDialog.getSaveFileName(self, "保存 CSV",
            str(self.video_path.parent / f"{self.video_path.stem}_log.csv"),
            "CSV 文件 (*.csv)")
        if not out: return

        s = self._settings
        try:
            ms = float(s["max_speed_edit"].text())
            ma = float(s["max_accel_edit"].text())
            bu = s["buffer_spin"].value()
            fa = float(s["force_aspect_edit"].text())
            pp = s["fill_width_spin"].value()
            monitor_enabled = s["monitor_checkbox"].isChecked()
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请检查数值参数。"); return

        # 断开旧线程信号，防止泄漏到新线程
        self._teardown_export_thread()

        self._export_btn.setEnabled(False); self._cancel_btn.setEnabled(True)

        # ── 检查可选依赖可用性 ──
        try:
            import decord  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "decord 加载失败",
                "视频解码需要自建 decord fork（PyPI 版不支持）。\n\n"
                "修复：运行 setup_venv.bat，或从 chr431/decord 获取发布产物到 _decord_build\\")
            self._finish_export()
            return
        self._export_thread = ExportThread(
            video_path=self.video_path,
            roi=roi,
            max_speed_kmh=ms, max_accel_mps2=ma,
            buffer_size=bu, decode_backend=config.DECODE_BACKEND_KEYS[
                s["backend_combo"].currentIndex()],
            ocr_backend=config.OCR_BACKEND_KEYS[
                s["ocr_backend_combo"].currentIndex()],
            fill_width=pp,
            force_aspect=fa,
            speed_format=self.speed_format,
            frame_start=s["frame_start_edit"].text(),
            frame_end=s["frame_end_edit"].text(),
            monitor_enabled=monitor_enabled,
            # YUV420 解码输出（decord ≥0.7.10）：分段/OCR 只取 Y 平面，
            # 代表帧保留 YUV 供最终检查前转 RGB 预览
            yuv_output=True,
            output_path=Path(out),
            parent=self,
        )
        self._export_thread.progress_updated.connect(self._on_progress)
        self._export_thread.finished.connect(self._on_done)
        self._export_thread.error_occurred.connect(self._on_error)
        self._export_thread.cancelled.connect(self._on_cancel)
        self._export_thread.pipeline_ready.connect(self._on_pipeline_ready)
        self._export_thread.start()

    def _on_pipeline_ready(self, pipeline, output_path) -> None:
        self._pipeline = pipeline
        self._review_output_path = output_path

    def _cancel_export(self) -> None:
        if self._export_thread: self._export_thread._cancel_flag = True
        self._cancel_btn.setEnabled(False); self._status_label.setText("正在取消...")

    def _on_progress(self, msg: str, pct: float) -> None:
        self._status_label.setText(msg); self._progress_bar.setValue(int(pct))

    def _on_done(self, mode: str) -> None:
        if self.sender() is not self._export_thread:
            return
        self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._export_thread = None
        self._show_final_check()

    def _show_final_check(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        out = getattr(self, "_review_output_path", None)
        if pipeline is None or out is None:
            return
        # 段级 review：段值 + 代表帧彩色预览（YUV 在最终检查前转 RGB），
        # 用户可改段值（段内不混速度，改段=改全段）
        try:
            pipeline.prepare_review_rgb()
        except Exception:
            pass  # 转换失败保留灰度/原数组，review 仍可用
        from gui_review import ReviewDialog  # 延迟导入（pyqtgraph ~0.8s）
        dlg = ReviewDialog(self, pipeline.segments,
                           pipeline._max_speed, pipeline._max_accel,
                           pipeline._fps or 1.0)
        self._gui_mark("final_check: before exec")
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        corrections = dlg.get_corrections()
        self._gui_mark("final_check: dialog closed")
        # finalize（写 CSV/统计）在 frozen 环境可能被安全扫描拖慢数秒 →
        # 后台执行，完成后回主线程收尾
        import threading as _th

        def _finalize_bg():
            self._gui_mark("final_check: finalize start")
            if accepted and corrections:
                vals = [seg["value"] for seg in pipeline.segments]
                for si, v in corrections.items():
                    if 0 <= si < len(vals):
                        vals[si] = v
                pipeline.finalize(out, vals,
                                  pinned_indices=set(corrections.keys()))
            else:
                pipeline.finalize(out)
            self._gui_mark("final_check: finalize done")
            QTimer.singleShot(0, lambda: (self._finish_export(),
                                           self._status_label.setText("最终检查完成 — 结果已保存。"),
                                           self._gui_mark("final_check: finish_export done")))
        _th.Thread(target=_finalize_bg, daemon=True).start()

    def _on_error(self, err: str) -> None:
        if self.sender() is not self._export_thread:
            return
        self._finish_export(); QMessageBox.critical(self, "导出失败", err)

    def _on_cancel(self) -> None:
        if self.sender() is not self._export_thread:
            return
        self._finish_export(); self._status_label.setText("已取消。")

    def _teardown_export_thread(self) -> None:
        """拆除导出线程：断开全部信号并释放引用（幂等）。"""
        if self._export_thread is not None:
            try:
                self._export_thread.progress_updated.disconnect()
                self._export_thread.finished.disconnect()
                self._export_thread.error_occurred.disconnect()
                self._export_thread.cancelled.disconnect()
                self._export_thread.pipeline_ready.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._export_thread = None

    def _finish_export(self) -> None:
        self._export_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._teardown_export_thread()
        # Release pipeline memory (crops etc.) on cancel/error
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            import logging
            _log = logging.getLogger("RaceVideoToLog.gui")
            try:
                from video_utils import rss_mb, sum_nbytes
                _raw_mb = sum_nbytes(list(pipeline.crops.values())) / 1e6
                _log.info("[MEM] _finish_export PRE-clear: crops=%d(%.1fMB) rss=%.0fMB",
                    len(pipeline.crops), _raw_mb, rss_mb())
            except Exception:
                pass
            pipeline.crops.clear()
            import gc; gc.collect()
            try:
                from video_utils import rss_mb
                _log.info("[MEM] _finish_export POST-clear: rss=%.0fMB", rss_mb())
            except Exception:
                pass
            self._pipeline = None
        self._gui_mark("finish_export: pipeline cleared")
