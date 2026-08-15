"""OCR 线程预算单测：auto_ocr_thread_count / _ocr_num_threads 语义。

锁定"根本性解决抢核"的预算规则：OCR 吃满全部物理核，解码走 NVDEC
卸载或 fork 默认线程（不抢核）；RVTOL_OCR_THREADS env 钩子优先。
"""
from __future__ import annotations

from segment_flow import SegmentPipeline
from ocr_native import auto_ocr_thread_count, cpu_physical_cores


def _pipe(**kw):
    defaults = dict(video_path="x", roi=(0, 0, 10, 10), max_speed_kmh=400.0,
                    max_accel_mps2=50.0, fps=30.0, frame_start=None,
                    frame_end=None)
    defaults.update(kw)
    return SegmentPipeline(**defaults)


def test_physical_cores_positive():
    assert cpu_physical_cores() >= 2


def test_auto_budget_is_all_physical_cores():
    # 两种解码后端统一：全物理核（实测满负荷正收益；超物理核不提升）
    assert auto_ocr_thread_count() == cpu_physical_cores()


def test_env_hook_priority(monkeypatch):
    p = _pipe()
    monkeypatch.setenv("RVTOL_OCR_THREADS", "6")
    assert p._ocr_num_threads() == 6, "显式 env 钩子优先于 auto 预算"


def test_auto_budget_without_env(monkeypatch):
    monkeypatch.delenv("RVTOL_OCR_THREADS", raising=False)
    p = _pipe()
    # 未打开解码器（_backend 未设）→ 保守按 CPU 预算 = 全物理核（统一规则）
    assert p._ocr_num_threads() == cpu_physical_cores()
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cpu_physical_cores()
    p._backend = "decord/CPU"
    assert p._ocr_num_threads() == cpu_physical_cores()
