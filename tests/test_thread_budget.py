"""OCR 线程预算单测：auto_ocr_thread_count / _ocr_num_threads 语义。

锁定"根本性解决抢核"的预算规则：OCR 吃满全部物理核，解码走 NVDEC
卸载或 fork 默认线程（不抢核）；RVTOL_OCR_THREADS env 钩子优先。
v2.15.2 新增少核 CPU 软解分核：物理核 ≤8 且 CPU 解码时 OCR 与 FFmpeg
各分 cores//2（实测 4 核 -15%、8 核 -14%）；核数多/GPU 解码保持全核。
"""
from __future__ import annotations

import config
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


def test_split_cores_on_low_core_cpu_decode(monkeypatch):
    """少核 + CPU 软解：OCR 与解码显式分核（cores//2）。"""
    monkeypatch.delenv("RVTOL_OCR_THREADS", raising=False)
    p = _pipe()
    p._backend = "decord/CPU"
    if cpu_physical_cores() <= config.CPU_CORES_SPLIT_THRESHOLD:
        assert p._ocr_num_threads() == max(2, cpu_physical_cores() // 2)
        assert p._decode_num_threads() == max(2, cpu_physical_cores() // 2)
    else:  # 本机核数多：保持全核，解码用 decord 默认（None）
        assert p._ocr_num_threads() == cpu_physical_cores()
        assert p._decode_num_threads() is None
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cpu_physical_cores()
    assert p._decode_num_threads() is None
