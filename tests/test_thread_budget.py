"""OCR 线程预算单测：auto_ocr_thread_count / _ocr_num_threads 语义。

锁定"根本性解决抢核"的预算规则：OCR 吃满全部物理核，解码走 NVDEC
卸载或 fork 默认线程（不抢核）；OCR_THREADS env 钩子优先。
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
    monkeypatch.setenv("OCR_THREADS", "6")
    assert p._ocr_num_threads() == 6, "显式 env 钩子优先于 auto 预算"


def test_auto_budget_without_env(monkeypatch):
    monkeypatch.delenv("OCR_THREADS", raising=False)
    p = _pipe()
    cores = cpu_physical_cores()
    # 未打开解码器（_backend 未设）→ 保守按 CPU 预算 = 全物理核（统一规则）
    assert p._ocr_num_threads() == cores
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cores
    p._backend = "decord/CPU"
    expect = max(2, cores // 2) if cores <= config.CPU_CORES_SPLIT_THRESHOLD \
        else cores
    assert p._ocr_num_threads() == expect


def test_split_cores_on_low_core_cpu_decode(monkeypatch):
    """少核 + CPU 软解：OCR 与解码显式分核（cores//2）。

    _decode_num_threads 只由 CPU 解码调用方（_open_vr 的 CPU 分支）使用，
    返回值与 _backend 无关（按物理核判定）。
    """
    monkeypatch.delenv("OCR_THREADS", raising=False)
    p = _pipe()
    p._backend = "decord/CPU"
    cores = cpu_physical_cores()
    if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        expect = max(2, cores // 2)
        assert p._ocr_num_threads() == expect
        assert p._decode_num_threads() == expect
    else:  # 核数多：保持全核，解码用 decord 默认（None）
        assert p._ocr_num_threads() == cores
        assert p._decode_num_threads() is None
    # GPU 解码：OCR 保持全核（_decode_num_threads 与 backend 无关，
    # GPU 分支在调用方不传 num_threads）
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cores


def test_av1_cpu_decode_allocates_more_to_decode(monkeypatch):
    """AV1 + CPU 软解：解码/OCR 对半分（cores//2，任何核数）。

    新 decord DLL（max_frame_delay≥16 恢复 dav1d 帧并行）后解码吞吐翻倍，
    平衡点回到对半分（实测 test6：16 核 dcd=8/ocrT=8 → 45.7s vs 12/4
    58.5s、8 核 dcd=4/ocrT=4 → 72.1s vs 6/2 91.9s）。
    """
    monkeypatch.delenv("OCR_THREADS", raising=False)
    p = _pipe()
    p._backend = "decord/CPU"
    p._codec = "av1"
    cores = cpu_physical_cores()
    expect = max(2, cores // 2)
    assert p._decode_num_threads(codec="av1") == expect
    assert p._ocr_num_threads() == expect
    # GPU 解码 AV1：OCR 全核（不触发 AV1 让核）
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cores
