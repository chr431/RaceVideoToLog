"""OCR 线程预算单测：auto_ocr_thread_count / _ocr_num_threads 语义。

锁定"根本性解决抢核"的预算规则：OCR 吃满全部物理核；OCR_THREADS env
钩子优先。v2.15.2 新增少核 CPU 软解分核：物理核 ≤8 且 CPU 解码时 OCR
与 FFmpeg 各分 cores//2（实测 4 核 -15%、8 核 -14%）。
引擎 0.8.0 起接管解码线程预算（_decode_num_threads），0.9.0 收敛为
恒分档（不再对多核返回 None）——分档规则见
test_split_cores_on_low_core_cpu_decode 的 docstring。
"""
from __future__ import annotations

import os

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
    """解码线程分档（引擎 0.8.0 起接管，0.9.0 收敛为恒分档）。

    _decode_num_threads 不再返回 None（旧"多核用 decord 默认 8 线程"
    白丢 ~28% 解码吞吐，见引擎 extractor._decode_num_threads 注释）：
      · OCR 在 GPU（默认 auto/TRT）：吃满逻辑核，钳 [8, 32]
      · OCR 在 CPU + 少核（物理 ≤8）：cores//2
      · OCR 在 CPU + 多核：stride>1 → 逻辑核 3/4 钳 [8,24]；
        stride==1 → 逻辑核 1/3 钳 [8,12]
    """
    monkeypatch.delenv("OCR_THREADS", raising=False)
    p = _pipe()
    p._backend = "decord/CPU"
    cores = cpu_physical_cores()
    logical = os.cpu_count() or cores
    if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        expect = max(2, cores // 2)
        assert p._ocr_num_threads() == expect
        assert p._decode_num_threads() == expect
    else:  # 核数多：OCR 保持全核；解码按 OCR 落点分档（默认 GPU → 逻辑核钳 [8,32]）
        assert p._ocr_num_threads() == cores
        assert p._decode_num_threads() == max(
            config.DECODE_THREADS_GPU_OCR_MIN,
            min(config.DECODE_THREADS_GPU_OCR_MAX, logical))
    # GPU 解码：OCR 保持全核（_decode_num_threads 与 _backend 无关，
    # 只看 _ocr_backend；GPU 分支在调用方不传 num_threads）
    p._backend = "decord/GPU"
    assert p._ocr_num_threads() == cores


def test_cpu_ocr_stride_tiers_decode_threads(monkeypatch):
    """OCR 在 CPU + 多核：解码线程按采样步长分档（引擎 0.9.0 契约）。"""
    monkeypatch.delenv("OCR_THREADS", raising=False)
    p = _pipe(ocr_backend="cpu")
    p._backend = "decord/CPU"
    cores = cpu_physical_cores()
    if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        expect = max(2, cores // 2)
        assert p._ocr_num_threads() == expect
        assert p._decode_num_threads() == expect
        return
    logical = os.cpu_count() or cores
    assert p._ocr_num_threads() == cores
    # stride==1（OCR 受限）：逻辑核 1/3 钳 [8, 12]
    assert p._decode_num_threads() == max(
        8, min(config.DECODE_THREADS_CPU_OCR_STRIDE1_MAX, logical // 3))
    # stride>1（解码受限）：逻辑核 3/4 钳 [8, 24]
    p._sample_stride = 8
    assert p._decode_num_threads() == max(
        8, min(config.DECODE_THREADS_CPU_OCR_MAX, logical * 3 // 4))


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
