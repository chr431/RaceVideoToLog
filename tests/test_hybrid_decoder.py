"""CPU+NVDEC 混合解码单测：后端识别 / 切分比例 / 解码 worker 与按序消费。

用假 reader（get_batch 返回确定性数组）锁定混合路径的契约，不需要
真实 decord/GPU：_decode_range_worker 的队列顺序、灰度/清晰度/bin
计算、哨兵终止、异常传播，以及 CPU/GPU 区间切分的无重叠全覆盖。
真实解码集成（含无 GPU 回退）见 test_decoder_integration.py。
"""
from __future__ import annotations

import threading
from queue import Queue

import numpy as np
import pytest

from segment_flow import (_decode_range_worker, _drain_queue,
                          _hybrid_ranges, SegmentPipeline)


class _Arr:
    """decord NDArray 最小替身（get_batch 返回对象带 asnumpy()）。"""

    def __init__(self, a: np.ndarray):
        self._a = a

    def asnumpy(self) -> np.ndarray:
        return self._a


class FakeReader:
    """假 VideoReader：get_batch(frames, roi=...) 返回确定性 (B,H,W,3)。"""

    def __init__(self, tag: int = 0):
        self.tag = tag

    def get_batch(self, frames, roi=None):
        h, w = 6, 8
        b = len(frames)
        arr = np.zeros((b, h, w, 3), dtype=np.uint8)
        for k, fi in enumerate(frames):
            arr[k, :, :, 0] = fi % 256           # 帧号可追溯
            arr[k, :, :, 1] = (fi // 256) % 256
            arr[k, :, :, 2] = self.tag
        return _Arr(arr)


def _pipe(**kw):
    defaults = dict(video_path="x", roi=(0, 0, 10, 10), max_speed_kmh=400.0,
                    max_accel_mps2=50.0, fps=30.0, frame_start=None,
                    frame_end=None)
    defaults.update(kw)
    return SegmentPipeline(**defaults)


# ═══════════════ 后端识别与切分比例 ═══════════════

@pytest.mark.parametrize("backend,expected", [
    ("cpu+nvdec", True), ("CPU+NVDEC", True), ("hybrid", True),  # 显式旧用法恒混合
    ("auto", False), ("cpu", False), ("nvdec", False), (None, False),
])
def test_is_hybrid_env_off(backend, expected):
    # 默认（RVTOL_HYBRID_DECODE 未设）：只有显式 cpu+nvdec/hybrid 才混合
    assert _pipe(decode_backend=backend)._is_hybrid() is expected


@pytest.mark.parametrize("backend,expected", [
    ("auto", True), ("nvdec", True), (None, True),   # GPU 系 → 混合
    ("cpu", False), ("CPU", False),                 # CPU 不受影响
])
def test_is_hybrid_env_on(monkeypatch, backend, expected):
    # RVTOL_HYBRID_DECODE=1：GPU 模式（auto/nvdec）内部改走混合
    import config
    monkeypatch.setenv(config.HYBRID_DECODE_ENV, "1")
    assert _pipe(decode_backend=backend)._is_hybrid() is expected


@pytest.mark.parametrize("bad", ["0", "off", "false", "no", "", "abc"])
def test_hybrid_env_off_values(monkeypatch, bad):
    import config
    monkeypatch.setenv(config.HYBRID_DECODE_ENV, bad)
    assert _pipe(decode_backend="auto")._is_hybrid() is False


def test_hybrid_split_default():
    import config
    assert _pipe()._hybrid_split() == config.HYBRID_CPU_SPLIT


def test_hybrid_split_env_priority(monkeypatch):
    monkeypatch.setenv("RVTOL_HYBRID_SPLIT", "0.42")
    assert _pipe()._hybrid_split() == pytest.approx(0.42)


def test_hybrid_split_env_invalid_falls_back(monkeypatch):
    import config
    p = _pipe()
    for bad in ("abc", "0", "1", "-0.5", "1.5", ""):
        monkeypatch.setenv("RVTOL_HYBRID_SPLIT", bad)
        assert p._hybrid_split() == config.HYBRID_CPU_SPLIT


def test_hybrid_split_av1_forces_pure_gpu(monkeypatch):
    # AV1 特判：CPU 软解 AV1 极耗核且与 GPU 段并发竞争拖慢 GPU 吞吐
    # （实测混合 19.1s vs 纯 GPU 14.4s）→ 返回 0（CPU 段空，等效纯 GPU）。
    # 该特判优先于 env 覆盖（任何 split 对 AV1 都是回归）。
    import config
    p = _pipe()
    p._hybrid_codec = "h264"
    assert p._hybrid_split() == config.HYBRID_CPU_SPLIT
    monkeypatch.setenv("RVTOL_HYBRID_SPLIT", "0.42")
    assert p._hybrid_split() == pytest.approx(0.42)  # 非 AV1 仍走 env
    p._hybrid_codec = "av1"
    assert p._hybrid_split() == 0.0                  # AV1 特判压过 env


# ═══════════════ 区间切分 ═══════════════

def test_hybrid_ranges_partition_no_overlap():
    frames = list(range(100))
    cpu_fis, gpu_fis = _hybrid_ranges(frames, 10, 0.55)
    assert cpu_fis == frames[10:55]
    assert gpu_fis == frames[55:]
    assert set(cpu_fis) | set(gpu_fis) == set(frames[10:]), "并集=全片"
    assert not (set(cpu_fis) & set(gpu_fis)), "零重叠"


def test_hybrid_ranges_tiny_video():
    # 校准窗口覆盖全片 → 无并行区间（两段皆空，生产走纯校准路径）
    frames = list(range(30))
    cpu_fis, gpu_fis = _hybrid_ranges(frames, 50, 0.55)
    assert cpu_fis == [] and gpu_fis == []


def test_hybrid_ranges_split_at_least_calib():
    # split 比例再小也不能切进校准窗口（校准帧由 CPU reader 独占）：
    # split_pos 夹到 calib_n → CPU 无额外段，GPU 解其余全部
    frames = list(range(100))
    cpu_fis, gpu_fis = _hybrid_ranges(frames, 40, 0.1)
    assert cpu_fis == []
    assert gpu_fis == frames[40:100]


# ═══════════════ 解码 worker + 按序消费 ═══════════════

def test_decode_range_worker_order_and_features():
    fis = [10, 11, 12, 13]
    q: Queue = Queue()
    err: list = []
    t = threading.Thread(target=_decode_range_worker,
                         args=(FakeReader(7), fis, q, (0, 0, 8, 6), None,
                               err, 2), daemon=True)
    t.start()
    items = list(_drain_queue(q))
    t.join()
    assert err == []
    assert [it[0] for it in items] == fis, "按帧号顺序入队"
    for it in items:
        fi, c, g, s, b = it
        assert c.shape == (6, 8, 3) and c.dtype == np.uint8
        assert g.shape == (6, 8) and g.dtype == np.uint8
        assert isinstance(s, float) and s >= 0.0
        assert b is None, "th=None 时 bin 不计算（参考路径）"


def test_decode_range_worker_bin_when_threshold():
    fis = [0, 1, 2]
    q: Queue = Queue()
    err: list = []
    t = threading.Thread(target=_decode_range_worker,
                         args=(FakeReader(), fis, q, (0, 0, 8, 6), 128,
                               err, 4), daemon=True)
    t.start()
    items = list(_drain_queue(q))
    t.join()
    assert err == []
    assert [it[0] for it in items] == fis
    assert all(it[4].shape == (6, 8) and it[4].dtype == bool for it in items)


def test_decode_range_worker_error_propagates():
    class BoomReader(FakeReader):
        def get_batch(self, frames, roi=None):
            raise RuntimeError("boom")

    q: Queue = Queue()
    err: list = []
    t = threading.Thread(target=_decode_range_worker,
                         args=(BoomReader(), [0, 1], q, (0, 0, 8, 6), None,
                               err), daemon=True)
    t.start()
    assert list(_drain_queue(q)) == [], "异常后只放哨兵，不放残缺数据"
    t.join()
    assert len(err) == 1 and "boom" in str(err[0])
