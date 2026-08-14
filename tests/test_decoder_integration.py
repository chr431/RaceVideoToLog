"""decord fork 解码集成测试。

CI 的 decoder-smoke job 从 chr431/decord GitHub release 下载 fork 并安装
后，本模块**真实运行**（CPU 软解，无需 GPU）；本机/CI 缺 decord 时显式
跳过并给出原因（不是静默通过）。PyPI 版 decord（缺 next_roi/get_batch/
get_codec）也会被 _require_fork_api 显式跳过。

测试视频：tests/fixtures/videos/smoke_speedo.mp4（60 帧 160×100 30fps
h264；生成参考脚本同目录 generate_smoke_video.py）。
参考哈希在 decord v0.7.4 上采集——软件解码同一构建逐位确定，可跨机器
比对；升级 decord 版本或重新生成视频导致哈希变化时需重新采集（属有意变更）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "decord",
    reason="decord fork 未安装（PyPI 版不支持）——解码集成测试未覆盖。"
           "本地: 运行 setup_venv.bat；CI: decoder-smoke job 自动下载安装。")

VIDEO = Path(__file__).parent / "fixtures" / "videos" / "smoke_speedo.mp4"
ROI = (10, 30, 130, 90)          # 半开区间 (x1, y1, x2, y2)
EXPECT_FRAMES = 60
EXPECT_FPS = 30.0
EXPECT_ROI_SHAPE = (60, 120, 3)  # (y2-y1, x2-x1, 3)
EXPECT_ROI0_SHA16 = "4045c7a10a945e95"  # decord v0.7.4 首帧 ROI 确定性校验


def _require_fork_api():
    from decord import VideoReader
    missing = [a for a in ("next_roi", "get_batch", "get_codec")
               if not hasattr(VideoReader, a)]
    if missing:
        pytest.skip(f"decord 非自建 fork（缺 {missing}）—— 解码集成测试未覆盖")


def _roi0_sha16(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def test_open_metadata():
    _require_fork_api()
    from decord import VideoReader, cpu
    vr = VideoReader(str(VIDEO), ctx=cpu(0))
    try:
        assert len(vr) == EXPECT_FRAMES
        assert abs(float(vr.get_avg_fps()) - EXPECT_FPS) < 0.5
        codec = vr.get_codec()
        assert codec and isinstance(codec, str)
    finally:
        del vr


def test_next_roi_shape_and_determinism():
    _require_fork_api()
    from decord import VideoReader, cpu
    vr = VideoReader(str(VIDEO), ctx=cpu(0))
    try:
        f0 = vr.next_roi(*ROI).asnumpy()
        assert f0.shape == EXPECT_ROI_SHAPE
        assert f0.dtype == np.uint8
        assert _roi0_sha16(f0) == EXPECT_ROI0_SHA16, (
            "首帧 ROI 哈希与 decord v0.7.4 参考不一致（解码行为变化，"
            "请确认是否有意）")
    finally:
        del vr


def test_get_batch_roi_matches_next_roi():
    """fork 关键 API：get_batch(roi) 与逐帧 next_roi 逐位一致（生产路径依赖）。"""
    _require_fork_api()
    from decord import VideoReader, cpu
    vr_b = VideoReader(str(VIDEO), ctx=cpu(0))
    try:
        fb = vr_b.get_batch([0, 10, 20, 30], roi=ROI).asnumpy()
        assert fb.shape == (4,) + EXPECT_ROI_SHAPE
    finally:
        del vr_b
    # 独立 reader 逐帧核对（get_batch 会推进读指针，不能复用同一 reader）
    vr_s = VideoReader(str(VIDEO), ctx=cpu(0))
    try:
        for want, arr in ((0, fb[0]), (10, fb[1]), (20, fb[2]), (30, fb[3])):
            vr_s.seek_accurate(want)
            single = vr_s.next_roi(*ROI).asnumpy()
            assert np.array_equal(arr, single), \
                f"帧 {want}: get_batch 与 next_roi 不一致"
    finally:
        del vr_s


def test_gray_output_single_channel():
    """fork 关键 API：output_format='gray'（CPU 软解省 RGB→灰转换）。"""
    _require_fork_api()
    from decord import VideoReader, cpu
    vr = VideoReader(str(VIDEO), ctx=cpu(0), output_format="gray")
    try:
        g = vr.next_roi(*ROI).asnumpy()
        assert g.shape == (EXPECT_ROI_SHAPE[0], EXPECT_ROI_SHAPE[1], 1)
        rgb = VideoReader(str(VIDEO), ctx=cpu(0)).next_roi(*ROI).asnumpy()
        gray = (rgb.astype(np.float32)
                @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        # 容差 2：色度下采样/舍入路径允许轻微偏差
        assert np.abs(g[..., 0].astype(np.float32) - gray).max() <= 2.0
    finally:
        del vr


def test_hybrid_decode_all_complete_and_close_to_cpu():
    """cpu+nvdec（无 GPU 自动回退 CPU）：_decode_all 产出完整有序帧集。

    灰度与纯 CPU 参考一致：无 GPU 时逐位相同；有 GPU 时混合后半段
    来自 GPU 解码（同帧跨后端灰度差实测 ≤38，且不产生假分段——
    完整漏斗门禁另测）。容差 40 防解码器舍入差异，抓错帧/错裁剪。
    """
    _require_fork_api()
    from segment_flow import SegmentPipeline
    roi_closed = (ROI[0], ROI[1], ROI[2] - 1, ROI[3] - 1)  # 闭合框语义
    pipe = SegmentPipeline(str(VIDEO), roi_closed, 400.0, 50.0,
                           30.0, None, None, decode_backend="cpu+nvdec")
    frames, crops, grays, sharp = pipe._decode_all()
    assert frames == list(range(EXPECT_FRAMES))
    assert list(crops) == frames, "crops 完整且有序"
    assert set(grays) == set(frames) == set(sharp)
    ref = SegmentPipeline(str(VIDEO), roi_closed, 400.0, 50.0,
                          30.0, None, None, decode_backend="cpu")
    _rf, _rc, rgrays, _rs = ref._decode_all()
    for fi in frames:
        d = int(np.abs(grays[fi].astype(int)
                       - rgrays[fi].astype(int)).max())
        assert d <= 40, f"帧 {fi} 灰度与纯 CPU 参考差 {d}"
