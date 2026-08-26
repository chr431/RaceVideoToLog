"""应用侧视频工具（引擎 0.7.0 移除的轻量 helper 回归应用层）。

video_ocr_engine 0.7.0「公共 API 清理」删除了 open_decord_vr /
VideoMetadata / format_duration / rss_mb / sum_nbytes（引擎仓库内零调用）。
RaceVideoToLog 的 GUI 预览与内存诊断仍需要这些轻量函数，统一放在本模块
（引擎不再承担，应用侧单一事实源）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    """视频元信息（GUI 状态栏/诊断展示）。"""

    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int


def format_duration(seconds: float) -> str:
    """秒 → "h:mm:ss" 或 "m:ss"。"""
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def open_decord_vr(video_path, force_cpu: bool = False):
    """打开 decord reader —— GPU (NVDEC) 优先，失败回退 CPU。

    Returns: (VideoReader, label) where label is 'GPU' or 'CPU'。
    DECORD_FORCE_CPU=1 或 force_cpu=True 时跳过 GPU。
    GUI 预览需要 RGB 帧，不传 output_format（默认 RGB）。
    """
    from decord import VideoReader as _VR

    _vr = None
    _label = "CPU"
    _force = force_cpu or os.environ.get(
        "DECORD_FORCE_CPU", "").strip() == "1"

    if not _force:
        try:
            from decord import gpu as _decord_gpu
            _vr = _VR(str(video_path), ctx=_decord_gpu(0))
            _label = "GPU"
        except Exception:
            pass

    if _vr is None:
        try:
            from decord import cpu as _decord_cpu
            _vr = _VR(str(video_path), ctx=_decord_cpu(0))
        except Exception:
            raise RuntimeError(
                f"无法打开视频（decord 不可用或路径错误）: {video_path}")
    return _vr, _label


def rss_mb() -> float:
    """当前进程 RSS（MB）；psutil 缺失返回 -1。"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def sum_nbytes(seq) -> int:
    """序列中 ndarray/bytes 元素的总字节数（兼容 (frame, bytes) 二元组）。"""
    s = 0
    for x in seq:
        if hasattr(x, "nbytes"):
            s += x.nbytes
        elif hasattr(x, "__len__") and len(x) == 2:
            if hasattr(x[1], "nbytes"):
                s += x[1].nbytes
    return s
