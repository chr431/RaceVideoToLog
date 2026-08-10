"""视频/数据通用工具。"""
from __future__ import annotations
import os as _os
from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

import numpy as np

from config import OCR_GAMMA as _OCR_GAMMA_DEFAULT

# OCR 预处理灰度权重（与 segment_flow._gray 一致）。
_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class VideoMetadata:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int
@dataclass
class SpeedObservation:
    timestamp: float
    raw_speed_kmh: int
    raw_text: str
    confidence: float = 0.0  # OCR model confidence [0, 1], 0 if unavailable
def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"
def _parse_int_or_none(s: str) -> int | None:
    """解析字符串为 int，空字符串返回 None。"""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None
def next_frame_roi(vr, x1: int, y1: int, x2: int, y2: int) -> "np.ndarray":
    """Grab next frame, returning the ROI crop (RGB uint8 HxWx3).

    Uses ``vr.next_roi(x1, y1, x2, y2)`` (half-open bounds, numpy slice
    semantics — pass closed bounds + 1) when the DLL provides it; the GPU
    path copies only the ROI from device memory.  Falls back to
    ``vr.next()`` + Python crop for old DLLs / CPU builds.  Raises
    StopIteration on EOF.
    """
    roi = getattr(vr, "next_roi", None)
    if roi is not None:
        arr = roi(x1, y1, x2, y2)
        f = arr.asnumpy()
        if f.ndim == 3 and f.shape[2] == 3:
            # GPU：next_roi 已裁剪；CPU reader 的 next_roi 回退全帧
            # （decord 的 NextFrameRoi 对 CPU 返回 NextFrameImpl()）→ 裁剪
            if f.shape[0] != y2 - y1 or f.shape[1] != x2 - x1:
                return f[y1:y2, x1:x2].copy()
            return f
        raise StopIteration()
    f = vr.next().asnumpy()
    # 必须 .copy()：视图会引用整个 6MB 全帧缓冲区，调用方持有返回值
    # （如 pipeline 的 raw_frames）时每帧泄漏一帧（实测 3000 帧 → 18GB）
    return f[y1:y2, x1:x2].copy()


def clamp_region(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    return x1, y1, x2, y2
def compute_video_hash(video_path: str | Path, chunk_size: int = 1_048_576) -> str:
    """计算视频文件的快速哈希（头尾各 1MB + 文件大小）。

    使用 SHA-256，足以唯一标识视频文件，同时避免读取整个大文件。
    """
    import hashlib
    video_path = Path(video_path)
    if not video_path.exists():
        return "N/A"
    file_size = video_path.stat().st_size
    h = hashlib.sha256()
    h.update(str(file_size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(chunk_size))
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)
            h.update(f.read(chunk_size))
    return h.hexdigest()[:16]  # 前 16 字符足够区分


@lru_cache(maxsize=64)
def _resize_map(src_w: int, src_h: int, new_w: int, new_h: int):
    """双线性坐标映射（缓存）：只依赖输入/输出尺寸，与像素无关。

    主流水线每帧同一 ROI 调 _np_resize（目标尺寸恒定）→ 映射缓存
    后每帧省去 arange/clip/cast 等 ~60% 的 numpy 工作量。
    """
    scale_x = src_w / new_w
    scale_y = src_h / new_h
    src_x = np.clip((np.arange(new_w) + 0.5) * scale_x - 0.5, 0, src_w - 1)
    src_y = np.clip((np.arange(new_h) + 0.5) * scale_y - 0.5, 0, src_h - 1)
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    return x0, x1, y0, y1, wx, wy


def _np_resize(img: "np.ndarray", new_w: int, new_h: int) -> "np.ndarray":
    """双线性 resize（float32），与 cv2.resize INTER_LINEAR 像素对齐一致。

    坐标映射复刻 OpenCV：src = (dst + 0.5) * scale - 0.5（像素中心对齐）。
    与 cv2 的数值差 <= 1e-5（浮点累加顺序），无实际影响；输出 float32。
    移除 cv2 依赖后的轻量替代（EXE -83MB）。
    """
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _resize_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    wx3 = wx[None, :, None]
    wy3 = wy[:, None, None]
    return ((1 - wx3) * (1 - wy3) * f[y0[:, None], x0[None, :]] +
            wx3 * (1 - wy3) * f[y0[:, None], x1[None, :]] +
            (1 - wx3) * wy3 * f[y1[:, None], x0[None, :]] +
            wx3 * wy3 * f[y1[:, None], x1[None, :]])


def _preprocess_standard(crop: "np.ndarray", target_h: int, pad: int,
                         max_width: int = 0,
                         gamma: "float | None" = None) -> "np.ndarray":
    """标准预处理：resize（numpy 双线性）+ 可选宽度限制 + 灰度 gamma + 边缘填充。

    max_width > 0 时限制宽度上限（px），用于纠正扁宽字体
    （如数字高度≈宽度时设为 96 可恢复 ~2:1 高宽比）。
    主识别（pipeline）与 re-OCR（correction）共用，保证一致。
    输出 float32（与 cv2 路径数值差 <= 1e-5）。

    gamma：灰度对比度增强指数（255*(gray/255)^g）。None = 用 env
    RVTOL_OCR_GAMMA，都没有则 config.OCR_GAMMA（正式默认 2.0）。
    白字黄底等背景色块场景放大高段分离，平滑无裁剪不侵蚀笔画。
    gamma <= 0 跳过灰度变换（保留 RGB，回退旧行为）；
    灰度权重 [0.299,0.587,0.114] 与 segment_flow._gray 一致。
    """
    h, w = crop.shape[:2]
    if target_h < 8:
        raise ValueError(f"target_h 必须 >= 8，当前为 {target_h}")
    new_w = max(1, int(w * target_h / h)) if h > 0 else w
    if max_width > 0:
        new_w = min(new_w, max_width)
    if abs(target_h / h - 1.0) > 0.02:
        resized = _np_resize(crop, new_w, target_h)
    else:
        resized = crop.astype(np.float32)
    if gamma is None:
        _env = _os.environ.get("RVTOL_OCR_GAMMA")
        gamma = float(_env) if _env else float(_OCR_GAMMA_DEFAULT)
    if gamma > 0:
        # 灰度 + gamma（正式预处理）：RGB 逐通道 gamma 视觉差异小、回归多
        # （tools/_gamma_misread_montage 对比），灰度版视觉更清晰、回归少。
        gray = resized @ _GRAY_W                          # (h, w) float32
        resized = 255.0 * np.power(gray / 255.0, gamma)
        resized = np.stack([resized] * 3, axis=-1)
    if pad > 0:
        resized = np.pad(resized, ((pad, pad), (pad, pad), (0, 0)),
                         mode="edge")  # 等价 cv2 BORDER_REPLICATE
    return resized


def open_decord_vr(video_path, force_cpu: bool = False):
    """Open video with decord — GPU (NVDEC) preferred, CPU fallback.

    Returns (VideoReader, label) where label is ``'GPU'`` or ``'CPU'``.
    Set ``DECORD_FORCE_CPU=1`` in the environment or pass *force_cpu=True*
    to skip GPU even when available.
    """
    from decord import VideoReader as _VR

    _vr = None
    _label = "CPU"
    _force = force_cpu or _os.environ.get("DECORD_FORCE_CPU", "").strip() == "1"

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
        except ModuleNotFoundError:
            raise RuntimeError(
                "decord 未安装（需要自建 fork，PyPI 版不支持）。"
                "请运行 setup_venv.bat 或从 chr431/decord 获取发布产物到 _decord_build\\")
        except Exception as _e:
            raise RuntimeError(f"decord 无法打开视频: {_e}")

    return _vr, _label


def rss_mb() -> float:
    """当前进程 RSS（MB）。psutil 缺失返回 -1。"""
    try:
        import psutil
        return psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
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
