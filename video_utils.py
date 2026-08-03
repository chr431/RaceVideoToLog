"""视频/数据通用工具。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import hashlib


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
def codec_from_fourcc(fourcc: float) -> str:
    value = int(fourcc)
    if value == 0:
        return "Unknown"
    chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
    codec = "".join(chars).strip("\x00").strip()
    return codec or "Unknown"
def safe_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
def _parse_int_or_none(s: str) -> int | None:
    """解析字符串为 int，空字符串返回 None。"""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None
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
