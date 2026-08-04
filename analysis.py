"""RaceVideoToLog — 数据分析工具函数（CSV 解析 / 平滑）。

GUI 图表已迁移至 gui_analysis.py（pyqtgraph）；本模块仅保留
parse_csv / smooth_data 等纯函数。CLI 图片导出已移除（v2.8.0+）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ocr_engine import _savgol_filter_np


# ═══════════════════════════════════════════════════════════════
# 模块级工具函数
# ═══════════════════════════════════════════════════════════════

def parse_csv(path: str | Path) -> tuple[list[float], list[float], list[float], list[int]]:
    """解析 CSV 文件，返回 (times_s, dists, speeds, flags)。

    每行格式: frame_index,distance,speed_kmh,flag
    - 从 # 注释头读取 fps，将帧号转换为实际时间（秒）
    - 跳过以 # 开头的注释行和空行
    - try/except 保护浮点转换
    - 裁剪起始零速帧，距离和时间归零
    """
    import re
    # ── 解析 CSV 头，提取 fps ──
    fps = 0.0
    with open(str(path), "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            m = re.search(r"\bfps=([\d.]+)", line)
            if m:
                try:
                    fps = float(m.group(1))
                except ValueError:
                    pass
    if fps <= 0:
        fps = 1.0  # fallback: treat frame index as-is

    times, dists, speeds, flags = [], [], [], []
    with open(str(path), "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    frame_idx = float(parts[0])
                    times.append(frame_idx / fps)  # frame → seconds
                    dists.append(float(parts[1]))
                    speeds.append(float(parts[2]))
                    flags.append(int(parts[3]) if len(parts) > 3 else 0)
                except ValueError:
                    continue
    # 裁剪起始零速帧，并将时间轴和距离轴归零
    start = 0
    for i, s in enumerate(speeds):
        if s > 0:
            start = i
            break
    if start > 0:
        times = times[start:]
        speeds = speeds[start:]
        dists = dists[start:]
        flags = flags[start:]
    # 始终归零时间轴和距离轴（不受 --frame-start 参数影响）
    if times:
        base_time = times[0]
        times = [t - base_time for t in times]
    if dists:
        base_dist = dists[0]
        dists = [d - base_dist for d in dists]
    return times, dists, speeds, flags


def smooth_data(xv: "np.ndarray | list[float]", yv: "np.ndarray | list[float]", strength: int) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay 滤波（纯 numpy 实现）：多项式滑动窗口拟合，保留峰谷形状。

    Args:
        xv: x 轴数据
        yv: y 轴数据
        strength: 平滑强度 (0-100)，0 表示不平滑
    Returns:
        (xv_array, smoothed_yv) — xv 保持不变，yv 平滑后
    """
    if strength <= 0 or len(xv) < 5:
        return np.array(xv, dtype=float), np.array(yv, dtype=float)
    win = int(len(xv) * strength / 100.0 * 0.0175)
    win = max(5, min(win, len(xv) - 2))
    if win % 2 == 0:
        win += 1
    polyorder = min(3, win - 1)
    sy = _savgol_filter_np(np.array(yv, dtype=float), win, polyorder)
    return np.array(xv, dtype=float), sy


# ═══════════════════════════════════════════════════════════════
# GUI: 数据分析 Tab
# ═══════════════════════════════════════════════════════════════
# 已迁移至 gui_analysis.py (PySide6)
# 从 gui_analysis import AnalysisTab

# ═══════════════════════════════════════════════════════════════
# CLI: 无头分析导出
# ═══════════════════════════════════════════════════════════════
