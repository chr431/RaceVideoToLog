"""信号处理：SG 滤波与邻帧一致性评分。"""
from __future__ import annotations
import math

import numpy as np

from config import (CONSISTENCY_TIME_WINDOW, CONSISTENCY_DECAY_TAU,
    CONSISTENCY_PINNED_WEIGHT)

# Savitzky-Golay 卷积系数缓存（按 (window, polyorder) 复用）
_sg_coeff_cache: dict = {}


def _savgol_filter_np(y: "np.ndarray", window_length: int, polyorder: int) -> "np.ndarray":
    """纯 numpy Savitzky-Golay 滤波 — 预计算卷积系数，O(N) 复杂度。

    等价于 scipy.signal.savgol_filter，但无 scipy 依赖。
    通过预计算伪逆系数 + np.convolve 实现，比逐点 lstsq 快 10-100x。
    """
    if window_length % 2 == 0 or window_length < 1:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be > polyorder")
    half = window_length // 2
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < window_length:
        return y.copy()

    # ── 预计算卷积系数（缓存复用）──
    cache_key = (window_length, polyorder)
    if cache_key not in _sg_coeff_cache:
        x = np.arange(-half, half + 1, dtype=float)
        A = np.vander(x, polyorder + 1, increasing=True)
        # pinv(A)[0] = 多项式常数项 a0 的系数 = 中心点的平滑值
        _sg_coeff_cache[cache_key] = np.linalg.pinv(A)[0]
    coeffs = _sg_coeff_cache[cache_key]

    # ── 卷积应用（O(N)）──
    result = np.convolve(y, coeffs[::-1], mode="same")

    # ── 边界处理：用最近的有效滤波值填充 ──
    if half > 0 and n > half:
        result[:half] = result[half]
        result[-half:] = result[-half - 1]

    return result
def _neighbor_consistency_score_lr(i: int, v: float, rows: list, times: list[float],
                    max_speed_kmh: float, max_accel_mps2: float,
                    time_window: float = CONSISTENCY_TIME_WINDOW, tau: float = CONSISTENCY_DECAY_TAU,
                    high_weight: set[int] | None = None) -> tuple[float, float]:
    """邻域左右分侧一致性评分。权重 = exp(-dt/tau)。

    Returns: (left_score, right_score)
    - 单侧无邻居时默认 1.0（不惩罚边界帧）
    - high_weight 中的帧额外 ×CONSISTENCY_PINNED_WEIGHT 权重
    """
    import math
    n = len(rows)
    if v < 0 or v > max_speed_kmh:
        return 0.0, 0.0
    t_i = times[i]
    hw = high_weight or set()

    def _scan(start: int, stop: int, step: int) -> float:
        votes = 0.0
        total = 0.0
        for j in range(start, stop, step):
            dt = abs(times[j] - t_i)
            if dt > time_window:
                break
            v_j = rows[j][2]
            if v_j < 0 or v_j > max_speed_kmh:
                continue
            max_dv = max_accel_mps2 * dt * MPS_TO_KMH
            exp_w = math.exp(-dt / tau)
            pin_w = CONSISTENCY_PINNED_WEIGHT if j in hw else 1.0
            total += exp_w * pin_w
            if abs(v - v_j) <= max_dv:
                votes += exp_w * pin_w
        return votes / total if total > 0 else 1.0

    left = _scan(i - 1, -1, -1)
    right = _scan(i + 1, n, 1)
    return left, right
def _neighbor_consistency_score(i: int, v: float, rows: list, times: list[float],
                            max_speed_kmh: float, max_accel_mps2: float,
                            time_window: float = CONSISTENCY_TIME_WINDOW, tau: float = CONSISTENCY_DECAY_TAU,
                            high_weight: set[int] | None = None) -> float:
    """邻域一致性合并分数（向后兼容）。左右侧权重合并后的单值分数。"""
    left, right = _neighbor_consistency_score_lr(i, v, rows, times, max_speed_kmh, max_accel_mps2,
                                    time_window, tau, high_weight)
    return (left + right) / 2.0
