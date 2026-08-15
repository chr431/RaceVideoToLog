"""信号处理：纯 numpy Savitzky-Golay 滤波。

邻帧一致性评分（_neighbor_consistency_score*）在 v2.13 段级化后已无调用者，
已随其 CONSISTENCY_* 参数一并移除。
"""
from __future__ import annotations

import numpy as np

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
