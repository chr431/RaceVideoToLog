"""YUV420(NV12) 工具测试：Y 平面 range 展开与 RGB 转换。"""
from __future__ import annotations

import numpy as np

from video_utils import (_nv12_luma, _nv12_luma_full, nv12_to_rgb)


def _gray_expected(raw_y: np.ndarray, color_range: int) -> np.ndarray:
    if color_range == 1:
        return raw_y
    v = (raw_y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def test_luma_full_limited_and_full():
    y = np.arange(256, dtype=np.uint8).reshape(16, 16)
    crop = np.zeros((16 + 8, 16), dtype=np.uint8)
    crop[:16] = y
    assert _nv12_luma(crop).shape == (16, 16)
    assert np.array_equal(_nv12_luma_full(crop, 1), y)
    assert np.array_equal(_nv12_luma_full(crop, 0), _gray_expected(y, 0))


def test_nv12_to_rgb_shape_even_and_odd():
    # 灰度 YUV（U=V=128）：R=G=B，形状为 (h, w, 3)
    for h, w in ((4, 6), (5, 7), (1, 1)):
        rows = h + (h + 1) // 2
        crop = np.zeros((rows, w), dtype=np.uint8)
        crop[:h] = 180
        crop[h:] = 128
        rgb = nv12_to_rgb(crop)
        assert rgb.shape == (h, w, 3)
        assert rgb.dtype == np.uint8
        # decord BT.601 矩阵语义：gray = clip(1.164383*(Y-16))
        expect = np.clip(np.rint(1.164383 * (180 - 16)), 0, 255)
        assert int(np.abs(rgb.astype(int) - int(expect)).max()) <= 1
        assert np.all(rgb[..., 0] == rgb[..., 1])
        assert np.all(rgb[..., 1] == rgb[..., 2])


def test_nv12_to_rgb_passthrough_rgb_array():
    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    arr[..., 0] = 1
    assert np.array_equal(nv12_to_rgb(arr), arr)
