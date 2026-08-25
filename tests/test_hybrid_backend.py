"""hybrid 解码后端选项回归：config 键位 / 引擎构造 / GUI CSV 导入映射。"""
from __future__ import annotations

import config
from video_ocr_engine import FieldExtractor


def test_decode_backend_keys_include_hybrid():
    assert "hybrid" in config.DECODE_BACKEND_KEYS
    assert config.DECODE_BACKEND_LABELS.get("hybrid"), \
        "hybrid 后端必须有 GUI 显示标签"


def test_field_extractor_accepts_hybrid():
    """构造即接受（不打开解码器）；引擎 `_open_vr` 负责激活/回退。"""
    ex = FieldExtractor("dummy.mp4", (0, 0, 10, 10), decode_backend="hybrid")
    assert ex._decode_backend == "hybrid"


def test_export_controller_backend_mapping():
    from export_controller import _decode_backend_combo_index

    assert _decode_backend_combo_index("hybrid") == \
        config.DECODE_BACKEND_KEYS.index("hybrid")
    assert _decode_backend_combo_index("decord/GPU+CPU-hybrid") == \
        config.DECODE_BACKEND_KEYS.index("hybrid")
    assert _decode_backend_combo_index("gpu+cpu-hybrid") == \
        config.DECODE_BACKEND_KEYS.index("hybrid")
    # 旧头兼容
    assert _decode_backend_combo_index("decord/cpu") == \
        config.DECODE_BACKEND_KEYS.index("cpu")
    assert _decode_backend_combo_index("decord/gpu") == \
        config.DECODE_BACKEND_KEYS.index("nvdec")
    # 未知标签回退 auto
    assert _decode_backend_combo_index("unknown") == \
        config.DECODE_BACKEND_KEYS.index("auto")
