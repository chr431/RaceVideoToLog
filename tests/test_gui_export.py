"""ExportThread 参数完整性：GUI 导出线程必须接受 gray/yuv 两种输出标志。"""
from __future__ import annotations

import inspect


def test_export_thread_accepts_yuv_output():
    from gui_export import ExportThread
    params = inspect.signature(ExportThread.__init__).parameters
    assert "yuv_output" in params
    assert "gray_output" in params
