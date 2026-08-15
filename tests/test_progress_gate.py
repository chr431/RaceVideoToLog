"""并行进度收敛测试：解码∥OCR 并行时进度不得回退/阶段文字反复横跳。"""
from __future__ import annotations

from segment_flow import _ProgressGate


def test_progress_gate_monotonic_and_phase_forward():
    seen: list[tuple[str, float]] = []
    gate = _ProgressGate(lambda msg, pct: seen.append((msg, pct)))

    gate("解码+分段...", 2.0)
    gate("正在构建引擎...", 2.5)          # 同阶段更高百分比 → 显示
    gate("解码+分段: 100/100", 58.0)      # 解码完成
    gate("[OCR] 段 1", 58.0)              # 进入 OCR 阶段，同百分比也要切换
    gate("[OCR] 段 2", 58.3)
    gate("解码+分段: 100/100", 58.0)      # 并行解码晚到，禁止倒回解码
    gate("[OCR] 段 3", 86.0)
    gate("[OCR] 段 4", 86.0)              # 同阶段同百分比重复 → 丢弃
    gate("检测纠正...", 88.0)
    gate("完成", 100.0)

    msgs = [m for m, _ in seen]
    pcts = [p for _, p in seen]
    assert pcts == sorted(pcts)
    assert msgs.index("[OCR] 段 1") > msgs.index("解码+分段: 100/100")
    assert msgs.index("检测纠正...") > msgs.index("[OCR] 段 3")
    assert "[OCR] 段 4" not in msgs, "同阶段同百分比重复消息应被丢弃"


def test_progress_gate_drops_backward():
    seen: list[float] = []
    gate = _ProgressGate(lambda msg, pct: seen.append(pct))
    gate("x", 80.0)
    gate("y", 40.0)   # 回退 → 丢弃
    assert seen == [80.0]
