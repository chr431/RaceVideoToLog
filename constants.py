"""共享常量：Flag 枚举与 OCR 数字正则。"""
from __future__ import annotations
import re


class Flag:
    """速度数据 flag 值 — 统一标记每帧数据的来源和可信度。

    信任层级（由低到高）:
        RAW (0)           — 原始 OCR 值，未经修正
        REOCR_AUTO (11)   — 重 OCR 自动修正
        FILL_INTERP (12)  — 物理插值填充
        PARTIAL_AUTO (13) — 部分数字模式推断修正
        HIGH_TRUST (21)   — Viterbi+物理验证，自动高可信帧
        PINNED (22)       — 用户手动修正，绝对真值
    """
    RAW: int = 0
    REOCR_AUTO: int = 11
    FILL_INTERP: int = 12
    PARTIAL_AUTO: int = 13
    HIGH_TRUST: int = 21
    PINNED: int = 22

    @classmethod
    def is_corrected(cls, flag: int) -> bool:
        """是否为自动纠错帧 (10-19)。"""
        return 10 <= flag <= 19

    @classmethod
    def is_trusted(cls, flag: int) -> bool:
        """是否为高可信帧 — HIGH_TRUST / PINNED (20-29)。"""
        return 20 <= flag <= 29

OCR_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")  # noqa: E305

# 常见 OCR 数字混淆映射（对称）。build_speed_candidates 和
# 低置信度短文本帧的 Phase-2 候选扩展使用。
CONFUSION_MAP: dict[str, list[str]] = {
    "0": ["8", "6", "9"],
    "1": ["7", "2", "4", "9"],
    "2": ["7", "1", "3", "9"],
    "3": ["8", "9", "2", "5"],
    "4": ["7", "9", "1"],
    "5": ["6", "3", "8", "9"],
    "6": ["8", "5", "0", "2"],
    "7": ["1", "2", "4"],
    "8": ["0", "6", "3", "5", "9"],
    "9": ["8", "3", "5", "0", "4", "1", "2"],
}
