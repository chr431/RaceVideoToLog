"""RaceVideoToLog 集中配置 — 常量、颜色、公共 API。"""
from __future__ import annotations

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子
KMH_TO_MPS: float = 1.0 / 3.6    # km/h → m/s 转换因子

# ═══════════════════ 模型默认值 ═══════════════════
DEFAULT_OCR_MODEL: str = "v6_tiny"     # 主 OCR 模型
DEFAULT_REOCR_MODEL: str = "v6_small"  # 重 OCR 模型

# ═══════════════════ 图表颜色 ═══════════════════
COLOR_BLUE: str = "#2196F3"
COLOR_ORANGE: str = "#FF5722"
COLOR_GREEN: str = "#4CAF50"
COLOR_RED: str = "#F44336"
COLOR_GRAY: str = "#888888"
COLOR_LIGHT_GRAY: str = "#666666"
COLOR_LIGHTER_GRAY: str = "#aaaaaa"
COLOR_BG_DARK: str = "#2a2a2a"
COLOR_BG_LIGHT: str = "#ffffff"
COLOR_FG_DARK: str = "#e0e0e0"
COLOR_FG_LIGHT: str = "#333333"

def chart_colors(dark: bool) -> tuple[str, str]:
	"""返回当前主题的 (background, foreground) 颜色对。"""
	return (COLOR_BG_DARK, COLOR_FG_DARK) if dark else (COLOR_BG_LIGHT, COLOR_FG_LIGHT)

# ═══════════════════ 速度单位转换 ═══════════════════
SOURCE_TO_KMH: dict[str, float] = {
	"m/s": MPS_TO_KMH,
	"km/h": 1.0,
	"mile/h": 1.609344,
}

# ═══════════════════ GPU 后端公共 API ═══════════════════
_gpu_backend: str = "CPU"

def get_gpu_backend() -> str:
	"""返回当前实际使用的 GPU 后端名称（CUDA 或 CPU）。"""
	return _gpu_backend

# ═══════════════════ LCS 局部一致性评分 ═══════════════════
LCS_TIME_WINDOW: float = 0.5         # 时间窗 (秒)
LCS_TAU: float = 0.06                 # 指数衰减常数 exp(-dt/tau)
LCS_HIGH_WEIGHT: float = 3.0         # pinned 帧权重倍率
LCS_ERROR_LOW: float = 0.3           # detect: < this = error
LCS_TRUST_HIGH: float = 0.7          # error/borderline 分界 & HIGH_TRUST 标记阈值
LCS_WARNING_THRESHOLD: float = 0.5   # 人工修正加速度警告阈值
LCS_CONFIDENCE_MIN_SCORE: float = 30.0  # find_problem_segments 默认 min_score
LCS_INTERP_WEIGHT: float = 0.25        # 插值接近度权重 (加性)
LCS_NOVELTY_WEIGHT: float = 0.10       # 新颖性权重（非原始OCR加分）

# ═══════════════════ 中值滤波参考剖面验证 ═══════════════════
# 在 HIGH_TRUST 标记前，用中值滤波剖面检测"一致性孤岛"
# ——局部物理自洽但偏离全局趋势的 OCR 误读
PROFILE_TIME_WINDOW: float = 0.5       # 中值滤波时间窗口 (秒)，根据实际帧间隔折算帧数
PROFILE_MIN_WINDOW: int = 5            # 最小滤波窗口 (帧数)，确保低帧率下仍有效
PROFILE_ABS_TOLERANCE: float = 4.0     # 绝对偏差容许 (km/h)
PROFILE_PCT_TOLERANCE: float = 0.02    # 相对偏差容许 (比例, 0.02=2%)

# ═══════════════════ 纠错迭代参数 ═══════════════════
CORRECTION_MAX_ROUNDS: int = 4         # Stage 4 最大迭代轮数
FILL_MAX_PASSES: int = 10              # Stage 5 最大填充轮数
CORRECTION_ACCEPT_MIN_SCORE: float = 0.35  # 接受修正的最低 LCS 分数
