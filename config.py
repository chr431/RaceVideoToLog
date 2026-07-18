"""RaceVideoToLog 集中配置 — 常量、默认值、命名颜色。

消除散布在多个文件中的魔法数字、硬编码颜色和重复默认值。
"""
from __future__ import annotations

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子
KMH_TO_MPS: float = 1.0 / 3.6    # km/h → m/s 转换因子

# ═══════════════════ 默认参数 ═══════════════════
DEFAULT_MAX_SPEED: float = 400.0       # km/h
DEFAULT_MAX_ACCEL: float = 50.0        # m/s²
DEFAULT_TARGET_H: float = 24.0         # OCR 高度 px
DEFAULT_PAD: float = 0.0               # 边缘填充 px
DEFAULT_DIV: int = 2                   # 采样间隔 1/N
DEFAULT_BUFFER: int = 8                # 缓冲队列大小
DEFAULT_BACKEND: str = "auto"          # OCR 后端
DEFAULT_OCR_MODEL: str = "v6_small"    # OCR 模型
DEFAULT_SPEED_FORMAT: str = "km/h"     # 速度格式
DEFAULT_SMOOTH: int = 0                # 图表平滑强度

# ═══════════════════ 纠错参数 ═══════════════════
# 锚点选择
ANCHOR_WINDOW_SEC: float = 0.3         # 自适应窗口覆盖的时间 (秒)
ANCHOR_MAX_DEV: float = 4.0            # 锚点最大偏差 (km/h)
ANCHOR_NEIGHBOR_MAX_DEV: float = 10.0  # 锚点邻居最大偏差 (km/h)
# 错误检测容限
DETECTOR_MAX_ACCEL_MARGIN: float = 1.2  # 邻帧跳变容限乘数
DETECTOR_VSHAPE_MARGIN: float = 2.5     # V 字形检测容限乘数
DETECTOR_CLIFF_MARGIN: float = 3.0      # 悬崖检测容限乘数
DETECTOR_CLIFF_FLAT_RATIO: float = 0.3  # 悬崖对侧平坦比例
DETECTOR_LOCAL_WINDOW: int = 5          # 局部趋势窗口 (帧数)
DETECTOR_LOCAL_MAX_DEV: float = 3.0     # 局部趋势最大偏差 (km/h)
DETECTOR_SPIKE_RANGE: int = 2           # 孤立离群检测范围
# 候选评分权重
CANDIDATE_NEIGHBOR_WEIGHT: float = 0.4
CANDIDATE_ANCHOR_WEIGHT: float = 0.35
CANDIDATE_SMOOTH_WEIGHT: float = 0.25
# 迭代限制
MAX_CORRECTION_ROUNDS: int = 3
MAX_FILL_PASSES: int = 10

# ═══════════════════ 置信度评分权重 ═══════════════════
CONFIDENCE_OCR_DEV_WEIGHT: float = 0.3
CONFIDENCE_ACCEL_WEIGHT: float = 0.4
CONFIDENCE_CORRECTED_PENALTY: float = 30.0
CONFIDENCE_SMOOTH_WEIGHT: float = 0.2
CONFIDENCE_MIN_SEGMENT_LEN: int = 3
CONFIDENCE_MIN_SCORE: float = 70.0

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
# 原 _gpu_backend 改为通过此 API 访问，不再需要 _oe._gpu_backend
_gpu_backend: str = "CPU"

def get_gpu_backend() -> str:
	"""返回当前实际使用的 GPU 后端名称（CUDA 或 CPU）。"""
	return _gpu_backend
