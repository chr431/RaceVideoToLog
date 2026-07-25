"""RaceVideoToLog 集中配置 — 常量、颜色、公共 API。"""
from __future__ import annotations

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子
KMH_TO_MPS: float = 1.0 / 3.6    # km/h → m/s 转换因子

# ═══════════════════ 用户可配置默认值 ═══════════════════
DEFAULT_BACKEND: str = "auto"           # GPU 后端 (auto / tensorrt / cpu)
DEFAULT_VIDEO_BACKEND: str = "cv2"     # 视频解码器 (cv2 / decord)，cv2 兼容性好且内存低
DEFAULT_OCR_MODEL: str = "v6_tiny"     # 主 OCR 模型
DEFAULT_REOCR_MODEL: str = "v6_small"  # 重 OCR 模型
DEFAULT_SPEED_FORMAT: str = "km/h"     # 速度单位 (km/h / m/s / mile/h)
DEFAULT_FRAME_DIV: int = 2             # 采样间隔 (1=每帧, 2=隔帧)
DEFAULT_MAX_SPEED: float = 400.0       # 最大速度 (km/h)
DEFAULT_MAX_ACCEL: float = 50.0        # 最大加速度 (m/s²)
DEFAULT_TARGET_H: int = 48             # OCR 预处理目标高度 (px)
DEFAULT_PAD: int = 0                   # OCR 预处理 padding (px)
DEFAULT_BUFFER_SIZE: int = 16          # 生产者-消费者队列缓冲大小
DEFAULT_LOG_LEVEL: str = "normal"      # 日志级别 (normal / detailed / debug)
DEFAULT_CORRECTION_MODE: str = "auto"  # 纠错模式 (auto / manual)
OCR_REC_BATCH_NUM: int = 12            # OCR 识别批处理大小

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

def set_gpu_backend(backend: str) -> None:
    """由 gpu_setup 调用，设置实际使用的 GPU 后端。"""
    global _gpu_backend
    _gpu_backend = backend

# ═══════════════════ LCS 局部一致性评分 ═══════════════════
LCS_TIME_WINDOW: float = 0.5         # 时间窗 (秒)
LCS_TAU: float = 0.06                 # 指数衰减常数 exp(-dt/tau)
LCS_HIGH_WEIGHT: float = 3.0         # pinned 帧权重倍率
LCS_ERROR_LOW: float = 0.3           # detect: < this = error
LCS_TRUST_HIGH: float = 0.75         # error/borderline 分界 & HIGH_TRUST 标记阈值
LCS_WARNING_THRESHOLD: float = 0.5   # 人工修正加速度警告阈值
LCS_CONFIDENCE_MIN_SCORE: float = 30.0  # find_problem_segments 默认 min_score
LCS_INTERP_WEIGHT: float = 0.25        # 插值接近度权重 (加性)
LCS_NOVELTY_WEIGHT: float = 0.10       # 新颖性权重（非原始OCR加分）

# ═══════════════════ 错误检测：多信号置信度评分 ═══════════════════
ERROR_DETECT_OCR_CONF_WEIGHT: float = 0.01   # OCR 模型内部置信度（几乎不可靠）
ERROR_DETECT_PHYSICS_WEIGHT: float = 0.20    # 物理可达性（仅可靠邻居）
ERROR_DETECT_LINEARITY_WEIGHT: float = 0.20  # 局部线性度
ERROR_DETECT_REOCR_AGREE_WEIGHT: float = 0.20 # OCR 读数自洽
ERROR_DETECT_TEXT_LEN_WEIGHT: float = 0.14   # 文本长度信号
ERROR_DETECT_ACCEL_SPIKE_WEIGHT: float = 0.40 # 加速度尖峰对检测（一致性孤岛）
ERROR_DETECT_SG_DEVIATION_WEIGHT: float = 0.15 # 中值滤波偏离度（辅助）
ERROR_DETECT_CANDIDATE_THRESHOLD: int = 65

# ═══════════════════ Viterbi DP ═══════════════════
VITERBI_OBS_WEIGHT: float = 0.3
VITERBI_ACCEL_WEIGHT: float = 1.0
VITERBI_SOFT_ANCHOR_CONFIDENCE: int = 85
VITERBI_MAX_CANDIDATES: int = 40

# ═══════════════════ 纠错参数 ═══════════════════
MANUAL_CORRECT_THRESHOLD: int = 40
AUTO_CORRECT_THRESHOLD: int = 80
CORRECTION_MAX_ROUNDS: int = 10        # Viterbi 多轮迭代
FILL_MAX_PASSES: int = 50
CORRECTION_MIN_DIFF: float = 0.5
AUTO_SMOOTH_CLUSTER_MAX: int = 5
AUTO_SMOOTH_DEVIATION_MULT: float = 5.0
REOCR_HEIGHTS: tuple = (24, 32, 48)

# ═══════════════════ 问题段检测参数 ═══════════════════
ACCEL_ANOMALY_THRESHOLD: float = 10.0  # 邻帧加速度异常阈值 (km/h)
MAX_SUGGESTED_FRAMES: int = 8          # 每个问题段最多建议帧数
PROBLEM_MIN_SEGMENT_LEN: int = 3       # 问题段最小连续帧数

# ═══════════════════ 部分数字扩展参数 ═══════════════════
MAX_PARTIAL_WILDCARDS: int = 2         # expand_partial 最大通配符数
