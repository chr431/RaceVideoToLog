"""RaceVideoToLog 集中配置 — 常量、颜色、公共 API。"""
from __future__ import annotations

__version__ = "2.6.1"

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

# ═══════════════════ 邻帧一致性评分（GUI 人工修正校验）═══════════════════
CONSISTENCY_TIME_WINDOW: float = 0.5    # 时间窗 (秒)
CONSISTENCY_DECAY_TAU: float = 0.06     # 指数衰减常数 exp(-dt/tau)
CONSISTENCY_PINNED_WEIGHT: float = 3.0  # 已固定帧权重倍率
MANUAL_EDIT_ACCEL_WARNING: float = 0.5  # 人工修正加速度警告阈值

# ═══════════════════ 错误检测：多信号置信度评分 ═══════════════════
ERROR_DETECT_OCR_CONF_WEIGHT: float = 0.01   # OCR 模型内部置信度（修复后可用）
ERROR_DETECT_PHYSICS_WEIGHT: float = 0.20    # 物理可达性
ERROR_DETECT_LINEARITY_WEIGHT: float = 0.20  # 局部线性度
ERROR_DETECT_TEXT_LEN_WEIGHT: float = 0.14   # 文本长度信号
ERROR_DETECT_ACCEL_SPIKE_WEIGHT: float = 0.40 # 加速度尖峰对检测
ERROR_DETECT_SG_DEVIATION_WEIGHT: float = 0.15 # 中值滤波偏离度（唯一全局信号）
ERROR_DETECT_CANDIDATE_THRESHOLD: int = 65

# ═══════════════════ Viterbi DP ═══════════════════
VITERBI_OBS_WEIGHT: float = 0.3
VITERBI_ACCEL_WEIGHT: float = 1.0
VITERBI_TRUSTED_BOUNDARY_CONFIDENCE: int = 85
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

# ═══════════════════ 平滑 + 自动对齐参数 ═══════════════════
SMOOTHNESS_MAX_ITERATIONS: int = 10       # _smoothness_pass 最大迭代轮数
AUTO_ALIGN_DIFF_MIN_KMH: int = 5         # auto-align 最小修正量 (km/h)
AUTO_ALIGN_DIFF_MAX_KMH: int = 25        # auto-align 最大修正量 (km/h)
AUTO_ALIGN_NUDGE_FACTOR: float = 0.8     # auto-align 向插值修正的比例
AUTO_ALIGN_MIN_CHANGE_KMH: int = 3       # auto-align 最小提交变化量 (km/h)
AUTO_ALIGN_SG_DEV_UPPER: int = 70        # SG偏离度上限（高于此值跳过 auto-align）
AUTO_ALIGN_SG_DEV_LOWER: int = 20        # SG偏离度下限（低于此值跳过 auto-align）
AUTO_ALIGN_FALLBACK_MAX_DV: float = 4.0  # 无法获取 fps 时的 fallback max_dv (km/h)

# ═══════════════════ Force-SG 平滑参数 ═══════════════════
FORCE_SG_MAX_ITERATIONS: int = 15        # _force_sg_smooth 最大迭代轮数
FORCE_SG_NUDGE_FACTOR: float = 0.7       # force-SG 向中值修正的比例
FORCE_SG_THRESHOLD_MULT: float = 1.2     # force-SG max_dv 阈值倍率
FORCE_SG_MIN_CHANGE_KMH: int = 1         # force-SG 最小变化量 (km/h)

# ═══════════════════ 候选值后过滤 ═══════════════════
CANDIDATE_POSTFILTER_PHYSICS_MIN: int = 90    # 自洽帧 physics 最低阈值
CANDIDATE_POSTFILTER_LINEARITY_MIN: int = 90  # 自洽帧 linearity 最低阈值
CANDIDATE_HUNDREDS_MAX_DIFF: int = 100         # 百位变体最大允许差值 (km/h)
CANDIDATE_ISLAND_SG_DEV_MAX: int = 20          # 孤岛帧 SG偏离度上限

# ═══════════════════ 参考值构建保护 ═══════════════════
REF_GUARD_PHYSICS_MIN: int = 90      # 跳过插值参考值的 physics 阈值
REF_GUARD_LINEARITY_MIN: int = 90    # 跳过插值参考值的 linearity 阈值
REF_GUARD_SG_DEV_MIN: int = 80       # 跳过插值参考值的 sg_dev 阈值
DISTANT_INTERP_MIN_DISTANCE: int = 30     # 远距离插值跳过帧数
DISTANT_INTERP_ISLAND_THRESHOLD: int = 30  # 孤岛检测距离阈值 (km/h)
REF_INTERP_MAX_KMH_DIFF: int = 50    # 插值参考值最大允许偏差 (km/h)
MANUAL_REF_CONFIDENCE_MAX: int = 40  # 手动模式构建参考值的置信度上限

# ═══════════════════ Viterbi 后处理 ═══════════════════
VITERBI_POST_TRUST_THRESHOLD: int = 70     # Viterbi 后信任判定最低分数
TRUST_WINDOW_FALLBACK_MAX_DV: float = 8.0  # 信任窗口 fallback max_dv (km/h)
TRUST_NEIGHBOR_SEARCH_WINDOW: int = 3      # 信任传播邻居搜索窗口（每侧3帧）
FILL_CONFIDENCE_THRESHOLD: int = 30        # Fill 阶段的置信度阈值
FINAL_CONF_BLEND_PHASE1: float = 0.7       # 最终置信度中 Phase 1 权重
FINAL_CONF_BLEND_VITERBI: float = 0.3      # 最终置信度中 Viterbi 权重

# ═══════════════════ Viterbi DP 内部常量 ═══════════════════
VITERBI_FALLBACK_DT: float = 1.0 / 30.0    # 时间戳无效时的 fallback dt (秒)
VITERBI_ANCHOR_COST: float = 0.1           # 锚点/边界帧的极低成本
VITERBI_CHANGE_THRESHOLD_KMH: float = 0.5   # Viterbi 结果与 raw 差异 < 此值不计为修正
VITERBI_OBS_COST_FALLBACK_MULT: float = 0.1 # raw<=0 时的观测代价倍率
VITERBI_MIN_MAX_COST: float = 0.01          # dp_cost 归一化初始最小值
VITERBI_COST_NORM_CONF_EXCLUDE: int = 80    # 排除出归一化池的 Phase1 置信度阈值
VITERBI_ANCHOR_CONF_THRESHOLD: int = 90     # 标记为锚点帧的 Phase1 置信度阈值
VITERBI_CONF_NORMAL_MIN: int = 80           # Viterbi 置信度"正常"等级下限
VITERBI_CONF_MARGINAL_MIN: int = 40         # Viterbi 置信度"存疑"等级下限

# ═══════════════════ 错误检测信号内部常量 ═══════════════════
PHYSICS_FALLBACK_DT: float = 0.02            # 物理信号 fallback dt (秒, ~50fps)
LINEARITY_NEIGHBOR_MAX_LOOK: int = 120       # 线性度信号邻居搜索最大帧数
LINEARITY_DECAY_FACTOR: float = 3.0          # 线性度指数衰减系数
ACCEL_SPIKE_VIOLATION_MULT: float = 2.0      # 加速度尖峰阈值倍率
ACCEL_SPIKE_SEARCH_WINDOW: int = 15          # 对立尖峰搜索窗口 (帧)
SG_WINDOW_HALF_MIN: int = 31                 # SG中值滤波窗口最小半宽
SG_WINDOW_HALF_FPS_MULT: float = 2.5         # SG窗口 fps 倍率
SG_WINDOW_HALF_FALLBACK: int = 150           # SG窗口半宽 fallback
SG_DEV_REL_THRESHOLD: float = 0.03           # SG偏离度相对阈值 (3%)
SG_DEV_ABS_THRESHOLD_KMH: float = 5.0        # SG偏离度绝对阈值 (km/h)
REOCR_CLUSTER_GAP_KMH: int = 2               # Re-OCR读数聚类间隔 (km/h)
CONF_TIER_LOW_MAX: int = 30                  # 置信度 low tier 上限
CONF_TIER_MEDIUM_MAX: int = 70               # 置信度 medium tier 上限
TEXTLEN_SCORE_3PLUS: float = 100.0           # 3+位数字的 text_len 得分
TEXTLEN_SCORE_2: float = 50.0                # 2位数字的 text_len 得分
TEXTLEN_SCORE_1: float = 20.0                # 1位数字的 text_len 得分
TEXTLEN_SCORE_0: float = 10.0                # 无数字的 text_len 得分
ACCEL_SCORE_NORMAL: float = 100.0            # 正常帧加速度信号得分
ACCEL_SCORE_NEAR_ONE: float = 50.0           # 近一个尖峰的得分
ACCEL_SCORE_SAME_DIR: float = 60.0           # 同向尖峰得分
ACCEL_SCORE_VIOLATION: float = 20.0          # 违反帧得分
ACCEL_SCORE_ISLAND_INTERIOR: float = 10.0    # 孤岛内部帧得分
SG_CLUSTER_SCORE_1: float = 70.0             # SG聚集 ≤1帧的得分
SG_CLUSTER_SCORE_3: float = 40.0             # SG聚集 ≤3帧的得分
SG_CLUSTER_SCORE_5: float = 30.0             # SG聚集 ≤5帧的得分
SG_CLUSTER_SCORE_10: float = 15.0            # SG聚集 ≤10帧的得分
SG_CLUSTER_SCORE_MANY: float = 5.0           # SG聚集 >10帧的得分
REOCR_AGREE_1CLUSTER: float = 100.0          # Re-OCR完全一致得分
REOCR_AGREE_2CLUSTER: float = 60.0           # Re-OCR两聚类得分
REOCR_AGREE_3PLUS: float = 20.0              # Re-OCR三维以上得分

# ═══════════════════ 向后兼容置信度权重 ═══════════════════
COMPAT_CONF_PHYSICS_WEIGHT: float = 0.60
COMPAT_CONF_TEXTLEN_WEIGHT: float = 0.40

# ═══════════════════ 部分数字扩展参数 ═══════════════════
MAX_PARTIAL_WILDCARDS: int = 2         # expand_partial 最大通配符数
