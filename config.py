"""RaceVideoToLog 集中配置 — 常量、颜色、公共 API。"""
from __future__ import annotations

__version__ = "2.12.1"

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子

# ═══════════════════ 用户可配置默认值 ═══════════════════
DEFAULT_BACKEND: str = "auto"           # GPU 后端 (auto / tensorrt / cpu)
DEFAULT_OCR_MODEL: str = "v6_tiny"     # 主 OCR 模型
DEFAULT_REOCR_MODEL: str = "v6_small"  # 重 OCR 模型（自动推导：主 tiny → 用此模型；主 small → 无重 OCR）
DEFAULT_SPEED_FORMAT: str = "km/h"     # 速度单位 (km/h / m/s / mile/h)
DEFAULT_FRAME_DIV: int = 2             # 采样间隔 (1=每帧, 2=隔帧)
DEFAULT_MAX_SPEED: float = 400.0       # 最大速度 (km/h)
DEFAULT_MAX_ACCEL: float = 50.0        # 最大加速度 (m/s²)
DEFAULT_TARGET_H: int = 48             # OCR 预处理目标高度 (px)
DEFAULT_PAD: int = 0                   # OCR 预处理 padding (px)
DEFAULT_MAX_WIDTH: int = 0              # 预处理最大宽度 px（0=不限）
DEFAULT_BUFFER_SIZE: int = 16          # 生产者-消费者队列缓冲大小
DEFAULT_LOG_LEVEL: str = "normal"      # 日志级别 (normal / detailed / debug)
MONITOR_ENABLED: bool = True           # 默认启用资源监控（--no-monitor / GUI 复选框 / RVTOL_MONITOR=0 关闭）
MONITOR_INTERVAL_S: float = 1.0        # 资源采样间隔（秒）
MONITOR_GPU: bool = True               # 是否采样 GPU 利用率/显存/温度

# ═══════════════════ 枚举值（CLI/GUI 单点定义） ═══════════════════
BACKEND_KEYS: list[str] = ["auto", "tensorrt", "cpu"]
BACKEND_LABELS: dict[str, str] = {"auto": "自动", "tensorrt": "TensorRT", "cpu": "CPU"}
OCR_FRAME_BATCH: int = 6               # 帧批处理大小（≤6 兼容 TRT profile 上限）
# OCR 输入 pad 宽度下限（px）：实际内容宽 × 高比超出该值时用实际宽。
# 速度数字是窄图（48 高后 78-160 宽），旧代码强制 pad 到 320 让 GPU 白算
# 2~4 倍。但 v6 tiny 对输入宽度敏感 —— test5(max_width=72) 在 72 宽下
# 精度 0.07%→0.54%（模型依赖 padding 的时序上下文）；small 在窄宽下反而
# 更好（test6 0.83%→0.33%）。默认 192 平衡：test4/5 精度不变，test6
# small 仍有 ~1.5x 推理加速。
#
# 两模型可独立设下限（精确优化实测值）。测试可用环境变量覆盖单模型：
# RVTOL_PAD_TINY / RVTOL_PAD_SMALL（像素，指定即优先于本表）。
#
# 实测（bench_decoder, TRT, 2026-08）：
# - tiny=192：test4 err 2.18%/falseT 5（224 时 falseT 翻倍到 10）、
#   test5(max_width=72) err 0.04%。192 是 tiny 的精度甜点（144 时 test4
#   2.45%，96 时 0.19%/2.42%，256 时 test5 0.14%）。
# - small=224：test6 err 0.09%（192 时 0.16%，48~96 时 0.69~1.19% —— small
#   并不窄宽鲁棒，宽 pad 更准；256 与 224 精度相同但更慢）。
# 跨模型 tiny+small 时 re-OCR(small@224) 让 test6 err 0.47%→0.38%。
OCR_PAD_WIDTH_MIN: int = 192
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
    "v6_tiny": 192,
    "v6_small": 224,
}

# ═══════════════════ 图表颜色 ═══════════════════
COLOR_BLUE: str = "#2196F3"
COLOR_ORANGE: str = "#FF5722"
COLOR_GREEN: str = "#4CAF50"
COLOR_RED: str = "#F44336"
COLOR_GRAY: str = "#888888"
COLOR_LIGHT_GRAY: str = "#666666"
# GUI 专用调色板（预览/图表/对话框共用）
PREVIEW_BG: str = "#111"                # 视频预览底色（暗）
PREVIEW_BG_LIGHT: str = "#e0e0e0"       # 视频预览底色（亮）
ROI_BOX_COLOR: str = "#ff5050"          # ROI 框颜色
CANVAS_BG_DARK: str = "#1f1f1f"         # 图表背景（暗）
CANVAS_BG_LIGHT: str = "#f5f5f5"        # 图表背景（亮）
CANVAS_FG_DARK: str = "#f0f0f0"         # 图表前景（暗）
CANVAS_FG_LIGHT: str = "#000000"        # 图表前景（亮）
CANVAS_FILL: str = "#151515"            # 预览画布填充
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
ERROR_DETECT_OCR_CONF_WEIGHT: float = 0.01   # OCR 模型内部置信度
ERROR_DETECT_ABS_WEIGHT: float = 0.15        # 带宽归一化绝对残差（取代 physics+linearity）
ERROR_DETECT_ACCEL_SPIKE_WEIGHT: float = 0.50 # 加速度尖峰对检测
ERROR_DETECT_FREQ_WEIGHT: float = 0.10        # 频域残差（高频内容 = 非物理）
# 最差信号地板：任一信号低于阈值时，组合分数上限
ERROR_DETECT_FLOOR_CAP: dict[float, float] = {30.0: 25.0, 50.0: 50.0, 70.0: 69.0}
# 带宽归一化绝对残差（Signal 2）参数
ABS_RESID_FLOOR: float = 3.0       # 带宽下限 (km/h)：巡航帧插值噪声 0.5-1.5，须 ≥3 防误报
ABS_RESID_WINDOW: int = 15         # 局部带宽滑动均值窗口（帧）
# 频域残差信号参数：真实速度低带宽（99% ≤2.5Hz）；高频残差 = 疑似误差
FREQ_RESID_SIGMA: float = 3.0        # 中心高斯低通标准差（帧）；短窗 → 捕捉单帧/短时 3-9 km/h 错误
FREQ_RESID_SCALE: float = 5.0        # 残差→分数衰减尺度：score=100*exp(-resid/scale)
FREQ_FLOOR_THRESHOLD: float = 50.0   # 频域专用 floor：freq 分数低于此 → 组合分压顶到 FREQ_FLOOR_CAP
FREQ_FLOOR_CAP: float = 50.0         # 频域 floor 的压顶值（< AUTO_CORRECT_THRESHOLD=70 才进入纠错）
FREQ_CORROBORATE_THRESHOLD: float = 80.0  # 协同 floor：freq 低仅当 min(abs,accel)<此值才压顶（避免真实变速被误伤）

# ═══════════════════ Viterbi DP ═══════════════════
VITERBI_OBS_WEIGHT: float = 0.3
VITERBI_ACCEL_WEIGHT: float = 1.0
VITERBI_TRUSTED_BOUNDARY_CONFIDENCE: int = 85
VITERBI_MAX_CANDIDATES: int = 40

# ═══════════════════ 纠错参数 ═══════════════════
AUTO_CORRECT_THRESHOLD: int = 70
CORRECTION_MAX_ROUNDS: int = 10        # Viterbi 多轮迭代
FILL_MAX_PASSES: int = 50
CORRECTION_MIN_DIFF: float = 0.5
AUTO_SMOOTH_CLUSTER_MAX: int = 5
AUTO_SMOOTH_DEVIATION_MULT: float = 5.0

# ═══════════════════ 平滑 + 自动对齐参数 ═══════════════════
SMOOTHNESS_MAX_ITERATIONS: int = 10       # _smoothness_pass 最大迭代轮数
AUTO_ALIGN_DIFF_MIN_KMH: int = 5         # auto-align 最小修正量 (km/h)
AUTO_ALIGN_DIFF_MAX_KMH: int = 25        # auto-align 最大修正量 (km/h)
AUTO_ALIGN_NUDGE_FACTOR: float = 0.8     # auto-align 向插值修正的比例
AUTO_ALIGN_MIN_CHANGE_KMH: int = 3       # auto-align 最小提交变化量 (km/h)

# ═══════════════════ Force-Median 平滑参数 ═══════════════════
FORCE_MEDIAN_MAX_ITERATIONS: int = 15     # _force_median_smooth 最大迭代轮数
FORCE_MEDIAN_NUDGE_FACTOR: float = 0.7    # 向中值修正的比例
FORCE_MEDIAN_THRESHOLD_MULT: float = 3.0  # max_dv 阈值倍率（3×：只改明显偏离的帧，避免 ±1-3 噪声被拉）
FORCE_MEDIAN_MIN_CHANGE_KMH: int = 1      # 最小变化量 (km/h)

# ═══════════════════ 候选值后过滤 ═══════════════════
CANDIDATE_POSTFILTER_ABS_MIN: int = 85        # 自洽帧 abs 最低阈值（绝对残差信号）
CANDIDATE_HUNDREDS_MAX_DIFF: int = 100         # 百位变体最大允许差值 (km/h)

# ═══════════════════ 参考值构建保护 ═══════════════════
REF_GUARD_ABS_MIN: int = 85          # 跳过插值参考值的 abs 阈值
INTERP_ANCHOR_CONF_MIN: int = 85     # _local_interp 锚点最低 Phase-1 置信度：
                                     # 低置信误读簇（物理自洽但错误）不得当插值锚点
                                     # （test5 1600/3998/5263 等 d>5 离群的根源）
DENSE_PIN_REF_MAX_DIFF: float = 30.0  # 单候选钉死值与 conf-gated 参考严重矛盾（>30）
                                     # 时解除钉死 —— 物理自洽的误读簇被钉死成错误值
                                     # （test5 5265/5267 raw=11 钉死拖垮 118 坡）
DISTANT_INTERP_MIN_TIME: float = 1.0      # 远距离插值最小时间距离 (秒)
FORCE_MEDIAN_WINDOW_TIME: float = 0.1      # force-median 中值窗口时间 (秒)
TRUST_WINDOW_TIME: float = 0.15            # 信任传播验证时间窗 (秒)
DISTANT_INTERP_ISLAND_THRESHOLD: int = 30  # 孤岛检测距离阈值 (km/h)
REF_INTERP_MAX_KMH_DIFF: int = 50    # 插值参考值最大允许偏差 (km/h)
REF_MIN_DIFF: float = 6.0            # 参考值最小偏差：raw 与插值差 < 此值时不设参考（raw 自洽）
                                     # 3.0→6.0：拦截阶梯显示上 ref 滞后 3-5 km/h 拖拽正确 raw
                                     # （test4 dense 回归根源）；真误读帧 ref 差巨大不受影响

# ═══════════════════ Viterbi 后处理 ═══════════════════
VITERBI_POST_TRUST_THRESHOLD: int = 70     # Viterbi 后信任判定最低分数
TRUST_WINDOW_FALLBACK_MAX_DV: float = 8.0  # 信任窗口 fallback max_dv (km/h)
FILL_CONFIDENCE_THRESHOLD: int = 30        # Fill 阶段的置信度阈值
FILL_CANDIDATE_MAX_DIFF: int = 12          # fill 候选优先的最大差值（与插值的距离保护）
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
LINEARITY_TIME_WINDOW: float = 0.25          # 插值期望值每侧搜索时间窗 (秒)
LINEARITY_MAX_NEIGHBORS: int = 10            # 插值期望值每侧最大邻居帧数
ACCEL_SPIKE_VIOLATION_MULT: float = 3.0      # 加速度尖峰阈值倍率
ACCEL_SPIKE_SEARCH_WINDOW: int = 15          # 对立尖峰搜索窗口 (帧)
CONF_TIER_LOW_MAX: int = 30                  # 置信度 low tier 上限
CONF_TIER_MEDIUM_MAX: int = 70               # 置信度 medium tier 上限
ACCEL_SCORE_NORMAL: float = 100.0            # 正常帧加速度信号得分
ACCEL_SCORE_NEAR_ONE: float = 100.0          # 近一个尖峰不惩罚（真实速度变化的单尖峰会误伤相邻正确帧）
ACCEL_SCORE_SAME_DIR: float = 60.0           # 同向尖峰得分
ACCEL_SCORE_VIOLATION: float = 20.0          # 违反帧得分
ACCEL_SCORE_ISLAND_INTERIOR: float = 10.0    # 孤岛内部帧得分

# ═══════════════════ 部分数字扩展参数 ═══════════════════
MAX_PARTIAL_WILDCARDS: int = 2         # expand_partial 最大通配符数
