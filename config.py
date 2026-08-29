"""RaceVideoToLog 应用配置 — 聚合引擎配置 + GUI 专属常量。

v2.16 起：引擎已拆分为独立仓库（chr431/video_ocr_engine），其配置单一事实源
在引擎的 engine_config.py，本文件 `from engine_config import *` 聚合再导出
（兼容所有 `import config; config.SEG_*` 的既有引用），并保留 GUI/应用专属
常量（颜色/窗口/图表/监控/日志）。

引擎接入方式：2026-08-30 前是 git submodule（third_party/video_ocr_engine，经
engine_bootstrap 注入 sys.path）；此后改为 **pip 依赖**，版本由本仓库
pyproject.toml 的 git tag 锁定，模块直接从已安装包 import。
"""
from __future__ import annotations

import os as _os

# ── 解码预处理策略（应用侧决策，必须先于引擎包 import 生效）──
# 引擎 0.9.0 起 import 时 setdefault DECORD_SKIP_LOOP_FILTER=all（CPU 软解
# 关去块滤波，HEVC 解码 -13%~-18%）。引擎级真值裁定为中性，但本应用
# CPU 解码路径实测（5 视频全量漏斗，2026-08-29，decord 0.7.14）：段界
# 漂移 +60 段、检出率 100%→97.9%、最终错误 0→7（test 3 / test2 4）——
# 最终错误数是本应用硬门禁，故在此 setdefault 回完整去块滤波（none）。
# 本 setdefault 先于引擎包 __init__ 的同名 setdefault 执行即生效；生产
# 默认 auto 走 NVDEC 本就不受该开关影响。如需引擎默认速度（hybrid/cpu
# 后端），可预先显式设置 DECORD_SKIP_LOOP_FILTER=all 覆盖。
_os.environ.setdefault("DECORD_SKIP_LOOP_FILTER", "none")

# 管线引擎域 + 共享常量（单一事实源在引擎仓库 engine_config.py）
from engine_config import *  # noqa: F401,F403 — 聚合导出兼容

# ── OCR 输入 pad 宽度下限：224（回退引擎 0.9.0 的 160 默认，P0-5）──
# 引擎按"逐帧全等准确率"将 224→160；本应用按生产漏斗口径（分段代表帧
# OCR + tol=1 + 检测/纠正链）实测（2026-08-29，5 视频全量帧，单变量
# OCR_PAD_SMALL）：160 使原始误读 150→190（test5 7→26、test6 17→32），
# 224 恢复 0.7.0 时代水平（≈149）且 auto 路径墙钟零差（test5/test6
# 3000 帧 4.1/2.8s 持平）。两处同步覆写：engine_config 模块属性供引擎
# 内部默认读取（extractor fill_width=None 分支），应用命名空间供 GUI
# 默认值（FILL_WIDTH_RANGE 内，用户仍可调）。
import engine_config as _engine_config  # noqa: E402
_engine_config.DEFAULT_FILL_WIDTH = 224
DEFAULT_FILL_WIDTH = 224

# ═══════════════════ 应用域常量（引擎 v0.3 重构后回归应用侧）═══════════════════
# 引擎重构清空了领域后处理/速度语义常量（引擎是零领域语义通用库），
# 以下常量原属 engine_config，现是本仓库（速度提取应用）的单一事实源。

# ── 速度单位 ──
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子
SOURCE_TO_KMH: dict[str, float] = {
    "m/s": MPS_TO_KMH,
    "km/h": 1.0,
    "mile/h": 1.609344,
}
DEFAULT_SPEED_FORMAT: str = "km/h"   # 速度单位 (km/h / m/s / mile/h)
DEFAULT_MAX_SPEED: float = 400.0     # 最大速度 (km/h)
DEFAULT_MAX_ACCEL: float = 50.0      # 最大加速度 (m/s²)

# ── 段级检测/纠错（速度领域参数，引擎不感知）──
SEG_WIN: int = 30               # 段级检测带宽窗口（换算成帧：×中位段间距，上限 120 帧）
SEG_WIN_MAX_FRAMES: float = 120.0  # _local_bandwidth 的帧窗口上限
SEG_MULT: float = 2.0           # 检测门限倍率：|值-中值| > 带宽×mult ⇒ suspect
SEG_MIN_DEV: float = 6.0        # 纠正最小偏差：|插值-当前| > 此值才改
SEG_MED_K: int = 10             # 中值滤波窗口半宽（段索引）：平滑值曲线，误读=尖峰
SEG_DETECT_FLOOR: float = 3.0   # 带宽下限 (km/h)
SEG_SINGLE_FLOOR: float = 2.0   # 单帧段专用带宽下限
SEG_ANCHOR_MAX_FRAMES: float = 120.0  # 纠错锚点最大帧距离

# ── 段级置信度（中值偏差 + 急动度加权，供 DP 锚定）──
SEG_CONF_W_MED: float = 0.7
SEG_CONF_W_JERK: float = 0.3
SEG_CONF_JERK_SCALE: float = 3.0
SEG_CONF_MIN_NEIGHBORS: int = 3
SEG_CONF_SHORT_NEIGHBOR: float = 30.0
SEG_CONF_EDGE: float = 100.0
SEG_CONF_MED_GATE: float = 50.0
SEG_CONF_MIN_CONSISTENT_FRAMES: int = 3
SEG_CONF_MIN_CONSISTENT_FRAMES_EXACT: int = 5
SEG_CONF_SHORT_RUN_CAP: float = 15.0
SEG_CONF_ISLAND_TOL: float = 2.0
SEG_CONF_ISLAND_DEV_MULT: float = 3.0

# ── 段级稠密格点 DP 纠正──
SEG_DP_OBS_WEIGHT: float = 1.0
SEG_DP_ACCEL_WEIGHT: float = 1.0
SEG_DP_MAX_DV_CAP: float = 4.0
SEG_DP_ANCHOR_COST: float = 0.1
SEG_DP_CHANGE_THRESHOLD: float = 3.0
SEG_DP_ANCHOR_CONF: float = 20.0
SEG_DP_DEANCHOR_JERK_MIN: float = 5.0
SEG_DP_DEANCHOR_JERK_MAX: float = 40.0
SEG_DP_DEANCHOR_CONF_MAX: float = 50.0

# ── 第二遍尖峰检测（孤立 2-off 单帧误读）──
SEG_SPIKE_K: int = 2
SEG_SPIKE_THRESH: float = 2.0
SEG_SPIKE_MIN_FIX: float = 2.0
SEG_SPIKE_MIN_NBR: int = 2
SEG_SPIKE_MIN_FPS: float = 40.0

# 版本：应用侧单一事实源（引擎独立版本线 0.3.x 不随应用 bump）；
# 运行时 CSV 头/控制台读 config.__version__（历史入口保持不变）
__version__ = "2.17.2"

# ═══════════════════ 应用/日志与监控 ═══════════════════
DEFAULT_LOG_LEVEL: str = "normal"      # 日志级别 (normal / detailed / debug)
MONITOR_ENABLED: bool = True           # 默认启用资源监控（--no-monitor / GUI 复选框 / RVTOL_MONITOR=0 关闭）
MONITOR_INTERVAL_S: float = 1.0        # 资源采样间隔（秒）
MONITOR_GPU: bool = True               # 是否采样 GPU 利用率/显存/温度

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

# ═══════════════════ GUI 参数范围 ═══════════════════
# GUI 段 review 的加速度容差倍率（允许输入值超出物理约束的倍数）
REVIEW_ACCEL_TOLERANCE: float = 3.0
# GUI 参数范围（gui_settings 使用；默认值仍取 DEFAULT_* 常量）
BUFFER_SIZE_RANGE: tuple[int, int] = (4, 256)
FILL_WIDTH_RANGE: tuple[int, int] = (160, 320)
# 分析 Tab 的 SG 平滑窗口换算系数（strength 0-100 → 窗口比例）
SMOOTH_WIN_FACTOR: float = 0.0175
SMOOTH_MIN_WIN: int = 5
