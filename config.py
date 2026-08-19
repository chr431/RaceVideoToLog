"""RaceVideoToLog 应用配置 — 聚合引擎配置 + GUI 专属常量。

v2.16 起：引擎已拆分为独立仓库（chr431/video_ocr_engine，git submodule
third_party/video_ocr_engine），其配置单一事实源在引擎的 engine_config.py，
本文件 `from engine_config import *` 聚合再导出（兼容所有 `import config;
config.SEG_*` 的既有引用），并保留 GUI/应用专属常量（颜色/窗口/图表/
监控/日志）。引擎模块经 engine_bootstrap 加入 sys.path。
"""
from __future__ import annotations

# 管线引擎域 + 共享常量（单一事实源在引擎仓库 engine_config.py）
from engine_config import *  # noqa: F401,F403 — 聚合导出兼容

# 版本：应用侧单一事实源（引擎独立版本线 0.1.x 不随应用 bump）；
# 运行时 CSV 头/控制台读 config.__version__（历史入口保持不变）
__version__ = "2.15.2"

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
