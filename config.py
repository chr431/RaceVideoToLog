"""RaceVideoToLog 集中配置 — 常量、颜色、公共 API。

v2.13 起：分段流水线（segment_flow.py）为唯一管线，原逐帧纠错参数
（Viterbi/多信号检测/对齐等）已删除。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "2.14.0"

# ═══════════════════ 数据目录 ═══════════════════

def app_data_dir() -> Path:
    """程序数据目录（本文件夹内，免安装/portable 设计）。

    引擎缓存（ocr_engines/）与运行日志（logs/）都放这里 —— 不写
    %LOCALAPPDATA%，卸载/移动时删除整个程序目录即清理干净。

    frozen: exe 所在目录；源码运行: 项目根目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_logs_dir() -> Path:
    """运行日志目录（本目录/logs，与数据目录一致）。"""
    return app_data_dir() / "logs"

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子

# ═══════════════════ 用户可配置默认值 ═══════════════════
DEFAULT_OCR_MODEL: str = "v6_small"     # 唯一 OCR 模型（v2.13 起移除 tiny / 重 OCR）
DEFAULT_SPEED_FORMAT: str = "km/h"     # 速度单位 (km/h / m/s / mile/h)
DEFAULT_BUFFER_SIZE: int = 128          # 解码∥OCR 流水线队列缓冲（段数）
                                        # 64→128：GPU 解码突发时缓冲背压，减少
                                        # 解码线程 q.put 阻塞等待（GPU+CPU wall
                                        # -0.3s；256 无进一步收益）
DEFAULT_DECODE_BACKEND: str = "auto"   # 解码后端 (auto / cpu / nvdec)
DECODE_BACKEND_KEYS: list[str] = ["auto", "cpu", "nvdec"]
DECODE_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU", "nvdec": "NVDEC"}
DEFAULT_OCR_BACKEND: str = "auto"      # OCR 推理后端 (auto / cpu / tensorrt)
OCR_BACKEND_KEYS: list[str] = ["auto", "cpu", "tensorrt"]
OCR_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU", "tensorrt": "TensorRT"}
DEFAULT_MAX_SPEED: float = 400.0       # 最大速度 (km/h)
DEFAULT_MAX_ACCEL: float = 50.0        # 最大加速度 (m/s²)
DEFAULT_FORCE_ASPECT: float = 0.0      # 强制横向宽高比（0=不启用；>0 时宽度
                                       # 强制 = 48×此值，纠正扁宽字体）
DEFAULT_FILL_WIDTH: int = 224          # OCR 输入 pad 宽度下限（引擎 _resize_norm
                                       # pad 到该总宽）。扫描（test2/5/6 全量）：
                                       # 320 raw 最优 0.53% vs 224 0.67%（test5 7→2、
                                       # test6 17→5），但端到端 224 最优（13 vs 16）
                                       # ——test5/6 的 raw 提升被 DP 吸收，test2 宽
                                       # pad 引入混杂邻域 DP 拉中间值（纠错 5）。
                                       # GUI 可调 160-320，默认 224
OCR_GAMMA: float = 2.0                 # OCR 预处理灰度 gamma 增强指数（正式预处理：
                                       # 白字黄底等背景色块场景放大高段分离；灰度
                                       # 先于 gamma——RGB 逐通道 gamma 视觉差异小、
                                       # 回归多。1.0=纯灰度不增强，0=保留 RGB）
DEFAULT_LOG_LEVEL: str = "normal"      # 日志级别 (normal / detailed / debug)
MONITOR_ENABLED: bool = True           # 默认启用资源监控（--no-monitor / GUI 复选框 / RVTOL_MONITOR=0 关闭）
MONITOR_INTERVAL_S: float = 1.0        # 资源采样间隔（秒）
MONITOR_GPU: bool = True               # 是否采样 GPU 利用率/显存/温度

# ═══════════════════ 段管线参数 ═══════════════════
SEG_C: float = 5.0              # 分段聚类阈值：max 3×3 窗口和 < C ⇒ 显示未变
SEG_WIN: int = 30               # 段级检测带宽窗口（换算成帧：×中位段间距，上限 120 帧）
SEG_MULT: float = 2.0           # 检测门限倍率：|值-中值| > 带宽×mult ⇒ suspect
SEG_MIN_DEV: float = 6.0        # 纠正最小偏差：|插值-当前| > 此值才改
SEG_MED_K: int = 10             # 中值滤波窗口半宽（段索引）：平滑值曲线，误读=尖峰
SEG_DETECT_FLOOR: float = 3.0   # 带宽下限 (km/h)：防 ±1-2 噪声被 flag
                                # （floor4×mult2=gate8 会漏 8-off 尖峰，如
                                # test.mp4 1499 段 160 在 168 平板上）
SEG_SINGLE_FLOOR: float = 2.0   # 单帧段专用带宽下限：单帧段误读率 4.2% vs
                                # 多帧 0.3%（12.6×，80% 误读是单帧段）→ 平缓区
                                # gate 4 抓 ≥5-off 单帧误读；弯曲区按实际带宽
                                # （↓到 1.5/1.0 虽提升 ±1 召回 94.5→96.7%，但
                                #  当前纠错把正确单帧段改错 → test 19→22/23 回归，
                                #  需配合纠错保守化（下一步）才可放宽）
SEG_ANCHOR_MAX_FRAMES: float = 120.0  # 纠错锚点最大帧距离：近锚点才插值（防远锚点误插值）

# ═══════════════════ 段级置信度（中值偏差 + 急动度加权，供 DP 锚定） ═══════════════════
SEG_CONF_W_MED: float = 0.7       # 中值偏差信号权重（主导锚定：紧邻误读的
                                  # 正确段中值分高 → 被 pin，防 DP 平滑拖走）
SEG_CONF_W_JERK: float = 0.3      # 急动度信号权重：辅助区分（刹车中值低但
                                  # 急动度高 → conf 中，raw 观测保其不变）
SEG_CONF_JERK_SCALE: float = 3.0  # 急动度分指数尺度 (km/h)：100*exp(-jerk/scale)

# ═══════════════════ 段级稠密格点 DP 纠正（对齐旧 viterbi_dense） ═══════════════════
# 观测 = 纯惩罚偏离 raw（旧系统 ref 来自重 OCR，重 OCR 已删 → ref 删除）。
# 观测存在的意义：惩罚任何改动，防止把正确的改错。DP 只在转移平滑性
# （加速度约束）强烈要求时移动值。
SEG_DP_OBS_WEIGHT: float = 1.0      # 观测权重：非锚点填向局部锚点插值（曲线），
                                    # 高权重让 DP 输出精确贴合曲线（锚点插值
                                    # 本身给基线，DP 再加全局平滑处理运行）
SEG_DP_ACCEL_WEIGHT: float = 1.0    # 转移权重：超加速度约束的二次惩罚
SEG_DP_MAX_DV_CAP: float = 4.0      # 每段转移最大变化 (km/h)：max_dv = min(
                                    # max_accel×dt×3.6, cap)。长段间距时
                                    # max_accel×dt 过松（8-off 跳变免费），
                                    # cap 保证误读跳变被惩罚、DP 拉正
SEG_DP_ANCHOR_COST: float = 0.1     # 高置信段锚定代价（固定到 raw）
SEG_DP_CHANGE_THRESHOLD: float = 3.0  # |DP输出 - raw| > 此值才修正：干净视频
                                      # 1-off 拉偏不提交；放宽到 3.0 消掉 2-off
                                      # 正确段被 DP 微调改错（gamma raw 下实测
                                      # 误改 2→0，漏纠不变，最终 15→13）
SEG_DP_ANCHOR_CONF: float = 20.0   # 锚定阈值：conf ≥ 此值的段固定到 raw
                                    # （门控 conf 后正确段 p10=72 干净分离，
                                    #  T=20 pin 100% 正确、仅 9% 误读）

# ═══════════════════ 孤立尖峰豁免（A4，13→12 实测） ═══════════════════
# conf∈[20,50) 的锚定段若 jerk（二阶差分）中等 → 解除锚定交给 DP。
# 判别依据（5 视频 722 段实测）：真刹车 jerk≈0（713 段全在 [0,9] 且
# 绝大多数 [0,4]）、丢位邻居污染 jerk≥80（9 段）、孤立尖峰误读 jerk 中等
# （如 test#74 raw=107 truth=103 jerk=9 —— 锚定会保留误读）。带通 [5,40]
# 只抓尖峰：解锚 24 段中 23 误读 + 1 正确（正确段也未被改坏），13→12
# 零误改。参数敏感性：下界 0 灾难（刹车全解锚，78 误改）、下界 3-8 ×
# 上界 20-60 全部稳定 12。0=禁用豁免。
SEG_DP_DEANCHOR_JERK_MIN: float = 5.0
SEG_DP_DEANCHOR_JERK_MAX: float = 40.0

# ═══════════════════ OCR 输入 pad 宽度下限 ═══════════════════
# 速度数字是窄图（48 高后 78-160 宽）。v6_small 在宽 pad 更准
# （test6：224→err 0.09%，192→0.16%，48~96→0.69~1.19%；256 精度相同但更慢）。
OCR_PAD_WIDTH_MIN: int = 224
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
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

# ═══════════════════ 邻帧一致性评分（signals 使用，GUI 已改段级）═══════════════════
CONSISTENCY_TIME_WINDOW: float = 0.5    # 时间窗 (秒)
CONSISTENCY_DECAY_TAU: float = 0.06     # 指数衰减常数 exp(-dt/tau)
CONSISTENCY_PINNED_WEIGHT: float = 3.0  # 已固定帧权重倍率
