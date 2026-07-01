# RaceVideoToLog — 项目知识总结

## 项目概述

从赛车视频中 OCR 提取速度数字，输出时间-速度-距离 CSV 文件。
- GPU (CUDA) / CPU 双后端
- GUI (PySide6 + qfluentwidgets Fluent Design) + CLI 无头模式
- 物理约束纠错 + 数据分析对比

## 代码结构

```
RaceVideoToLog.py    # 入口: arg 解析 + GUI/CLI 调度
gui.py               # PySide6 主窗口 (Fluent Design 主题)
gui_analysis.py      # 数据分析 Tab (matplotlib 图表)
analysis.py          # 数据分析业务逻辑 (解析/平滑/绘图)
ocr_engine.py        # OCR 引擎、预处理、纠错算法、锚点选择、GPU DLL 加载
correction.py        # 纠错流水线 (5 阶段)
headless.py          # CLI 无头 OCR 模式
RaceVideoToLog.spec  # PyInstaller onedir 打包配置
```

### 依赖

- PySide6 6.9+ (Qt 6 GUI)
- qfluentwidgets 1.11+ (Microsoft Fluent Design 组件库)
- rapidocr-onnxruntime (PP-OCR v6_small ONNX 模型)
- onnxruntime (CUDA/CPU 推理)
- opencv-python (图像 I/O 和预处理)
- matplotlib (数据分析绘图, QtAgg 后端)
- numpy (数值计算)

### OCR 模型

- v6_small: 唯一使用模型 (PP-OCRv6 轻量版 ONNX)
- 预处理: 灰度化 + 缩放到 target_h (默认 24px) → PP-OCR 内置归一化
- Fallback: OTSU 二值化 → 数字仪表后备 OCR

## GUI 架构

### 主窗口 (gui.py)

- `RaceVideoToLogApp(QMainWindow)` — 继承 QMainWindow
- 双 Tab: OCR 处理 + 数据分析 (Pivot 切换)
- 亮/暗双主题: `setTheme(Theme.LIGHT/DARK)` (qfluentwidgets)
- 左侧面板: 参数配置 (CardWidget 容器)
- 右侧面板: ROI 输入 + 视频预览 (带拖拽选 ROI)
- `_ExportThread(QThread)`: 后台 OCR + 纠错 + CSV 写出

### 数据分析 Tab (gui_analysis.py)

- `AnalysisTab` — 嵌入 QStackedWidget
- 3 个 CSV 导入槽位, 自动解析
- v-t / v-x / Δt-x 三种图表模式
- SpanSelector 拖拽范围选择 (积分距离/用时)
- 平滑滑块 + 诊断信息 (红色=自动纠错/绿色=人工纠错)
- 修改后自动刷新图表 (无需手动点击渲染)

### 主题系统

- 全局初始化: `setTheme(Theme.AUTO)` (RaceVideoToLog.py, QApplication 创建后)
- 手动切换: `_toggle_theme()` → `setTheme(Theme.LIGHT/DARK)`
- 标题栏同步: `DwmSetWindowAttribute` (Windows 11 DWMWA_USE_IMMERSIVE_DARK_MODE)
- Palette 背景色: `_sync_titlebar()` 设置 Window/Base/Text 色值
- matplotlib 同步: `_sync_figure_theme()` (figure/axes/spines 颜色)

## 关键算法

### 1. 预处理 (`_preprocess` / `_preprocess_fb`)
灰度化 → 缩放到 target_h → 转 BGR。备选: OTSU 二值化。

### 2. 数字仪表后备 OCR (`ocr_digital_fallback`)
当主 OCR 失败时: CLAHE+OTSU × 多高度 / use_det=False。

### 3. 纠错算法 (`correct_with_anchors`)
- 5 阶段: 错误检测 → 重 OCR → 评分 → 迭代 → 级联填充
- 6 种检测器: 邻帧跳变、V 字形、悬崖、锚点趋势偏离、孤立离群、局部趋势偏离

### 4. 锚点选择 (`auto_select_anchors`)
局部中位数过滤器 + 加速度验证 (max_accel_mps2=50 m/s²)。

### 5. 视频哈希 (`compute_video_hash`)
SHA-256 头尾各 1MB + 文件大小, 前 16 位。

## CSV 格式

```
# RaceVideoToLog
# video_hash=..., video=...
# roi=..., format=..., max_speed=..., ...
timestamp,distance,speed_kmh,flag
```
Flag: 0=可信, 1=自动纠错, 2=锚点(自动/人工)。

## 测试文件

### test4.mp4
- ROI: 858,939,964,1004, fps: 60.00
- 参数: max_speed=400, max_accel=100, div=2

### test3.mp4
- ROI: 868,948,956,997, fps: 53.45
- 参数: max_speed=350, max_accel=50, div=4

## 常用命令

```bash
# GUI
python RaceVideoToLog.py

# CLI
python RaceVideoToLog.py test4.mp4 --roi 858 939 964 1004 --max-speed 400 --max-accel 100 --div 2 -o out.csv

# 数据分析
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix

# 打包
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```
