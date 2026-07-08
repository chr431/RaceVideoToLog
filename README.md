# RaceVideoToLog

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。支持 GPU (CUDA) / CPU 后端，提供 PySide6 Fluent Design GUI 和 CLI 两种界面。GUI 与 CLI 共用统一的 `ProcessingPipeline` 后端，保证行为一致。

## 安装

```bash
pip install rapidocr_onnxruntime onnxruntime opencv-python numpy matplotlib pyside6 qfluentwidgets
```

**GPU 加速**（可选）：安装 NVIDIA CUDA Toolkit 12.x + cuDNN 9.x，程序自动检测。

## 使用方式

### GUI

```bash
python RaceVideoToLog.py
```

1. 导入视频文件，框选仪表盘数字区域
2. 选择纠错模式（自动锚点 / 人工基准）
3. 设置参数，导出 CSV

GUI 内置数据分析工具，支持导入多个 CSV 进行 v-t / v-x / Δt-x 对比，拖拽选择范围自动计算积分距离/用时。

### CLI

```bash
python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 [options] -o output.csv
```

CLI 与 GUI 共用 `pipeline.py` 中的 `ProcessingPipeline`，在原生线程中运行以保证 CUDA 推理性能。

### 数据分析

```bash
# 比较两个 CSV，生成 3 张对比图
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix
```

## 项目结构

```
RaceVideoToLog/
├── RaceVideoToLog.py    # 入口: arg 解析 + GUI/CLI 调度
├── gui.py               # PySide6 主窗口 (Fluent Design)
├── gui_review.py        # 人工审核对话框
├── gui_analysis.py      # 数据分析 Tab
├── analysis.py          # 数据分析业务逻辑
├── pipeline.py          # 统一处理流水线 (GUI/CLI 共用)
├── correction.py        # 纠错流水线
├── ocr_engine.py        # OCR 引擎、预处理、锚点选择
├── theme_manager.py     # 主题回调管理器
├── headless.py          # CLI 入口 (委托给 pipeline)
├── RaceVideoToLog.spec  # PyInstaller 打包配置
└── README.md
```

## 输出格式

```csv
# RaceVideoToLog
# video_hash=709674b9c34665ea, video=test.mp4
# roi=876,933,962,982, format=km/h
# max_speed=400.0, max_accel=50.0, div=4, ...
timestamp,distance,speed_kmh,flag
10.48,0.00,0.00,2
10.55,0.00,0.00,0
```

Flag: `0` = 原始 OCR, `1` = 自动纠错, `2` = 锚点, `3` = 需人工审核 (skip_fill 模式)。

## CLI 参数

```
python RaceVideoToLog.py [video] [options]

位置参数:
  video                    视频文件（省略启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y1        识别范围
  --format {m/s,km/h,mile/h}  速度单位 (默认: km/h)
  --div N                  采样间隔 1/N (默认: 2)
  --max-speed N            最大速度 km/h (默认: 400)
  --max-accel N            最大加速度 m/s² (默认: 50)
  --target-h N             OCR 高度 px (默认: 24)
  --pad N                  边缘填充 px (默认: 0)
  --buffer N               缓冲队列大小 (默认: 8)
  --backend {auto,cuda,cpu}  OCR 后端 (默认: auto)
  -o, --output PATH        输出 CSV 路径
  --frame-start N          起始帧号
  --frame-end N            结束帧号
  --analysis CSV1 CSV2     分析模式
  --analysis-out PREFIX    分析输出前缀
```

## 打包

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

生成 `dist/RaceVideoToLog/` (onedir 模式)。GPU 用户需自行安装 CUDA Toolkit + cuDNN。

## License

MIT
