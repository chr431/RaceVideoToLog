# RaceVideoToLog

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。支持 GPU (CUDA) / CPU 后端，提供 GUI 和 CLI 两种界面。

## 安装

```bash
pip install rapidocr_onnxruntime onnxruntime opencv-python numpy matplotlib
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

GUI 内置数据分析工具，可导入多个 CSV 进行 v-t / v-x / Δt-x 对比。

### CLI

```bash
python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 [options] -o output.csv
```

CLI 与 GUI 自动锚点模式共用同一纠错后端。完整参数见下文。

### 数据分析

```bash
# 比较两个 CSV，生成 3 张对比图
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix
```

## 纠错算法

### 自动锚点（推荐）

全自动流程，无需人工干预：

```
OCR → auto_select_anchors → Correction 流水线 → CSV
```

**锚点选择**：局部中位数过滤器（自适应窗口覆盖 ≈0.3 秒），帧值偏离中位数 ≤4 km/h 视为可靠 OCR 结果，用作后续纠错的硬约束。

**Correction 流水线**（5 阶段）：

| 阶段 | 功能 |
|------|------|
| 1. 错误检测 | 6 种检测器并行扫描：邻帧跳变、V 字形、悬崖、锚点趋势偏离、孤立离群、局部趋势偏离 |
| 2. 重 OCR | 4 种预处理变体（灰度/CLAHE+OTSU/OTSU反相/后备链）重新识别，有缓存 |
| 3. 最优选择 | 评分函数 = 邻帧一致性(0.4) + 锚点插值(0.35) + 平滑度(0.25) |
| 4. 多轮迭代 | 逐轮重新检测修复，最多 3 轮，空集退出 |
| 5. 级联填充 | 不可恢复帧的线性插值 + 加速度裁剪，while 循环收敛 |

### 人工基准

用户按固定间隔（如 1/10）手动标注部分帧，系统自动用锚点选择补充其余可靠帧，Correction 在混合锚点基础上纠错。人工标注值不会被覆盖。

## OCR 流水线

每帧经过三步后备链，前一步成功则跳过后续：

```
1. 灰度化 + 缩放(target_h) → RapidOCR 标准识别
2. OTSU 二值化 + 缩放 → RapidOCR 标准识别          (步骤 1 失败时)
3. ocr_digital_fallback:
   a. CLAHE+OTSU × 3 高度(28/32/48px) → 标准识别
   b. 4 变体 × 2 高度(32/48px) → use_det=False  (步骤 3a 失败时)
```

OCR 失败帧标记为 -1.0，保证帧索引对齐。

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

Flag: `0` = 原始 OCR，`1` = 自动纠错，`2` = 锚点（自动或人工）。

## 项目结构

```
RaceVideoToLog/
├── RaceVideoToLog.py    # GUI + CLI 入口
├── correction.py         # 纠错流水线（GUI/CLI 共用）
├── ocr_engine.py         # OCR 引擎、预处理、锚点选择、后端管理
├── analysis.py           # 数据分析（GUI tab + CLI 导出）
├── headless.py           # CLI OCR 流水线
├── RaceVideoToLog.spec   # PyInstaller 打包配置
└── README.md
```

## CLI 参数

```
python RaceVideoToLog.py [video] [options]

位置参数:
  video                    视频文件（省略启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2        识别范围（像素坐标，CLI 必填）
  --format {m/s,km/h,mile/h}  速度单位 (默认: km/h)
  --div N                  采样间隔 1/N, 1-10 (默认: 2)
  --max-speed N            最大合理速度 km/h (默认: 400)
  --max-accel N            最大合理加速度 m/s² (默认: 50)
  --target-h N             OCR 目标高度 px (默认: 24)
  --pad N                  边缘填充 px (默认: 0)
  --workers N              并行线程数 (默认: 4, ≤32)
  --backend {auto,cuda,cpu}  OCR 后端 (默认: auto)
  -o, --output PATH        输出 CSV 路径
  --frame-start N          起始帧号
  --frame-end N            结束帧号
  --analysis CSV1 CSV2     分析模式（比较两个 CSV）
  --analysis-out PREFIX    分析输出前缀
```

## 打包

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

生成 `dist/RaceVideoToLog.exe`（≈333MB）。GPU 用户需自行安装 CUDA Toolkit + cuDNN。

## License

MIT
