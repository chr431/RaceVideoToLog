# RaceVideoToLog

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。支持 GPU (CUDA) / CPU 两种后端，提供 GUI 和无头 CLI 两种使用方式。

## 功能概述

- **OCR 识别**：PP-OCR v5_mobile 识别仪表盘七段数码管速度
- **自动锚点纠错**：全自动模式，自动识别可靠帧并使用物理约束纠正错误
- **人工基准标注**：手动标注部分帧作为锚点，结合自动锚点实现最高精度
- **数据分析**：GUI 内置多 CSV 对比图表（v-t / v-x / Δt-x），支持缩放、平移、范围选择
- **GPU 加速**：自动检测 CUDA / cuDNN，支持并行推理

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.10 | 运行环境 |
| rapidocr_onnxruntime | latest | PP-OCR 引擎 |
| onnxruntime | ≥1.26 | ONNX 推理 |
| OpenCV | ≥4.10 | 视频读取、图像预处理 |
| NumPy | ≥2.0 | 数值计算 |
| Matplotlib | ≥3.10 | 数据分析图表 |
| Tkinter | ≥8.6 | GUI 界面 |

**可选（GPU 加速）**：
- NVIDIA CUDA Toolkit 12.x
- NVIDIA cuDNN 9.x

## 快速开始

### GUI 模式
```bash
python RaceVideoToLog.py
```
1. 导入视频文件
2. 用鼠标框选仪表盘数字区域
3. 选择纠错模式（自动锚点 / 人工基准）
4. 点击导出，选择保存路径

### 无头 CLI 模式
```bash
python RaceVideoToLog.py test.mp4 \
  --roi 876 935 962 982 \
  --div 4 \
  --frame-start 600 \
  --frame-end 4300 \
  -o output.csv
```

### 数据分析 CLI
```bash
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out comparison
# 生成 comparison_v-t.png, comparison_v-x.png, comparison_Δt-x.png
```

## 纠错模式

### 自动锚点纠错（推荐默认）

全自动流程，无需人工干预：

```
OCR 识别 → 自动选择可靠锚点 → Correction B 纠错 → 输出 CSV
```

- 使用**局部中位数过滤器**自动识别 ≈90% 帧作为可靠锚点
- 对剩余 ≈10% 帧运行多阶段物理约束纠错
- 实测准确率：97.3% exact, 98.9% within 1 km/h, max error 9 km/h

### 人工基准标注

手动标注部分帧 + 自动锚点补充，适合需要最高精度的场景：

```
OCR 识别 → 人工标注采样帧 → 自动锚点补充 → 混合锚点 Correction B → 输出 CSV
```

- 用户每 N 帧标注一次正确速度（N 可配置，默认 10）
- 标注完成后自动补充可靠帧作为额外锚点
- 人工锚点作为绝对约束，Correction B 在混合锚点基础上纠错

## 工作流程详解

### 1. 视频帧提取

按 `div` 参数等间隔采样视频帧（div=1 表示每帧都处理，div=4 表示每 4 帧处理 1 帧）。用户可指定起止帧范围。

### 2. OCR 识别

每帧经过以下流水线：
1. **灰度化 + 缩放**（目标高度 24px）
2. **PP-OCR v5_mobile** 推理（CUDA 或 CPU）
3. **后备链**：若主识别失败 → CLAHE + OTSU 增强 → EasyOCR 模式（`use_det=False`）
4. 提取数值并转换为 km/h

确保每帧都有识别结果（失败帧标记为 -1.0）。

### 3. 纠错算法

#### Correction B — 5 阶段流水线

**阶段 1 — 错误检测**（5 种检测器并行）：

| 检测器 | 检测模式 | 原理 |
|--------|----------|------|
| A. 邻帧跳变 | 与前后帧加速度超限 | `abs(dv/dt) > max_accel × 1.2` |
| A2. V 字形 | 急减速后立即急加速 | 单侧加速度 > 2.5× 物理极限，对侧反向 |
| A3. 悬崖 | 单侧极端跳变 + 对侧平坦 | 一侧 > 3× 极限，对侧 < 0.3× 极限 |
| B. 锚点趋势偏离 | 偏离锚点间线性插值 | `abs(v - interp) > 3 × max_accel × dt` |
| C. 孤立离群 | 邻帧一致的异常值 | 两侧邻居一致但本帧偏离 > 阈值 |
| D. 局部趋势偏离 | 5 帧中位数偏差 | 局部中位数偏离 > 3 km/h 且邻帧一致 |

**阶段 2 — 重 OCR 获取备选**：
- 4 种预处理变体：灰度、CLAHE+OTSU、OTSU 反相、EasyOCR 后备
- 有缓存避免重复处理
- 若所有变体返回相同值（OCR 一致性错误），跳过 OCR 直接使用插值

**阶段 3 — 最优值选择**：
- 对每个候选值评分：邻帧一致性（0.4）+ 锚点插值接近度（0.35）+ 平滑度（0.25）
- 选最高分，更新帧值

**阶段 4 — 多轮迭代**：
- 最多 3 轮，逐轮检测并修复新出现的错误
- 空集提前退出

**阶段 5 — 级联填充**：
- 左到右有序处理不可恢复帧
- 线性插值 + 加速度裁剪
- while 循环直到收敛（消除级联尖刺）

#### auto_select_anchors — 自动锚点选择

**两阶段过滤**：
1. **局部中位数过滤器**：自适应窗口（覆盖 ≈0.3 秒），帧值偏离中位数 ≤4 km/h 视为可靠
2. **锚点后过滤**：移除偏离邻帧 >10 km/h 的极端离群锚点

## 输出格式

CSV 文件包含注释头和数据行：

```
# RaceVideoToLog
# video_hash=709674b9c34665ea, video=test.mp4
# roi=876,933,962,982, format=km/h
# max_speed=400.0, max_accel=50.0, div=4, target_h=24.0, ...
timestamp,distance,speed_kmh,flag
10.48,0.00,0.00,2
10.55,0.00,0.00,0
```

**Flag 含义**：
- `0` — 原始 OCR 值（未修改）
- `1` — 自动纠错值
- `2` — 锚点值（自动锚点或人工确认）

## 项目结构

```
RaceVideoToLog/
├── RaceVideoToLog.py    # 主 GUI 应用和 CLI 入口
├── analysis.py           # 数据分析模块（GUI tab + CLI 无头分析）
├── headless.py           # 无头 OCR 流水线
├── ocr_engine.py         # OCR 引擎、纠错算法、辅助函数
├── RaceVideoToLog.spec   # PyInstaller 打包配置
└── README.md
```

### 核心模块

| 文件 | 行数 | 功能 |
|------|------|------|
| `RaceVideoToLog.py` | ~1700 | GUI（Tkinter）、5 阶段 Correction B、基线标注窗口、导出流程 |
| `ocr_engine.py` | ~1500 | GPU 后端管理、OCR 预处理、`auto_select_anchors`、`correct_speed_series_v2`、候选生成、辅助函数 |
| `analysis.py` | ~580 | `AnalysisTab` GUI 类、`parse_csv`、`smooth_data`、`run_analysis_headless` |
| `headless.py` | ~270 | 无头 CLI OCR 流水线、多预处理重试、并行推理 |

## 打包为 EXE

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

生成 `dist/RaceVideoToLog.exe`（≈333MB）。已配置：
- 排除 NVIDIA CUDA DLL（用户自行安装 CUDA Toolkit）
- 排除 DirectML provider、v5_server 模型
- scipy 完全排除（使用纯 NumPy 实现 Savitzky-Golay）
- UPX 压缩（关键 DLL 除外）
- 字节码优化 optimize=2

## CLI 完整参数

```
python RaceVideoToLog.py [video] [options]

位置参数:
  video               视频文件路径

可选参数:
  --roi X1 Y1 X2 Y2   识别范围（像素坐标）
  --format FORMAT     速度单位: m/s, km/h, mile/h (默认: km/h)
  --div N             采样间隔 1/N (1-10, 默认: 2)
  --max-speed N       最大合理速度 km/h (默认: 400)
  --max-accel N       最大合理加速度 m/s² (默认: 50)
  --target-h N        OCR 目标高度 px (默认: 24)
  --pad N             边缘填充 px (默认: 0)
  --workers N         并行线程数 (默认: 4, 最大: 32)
  --backend BACKEND   auto, cuda, cpu (默认: auto)
  --ocr-model MODEL   v5_mobile (默认)
  -o, --output PATH   输出 CSV 路径
  --frame-start N     起始帧
  --frame-end N       结束帧
  --baseline-freq N   人工基准抽样频率 1/N (1=全部人工)
  --analysis CSV1 CSV2 数据分析模式（比较两个 CSV）
  --analysis-out PREFIX 分析输出前缀
```

## License

MIT
