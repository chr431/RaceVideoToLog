# RaceVideoToLog

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。支持 GPU (CUDA) / CPU 两种后端，提供 GUI 和 CLI 两种使用方式。

## 功能概述

- **OCR 识别**：PP-OCR v5_mobile 识别仪表盘七段数码管速度
- **自动锚点纠错**（GUI）：全自动，自动识别可靠帧并使用物理约束纠正错误
- **人工基准标注**（GUI）：手动标注部分帧为锚点，系统自动补充可靠帧，实现最高精度
- **数据分析**（GUI）：多 CSV 对比图表（v-t / v-x / Δt-x），支持缩放、平移、范围选择
- **无头 CLI**：命令行批量处理视频，输出 CSV

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.10 | |
| rapidocr_onnxruntime | latest | PP-OCR 引擎 (v5_mobile) |
| onnxruntime | ≥1.26 | ONNX 推理后端 |
| OpenCV | ≥4.10 | 视频读取、图像预处理 |
| NumPy | ≥2.0 | 数值计算 |
| Matplotlib | ≥3.10 | 数据分析图表（GUI 模式） |
| Tkinter | ≥8.6 | GUI 界面 |

**可选（GPU 加速）**：
- NVIDIA CUDA Toolkit 12.x + cuDNN 9.x

## 快速开始

### GUI 模式
```bash
python RaceVideoToLog.py
```
1. 导入视频，框选仪表盘数字区域
2. 选择纠错模式（自动锚点 / 人工基准）
3. 设置参数（采样间隔、速度/加速度限制等）
4. 点击导出

### CLI 模式
```bash
python RaceVideoToLog.py test.mp4 \
  --roi 876 935 962 982 \
  --div 4 \
  --frame-start 600 \
  --frame-end 4300 \
  -o output.csv
```
CLI 模式使用 `correct_speed_series_v2` 物理约束纠错，不支持人工基准标注。

### 数据分析（CLI / GUI）
```bash
# CLI: 比较两个 CSV，生成 3 张对比图
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out comparison

# GUI: 数据分析 tab 支持导入最多 3 个 CSV，交互式查看
```

## 工作流程

### 1. 视频帧提取

按 `div` 参数等间隔采样（div=4 表示每 4 帧处理 1 帧），可指定起止帧范围。

### 2. OCR 识别（三步后备链）

每帧经过以下流水线，前一步成功则跳过后续步骤：

```
步骤 1: 灰度化 + 缩放到 target_h → RapidOCR 标准检测识别
  ↓ 失败
步骤 2: OTSU 二值化 + 缩放 → RapidOCR 标准检测识别
  ↓ 失败
步骤 3: ocr_digital_fallback 后备链
  ├─ 3a. CLAHE + OTSU 增强 → 3 种高度(28/32/48px) → 标准 RapidOCR
  └─ 3b. 4 种预处理变体 × 2 种高度(32/48px) → RapidOCR(use_det=False)
       变体: clahe_otsu, 反相灰度, otsu, otsu 反相
```

确保每帧都有识别结果（彻底失败时标记为 -1.0），保证帧索引对齐。

### 3. 纠错（GUI 模式）

GUI 提供两种纠错模式，用户必须选择其一。

#### 自动锚点纠错（推荐）

```
OCR → auto_select_anchors → Correction 5 阶段 → CSV
```

实测准确率：97.3% exact, 98.9% within 1 km/h, max error 9 km/h。

#### 人工基准标注

```
OCR → 人工标注采样帧 → 自动锚点补充 → 混合锚点 Correction → CSV
```

用户每 N 帧手动标注一次（N 可配置），标注完成后自动补充可靠帧作为额外锚点，人工锚点值不会被覆盖。

#### Correction — 5 阶段流水线

**阶段 1 — 错误检测**（6 种检测器并行）：

| 检测器 | 检测模式 | 条件 |
|--------|----------|------|
| A. 邻帧跳变 | 与前后帧加速度同时超限 | `abs(dv/dt) > max_accel × 1.2`（双向） |
| A2. V 字形 | 急减速后立即急加速 | 单侧 `> 2.5×` 极限，对侧反向 |
| A3. 悬崖 | 单侧极端跳变 + 对侧平坦 | 一侧 `> 3×` 极限，对侧 `< 0.3×` 极限 |
| B. 锚点趋势偏离 | 偏离锚点间线性插值过多 | `abs(v - interp) > 3 × max_accel × dt` |
| C. 孤立离群 | 邻帧一致但本帧异常 | 邻居彼此一致，本帧与两边都冲突 |
| D. 局部趋势偏离 | 5 帧中位数偏离 | 偏离中位数 `> 3 km/h` 且邻帧与中位数一致 |

**阶段 2 — 重 OCR 获取备选**：
- 对每个错误帧，用 4 种预处理变体重新 OCR（灰度、CLAHE+OTSU、OTSU 反相、`ocr_digital_fallback`）
- 有缓存避免重复处理同一帧
- 若所有变体返回相同值（OCR 一致性错误），跳过 OCR 直接使用锚点插值

**阶段 3 — 最优值选择**：
评分函数 = 邻帧一致性(0.4) + 锚点插值接近度(0.35) + 平滑度(0.25)，选最高分候选。

**阶段 4 — 多轮迭代**（最多 3 轮）：
逐轮重新检测 → 修复，空集提前退出。

**阶段 5 — 级联填充**：
左到右有序处理不可恢复帧，线性插值 + 加速度裁剪，while 循环直到收敛（消除级联尖刺）。

#### auto_select_anchors — 自动锚点选择

1. **局部中位数过滤器**：自适应窗口（覆盖 ≈0.3 秒），帧值偏离中位数 ≤4 km/h 视为可靠
2. **锚点后过滤**：移除偏离邻帧 >10 km/h 的离群锚点

### 4. CLI 模式纠错

CLI 使用 `correct_speed_series_v2`（物理约束纠错）而非 Correction：
1. 自适应可达性扫描（窗口覆盖 ≈0.5 秒）
2. 尖峰/显示保持/加速区检测
3. 可疑段 DP 修正
4. `_retry_suspect_frames`：对修正后仍超限的帧进行 5 种预处理变体重 OCR
5. 后处理离群值检测

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

**Flag 含义**：
- `0` — 未修改的 OCR 值
- `1` — 自动纠错值
- `2` — 锚点值（自动锚点或人工确认）

## 项目结构

```
RaceVideoToLog/
├── RaceVideoToLog.py    # GUI 应用 + CLI 入口
├── correction.py         # 物理约束纠错流水线（GUI/CLI 共用）
├── ocr_engine.py         # OCR 引擎、预处理、锚点选择、后端管理
├── analysis.py           # 数据分析模块（GUI tab + CLI 分析导出）
├── headless.py           # 无头 CLI OCR 流水线
├── RaceVideoToLog.spec   # PyInstaller 打包配置
└── README.md
```

## CLI 参数

```
python RaceVideoToLog.py [video] [options]

位置参数:
  video                   视频文件路径（省略则启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2       识别范围像素坐标
  --format {m/s,km/h,mile/h}  速度单位 (默认: km/h)
  --div N                 采样间隔 1/N, 1-10 (默认: 2)
  --max-speed N           最大合理速度 km/h (默认: 400)
  --max-accel N           最大合理加速度 m/s² (默认: 50)
  --target-h N            OCR 目标高度 px (默认: 24)
  --pad N                 边缘填充 px (默认: 0)
  --workers N             并行线程数 (默认: 4, ≤32)
  --backend {auto,cuda,cpu}  OCR 后端 (默认: auto)
  -o, --output PATH       输出 CSV 路径
  --frame-start N         起始帧号
  --frame-end N           结束帧号
  --analysis CSV1 CSV2    分析模式：比较两个 CSV 生成 PNG
  --analysis-out PREFIX   分析输出文件名前缀

仅 GUI 可用:
  --baseline-freq N       人工基准抽样频率 1/N
```

## 打包

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

生成 `dist/RaceVideoToLog.exe`（≈333MB）。已排除 NVIDIA DLL、DirectML、scipy、v5_server 模型。GPU 用户需自行安装 CUDA Toolkit + cuDNN。

## License

MIT
