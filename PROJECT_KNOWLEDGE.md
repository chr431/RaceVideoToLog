# RaceVideoToLog — 项目知识总结

## 项目概述

从赛车视频中 OCR 提取速度数字，输出时间-速度-距离 CSV 文件。
- GPU (CUDA) / CPU 双后端
- GUI (PySide6 + qfluentwidgets Fluent Design) + CLI 无头模式
- 两段式人工审核：pass1 轻量纠错 → 人工标记 → pass2 重新纠错
- 物理约束纠错 + 数据分析对比

## 代码结构

```
RaceVideoToLog.py    # 入口：arg 解析 + GUI/CLI 调度
pipeline.py          # 统一处理流水线 (ProcessingPipeline)，GUI/CLI 共用
gui.py               # PySide6 主窗口 (Fluent Design 主题)
gui_review.py        # 人工审核对话框 (问题段审核 + 校正)
gui_analysis.py      # 数据分析 Tab (matplotlib 图表)
analysis.py          # 数据分析业务逻辑 (解析/平滑/绘图)
ocr_engine.py        # OCR 引擎、预处理、候选生成、锚点选择、Flag 枚举、SG 滤波
correction.py        # 纠错流水线 (检测器、重 OCR、评分、填充、置信度)
config.py            # 集中常量：颜色、默认值、物理常量、纠错参数、GPU 后端
gpu_setup.py         # GPU DLL 加载 + ONNX Runtime CUDA EP 配置
widget_utils.py      # 共享 GUI 组件：静态卡片、matplotlib 缩放/平移
theme_manager.py     # 主题注册/回调系统
headless.py          # CLI 无头 OCR 模式
RaceVideoToLog.spec  # PyInstaller onedir 打包配置
build_exe.bat        # 一键构建 EXE 脚本
tests/
  test_correction.py # 37 个单元测试 (检测器、评分、锚点、SG、Flag、解析)
```

### 依赖

- PySide6 6.9+ (Qt 6 GUI)
- qfluentwidgets 1.11+ (Fluent Design 组件库)
- rapidocr-onnxruntime (PP-OCRv6 ONNX 模型)
- onnxruntime-gpu (CUDA 推理, ORT 1.27+) / onnxruntime (CPU fallback)
- opencv-python-headless (图像 I/O 和预处理)
- matplotlib (数据分析绘图, QtAgg 后端)
- numpy (数值计算)
- pyinstaller (打包, 开发依赖)

### OCR 模型

- 默认组合 v6_tiny (主 OCR) + v6_small (重 OCR): PP-OCRv6 ONNX 模型，GUI 和 CLI 均支持独立选择
- 预处理: 灰度化 + 缩放到 target_h (默认 24px) → 转 BGR → PP-OCR 内置归一化
- 重 OCR: h=32（与主 OCR h=24 不同高度，约 10% 概率产生不同读数）
- 双模型策略 (已验证): tiny 主 OCR + small 重 OCR 比全程 small 快 28%，
  99.6% 帧差异 <2 km/h，平均差异 0.022 km/h。tiny 的 OCR 快 42%（17s vs 29s），
  但原始 OCR 遗漏略多导致纠错负担增加（9s vs 6s），净效果仍然正面
- 已移除: OCR 原生置信度 (检测置信度 ≠ 识别准确度，本场景无区分力)；
  多重预处理变体 (OTSU/CLAHE/反转, 24.8x 更慢仅多覆盖 9.7%)

## ProcessingPipeline 架构 (pipeline.py)

`ProcessingPipeline` 是 GUI 和 CLI 的共享处理核心，运行在调用者线程中。

### 两种模式

**自动锚点模式** (`run_auto`): OCR → 锚点选择 → 完整 5 阶段纠错 → 距离积分 → CSV。用于 CLI 和 GUI 一键导出。

**人工辅助模式** (两段式):

- `run_review_pass1`: OCR → 锚点选择 → 轻量纠错(仅 h=32 重 OCR) → 置信度评分 → 问题段检测 → 返回结果给 `ReviewDialog`
- `run_review_pass2`: 合并人工修正 → 完整 5 阶段纠错 → 距离积分 → CSV

### 关键组件

- 生产者-消费者 OCR：Queue 流水线重叠 I/O 与 GPU 推理
- 重 OCR 缓存：绑定到 Pipeline 实例生命周期，基于帧图像哈希
- 性能计时：`_timing` dict 记录 OCR/纠错/写入耗时
- 调试模式：`_debug_raw_text` 在 CSV 中输出原始 OCR 文本

## 性能基准与优化记录（2026-07-20）

### 推理后端对比

| 后端 | v6_tiny | v6_small | 说明 |
| --- | --- | --- | --- |
| ONNX Runtime 1.27 | 534fps | 248fps | 当前生产后端 |
| rapidocr 3.9.1 + ONNX | 420fps | 149fps | 未启用 gpu_setup 优化 |
| rapidocr 3.9.1 + PaddlePaddle | 255fps | 138fps | 需 CUDA 12.9 专用轮子 |
| PaddleOCR 3.7.0 + PaddlePaddle | 95fps | 90fps | Pipeline 开销过大 |
| rapidocr-paddle 1.4.5 (v4) | — | — | 旧模型, 9fps |

**结论: ONNX Runtime 在 PP-OCRv6 tiny/small 模型上具有压倒性优势。**
PaddlePaddle 框架开销（设备同步、executor 调度）对小模型每次推理占主导。

### PaddlePaddle 安装备忘

- `paddlepaddle-gpu` 仅到 2.6.2, 不支持 CUDA 12.9。
- `paddlepaddle` 3.x 统一包为 CPU-only。GPU 版需从专用索引安装:
  `pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`
- rapidocr-paddle 捆绑 PP-OCRv4 模型。
- PaddleOCR 3.x 需 PaddlePaddle ≥3.0。
- Python 版本限制: rapidocr-paddle <3.13; PySide6-Fluent-Widgets 3.12+ 无 wheel。

### 视频解码方案

| 方案 | 速度 | ONNX兼容 | 说明 |
| --- | --- | --- | --- |
| cv2 FFMPEG | ~450fps | ✅ | 当前方案 |
| decord 软件 | 600-750fps | ❌ | FFmpeg DLL 与 ORT 冲突 |
| decord 子进程 | ~250fps | ✅ | IPC 开销抵消解码收益 |
| PyAV | 133-203fps | ✅ | 太慢 |

**decord+ONNX DLL 冲突**: decord 捆绑 FFmpeg 4.x DLL 与 ORT CUDA 依赖链冲突。
**子进程隔离**: Pipe/共享内存传帧增加 ~1ms/帧 IPC 开销, 整体反降 25%。

### 全管线吞吐量 (v6_tiny, test4, div=1)

| 阶段 | 吞吐量 |
| --- | --- |
| 纯 OCR 推理 | 1118fps |
| 视频解码上限 | 458fps |
| 管线实际 (buffer=16) | ~340fps |
| 管线实际 (buffer=8) | ~310fps |

瓶颈: Python 线程同步 (Queue mutex + GIL) 造成 ~26% 效率损失。

### 已排除的无效优化

- 多 consumer 并行: cuDNN crash
- 分离 det/rec 线程: GPU 上下文切换开销 > 收益
- Lock-free SPSC: Python spin-wait 烧 CPU
- 预解码全帧: 不如流水线
- CUDA Graph: 需绕过 RapidOCR
- FP16/INT8: 需模型转换

### 已实施的优化

| 优化 | 效果 |
| --- | --- |
| v6_tiny 默认主 OCR | 推理 2.2x 提速 |
| buffer 8→16 | 管线 ~7% 提速 |
| 跳过 detection 模型 | GPU 显存 -8MB |
| 默认 tiny+small 组合 | 整体 ~28% 提速 |

## GUI 架构

### 主窗口 (gui.py)

- `RaceVideoToLogApp(QMainWindow)` — Fluent Design 主窗口
- 双 Tab: OCR 处理 + 数据分析 (Pivot 切换)
- 亮/暗双主题、ROI 拖拽选择、视频预览
- `_ExportThread(QThread)`: 后台运行 ProcessingPipeline
- `_Pass2Thread(QThread)`: 非阻塞 pass2 重纠错

### 人工审核对话框 (gui_review.py)

- `ReviewDialog(QDialog)`: 左侧问题段列表 + 右侧速度曲线 + 原始图像 + 修正控件
- ← → 键在当前段内逐帧导航
- 支持完整修正 (输入确切值) 和部分修正 (输入 "12x" 含缺失位)
- matplotlib 图表：增量更新 (set_offsets)，仅段切换/主题变化时完全重建
- 建议帧：段首尾 + 最低置信度 + 加速度异常点 (最多 8 个)

### 数据分析 Tab (gui_analysis.py)

- 3 个 CSV 导入槽位, 自动解析
- v-t / v-x / Δt-x 三种图表模式
- SpanSelector 拖拽范围选择 (积分距离/用时)
- 平滑滑块 + 诊断信息 (红色=自动纠错/绿色=人工纠错)
- 缓存机制：仅 csv/mode/coastdown 变化时重建，平滑变化增量更新

### 主题系统 (theme_manager.py)

- 全局初始化: `setTheme(Theme.AUTO)` (QApplication 创建后)
- `ThemeManager.register(callback)` 回调注册，暗色变化时自动通知所有组件
- `_sync_figure_theme()` 同步 matplotlib figure/axes/spines 颜色

## 关键算法

### 1. 预处理 (`_preprocess_standard`)
灰度化 → 缩放到 target_h → 转 BGR。无额外 fallback（验证过完整 fallback 链性价比极低）。

### 2. 候选生成 (`build_speed_candidates`)

三种策略：原始值 → 后缀扩展 (处理丢位, 如 "60"→60/160/260) → OCR 字符混淆替换 (如 6↔8, 3↔8)。

### 3. 锚点选择 (`auto_select_anchors`)
5 阶段：自适应窗口 (覆盖 ~0.3s) → 中位数筛选 (center + boundaries) → 邻居验证 (相邻 ≤10 km/h) → 宽窗口去漂移 (±30 帧中位数) → **图连通性验证**

图连通性：候选锚点建图，物理可达 (加速度不超限) 者连边，DFS 找最大连通分量。比旧版逐点加速度验证更鲁棒，自动剔除与多数锚点物理矛盾的孤立异常点。

### 4. 纠错算法 (`correct_with_anchors`, 5 阶段)

1. **投票制错误检测**：8 种检测器全部投票，≥2 票 = 确定错误，1 票 + 锚点稀疏 (>30 帧 gap) = 错误
   - A. 邻帧跳变  A2. V 字形  A3. 悬崖  E. 卡值 (连续 ≥3 帧相同但上下文变化)
   - B. 锚点趋势偏离  C. 孤立离群  D. 局部趋势偏离
2. **h=32 重 OCR**：与主 OCR h=24 不同高度，约 10% 概率产生不同值
3. **动态权重评分**：锚点密集区锚点权重高 (~0.5)，稀疏区邻居/平滑权重上升
4. **多轮迭代**：最多 3 轮
5. **级联填充**：对无法修复帧，线性插值 + 加速度钳制，最多 10 轮

**轻量模式** (light_mode=True): 仅阶段 1-3，只选重 OCR 值或原始值，不生成混淆/推断/插值候选，不迭代不填充。用于 pass1。

### 5. 置信度评分 (`compute_confidence`)
4 维度：OCR 偏差 (0.3) + 邻帧加速度 (0.4) + 纠错标记惩罚 (-30) + SG 平滑偏差 (0.2)。→ `find_problem_segments` 聚合成问题段，建议帧 = 段首尾 + 最低分 + 加速度异常点。

### 6. 部分数字推断 (`_infer_partial_pattern`)

OCR 读到不完整数字 (如 "21" 而邻居约 221) 时自动推断缺失位。模式 "x21" (首位缺失) / "21x" (末位缺失) / "2x1" (误读)。仅在与预期值偏差 <20% 时采纳。

## Flag 枚举 (CSV 第 4 列)

| 值 | 常量 | 含义 |
| --- | --- | --- |
| 0 | RAW | 原始 OCR 输出，未纠错 |
| 11 | REOCR_AUTO | 重 OCR 自动修正 |
| 12 | FILL_INTERP | 级联插值填充 |
| 13 | PARTIAL_AUTO | 部分数字模式推断修正 |
| 21 | ANCHOR_AUTO | 自动锚点 (硬约束) |
| 22 | ANCHOR_MANUAL | 人工修正锚点 |
| 23 | CONFIRMED_SEG | 人工确认段内帧 |
| 30 | FLAGGED_REVIEW | 标记待人工审核 |

辅助方法: `Flag.is_corrected(f)` (10-19), `Flag.is_anchor(f)` (≥20)

## GPU 后端 (gpu_setup.py)

- `select_backend("cuda"|"cpu"|"auto")`: 扫描 CUDA 安装目录 (v13.3-v11.6)，ctypes 加载 DLL，monkey-patch `OrtInferSession.__init__` 配置 CUDA EP (HEURISTIC cudnn search, ORT_PARALLEL)
- `reset_backend()`: 重置状态以切换后端
- `get_gpu_backend()`: 公共访问器，返回 "CUDA" 或 "CPU"

## CSV 格式

```
# RaceVideoToLog v2.4.0
# video_hash=..., video=...
# roi=..., format=..., frame_start=, frame_end=
# max_speed=..., max_accel=..., div=..., target_h=..., pad=..., buffer=...
# backend=..., model=... [, reocr_model=...]
# auto_anchor=1 | manual_anchor=1
# stats: total=..., anchors=..., corrected=...
# timing: ocr=....0s, correction=....0s, ...
timestamp,distance,speed_kmh,flag
```

## 人工审核流程

1. 用户加载视频、设置参数、点击"导出"
2. `_ExportThread` 运行 `run_review_pass1`: OCR → 轻量纠错 → 置信度 → 问题段
3. 若有问题段，弹出 `ReviewDialog`
4. 用户逐段审核：← → 键导航、建议帧按钮、输入修正值 (完整或部分如 "12x")
5. 点击"完成审核" → `run_review_pass2`: 合并修正 → 完整纠错 → 写 CSV

## 测试

```bash
python -m pytest tests/ -v    # 37 个单元测试
```

覆盖：SG 滤波、expand_partial、Flag 枚举、normalize_ocr_text、safe_int/float、parse_csv_header、build_speed_candidates、find_neighbor_anchors、8 种错误检测器 (独立 + 集成)、锚点选择 (窗口/中心/邻居)、compute_confidence、score_candidate。

## 常用命令

```bash
# GUI
python RaceVideoToLog.py

# CLI: 自动锚点
python RaceVideoToLog.py test4.mp4 --roi 858 939 964 1004 --max-speed 400 --max-accel 100 --div 2 -o out.csv

# CLI: 指定 OCR 模型
python RaceVideoToLog.py test4.mp4 --roi 858 939 964 1004 --ocr-model v6_tiny -o out.csv

# CLI: 双模型 (tiny 主 OCR + small 重 OCR，推荐快速模式)
python RaceVideoToLog.py test4.mp4 --roi 858 939 964 1004 --ocr-model v6_tiny --reocr-model v6_small -o out.csv

# CLI: 从已有 CSV 继续 (跳过 OCR)
python RaceVideoToLog.py --from-csv existing.csv test4.mp4 -o out.csv

# 数据分析
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix

# 打包
build_exe.bat
# 或手动: python -m PyInstaller RaceVideoToLog.spec --noconfirm
```
