# RaceVideoToLog — 项目知识总结

> **⚠️ 注意：本文档部分内容已过时**（如 LCS 5阶段纠错算法、onnxruntime-gpu 依赖、decord 默认解码器等）。
> 最新架构参见 `.claude/memory/viterbi-v5-architecture.md` 和 `consistency-island-detection.md`。
> 最后更新：2026-07 针对 v2.5 之前的版本。

## 项目概述

从赛车视频中 OCR 提取速度数字，输出时间-速度-距离 CSV 文件。

- decord (NVDEC) 硬件加速视频解码，比 cv2 快 ~60%
- GPU (CUDA) / CPU 双后端，RapidOCR 3.9.1 + ONNX Runtime
- GUI (PySide6 + PySide6-Fluent-Widgets Fluent Design) + CLI 无头模式
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
ocr_engine.py        # OCR 引擎、预处理、候选生成、LCS 评分、Flag 枚举、SG 滤波
correction.py        # 纠错流水线 (LCS 检测、重 OCR、评分、填充、置信度)
config.py            # 集中常量：颜色、默认值、物理常量、纠错参数、GPU 后端
gpu_setup.py         # GPU DLL 加载 + ONNX Runtime CUDA EP 配置
widget_utils.py      # 共享 GUI 组件：静态卡片、matplotlib 缩放/平移
theme_manager.py     # 主题注册/回调系统
headless.py          # CLI 无头 OCR 模式
RaceVideoToLog.spec  # PyInstaller onedir 打包配置
build_exe.bat        # 一键构建 EXE 脚本
tests/
  test_correction.py # 单元测试 (LCS 评分、Flag、SG、候选生成、解析)
```

### 依赖

- **rapidocr 3.9.1** (统一 OCR 包，取代已弃用的 rapidocr-onnxruntime)
- **onnxruntime-gpu 1.27+** (CUDA 推理) / onnxruntime (CPU fallback)
- **opencv-python-headless 5.x** (图像预处理: resize, copyMakeBorder, cvtColor)
- **decord 0.6.0** (NVDEC 硬件加速视频解码，比 cv2 快 ~60%)
  - ⚠️ 必须在 ORT 之后导入，否则 DLL 初始化失败
  - decord 无 CPU 软件解码；无 GPU 时自动回退 cv2.VideoCapture
- **PySide6 6.11+** (Qt 6 GUI) + **PySide6-Fluent-Widgets 1.11+** (Fluent Design)
- **matplotlib 3.10+** (数据分析绘图, QtAgg 后端)
- **numpy 2.x** (数值计算)
- **pyinstaller** (打包, 开发依赖)

### OCR 模型

- 默认组合 v6_tiny (主 OCR) + v6_small (重 OCR): PP-OCRv6 ONNX 模型
- **只用 rec 模型**（`use_det=False` + `use_cls=False`）：ROI 已紧密裁剪，不需要检测
- **Monkey-patch** `RapidOCR._initialize`：跳过 det/cls 模型的 ONNX session 创建，
  减少 2 个 CUDA session（~10MB 显存 + 初始化时间）
- 预处理: BGR resize 到 target_h (默认 24px) + cv2.copyMakeBorder 填充
  - ⚠️ rapidocr 3.9.1 识别模型需要 BGR 输入，不能用灰度
- 重 OCR: h=32（与主 OCR h=24 不同高度，约 10% 概率产生不同读数）
- 双模型策略 (已验证): tiny 主 OCR + small 重 OCR 比全程 small 快 28%，
  99.6% 帧差异 <2 km/h，平均差异 0.022 km/h
- 已移除: OCR 原生置信度、多重预处理变体 (OTSU/CLAHE/反转)

## ProcessingPipeline 架构 (pipeline.py)

`ProcessingPipeline` 是 GUI 和 CLI 的共享处理核心，运行在调用者线程中。

### 两种模式

**自动纠错模式** (`run_auto`): OCR → LCS 评分 → 完整 5 阶段纠错 → 距离积分 → CSV。用于 CLI 和 GUI 一键导出。

**人工辅助模式** (两段式):

- `run_review_pass1`: OCR → 轻量纠错(仅 h=32 重 OCR) → 置信度评分 → 问题段检测 → 返回结果给 `ReviewDialog`
- `run_review_pass2`: 合并人工修正 → 完整 5 阶段纠错 → 距离积分 → CSV

### 关键组件

- **视频源**: decord (NVDEC) 优先，失败时自动回退 cv2.VideoCapture (CPU)
  - decord 使用 `cpu(0)` 上下文（GPU 解码 + 复制到 CPU 内存），支持随机访问 `vr[i]`
  - cv2 回退使用 `grab()`/`retrieve()` 顺序读取
  - 导入顺序：ORT (select_backend) → decord（二者颠倒会导致 DLL init 失败）
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

| 方案 | 纯解码速度 | ONNX兼容 | 说明 |
| --- | --- | --- | --- |
| **decord (NVDEC) 当前方案** | **734-767fps** | ✅ | ORT 先导入即可共存 |
| cv2 FFMPEG (CPU 回退) | 457-497fps | ✅ | 无 GPU 时自动启用 |
| decord 子进程 | ~250fps | ✅ | IPC 开销抵消解码收益 |
| PyAV | 133-203fps | ✅ | 太慢 |

**decord vs cv2 纯解码速度 (1080p)**:

| 编码 | cv2 | decord | 提升 |
| --- | --- | --- | --- |
| HEVC (test/test2) | ~457fps | ~743fps | **+63%** |
| H264 (test3/test4) | ~482fps | ~758fps | **+57%** |

**decord + ONNX Runtime 兼容性**:
- decord 捆绑 FFmpeg 4.x DLL (`avcodec-58.dll`, `avutil-56.dll` 等)
- **冲突原因**: decord 先加载的 VC++ runtime 导致 ORT 的 `onnxruntime_pybind11_state` DLL 初始化失败
- **解决**: 导入顺序 ORT → decord（`select_backend()` 先于 `_run_ocr()` 调用）
- **CPU 回退**: decord 没有软件解码路径；`cpu(0)` 仍用 NVDEC 硬件，只是帧数据放 CPU 内存
  - 无 GPU 时自动回退 `cv2.VideoCapture`

### 全管线吞吐量 (v6_tiny, test.mp4, div=2)

| 阶段 | 吞吐量 |
| --- | --- |
| 纯 OCR 推理 | ~534fps |
| decord 视频解码 | ~750fps |
| 管线实际 OCR 阶段 | ~310fps |

瓶颈: OCR ONNX 推理是管线限速步骤，非视频解码。Producer-consumer Queue 缓冲已掩盖解码延迟。

### 已排除的无效优化

- 多 consumer 并行: cuDNN crash (ORT 1.27 已修复但无收益)
- 分离 det/rec 线程: GPU 上下文切换开销 > 收益
- Lock-free SPSC: Python spin-wait 烧 CPU
- 预解码全帧: 不如流水线
- CUDA Graph: 需绕过 RapidOCR
- FP16/INT8: 需模型转换
- decord 子进程隔离: IPC 开销 ~1ms/帧，整体反降 25%

### 已实施的优化

| 优化 | 效果 |
| --- | --- |
| v6_tiny 默认主 OCR | 推理 2.2x 提速 |
| buffer 8→16 | 管线 ~7% 提速 |
| skip det + cls 模型加载 | 初始化快 ~200ms, GPU 显存 -10MB |
| 默认 tiny+small 组合 | 整体 ~28% 提速 |
| decord NVDEC 视频解码 | 纯解码 +60%，管线瓶颈不在此 |
| cv2 CPU 自动回退 | 无 GPU 环境零配置运行 |


## 准确率基准 (2026-07-23)

在 test4.mp4 上以 ground_truth_csv/test4_truth.csv 为基准的自动化测试
(参数: v6_tiny+v6_small, TensorRT, div=1, max_accel=70):

| 指标 | 值 |
|------|-----|
| 总帧数 | 6203 |
| 错误率 | ~4.1% |
| False trusted (flag≥21 但错误) | ~1 帧 |
| 主要残留错误模式 | 小幅度 OCR 误读 (224→220) |

test.mp4 基准 (参数相同, max_accel=50):
| 指标 | 值 |
|------|-----|
| 总帧数 | 3573 |
| 错误率 | ~3.1% |
| False trusted | ~9 帧 (均 ≤5 km/h, 模糊区小误差) |

### 错误识别机制改进 (2026-07-23)

**问题**:
1. 一致性孤岛：OCR 对多帧连续误读为相同错误值 → LCS 无法检测
2. 级联错误传播：边界帧 OCR 故障通过 fill 阶段向左传播，破坏正确帧
3. HIGH_TRUST 帧在后续阶段可被修改（flag 保留但值改变）

**改进**:
1. **中值滤波剖面验证**：滑动中值滤波 (15 帧窗口) 检测偏离全局趋势的帧，
   阻止一致性孤岛被标记为 HIGH_TRUST。中值滤波天然抗离群值，优于 SG 滤波。
2. **HIGH_TRUST 帧保护**：`_fix_errors` 和 `_fill_unrecoverable` 不再修改已标记
   为 HIGH_TRUST (flag≥20) 的帧，阻止级联错误传播。
3. **修正后剖面重算**：最终 HIGH_TRUST 标记前使用修正后的值重新计算中值剖面，
   避免被原始 OCR 离群值污染。

验证工具: `tools/verify_accuracy.py`

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

### 3. LCS 局部一致性评分 (`compute_lcs_scores`)
对每帧计算指数加权时间窗投票分数。权重 = exp(-dt/0.06s)，时间窗 0.5s。
score >= 0.7 → HIGH_TRUST 自动标记；score < 0.3 → 错误；0.3-0.7 → borderline（一并纠错）。

pinned 帧（用户手动修正）在评分中获得 3× 权重，确保人工修正值影响邻近帧的评分。

### 4. 纠错算法 (`correct_with_trust`, 5 阶段)

1. **LCS 左右分侧错误检测**：compute_lcs_scores_lr 分别计算左右侧分数，
   lcs_detect_errors：任一侧 < LCS_ERROR_LOW(0.3) → error，
   任一侧 < LCS_TRUST_HIGH(0.7) → borderline。两侧均 >= TRUST_HIGH → 可信。
2. **候选生成**：
   - 重 OCR：尝试 3 种高度 (24,32,48)，取所有结果的并集
   - 混淆替换：CONFUSION_MAP 覆盖常见 OCR 误读 (1↔9, 2↔9, 0↔8 等)
   - 缺位扩展：`_auto_expand_digits` 对 1-3 位输入做插入+替换展开
   - 线性插值：以左右 HIGH_TRUST/PINNED 帧为基准估计
3. **统一评分选择**：候选 + 插值 + 当前值统一参与评分，
   综合 LCS 物理一致性 + 插值接近度 + 新颖性加成，选最高分。
   边界优先处理：按距可信帧距离排序，簇边界先修有助于级联修复。
4. **多轮迭代**：最多 4 轮，每轮重新 LCS 评分
5. **级联填充**：对无法修复帧，以左右 HIGH_TRUST/PINNED 帧为约束插值，最多 10 轮

**轻量模式** (light_mode=True): 仅阶段 1-3，只选重 OCR 值或原始值，不迭代不填充。用于 pass1。

**已知局限**："一致性孤岛"——当连续多帧被 OCR 误读为相同错误值时，LCS 在局部
无法区分对错。当前通过候选生成和评分加成缓解，但仍有约 0.2% 的此类错误需要人工审核。

### 5. 置信度评分 (`compute_confidence`)
4 维度：OCR 偏差 (0.3) + 邻帧加速度 (0.4) + 纠错标记惩罚 (-30) + SG 平滑偏差 (0.2)。→ `find_problem_segments` 聚合成问题段，建议帧 = 段首尾 + 最低分 + 加速度异常点。

### 6. 部分数字扩展 (`_auto_expand_digits`)

OCR 读到 1-2 位数字时，暴力生成所有可能的缺位扩展（如 "21" → [21, 121, 221, 321]）。
不依赖插值猜测，由 LCS 评分自动选择最优候选。

## Flag 枚举 (CSV 第 4 列)

| 值 | 常量 | 含义 |
| --- | --- | --- |
| 0 | RAW | 原始 OCR 输出，未纠错 |
| 11 | REOCR_AUTO | 重 OCR 自动修正 |
| 12 | FILL_INTERP | 物理插值填充 |
| 13 | PARTIAL_AUTO | 部分数字模式推断修正 |
| 21 | HIGH_TRUST | LCS 高可信帧 |
| 22 | PINNED | 用户手动修正 (绝对真值) |
| 23 | CONFIRMED_SEG | 人工确认段内帧 |
| 30 | FLAGGED_REVIEW | 标记待人工审核 |

辅助方法: `Flag.is_corrected(f)` (10-19), `Flag.is_trusted(f)` (≥20), `Flag.is_anchor(f)` (backward-compat)

## GPU 后端 (gpu_setup.py)

- `select_backend("cuda"|"cpu"|"auto")`: 扫描 CUDA 安装目录 (v13.3-v11.6)，ctypes 加载 DLL，monkey-patch `OrtInferSession.__init__` 配置 CUDA EP (HEURISTIC cudnn search, ORT_PARALLEL)
- `reset_backend()`: 重置状态以切换后端
- `get_gpu_backend()`: 公共访问器，返回 "CUDA" 或 "CPU"

## CSV 格式

```
# RaceVideoToLog v2.5.0
# video_hash=..., video=...
# roi=..., format=..., frame_start=, frame_end=
# max_speed=..., max_accel=..., div=..., target_h=..., pad=..., buffer=...
# backend=..., model=... [, reocr_model=...]
# pinned=N (仅当存在用户修正帧时)
# stats: total=..., trusted=..., corrected=...
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

覆盖：SG 滤波、expand_partial、Flag 枚举、normalize_ocr_text、safe_int/float、
parse_csv_header、build_speed_candidates、LCS 左右分侧评分、compute_confidence、
_auto_expand_digits、候选生成与选择。

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

# 打包 (自动安装 PyInstaller + 清理 + 构建)
build_exe.bat
# 或手动: python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

## 打包系统 (RaceVideoToLog.spec + build_exe.bat)

### build_exe.bat

一键构建脚本：自动创建/激活 .venv → 检查安装 PyInstaller → 清理旧构建 → 运行 PyInstaller。
需在项目根目录运行。

### PyInstaller 配置要点

- **CUDA DLL 不打包**：构建时从 PATH 移除 CUDA 目录，运行时用户自行安装 CUDA Toolkit
- **NVIDIA DLL 过滤**：`_NVIDIA_DLL_PREFIXES` 移除 cublas/cudnn 等（用户系统提供）
- **Qt6 精简**：仅打包 Widgets/Core/Gui/Xml/Svg，移除 Quick/Qml/Multimedia/WebEngine 等
- **PySide6 FFmpeg DLL 排除**：`avcodec-61.dll`, `avformat-61.dll` 等（decord 提供自己的 FFmpeg 4.x）
- **decord FFmpeg DLL 保留 + UPX 排除**：`avcodec-58.dll`, `avformat-58.dll`, `avutil-56.dll`, `swresample-3.dll`, `swscale-5.dll`
- **不需要的 ONNX 模型排除**：v5/v3 旧版、det 模型、medium 变体、DirectML/TensorRT provider
- **onnxruntime 非推理模块排除**：transformers/tools/quantization/datasets/backend
- **scipy/tkinter 完全排除**（SG 滤波已用纯 numpy 替代）
