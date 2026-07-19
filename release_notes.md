## v2.4.0 更新日志

### 🚀 双模型流水线

- **主 OCR + 重 OCR 模型独立选择**：GUI 和 CLI 均支持分别指定主 OCR 模型（首次识别）和重 OCR 模型（纠错阶段）
- **默认组合 v6_tiny + v6_small**：tiny 主 OCR 快 42%，small 重 OCR 保证纠错质量，整体提速 28%，99.6% 帧与全程 small 完全一致（平均差异 0.022 km/h）
- CLI 新增 `--reocr-model` 参数，GUI 新增重 OCR ComboBox

### 🧠 纠错算法增强

- **短暂偏离检测器** (`_detect_brief_excursion`)：捕获"下探又弹回"模式（如 219→212→219），解决此类偏离在物理容限内但实为 OCR 误读的问题
- **卡值检测器** (`_detect_stuck_value`)：检测连续 ≥3 帧完全相同但上下文在变化的模式（OCR 重复输出同一错误值）
- **投票制错误检测**：从"首个命中即退出"升级为 8 种检测器全部投票（≥2 票确定错误，1 票 + 锚点稀疏 >30 帧 = 错误），减少假阳性
- **动态评分权重**：候选评分权重根据到最近锚点的距离动态调整——锚点密集区锚点权重主导 (~0.5)，稀疏区邻居/平滑权重上升
- **时间缩放 anchor_score 阈值**：从固定 `max_accel * 3.6`（如 252 km/h）改为 `max_accel * time_to_anchor * 3.6 * 3`，近锚点处（1 帧约 13 km/h）更严格
- **最低分数阈值**：候选分必须 >0.3 才接受修正，防止低置信度修正引发连锁漂移

### 🎯 锚点选择改进

- **宽窗口一致性检查**（新增第 4 阶段）：±30 帧中位数偏离 >20 km/h 的锚点被剔除，解决局部验证无法检测的"多步渐进漂移"（如 216→117 在 0.12s 内累计 99 km/h 但每步都在容限内）
- **图连通性验证**（第 5 阶段）：替换逐点加速度验证，DFS 找最大连通分量，保留物理自洽的锚点集合
- 锚点选择从 4 阶段升级为 5 阶段

### 🔧 重 OCR 策略优化

- 重 OCR 高度从 h=24 改为 **h=32**（与主 OCR h=24 不同，约 10% 概率产生不同读数）
- 基准测试验证：h=24 重 OCR 与主 OCR **永远产生相同值**（0/309 帧差异），纯属无效开销
- 完整 fallback 链（OTSU/CLAHE/反转/use_det=False × 多高度）24.8x 更慢，仅多覆盖 9.7% 错误帧——已确认不值得

### 🧹 代码清理

- **config.py** 精简 60%：移除 50+ 行从未被导入的常量（检测器阈值、评分权重、置信度参数——均硬编码在对应模块中）
- 移除死代码：`ocr_digital_fallback`、`_estimate_raw_trust`、`_gpu_backend`/`_gpu_patched`/`_sync_gpu_backend` 兼容桥接
- 移除死导入：gui.py（csv, traceback, SpeedObservation, _oe）、pipeline.py（_oe）、ocr_engine.py（os）
- 版本号统一 v2.3.0 → v2.4.0

### 📄 文档

- `PROJECT_KNOWLEDGE.md` 彻底重写：当前架构、ProcessingPipeline 双模式、8 检测器、5 阶段锚点选择、双模型策略验证数据
- README 默认模型和 CLI 参数更新

---

## v2.3.0 更新日志

### 🔧 依赖升级

- **onnxruntime-gpu** 1.26 → 1.27：CUDA 12.x 原生构建，修复多线程 cuDNN 冲突
- 移除冗余 `onnxruntime` CPU-only 包（`onnxruntime-gpu` 已内置 CPU 推理）
- 移除未使用依赖 `easyocr`、`torch`、`scipy`、`pandas` 等，venv 缩减 ~54%

### 🧹 代码结构优化

- **`widget_utils.py`** — 提取共享 GUI 组件（`make_static_card`、`setup_chart_zoom_pan`），消除 3 个文件间 ~80 行重复代码
- **`Flag` 枚举** (`ocr_engine.Flag`) — 统一管理 flag 常量（RAW=0, REOCR_AUTO=11 等），附带 `is_corrected()` / `is_anchor()` 方法，消除散布的魔法数字
- **Wildcard import 清理** — `gui.py` 中 `from ocr_engine import *` 替换为显式导入
- **`_reocr_cache` 生命周期** — 从模块级全局变量绑定到 Pipeline 实例
- **ROI 拖拽重绘节流** — 16ms 单次定时器避免高频鼠标事件过度重绘

### ⚡ 性能优化

- **Savitzky-Golay 滤波**：逐点 lstsq (O(N×W³)) → 预计算卷积系数 + `np.convolve` (O(N))，10-100x 加速
- **`_reocr_cache` 键值**：`hash(tobytes())` 完整数组拷贝 → `hash(data[:256])`

### 📊 调试与可观测性

- **日志系统**：`print()` → `logging` 模块，静默异常全部补充日志
- **CSV 统计行**：`# stats: total=N, anchors=M, corrected=K`
- **CSV 计时行**：`# timing: ocr=Xs, correction=Ys, ...`
- **Pipeline 阶段计时**：各阶段耗时自动记录，CLI 输出详细分解
- **CUDA v13 路径扫描**：`_register_gpu_dlls()` 新增 v13.x 检测

### 📦 构建优化

- TensorRT EP 代码移除（PP-OCRv6_small 上无加速效果）
- Qt6 未用模块排除（Quick/Qml/Pdf/Network 等，-60MB）
- ORT 非推理子目录排除（transformers/tools/quantization，-4MB）
- 最终构建：~567 MB (onedir)

---

## v2.2.0 更新日志

### 🧠 自动部分数字推断

OCR 经常漏读或误读个别数字位（如 221 被识别为 21），现在系统会**自动推断**缺失位置：

- 根据邻居帧插值速度与 OCR 原始文本，自动生成部分数字模式（如 OCR 读到 "21"、邻居约 221 → 自动推断 "x21"）
- 候选值优先于 re-OCR 和混淆候选，但不独占，最终由评分函数选择最优值
- 新增 `_infer_partial_pattern()` 函数，支持插入缺失位和替换误读位两种模式
- Flag 13 标记自动部分数字修正

### 📊 人工审核窗口改进

- **图表缩放与平移**：滚轮缩放 + 右键拖拽平移，与数据分析 Tab 交互一致
- **散点图**：替换折线图为散点图，当前段橙色高亮 + 背景色块，已确认段绿色，问题段红色
- **布局稳定**：matplotlib `layout='none'` 消除画布尺寸乒乓效应，拖拽增加 40ms 节流
- **图像预览优化**：识别区域预览不再被压扁，比例更贴近实际裁剪区域

### 📄 CSV 头格式优化

- **按语义分行**：时空范围 → 处理参数 → 推理引擎 → 纠错模式
- **`parse_csv_header()`** 改用正则解析，兼容逗号空格和纯逗号两种分隔符，正确处理 ROI 值和空值
- **向后兼容**：旧格式 CSV 也能完整解析（`test4_truth.csv` 从 2 个 key 提升到 14 个）
- 修复 `auto_anchor` 标签在 `max_speed<=0` 和 review 无问题段时的遗漏

### 🐛 Bug 修复

- **导入设置按钮变量复用**："导入视频"和"导入设置"按钮共用一个变量名，分别改为 `_import_video_btn` 和 `_import_settings_btn`
- **Radio button 信号**：`clicked` → `toggled`，解决 `setChecked()` 不触发模式切换的问题
- **ROI 导入后预览不刷新**：添加 `_redraw()` 调用
- **`_fix_errors` 重复计算**：提升 `interp_cand` 和 `reocr_set` 到循环顶部，消除重复 re-OCR 调用
- **锚点加速度验证**：改为对比任意有效速度帧，不再仅限于锚点之间

### ⚡ 性能优化

- QThread → 原生 `threading.Thread`，消除 CUDA ONNX 推理的 4.6x 性能损失
- 初次 OCR 使用纯 BGR resize（无灰度转换、无 OTSU 二值化）
- 移除 digital fallback 和 re-OCR OTSU/CLAHE 变体
- 解码 + OCR 合并为单流水线循环，生产者解码预处理，消费者推理

### 📋 Flag 系统

| Flag | 含义 |
| ---- | ---- |
| 0    | 原始 OCR |
| 11   | re-OCR 自动修正 |
| 12   | 插值填充 |
| 13   | 部分数字自动修正 |
| 21   | 自动锚点帧 |
| 22   | 人工修正帧 |
| 23   | 人工确认段 |
| 30   | 待人工审核（skip_fill 模式） |

---

🔗 [完整提交历史](https://github.com/chr431/RaceVideoToLog/compare/v2.1.0...v2.2.0)
