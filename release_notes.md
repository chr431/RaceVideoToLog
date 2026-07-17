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
