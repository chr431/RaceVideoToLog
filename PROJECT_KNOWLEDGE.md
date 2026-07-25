# RaceVideoToLog — 项目知识总结 v2.6

## 项目概述

从赛车视频中 OCR 提取速度数字，输出时间-速度-距离 CSV 文件。

- 默认 cv2 视频解码，可选 decord (NVDEC) 硬件加速
- TensorRT / CPU 双后端自动选择，RapidOCR + ONNX Runtime
- GUI (PySide6 Fluent Design) + CLI 无头模式
- 两阶段纠错：Phase 1 多信号置信度评分 → Phase 2 Viterbi 动态规划全局最优
- 自动模式额外包含强制 SG 平滑，目标最小化帧间速度跳变

## 代码结构

```
RaceVideoToLog.py    # 入口：arg 解析 + GUI/CLI 调度
pipeline.py          # 统一处理流水线 (ProcessingPipeline)，GUI/CLI 共用
gui.py               # PySide6 主窗口 (Fluent Design 主题)
gui_review.py        # 最终检查对话框 (全帧速度曲线 + 逐帧修正)
gui_analysis.py      # 数据分析 Tab (matplotlib 图表)
analysis.py          # 数据分析业务逻辑
ocr_engine.py        # OCR 引擎、候选生成、Flag 枚举、SG 滤波、LCS 评分（GUI校验用）
correction.py        # Phase 2 纠错流水线 (Viterbi + fill + smoothness + auto-align + force_smooth)
error_detection.py   # Phase 1 多信号置信度评分 (7 个信号，只读)
viterbi.py           # Viterbi 动态规划全局最优路径选择
config.py            # 集中常量：颜色、默认值、物理常量、纠错参数
gpu_setup.py         # GPU DLL 加载 + TensorRT/CPU 后端选择
widget_utils.py      # 共享 GUI 组件
theme_manager.py     # 主题注册/回调系统
headless.py          # CLI 无头模式
RaceVideoToLog.spec  # PyInstaller 打包配置
build_exe.bat        # 一键构建 EXE
```

## 两阶段纠错架构

### Phase 1 — 错误检测 (`error_detection.py`)
**只读**：不修改任何值或 flag。对每帧计算 [0,100] 连续置信度。

7 个信号：
| 信号 | 权重 | 说明 |
|------|------|------|
| ocr_conf | 0.01 | OCR 模型内部置信度（几乎不可靠） |
| physics | 0.20 | 物理可达性（仅可靠邻居） |
| linearity | 0.20 | 局部线性度 |
| reocr_agree | 0.20 | OCR 读数自洽 |
| text_len | 0.14 | 文本长度信号 |
| accel_spike | 0.40 | 加速度尖峰对检测（一致性孤岛） |
| sg_dev | 0.15 | SG 中值滤波偏离度 |

### Phase 2 — 错误纠正 (`correction.py`)
接收 Phase 1 置信度，执行：
1. **候选生成**：原始值 + 重OCR + 百位变体 + 插值
2. **Viterbi DP**：在每帧候选集上寻找全局最优路径。观测代价（偏离参考值）+ 转移代价（加速度超标平方惩罚）
3. **Fill**：对无法修复帧做物理约束插值
4. **Smoothness**：检测并平滑孤立尖峰
5. **Auto-align**：对 SG 适度偏离的帧向局部插值微调
6. **Force-SG**（仅自动模式）：5帧滑动中值滤波，无视所有 flag，最小化帧间速度跳变

### 两种模式

| 特性 | 手动模式 | 自动模式 |
|------|---------|---------|
| 纠错阈值 | 40 | 80 |
| Fill + Smoothness + Auto-align | ✅ | ✅ |
| Force-SG 平滑 | ❌ | ✅ |
| 目标 | 匹配真值 | 最小化 max_dv |

## 关键算法

### 一致性孤岛检测与修正
- **检测**：sg_dev ≤ 20 且 physics ≥ 90 → 帧内一致但与全局 SG 剖面显著偏离
- **远距离插值**：跳过孤岛（min_distance=30）找到外部正确锚点，作为 Viterbi 候选
- **孤岛轮次限制**：孤岛候选存在时 Viterbi 限 1 轮，防止 r1 修正→r2 回退

### 参考值保护
physics ≥ 90 且 linearity ≥ 90 且 sg_dev ≥ 80 的帧与邻居和全局趋势均一致，
跳过插值参考值，防止 Viterbi 将正确值（如 168）误改为 68。

### 候选值过滤
内部一致的 3 位数字帧移除相差 ≥100 的百位变体，防止 Viterbi 选择错误变体。

## 准确率基准（2026-07-25，vs ground_truth_csv）

test.mp4（max_accel=50）：
| 指标 | 手动 | 自动 |
|------|------|------|
| Correct (d≤2) | 99.2% | 99.6% |
| Severe (d≥5) | 22 | 1 |
| Severe (d≥20) | 6 | 1 |
| RealMax | 119 | 100 |
| max_dv | 120 | 100 |

test4.mp4（max_accel=70）：
| 指标 | 手动 | 自动 |
|------|------|------|
| Correct (d≤2) | — | 99.3% |
| Severe (d≥5) | — | 24 |
| Severe (d≥20) | — | 5 |
| RealMax | — | 55 |

## Flag 枚举

| 值 | 常量 | 含义 |
| --- | --- | --- |
| 0 | RAW | 原始 OCR 输出 |
| 11 | REOCR_AUTO | Viterbi/重OCR 自动修正 |
| 12 | FILL_INTERP | 物理插值填充 |
| 13 | PARTIAL_AUTO | 部分数字推断修正 |
| 21 | HIGH_TRUST | Viterbi+物理验证高可信帧 |
| 22 | PINNED | 用户手动修正（绝对真值） |
| 23 | CONFIRMED_SEG | 人工确认段内帧 |
| 30 | FLAGGED_REVIEW | 待人工审核 |

## 依赖

- rapidocr 3.9+ (TensorRT/ONNX OCR)
- onnxruntime (CPU 推理)
- opencv-python-headless (图像预处理 + 默认视频解码)
- decord (可选，NVDEC 硬件解码)
- PySide6 + PySide6-Fluent-Widgets (GUI)
- matplotlib (数据分析绘图)
- numpy (数值计算)

## 常用命令

```bash
# GUI
python RaceVideoToLog.py

# CLI 自动模式（默认）
python RaceVideoToLog.py test.mp4 --roi 877 935 961 986 --max-speed 400 --max-accel 50 -o out.csv

# CLI 手动模式
python RaceVideoToLog.py test.mp4 --roi 877 935 961 986 --mode manual -o out.csv

# 指定解码器和后端
python RaceVideoToLog.py test.mp4 --roi ... --video-backend decord --backend tensorrt -o out.csv

# 打包
build_exe.bat
```
