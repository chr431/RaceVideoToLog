# Release Notes

## v2.6.0 (2026-07-25)

### 核心变更：两阶段 Viterbi 纠错架构

完全重写纠错系统，从 LCS 局部一致性评分 5 阶段流水线替换为两阶段 Viterbi 动态规划：

**Phase 1 — 错误检测** (`error_detection.py`，新文件)
- 只读：不修改任何值或 flag
- 7 个信号连续置信度评分 [0,100]：ocr_conf、physics、linearity、reocr_agree、text_len、accel_spike、sg_dev
- 加速度尖峰对检测识别一致性孤岛边界
- SG 中值滤波偏离度检测孤岛内部（全局趋势不一致但局部物理自洽）

**Phase 2 — 错误纠正** (`correction.py`，重写)
- Viterbi DP 全局最优路径选择 (`viterbi.py`，新文件)：在每帧候选集上联合优化，不再依赖邻居当前值
- 观测代价（偏离参考值）+ 转移代价（加速度超标平方惩罚）
- 多轮 Viterbi + Fill + Smoothness + Auto-align + Force-SG 六段式后处理

旧函数 `correct_with_trust` 保留为 deprecated wrapper，`_detect_errors` 和 `find_problem_segments` 已移除。

### 一致性孤岛修正

针对连续多帧 OCR 同向误读的"一致性孤岛"问题：
- **远距离插值**：当局部邻居也在孤岛内时，跳过 30 帧寻找外部正确锚点
- **候选值过滤**：内部一致的 3 位数字帧移除相差 ≥100 的百位变体（防止 168→68 误修）
- **孤岛轮次限制**：孤岛候选存在时 Viterbi 限 1 轮（防止 r1 修正→r2 回退）
- **参考值保护**：physics≥90 + linearity≥90 + sg_dev≥80 的帧跳过插值参考值

### 模式重构

| 特性 | v2.5 手动 | v2.6 手动 | v2.6 自动 |
|------|----------|----------|----------|
| 纠错管线 | 仅 re-OCR | 完整管线 | 完整管线 |
| Force-SG 平滑 | — | — | ✅ |
| 目标 | 保守 | 匹配真值 | 最小化 max_dv |

- 手动模式从仅重 OCR 升级为完整管线（fill + smoothness + auto-align）
- 自动模式新增 Force-SG：5 帧滑动中值滤波，无视 flag，迭代平滑至收敛
- CLI 新增 `--mode {auto,manual}` 参数

### 准确率改进（vs ground truth）

| 指标 | test.mp4 v2.5 | test.mp4 v2.6 自动 | test4.mp4 v2.5 | test4.mp4 v2.6 自动 |
|------|-------------|-------------------|---------------|-------------------|
| Correct (d≤2) | 99.1% | **99.6%** | 99.1% | **99.0%** |
| Severe (d≥5) | 31 | **1** | 48 | **24** |
| Severe (d≥20) | 18 | **1** | ~25 | **5** |
| RealMax | ~100 | 100 (边界帧) | ~200 | **55** |

### 可配置视频解码器

- 新增 `--video-backend {cv2,decord}` CLI/GUI 选项
- 默认 cv2（兼容性好、内存低），decord 可选（NVDEC 硬件加速，~2.6GB 显存）
- CSV 头写入 `video_backend` 字段
- `--from-csv` 支持导入 video_backend 设置

### 日志与调试

- 新增 `--log-level {normal,detailed,debug}` CLI/GUI 选项
- normal 级别仅输出主 CSV
- detailed/debug 额外输出 `_stage_report.csv`（逐帧 7 信号）+ `_summary.json`
- debug 额外输出 `_diagnostics.csv`

### Bug 修复

- **最终检查无图像**：`_raw_frames` 在 `finalize()` 中过早清空 → 延迟到 GUI `_finish_export()` 清空
- **Re-OCR 失效**：`_raw_frames = {}` 破坏候选生成 → 恢复列表传递
- **诊断文件帧号错位**：使用顺序索引而非实际帧号 → 存储实际帧号
- **decord 内存泄漏**：VR 对象缓存所有解码帧 → OCR 完成后显式释放

### 命名重构

旧名（基于实现）→ 新名（基于功能）：
- `LCS_TIME_WINDOW` → `CONSISTENCY_TIME_WINDOW`
- `LCS_TAU` → `CONSISTENCY_DECAY_TAU`
- `LCS_HIGH_WEIGHT` → `CONSISTENCY_PINNED_WEIGHT`
- `LCS_WARNING_THRESHOLD` → `MANUAL_EDIT_ACCEL_WARNING`
- `_lcs_score_for_value` → `_neighbor_consistency_score`
- `VITERBI_SOFT_ANCHOR_CONFIDENCE` → `VITERBI_TRUSTED_BOUNDARY_CONFIDENCE`
- `Flag.ANCHOR_AUTO` / `Flag.ANCHOR_MANUAL` → 已移除
- GUI 设置键 `manual_anchor`/`auto_anchor` 保留向后兼容

### 死代码清理

移除：`compute_lcs_scores`、`compute_lcs_scores_lr`、`lcs_detect_errors`、`find_problem_segments`、`LCS_ERROR_LOW`、`LCS_TRUST_HIGH`、`LCS_INTERP_WEIGHT`、`LCS_NOVELTY_WEIGHT`、`LCS_CONFIDENCE_MIN_SCORE`、`ACCEL_ANOMALY_THRESHOLD`、`MAX_SUGGESTED_FRAMES`、`PROBLEM_MIN_SEGMENT_LEN`、`skip_fill` 参数链路

### 文档

- README 重写为用户文档（安装→用法→输出→参数）
- PROJECT_KNOWLEDGE 重写为开发知识库（架构→算法→性能→已知局限→历史演进）
- 记忆库新增 `viterbi-v5-architecture.md` 和 `consistency-island-detection.md`

---

## v2.5.0

### 初始版本

- LCS 局部一致性评分 5 阶段纠错流水线
- 中值滤波参考剖面验证一致性孤岛检测
- TensorRT + ONNX Runtime 双后端
- PySide6 Fluent Design GUI + CLI 无头模式
- decord NVDEC 硬件解码（默认）
- 人工审核对话框：问题段列表 + 部分数字通配符输入 `12x`
- PP-OCRv6 tiny/small 双模型策略
