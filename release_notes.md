# Release Notes

## v2.6.1 (2026-07-27)

### 全面代码清理与工程优化

**代码结构**
- GUI 拆分：`gui_export.py`（导出线程）+ `gui_settings.py`（设置面板工厂）
- `ocr_engine.py` 延迟导入重构为 `_init_rapidocr()`，消除 Pylance 警告
- 60+ 魔法数字迁入 `config.py`（总计 124 配置项），所有算法参数可配置
- 版本号集中管理：`config.__version__`，CI 自动校验与 `pyproject.toml` 一致
- CSV 加载去重：`parse_csv_header` + `parse_csv_setting` + `csv_field_dest` 统一 CLI/GUI
- 候选生成逻辑去重，spinbox flyout 统一使用 `make_int_spinbox`

**构建系统**
- 新增 `setup_venv.bat`：一键创建 venv + 安装依赖
- `build_exe.bat` 重写：6 步全自动（检查 Python → 创建 venv → 安装依赖 → 验证 → 构建 → 完成）
- `pyproject.toml`：显式 `py-modules` 声明，`setuptools` 不再误识别目录
- TRT 10.x 锁定：`tensorrt>=10,<11`，`--no-deps` 安装只取 Python 绑定（~2MB），排除 DLL 包（~2.2GB）
- `cuda-python` + `tensorrt` 移入主依赖（共 ~35MB）

**测试**
- 测试重写为纯单元测试（250 行，原 464 行），不再依赖配置常量和算法内部行为
- 移除对已删除函数（`compute_lcs_scores`、`_detect_errors` 等）的引用

**性能优化**
- TRT 默认 FP32（构建 80s vs FP16 178s，推理速度持平）
- decord 解码器 `del _frame` 立即释放全帧数组，系统内存降低 ~50%
- 解码/推理分时统计：`_timing["decode"]` + `_timing["inference"]`
- 新增 `tools/bench_decoder.py`：cv2 vs decord 自动对比
- 新增 `tools/bench_trt_build.py`：FP16 vs FP32 引擎构建速度对比

**CSV 格式**
- 头新增 `fps` 字段，`analysis.py:parse_csv()` 自动将帧号转为秒
- `video_backend` 写入实际使用值（非用户选择值）

**其他**
- 许可证：GPLv3（因依赖 PySide6-Fluent-Widgets GPLv3）
- CI/CD：GitHub Actions（单元测试 + 版本一致性校验）
- GUI：decord/TensorRT 缺失时弹出明确安装指引
- `--from-csv` 和 GUI 导入设置统一解析逻辑，`video_backend` 修复遗漏
- 修复 `build_exe.bat` LF 行尾导致 cmd 闪退
- 修复 `pipeline.py` `config` 未导入、`viterbi.py` `max_speed_kmh` 参数缺失

### Phase 1 信号重构（2026-07-28~29）

**Physics 信号重新设计**
- 移除文本长度门控：不再只接受 ≥3 位数的邻居，2 位低速读数也能参与物理检查
- 时间窗搜索（±0.25s）替代固定紧邻帧：自适应帧率变化
- 连续指数衰减打分（`100·exp(-excess·2.0)`）替代 4 档离散打分
- `nv >= 0` 将停车帧（速度=0）视为可靠锚点

**Linearity 信号：中位数鲁棒插值**
- 全新实现：每侧取 K 个有效帧，K×K 配对插值，取中位数期望值
- 天然抗异常值，无需可靠性门控（移除有缺陷的 `_find_reliable_neighbor`）
- 时间窗搜索（±0.25s，最多 10 帧/侧）+ `nv >= 0`
- 所有速度范围同等有效（不再依赖文本长度）

**最差信号地板规则**
- 加权平均易被高分信号掩盖问题 → 新增地板规则：任一信号 <30 封顶 25，<50 封顶 50，<70 封顶 69
- 帧要获得 `high confidence` (≥70)，所有 4 个信号必须全部 ≥70
- 检测漏检率大幅下降：div=1 下严重漏检归零，div=2 从 18.2%→7.3%

### Phase 2 纠正算法优化

**候选生成改进**
- `build_speed_candidates` 混淆映射 `_CONFUSION_MAP` 提升为模块级常量（为精准候选扩展做准备）
- 百位变体生成扩展到所有数字位数（不再仅限 3 位）

**参数调优**
- `AUTO_CORRECT_THRESHOLD` 80→70：conf≥70 已满足所有信号正常的约束，不应再被 Viterbi 干预
- `MANUAL_CORRECT_THRESHOLD` 保持 40（经测试确认最优）

**Force-Median 自适应收敛**
- 新增提前终止：连续两轮修改 ≤2 帧即停止（防止过度平滑）
- 局部中位数天然无法处理大错误岛内部——由 Viterbi + auto-align 在前序步骤解决

### 代码清理（P0/P1/P2 Audit）

**架构缺陷修复**
- 移除 **8 处文本长度歧视**（`correction.py` + `pipeline.py`）：`len(raw_text) < 3` 门控全部替换
- 修复 **3 处 sg_dev 僵尸引用**：Phase 1 不再计算的信号，Phase 2 改用 accel/linearity 信号替代
- 清理 **18 个孤常量**：`TEXTLEN_SCORE_*`、`REOCR_AGREE_*`、`SG_CLUSTER_SCORE_*`、`PHYSICS_FALLBACK_DT` 等

**固定帧窗口→时间窗（3 处）**
- 信任传播：`TRUST_WINDOW_TIME=0.15s`（替代硬编码 ±3 帧）
- Force-Median：`FORCE_MEDIAN_WINDOW_TIME=0.1s`（替代硬编码 ±2 帧）
- 远距离插值：`DISTANT_INTERP_MIN_TIME=1.0s`（替代硬编码 30 帧）

**弃用代码移除**
- 移除 `correct_with_trust()` 函数（v2.5 起标记 deprecated，无调用者）
- 重命名 `_re_ocr_frame` → `_multi_height_ocr`（准确描述多高度缩放行为）
- 重命名 `_force_sg_smooth` → `_force_median_smooth`（实际使用中值滤波而非 SG）
- 重命名 `FORCE_SG_*` → `FORCE_MEDIAN_*` 常量
- 修复 `COMPAT_CONF_OCR_WEIGHT` 缺失（被 `compute_confidence` 引用）

### 性能优化

**ONNX CPU 后端（+45%）**
- `intra_op_num_threads=cpu_count//2`（限制线程数避免调度损耗）
- `inter_op_num_threads=2`、`enable_cpu_mem_arena=True`
- `OMP_WAIT_POLICY=PASSIVE`（避免线程忙等）
- 实测推理速度 314→456 fps

**decord 内存优化**
- 顺序读取替代随机访问：`_vr.next()` 代替每帧 `_vr[fi]`（seek_accurate）
- 消除内部帧缓存，内存从 ~5GB 降至 ~1GB（与 cv2 持平）
- 解码速度保持 343 fps

### Bug 修复
- **测试无法运行**：移除对已删除 LCS 函数的导入
- **数据分析横轴错误**：帧号未转为时间 → CSV 头写入 fps，`parse_csv` 自动转换
- **GUI 导入 CSV 时 video_backend 不生效**：遗漏映射已补全
- **PyInstaller spec 绝对路径**：改为 `os.path.abspath`
- **decord 静默回退 cv2**：`ModuleNotFoundError` 单独捕获并提示安装
- **`_diag_notes` 未初始化**：`log_level=normal` 时 `AttributeError`
- **pyname `config` 未导入**：`pipeline.py` 中 `config.__version__` 缺少 `import config`
- **viterbi `max_speed_kmh` 参数缺失**：`_compute_confidence_scores` 签名遗漏参数
- **matplotlib 布局警告**：`constrained` → `tight` 布局引擎

### 准确率（vs ground truth，max_accel=50，div=2）

| 指标 | v2.6.1 自动 | v2.6.1 手动 |
|------|:----------:|:----------:|
| test.mp4 ≤0.5 | 97.8% | 97.9% |
| test.mp4 >5 | 4 帧 | 4 帧 |
| test2.mp4 ≤0.5 | 99.2% | 99.0% |
| test2.mp4 >5 | 0 | 0 |
| test3.mp4 误报 | 0 low-conf | — |

**版本变更（v2.6.0 → v2.6.1）**：基础版本号同步，见上方

---

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
