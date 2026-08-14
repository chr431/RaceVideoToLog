# RaceVideoToLog — 项目知识（从 Claude Code 项目记忆提炼）

从赛车视频 OCR 提取速度，输出时间-速度-距离 CSV。Python 3.11+，PySide6 GUI +
CLI 双入口。段级流水线（segment_flow.py）是唯一生产管线。

## 回归门禁（改动后必跑）

- **准确率漏斗是真正的测试门禁**：`tools/accuracy_breakdown.py` 跑
  test/test2/test3/test5/test6 + ground_truth_csv，与 `tools/baseline.json`
  对比，当前基线 **12 错误**（test 4 / test2 8 / test3/5/6 0，TOL±1，
  A4 豁免后）。任一视频或总量的最终错误数增加 → 退出码 1（失败即红）。
  有意改进后：确认无意外回归 → `--update-baseline` 更新基线，并同步
  重新生成回归夹具（`tools/make_regression_fixtures.py`）与更新本节基线描述。
- **CI 回归夹具（无视频/无 decord 可跑）**：`tests/fixtures/` ——
  - `seg_series/*.json`：生产 run() 的全量段级序列（test/test2/test3/5/6），
    `tests/test_seg_series.py` 重构 _confidence+_dense_correct 并逐段断言
    与基线一致 + 最终错误集一致（12 案例口径）。
  - `ocr_frames/`：12 错误案例代表帧的原始 ROI 裁剪（.npy）+ manifest，
    `tests/test_ocr_fixtures.py` 用 onnxruntime CPU 锁定 OCR 行为基线。
  夹具版本必须与 config.__version__ 一致（测试强制）。任何算法/预处理/
  模型改动使夹具读数变化 → 测试失败，属有意改动时先跑完整漏斗确认无
  回归再重新生成夹具。
- **测试视频在 `D:\Videos\racelog_test`**（test~test6.mp4；truth 在仓库
  ground_truth_csv/，test5/test6 用 *_ref.csv）。
- **test4 truth 不可用**（巨大误差，用户确认）——不作准确率依据，仅用于
  "不把物理正确的帧改坏"。
- 剩余 12 错误构成：11 个 2-off（平滑偏移/DP 部分纠正/truth 瞬时跳变，
  信息论极限）+ test2 2 个"纠错"（改错，change_threshold=3 的 tradeoff 面）。
  12 案例全部是 len=1 单帧段误读（明细可跑 tools/final_err_dump.py）。
- pytest：`tests/test_segment_flow.py`（_detect/_correct）、
  `test_correction_chain.py`（conf/DP/build_rows/预处理）、`test_csv_io.py`、
  `test_from_csv.py`（from-csv 显式参数优先语义）、`test_packaging.py`
  （py-modules 完整性）、`test_seg_series.py`、`test_ocr_fixtures.py`、
  `test_decoder_integration.py`（缺 decord 显式跳过）。
- CI（.github/workflows/ci.yml）：test job 跑全量 pytest；decoder-smoke job
  从 chr431/decord release v0.7.5 下载 fork 真实跑解码集成测试（下载失败
  显式跳过不红）；version-check 跑 tools/version.py。
- 1-2 km/h 平滑偏移漏纠是信息论极限（与真实平滑不可区分）——局部物理约束
  无解，勿重复探索。

## 已验证的死路（不要重新投入）

- medium 模型（用户否决）；fp16/INT8（tiny/small 非算力受限，无效）
- Otsu/二值化/对比度拉伸/锐化喂 OCR（PP-OCRv6 训练于自然 RGB）
- 多预处理自动选择（固定 gray+gamma2.0 是全量最优，1.15% 误读）
- 窗口重 OCR 自动化（"至少一窗口读对"是幸存者偏差；仅人工辅助有价值）
- 结构相似 obs（fill 锚点插值已是更强先验）；scipy（纯 numpy win3 替代）

## 已锁定的参数（勿随意改动）

- OCR 预处理：resize 48 高 + **灰度 gamma 2.0**（config.OCR_GAMMA，正式预处理；
  分段/代表帧选择仍用 raw 灰度，已知不一致但已接受——SEG_GAMMA=0 锁定；
  对照实验钩子 config.SEG_GAMMA / env RVTOL_SEG_GAMMA，结论见下）
- SEG_GAMMA=0（分段 raw 灰度）为锁定默认。RVTOL_SEG_GAMMA=2.0 对照实验
  结论（v2.14 实测）：**净负，不做** —— 段数 1530/1267/1146→1407/1157/927
  （gamma 改变 Otsu/二值化合并了段），原始误读不变（155），但 test2 8→9
  （新增 1 误改）、总量 12→13。raw 灰度 + Otsu 的组合在分割层面更优；
  gamma 仅保留给 OCR 预处理。钩子留作实验入口，勿设默认。
- 性能基线（test5 7223 帧，decord v0.7.5 + onnxruntime 1.29，2026-08 实测）：
  **CPU+CPU 9.0s / GPU+CPU 8.6s / CPU+TRT 6.8s / GPU+TRT 8.1s**。
  生产者各 Python/numpy 子步骤合计仅 ~4%（GPU ~1000fps / CPU ~1260fps
  ROI-only）。DLL 在 _decord_build + site-packages 两处
- **decord v0.7.5（ROI-first + 撕裂帧竞态修复，fork 仓库 D:\Repo\decord）**：
  解码器只输出 ROI 矩形：CPU filter crop 先于 format（yuv420p 上
  x/y/w/h 全偶数约束，用偶数超集+顶部精裁绕过）；GPU kernel 只算 ROI
  窗口 + 输出池 ROI 尺寸（asnumpy 单次批量 D2H）。**同步 D2H 是正确性
  关键**：v0.7.4 的线程本地非阻塞拷贝流在并发管线中产生孤立帧撕裂
  （~0.2-0.9% 帧部分行旧内容 → 分段/OCR 漂移，门禁 12→24-41 波动），
  改回同步 cudaMemcpy 后 A/B 5 轮 0 分歧、门禁可复现 12。剩余余量：
  GPU 真批量异步解码（display 回调每帧 sync 仍在）。
- **线程预算规则（v2.14 起代码内置，_ocr_num_threads/auto_ocr_thread_count）**：
  OCR = 全部物理核（16C32T → 16），解码用 fork 默认（FFmpeg 帧线程 2 +
  filter auto≈2，落在 SMT 份额上不抢物理核）。实测 OCR 8→16 线程：
  GPU 解码 11.3→9.0s、CPU 解码 12.8→9.5s，满负荷正收益；超物理核不再提升。
  RVTOL_OCR_THREADS env 钩子优先（实验用）。**旧"FFMPEG8 + filter1 +
  OCR8"组合在当前栈上 13.3s/9.8GB 病态，已废弃**（filter=1 是元凶，
  单线程 sws 全帧转换 ~0.8ms/帧）。DECORD_FFMPEG_THREAD_COUNT=4 是坏点
  （13.5s）。逐帧 next_roi 780fps vs 批量 1247fps（固定成本摊薄）
- **pad 宽 224 最优**（降宽省推理但 +19 误读）；buffer 128 最优
- RVTOL_OCR_THREADS / RVTOL_OCR_GAMMA / DECORD_PREFETCH_DEPTH /
  DECORD_FILTER_THREADS env 钩子用于实验
- SEG_DP_CHANGE_THRESHOLD=3.0（gamma raw 调参产物，消掉 DP 微调误改）
- **A4 孤立尖峰豁免**（13→12）：SEG_DP_DEANCHOR_JERK_MIN/MAX=5/40 ——
  conf∈[20,50) 的锚定段若 jerk（二阶差分）∈带通则解锚交给 DP。判别实测：
  真刹车 jerk≈0、丢位污染 jerk≥80、孤立尖峰 jerk 中等（test#74 jerk=9）。
  **jerk_score 无判别力**（正确段被丢位邻居污染后与尖峰同形）——必须用
  原始 jerk 值；带通下界 0 是灾难（78 误改），上界 ≥80 引入污染段

## 架构要点

- 生产路径：`run()` → `_run_pipelined()`（解码∥分段∥段值 OCR 流水线，
  有界队列背压）→ `_confidence` → `_dense_correct`（稠密格点 DP）→ `_build_rows`
- **串行方法（_decode_all/_segment/_ocr_segments/_detect/_correct）是实验参考
  路径**，只被 tools/ 与测试使用，生产不走——改动需同时考虑两侧
- 分段：raw 灰度（`_gray_seg`，SEG_GAMMA=0；RVTOL_SEG_GAMMA 可实验 gamma）+
  Otsu 阈值 + 逐帧异或 + `_cluster_win3`（C=5，纯 numpy）
- flag 语义：0 RAW / 11 DP_CORRECTED / 12 FILL_INTERP / 21 HIGH_TRUST /
  22 PINNED（20-29 高可信绿点，10-19 自动纠错红点，GUI 按区间着色）
- TRT 引擎缓存：`<程序目录>/ocr_engines/`（免安装便携），旧 LOCALAPPDATA
  只读回退；engine 文件名含 sm89（本机 RTX 4060），换卡靠"加载失败→删除→
  重建"兜底；TRT 10/11 双兼容（getattr 回退 + 输出张量 profile 守卫）
- **decord fork 解码层状态（fork 仓库 D:\Repo\decord，本地可重建）**：
  - ✅ v0.7.5 已做：CPU filter crop 先于 format；GPU kernel ROI 窗口 +
    ROI 输出池；同步 D2H（撕裂帧竞态修复）。
  - 剩余余量：GPU 真批量异步解码 —— GetBatch 逐帧 NextFrameImpl +
    display 回调每帧 cudaStreamSynchronize；批内单次 sync + 批量 D2H
    可将 GPU decode 阶段（8.1s）进一步压缩。

## 工作流约束

- **git 安全**：绝不删分支（曾因 git init 毁掉整个 .git）；force push /
  reset --hard 前必须确认；merge 后不删实验分支（保留参考）
- 所有 .py 用 4 空格缩进（tab 曾致 Edit 失败）
- **from-csv 显式参数优先已修复并被测试锁定**（`RaceVideoToLog.apply_csv_settings`
  + `tests/test_from_csv.py`：命令行显式写出即使等于默认值也不被 CSV 覆盖）。
  调用 CLI 的工具脚本仍必须显式传参（防"未指定"走 CSV/默认路径）。
- **tools/ 布局**：永久工具在 tools/（accuracy_breakdown.py 门禁、
  detect_eval.py 召回评估、final_err_dump.py 错误明细、extract_roi.py
  ROI 复查导出、make_regression_fixtures.py 夹具生成、version.py、
  bench_*/decord_smoke/ocr_thread_safety_check）；一次性实验脚本在
  tools/archive/（`_` 前缀 + proto_* + eval_phase1，保留参考勿删）。
- 测试脚本复用 OcrEngine 实例（每次新建 TRT 加载 ~25s）
- 版本号以 config.__version__ 为准，tools/version.py 强制一致性；发布流程
  走 .github/workflows/release.yml
- decord 是自建 fork 硬依赖（PyPI 版不支持），缺失 `_decord_build/` 报错退出
- 新增纠错阶段前先测"对已修正帧是否净正"（align/forceSG 曾毁正确帧）
