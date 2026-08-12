# RaceVideoToLog — 项目知识（从 Claude Code 项目记忆提炼）

从赛车视频 OCR 提取速度，输出时间-速度-距离 CSV。Python 3.11+，PySide6 GUI +
CLI 双入口。段级流水线（segment_flow.py）是唯一生产管线。

## 回归门禁（改动后必跑）

- **准确率漏斗是真正的测试门禁**：`tools/_accuracy_breakdown.py` 跑
  test/test2/test3/test5/test6 + ground_truth_csv，当前基线 **12 错误**
  （test 4 / test2 8 / test3/5/6 0，TOL±1，A4 豁免后）。任何算法改动不得破基线。
- **测试视频在 `D:\Videos\racelog_test`**（test~test6.mp4；truth 在仓库
  ground_truth_csv/，test5/test6 用 *_ref.csv）。
- **test4 truth 不可用**（巨大误差，用户确认）——不作准确率依据，仅用于
  "不把物理正确的帧改坏"。
- 剩余 12 错误构成：11 个 2-off（平滑偏移/DP 部分纠正/truth 瞬时跳变，
  信息论极限）+ test2 2 个"纠错"（改错，change_threshold=3 的 tradeoff 面）。
- pytest：`tests/test_segment_flow.py`（13 用例，覆盖 _detect/_correct）。
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
  分段/代表帧选择仍用 raw 灰度，已知不一致但已接受）
- 性能基线（test5）：GPU+TRT 8.8s / GPU+CPU 10.1s / **CPU+CPU 9.6s**
  （decord v0.7.2 批量解码+预取 + 批量特征 + OCR 流水线化后，原 12.3s；
  -22%）。decord v0.7.2 是硬依赖（get_batch_roi 批量解码）；DLL 在
  _decord_build + site-packages 两处
- 线程配置：CPU 解码用 DECORD_FFMPEG_THREAD_COUNT=8 + DECORD_FILTER_THREADS=1
  最优（批量模式下 8 帧线程并行有效）；OCR 8 线程 + 攒批 B=16 最优。
  逐帧 next_roi 780fps vs 批量 1247fps（固定成本摊薄后帧线程并行才生效）
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
- 分段：raw 灰度 + Otsu 阈值 + 逐帧异或 + `_cluster_win3`（C=5，纯 numpy）
- flag 语义：0 RAW / 11 DP_CORRECTED / 12 FILL_INTERP / 21 HIGH_TRUST /
  22 PINNED（20-29 高可信绿点，10-19 自动纠错红点，GUI 按区间着色）
- TRT 引擎缓存：`<程序目录>/ocr_engines/`（免安装便携），旧 LOCALAPPDATA
  只读回退；engine 文件名含 sm89（本机 RTX 4060），换卡靠"加载失败→删除→
  重建"兜底；TRT 10/11 双兼容（getattr 回退 + 输出张量 profile 守卫）

## 工作流约束

- **git 安全**：绝不删分支（曾因 git init 毁掉整个 .git）；force push /
  reset --hard 前必须确认；merge 后不删实验分支（保留参考）
- 所有 .py 用 4 空格缩进（tab 曾致 Edit 失败）
- **from-csv 会静默覆盖显式参数**（值==默认值被误判为"未指定"）——测 CLI
  性能前先验证引擎/参数真实生效；调用 CLI 的工具脚本必须显式传参
- 测试脚本复用 OcrEngine 实例（每次新建 TRT 加载 ~25s）
- 版本号以 config.__version__ 为准，tools/version.py 强制一致性；发布流程
  走 .github/workflows/release.yml
- decord 是自建 fork 硬依赖（PyPI 版不支持），缺失 `_decord_build/` 报错退出
- 新增纠错阶段前先测"对已修正帧是否净正"（align/forceSG 曾毁正确帧）
