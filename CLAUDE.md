# RaceVideoToLog — 项目知识（从 Claude Code 项目记忆提炼）

从赛车视频 OCR 提取速度，输出时间-速度-距离 CSV。Python 3.11+，PySide6 GUI +
CLI 双入口。段级流水线（segment_flow.py）是唯一生产管线。

## 回归门禁（改动后必跑）

- **准确率漏斗是真正的测试门禁**：`tools/accuracy_breakdown.py` 跑
  test/test2/test3/test5/test6 + ground_truth_csv，与 `tools/baseline.json`
  对比，当前基线 **11 错误**（test 3 / test2 8 / test3/5/6 0，TOL±1，
  A4 豁免后；v2.15 灰度统一后 test 4→3）。任一视频或总量的最终错误数
  增加 → 退出码 1（失败即红）。有意改进后：确认无意外回归 →
  `--update-baseline` 更新基线，并同步重新生成回归夹具
  （`tools/make_regression_fixtures.py`）与更新本节基线描述。
- **CI 回归夹具（无视频/无 decord 可跑）**：`tests/fixtures/` ——
  - `seg_series/*.json`：生产 run() 的全量段级序列（test/test2/test3/5/6），
    `tests/test_seg_series.py` 重构 _confidence+_dense_correct 并逐段断言
    与基线一致 + 最终错误集一致（11 案例口径）。
  - `ocr_frames/`：错误案例代表帧的原始 ROI 裁剪（.npy）+ manifest，
    `tests/test_ocr_fixtures.py` 用 onnxruntime CPU 锁定 OCR 行为基线。
  夹具版本必须与 config.__version__ 一致（测试强制）。任何算法/预处理/
  模型改动使夹具读数变化 → 测试失败，属有意改动时先跑完整漏斗确认无
  回归再重新生成夹具。
- **测试视频在 `D:\Videos\racelog_test`**（test~test6.mp4；truth 在仓库
  ground_truth_csv/，test5/test6 用 *_ref.csv）。
- **test4 truth 不可用**（巨大误差，用户确认）——不作准确率依据，仅用于
  "不把物理正确的帧改坏"。
- 剩余 11 错误构成：9 个 2-off（平滑偏移/DP 部分纠正/truth 瞬时跳变，
  信息论极限）+ test2 2 个"纠错"（改错，change_threshold=3 的 tradeoff 面）。
  11 案例全部是 len=1 单帧段误读（明细可跑 tools/final_err_dump.py）。
- pytest：`tests/test_segment_flow.py`（_detect/_correct）、
  `test_correction_chain.py`（conf/DP/build_rows/预处理）、`test_csv_io.py`、
  `test_from_csv.py`（from-csv 显式参数优先语义）、`test_packaging.py`
  （py-modules 完整性）、`test_seg_series.py`、`test_ocr_fixtures.py`、
  `test_hybrid_decoder.py`（cpu+nvdec 识别/切分/解码 worker）、
  `test_decoder_integration.py`（缺 decord 显式跳过）。
- CI（.github/workflows/ci.yml）：test job 跑全量 pytest；decoder-smoke job
  从 chr431/decord release v0.7.8 下载 fork 真实跑解码集成测试（下载失败
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
- 性能基线（test5 7223 帧，decord v0.7.8 + onnxruntime 1.29，2026-08 实测）：
  **CPU+CPU 9.0s / GPU+CPU 8.6s / CPU+TRT 6.8s / GPU+TRT 8.1s /
  CPU+NVDEC+TRT 7.0s / CPU+NVDEC+CPU 8.1s**（v2.15 新增混合后端）。
  生产者各 Python/numpy 子步骤合计仅 ~4%（GPU ~1000fps / CPU ~1260fps
  ROI-only）。DLL 在 _decord_build + site-packages 两处
- **CPU+NVDEC 混合解码（实验开关，v2.15，_open_hybrid_vrs）**：不暴露
  GUI/CLI 参数；环境变量 RVTOL_HYBRID_DECODE=1（config.HYBRID_DECODE_ENV，
  默认关）开启后，GPU 模式（auto/nvdec）内部改走混合；显式传
  decode_backend='cpu+nvdec'/'hybrid'（旧程序化用法）恒为混合。CPU
  reader 覆盖前 10%（calib 后，保守分法）、GPU reader 覆盖后 90%
  （独立 seek_accurate 到段首），两 worker 线程并行填有界队列、消费者
  按序合并（帧序与单解码器一致）。**v0.7.8 起两后端灰度逐位一致**
  （GPU 也直出 Y，range 语义同 CPU swscale）→ 接缝无跨后端差异，
  auto 与混合门禁结果逐位相同（11 错）。切分比例 config.HYBRID_CPU_SPLIT
  =0.10（env RVTOL_HYBRID_SPLIT）。**AV1 特判**：CPU 软解 AV1 极耗核且
  与 GPU 段并发竞争拖慢 GPU 吞吐（混合 19.1s vs 纯 GPU 14.4s）→ 不
  打开 CPU reader、按纯 GPU 分支走（_open_hybrid_vrs 返回 (vr_gpu,
  vr_gpu)，调用方见 vr_gpu is vr 置 hybrid=False；_hybrid_split 返回
  0 兜底）。实测（venv+TRT，decode 阶段）：HEVC 2.5 vs GPU 2.6s、
  h264 3.1 vs 3.3s / 7.2 vs 7.7s、AV1 14.4 vs 14.1s —— 三种编码均
  不弱于纯 GPU。GPU 不可用自动回退纯 CPU。
- **OCR 混合（实验开关，v2.15，RVTOL_HYBRID_OCR=1，config.HYBRID_OCR_ENV，
  默认关）**：TRT（GPU）+ onnxruntime（CPU）双引擎并发处理段批。与解码
  不同，OCR 无状态约束——结果按段索引聚合、批顺序无关 → 实现只需共享
  一个 infer_q、两个推理线程各持一引擎（谁空闲谁取批），无 seek/背压
  复杂度。实测（test6，GPU 解码，venv）：ONNX 22.9s → 混合 **15.3s**
  （≈ 纯 TRT 14.7s；ONNX 16 线程与解码抢核是 ONNX 慢的主因，一半 OCR
  移上 GPU 后解码阶段 22.2→14.6s）。**结论：TRT 可用时 auto 已最优**
  （纯解码 13.8s vs auto 14.7s，OCR 隐藏于解码之下非瓶颈），混合仅对
  "TRT 可用但强制 OCR=cpu"有意义；准确率无扰动（test2 8/8、test6
  0/0 与基线一致）。
- **decord v0.7.8（ROI-first + 撕裂帧 + seek VFR 越位 + 双解码器背压 +
  GPU 色度 siting 修复 + GPU gray 输出，fork 仓库 D:\Repo\decord）**：
  - v0.7.5 起：解码器只输出 ROI 矩形（CPU filter crop 先于 format，yuv420p
    x/y/w/h 全偶数约束用偶数超集+顶部精裁绕过；GPU kernel 只算 ROI 窗口
    + 输出池 ROI 尺寸）；同步 D2H 撕裂帧修复（v0.7.4 非阻塞流并发管线
    孤立撕裂 ~0.2-0.9% → 门禁 12→24-41；同步 cudaMemcpy 后 A/B 5 轮
    0 分歧、门禁可复现 12）。
  - **v0.7.6 seek_accurate VFR 关键帧越位修复**：稀疏 GOP VFR 视频
    （test3 4264 帧仅 17 关键帧）帧中段 seek 落到下一个关键帧（+267=
    一个 GOP）静默错位（混合解码 GPU 段曾 433 误读）。三层：① Seek 关键帧
    强制 AVSEEK_FLAG_BACKWARD（正投到恰等于关键帧自身 PTS 会落到下一
    关键帧）；② 索引磁盘缓存命中路径从未重建 pts_frame_map_（仅全扫描
    IndexKeyframes 建）→ 缓存版本 1→2 + LoadCachedIndex 重建 map；
    ③ SeekAccurate 落点校验（解码一帧 PTS 对索引 ±2 tick，残差精确补跳，
    校验帧交还 NextFrame / 记账前移防吞帧）。验证：5 视频×采样位置×
    CPU/GPU 逐位一致（CPU 全 0）。
  - **v0.7.7 双解码器并发背压**：EnqueueRawFrame 加背压
    （raw_queue_+frame_queue_ ≥ max_queue_frames_+4 时等待）。raw/pkt/
    buffer 队列是无界 ConcurrentBlockingQueue，唯一有界点是 frame_queue_
    （32）；双并发解码器在 CPU 争抢下解码线程跑在消费者前，raw_queue_
    无限堆积整帧（1080p ~1.9MB/帧 → 8-11GB，del reader 才释放）。
    背压把消费者节奏传回 Push()，实测 11GB→500MB，帧序正确性不变。
  - **v0.7.8 GPU 色度 siting 修复 + GPU gray 输出**：
    ① improc.cu 色度采样 (src_x/2)+0.5 在奇数 luma 像素落色度 texel
    边界被 cudaFilterModeLinear 50/50 混合（与 CPU swscale MPEG-2
    siting 取所属 2x2 块单 texel 不一致）→ RGB 彩色边缘差 40+、|Δ|>=8
    像素 77-91% 在奇行/奇列。修复取 int(src_x/2)+0.5 → 同帧 GPU/CPU
    RGB 收敛 max≤3（纯舍余）。
    ② GPU 路径响应 output_format='gray'（上游 API 已有、此前忽略）：
    kernel 直出 Y 平面，**按流 color_range 展开**（tv→(Y-16)*255/219，
    与 CPU swscale GRAY8 逐位一致，实测 maxΔ=0）→ 分段灰度跨后端
    统一，auto 与混合门禁结果逐位相同（11 错，v2.15 起 test 4→3）。
    **range 坑**：测试视频标注 color_range=tv 但实际 Y 数据 full-range
    （0-255）——CPU swscale 遵循标注做 limited→full 展开，GPU 直出
    原始 Y 会与 CPU 差 mean 7-10，故 kernel 必须按 dec_ctx->color_range
    决定展开。
  - 剩余余量：GPU 真批量异步解码 —— GetBatch 逐帧 NextFrameImpl +
    display 回调每帧 cudaStreamSynchronize；批内单次 sync + 批量 D2H
    可将 GPU decode 阶段（8.1s）进一步压缩。
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
  - ✅ v0.7.5：CPU filter crop 先于 format；GPU kernel ROI 窗口 + ROI
    输出池；同步 D2H（撕裂帧竞态修复）。
  - ✅ v0.7.6：seek_accurate VFR 关键帧越位修复（seek 三层修复）。
  - ✅ v0.7.7：双解码器并发 raw 队列背压（内存爆炸修复）。
  - ✅ v0.7.8：GPU 色度 siting 修复（RGB 跨后端收敛 ±3）+ GPU
    output_format='gray' 直出 Y（按流 range 展开，与 CPU GRAY8
    逐位一致）。
  - 剩余余量：GPU 真批量异步解码 —— GetBatch 逐帧 NextFrameImpl +
    display 回调每帧 cudaStreamSynchronize；批内单次 sync + 批量 D2H
    可将 GPU decode 阶段（8.1s）进一步压缩。

## 工作流约束

- **git 安全**：绝不删分支（曾因 git init 毁掉整个 .git）；force push /
  reset --hard 前必须确认；merge 后不删实验分支（保留参考）
- **每次 push / release 后必须主动查 CI**（曾 6 连败而不自知）：
  `gh run list --repo chr431/RaceVideoToLog --workflow ci.yml --limit 3`
  确认最新 run 为 success；release 流程（decord fork / 本仓库）构建约
  10-15 分钟，发布后同样核对 release 产物存在。失败特征：workflow 级
  秒失败（created==updated、jobs=[]）多为 ci.yml 自身 YAML/解析问题；
  job 级失败看具体步骤日志 `gh run view <id> --log`。
- **CI YAML 教训（2026-08-14 踩坑）**：YAML plain scalar 不能含
  `: `（冒号+空格）—— `name: Install ... (fork: DLLs + python layer)`
  未加引号导致整个 workflow 解析失败、jobs=[] 秒红。步骤名含 `: ` 必须
  用引号包住。诊断 bisect 时注意：删除整个步骤会同时删掉 name 与内容，
  别把 name 问题误判成步骤内容问题。
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
