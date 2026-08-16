# RaceVideoToLog — 项目知识（从 Claude Code 项目记忆提炼）

从赛车视频 OCR 提取速度，输出时间-速度-距离 CSV。Python 3.11+，PySide6 GUI +
CLI 双入口。段级流水线（segment_flow.py）是唯一生产管线。

## 回归门禁（改动后必跑）

- **准确率漏斗是真正的测试门禁**：`tools/accuracy_breakdown.py` 跑
  test/test2/test3/test5/test6 + ground_truth_csv，与 `tools/baseline.json`
  对比，当前基线 **0 错误**（全部视频 0，TOL±1；v2.16 第二遍尖峰检测
  11→5，test2 truth 晋升人工复核 log 后 5→0）。任一视频或总量的最终
  错误数增加 → 退出码 1（失败即红）。有意改进后：确认无意外回归 →
  `--update-baseline` 更新基线，并同步重新生成回归夹具
  （`tools/make_regression_fixtures.py`）与更新本节基线描述。
- **CI 回归夹具（无视频/无 decord 可跑）**：`tests/fixtures/` ——
  - `seg_series/*.json`：生产 run() 的全量段级序列（test/test2/test3/5/6），
    `tests/test_seg_series.py` 重构 _confidence+_dense_correct+
    _spike_second_pass 并逐段断言与基线一致 + 最终错误集一致（0 案例口径）。
  - `ocr_frames/`：错误案例代表帧的 ROI 裁剪（.npy，YUV420 生产模式存
    Y 平面 H,W,1 即 OCR 实际输入）+ manifest（当前 0 案例），
    `tests/test_ocr_fixtures.py` 用 onnxruntime CPU 锁定 OCR 行为基线。
  夹具版本必须与 config.__version__ 一致（测试强制）。任何算法/预处理/
  模型改动使夹具读数变化 → 测试失败，属有意改动时先跑完整漏斗确认无
  回归再重新生成夹具。
- **测试视频在 `D:\Videos\racelog_test`**（test~test6.mp4；truth 在仓库
  ground_truth_csv/，test5/test6 用 *_ref.csv）。
- **test4 truth 不可用**（巨大误差，用户确认）——不作准确率依据，仅用于
  "不把物理正确的帧改坏"。
- **test2 truth 晋升（v2.16）**：ground_truth_csv/test2_truth.csv 已替换为
  人工复核的 test2_log.csv（用户逐帧复核，含 22 个手动 PINNED 帧）。
  旧 truth 在 1841/2466/2871/3095 四帧系统性偏 2（证实 4 个"最终错误"
  是假错误，生产输出正确）；参数随晋升更新：roi=881,940,958,982（用户
  复核时微调）、max_accel=50、frame 670-4176。对比脚本
  tools/archive/_truth_vs_log.py（14 差异帧明细）。
- pytest：`tests/test_segment_flow.py`（_detect/_correct）、
  `test_correction_chain.py`（conf/DP/build_rows/预处理）、`test_csv_io.py`、
  `test_from_csv.py`（from-csv 显式参数优先语义）、`test_packaging.py`
  （py-modules 完整性）、`test_seg_series.py`、`test_ocr_fixtures.py`、
  `test_hybrid_decoder.py`（cpu+nvdec 识别/切分/解码 worker）、
  `test_decoder_integration.py`（缺 decord 显式跳过）。
- CI（.github/workflows/ci.yml）：test job 跑全量 pytest；decoder-smoke job
  从 chr431/decord release v0.7.10 下载 fork 真实跑解码集成测试（下载失败
  显式跳过不红）；version-check 跑 tools/version.py。
- 2-off 漏纠的剩余类型是信息论极限（truth 瞬时跳变/值在邻域重复/贴合
  一侧的平滑偏移，与真实曲线不可区分）——孤立 2-off 尖峰（值不重复）
  已由第二遍尖峰检测修复（v2.16），勿重复探索其余类型。

## 已验证的死路（不要重新投入）

- **加速度分节重建纠错（用户提案，2026-08 实验）**：分节（相邻节首尾重合
  一段、节内相邻对加速度 max-min ≤ diff）→ 可信节（帧数 ≥ min_length 且
  |mean_accel| ≤ max_accel）锚定 → 按自身 mean_accel 延长线相交重建轮廓 →
  未锚定段 |raw−rec| > gap 才填充。离线扫参 2400 组（diff 0.5-200 / minL
  5-80 / gap 2-12 / max_accel_factor 0.1-1.0 / min_pairs 1-2）最优 **33 vs
  基线 11**（test 9 / test2 21 / test6 3）；节锚定+生产 DP 混合最优 23；
  作为生产 dense_correct 后第二阶段 12（净负，残留 11 修 0 破 1）。根因：
  ① 段间 dt≈0.017s（57fps）使 ±1 km/h OCR 抖动 = ±57 km/h/s 加速度，节被
  噪声碎成 2-3 段（diff≤30 时 60%+ 是 2 段节，mean_accel 退化为单对均值）；
  ② 2 段节含单个极端加速度对时（|pair| < max_accel 上限即过信）误读被锚定
  不可改（test#330 raw=160 节对 -91.6 可信）；min_pairs=2 能拒这类但刹车区
  失去全部锚点 → test2 爆炸；③ gap 松则小误读（2-8-off）漏纠（test#74
  rec=102 很准但 |107−102|≤8 保留），紧则正确段被轮廓偏差误改（刹车/转折
  区 rec 偏 2-6：test6 #4964 rec=79 vs truth=81、test2 #1017-1018 rec=112
  vs 119-120）；④ "等价于可变窗口中值滤波"不成立——重建是外推不是中值。
  结论：锚点来源不是当前系统瓶颈（conf+局部锚点插值+DP 已更强），勿再投入。
  实验脚本 tools/archive/proto_accel_section.py（--sweep/--detail/--hybrid/
  --ctx）+ _proto_stage2.py，数据用 tests/fixtures/seg_series（corr 复现
  基线 11），不重跑解码+OCR。
- medium 模型（用户否决）；fp16/INT8（tiny/small 非算力受限，无效）
- Otsu/二值化/对比度拉伸/锐化喂 OCR（PP-OCRv6 训练于自然 RGB）
- 多预处理自动选择（固定 gray+gamma2.0 是全量最优，1.15% 误读）
- 窗口重 OCR 自动化（"至少一窗口读对"是幸存者偏差；仅人工辅助有价值）
- 结构相似 obs（fill 锚点插值已是更强先验）；scipy（纯 numpy win3 替代）

## 性能实验轮次：2026-08-15（dev，v2.15 基线，7945HX + RTX 4060 Laptop）

### 测量基础设施（本轮新增，已通过 118 pytest + 漏斗门禁）
- `segment_flow.py` 新增 `RVTOL_PROFILE=1` 细粒度剖面（默认关闭、零开销）：
  producer 的 open_and_fps / calib / decode_batch / gray / sharp / bin /
  segmentation / q_put_block / consumer_total；OCR 线程的 engine_init /
  q_get_wait / preprocess / infer / ctc_decode。
- **decode_s 口径修正**：原实现把 `ocr_thread.join()` 收尾时间计入 decode_s
  （导致多数记录 decode≈ocr 的假象）；现 decode_s 只计生产者消费流结束，
  OCR 收尾单列 `ocr_tail`（实测仅 ~0.1s）。
- `tools/bench_decoder.py` 显式传 `--buffer`（修复 from-csv 旧 truth 头
  buffer=16 静默覆盖默认 128 的测量口径问题），并新增 `--no-monitor`。
- 新增工具：`bench_decoder_raw.py`（纯解码吞吐 + 逐帧 sha256 校验）、
  `bench_engine_load.py`、`bench_seg_proto.py`。
- 实验钩子：`RVTOL_OCR_BATCH`（OCR 批大小）、`RVTOL_TRT_BATCH_PROFILE`
  （TRT 引擎 batch profile，独立 `_pbN` 缓存文件，实验后已删除）。

### E0 归因（auto=GPU + TRT，buffer=128，warm run）
| 视频 | total | decode_batch(纯CAPI) | consumer 占比 | OCR infer | OCR 等待喂料 |
|---|---|---|---|---|---|
| test3 h264 3190f | 3.6s | 2.89s | 92.0% | 1.74s | 2.34s (73%) |
| test5 h264 7223f | 8.1s | 7.18s | 95.4% | 3.65s | 6.57s (88%) |
| test6 AV1 23441f | 15.8s | 11.83s | 78.6% | 13.93s | 5.15s |

结论：test3/test5 是纯 NVDEC 解码硬瓶颈（Python/numpy/分段 <5%，
任何 Python 层优化无效）；test6 是解码+TRT OCR 双瓶颈，且生产者会因
OCR 慢在队列上阻塞 1.67s。

### 异步批量解码复测（fork experiment/async-batch-decode，A/B 受控）
本轮重新构建该分支 Release DLL（CUDA 13.3 / FFmpeg 8，与 dev 同配置），
隔离 DLL 目录 + DECORD_LIBRARY_PATH 做 sync/async A/B，输出全部逐位一致：
- test3 h264 灰 ROI：sync 989.6 vs async 978.9 fps（-1.1%）
- test5 h264 灰 ROI：sync 950.5 vs async 941.3 fps（-1.0%）
- test6 AV1 灰 ROI：1674.1 vs 1674.0 fps（0.0%）
- test5 全帧 RGB（含 D2H）：577.4 vs 556.6 fps（-3.6%）
- test5 全帧 RGB（仅解码）：907.1 vs 890.9 fps（-1.8%）
- test5 端到端：8.0s vs 8.1s
**结论：异步批解码无任何收益（多数场景还慢 1-2%），原"0% 收益"结论
复测确认且偏保守；此方向正式封板，勿再投入。**

### 解码后端矩阵（本机诊断；产品策略不变）
| 视频 | auto=GPU | CPU | hybrid10 |
|---|---|---|---|
| test3 h264 | 3.6s | 3.4s | 3.7s |
| test5 h264 | 8.1s | 6.7s | 8.0s |
| test HEVC | 3.3s | 5.9s | 4.0s |
| test6 AV1 | 15.1s | 71s（历史） | 15.1s（AV1 特判退回纯 GPU） |

h264 上 CPU 更快只是本机 7945HX（16 物理核）特例且 RSS +~300MB；
HEVC/AV1 CPU 明显更慢，hybrid10 在 HEVC 还拖慢 0.7s。**维持 auto=GPU、
混合默认关闭**（典型用户 CPU 解码明显慢于 NVDEC，用户确认）。

### OCR 批参数 / TRT engine profile 扫描
- OCR 批 B：test5 无感（解码瓶颈）；test6 B=16 附近最优，**B=8 退化到
  16.9s**（15.1→16.9）。保持 B=16。
- buffer：64/128/256 无显著差异（test6 15.3/15.1-15.8/15.1）。保持 128。
- TRT engine batch profile（pb8/12/16 独立引擎，受控交替测量）：
  default 15.3/15.2 < pb8 15.5 < pb12 16.0 < pb16 16.5 —— 单调变差。
  pb16 虽把 OCR infer 13.9s→9.4s（-32%），但更大 TRT kernel 与 NVDEC
  争抢 GPU，把 decode_batch 11.8s→14.1s，净亏。**保持 batch=6 默认
  引擎 profile**；"OCR 与解码共享一块 GPU"是零和约束，勿单边优化。

### E5/E6 与原型
- 引擎加载：TRT 冷 0.44s / 热 0.24s；ONNX 冷 0.30s / 热 0.06s ——
  已被解码并行期吸收，无需 GUI 引擎复用。
- 监控采样（nvidia-smi 1s 间隔）：开 8.1s vs 关 8.1s，零可测开销，保持开启。
- 批量向量化 `_cluster_win3` 原型：数值逐位一致但 0.29s vs 0.33s 并不
  更快（管线中 segmentation 0.89s 主要是 GIL 竞争而非算法本身），不采纳。

### 门禁
- pytest 118 passed；准确率漏斗 final=11（test 3 / test2 8 / 其余 0），
  与基线完全一致，无回归。

## 性能实验轮次：2026-08-16（dev，v2.15.2，后端矩阵 × 核心数模拟）

### 测量方法
- tools/archive/_bench_matrix.py：bench_decoder.py（生产 headless 管线）
  驱动；**少核模拟 = psutil cpu_affinity 限制前 N 逻辑核 + RVTOL_OCR_THREADS
  =N**（子进程继承亲和性）。注意模拟伪影：cpu_physical_cores() 检测系统
  物理核（16）不受 affinity 影响，全自动模式在模拟下不会触发分核——验证
  分核逻辑须显式设线程 env 或跑真实少核机器。

### 后端矩阵（test5 h264 / test6 AV1，本机 16 核）
| 组合 | test5 | test6 |
|---|---|---|
| GPU+TRT（auto，生产默认） | 7.8s | 18.0s |
| GPU+ONNX | 8.5s | 27.4s |
| CPU+TRT | 6.9s | 75.5s（AV1 CPU 灾难） |
| CPU+ONNX | 9.6s | 87.4s |
**auto（GPU+TRT）在任何核心数（4/8/16 模拟）下都是最优或接近最优**——
GPU 干活，与核心数无关 → auto 决策无需按核心数调整。

### 少核分核（落地：CPU_CORES_SPLIT_THRESHOLD=8）
- CPU 软解 + 物理核 ≤8：OCR 线程与 decord FFmpeg 线程（num_threads 参数，
  不污染全局 env）各分 cores//2。实测（test5，affinity 模拟）：
  - 4 核 CPU+ONNX：ocrT=2/dcd=2 → 28.0s vs 现状（ocrT=4/dcd=2）33.1s（-15%）
  - 8 核 CPU+ONNX：ocrT=4/dcd=4 → 17.8s vs 20.7s（-14%）
  - 16 核：分核反而差（12.0 vs 9.5s）→ 保持现状（OCR=全核、FFmpeg 默认
    2 帧线程落 SMT 份额）；GPU(NVDEC) 解码不抢 CPU → 保持现状
  - GPU+ONNX 少核：ocrT=全核最优（4 核 ocrT=4 → 19.8s）→ 不分
  - 4 核 CPU+TRT：dcd=4 → 13.2s（TRT 不吃 CPU，解码可分更多核），但收益
    在波动内且 TRT 少核机器罕见 → 统一 cores//2 分核
- 线程数变化可能改变 ONNX 浮点归约顺序 → OCR 读数 ±1 变化（本机 16 核
  规则不触发 → 漏斗 0 无回归已确认；少核用户读数可能与本机不同，属既有
  RVTOL_OCR_THREADS 同类差异）

### 串行 vs 并行（CPU+ONNX，勿再投入）
- 串行（解码全帧 dcd=cores → 分段 → OCR 全核 ocrT=cores，脚本
  tools/archive/_bench_serial.py）与并行分核**持平**：test5 4 核 28.2 vs
  28.0s、8 核 17.2 vs 17.8s、16 核 10.3 vs 9.6s（±7% 波动内）。
- 原因：**CPU 工作量守恒**——解码+OCR 的 CPU 总时间固定，并行分核（每
  阶段半核同时跑）与串行全核（每阶段快一倍依次跑）只是同一份工作的
  空分/时分，负载相近时数学等价。串行还有全帧驻留内存（test6 ~100MB+）、
  引擎加载无法隐藏、进度条退化等代价 → 不落地，分核并行已是最优近似。

### AV1 解码多分核（落地：codec 感知分配）
- CPU 软解 AV1 吞吐极低（~270fps vs h264 ~1247fps），解码是绝对瓶颈，
  平分（cores//2）被解码拖死。规则：**AV1 + CPU 解码且 cores>4 →
  dcd = max(2, min(cores*3//4, cores-2))、ocrT = max(2, cores-dcd)**；
  4 核不分（ocrT=1 是灾难：ONNX 单线程追不上段率反而更慢）。
  实现：_open_vr CPU 分支打开后 get_codec() 探测，AV1 时按规则重开
  reader（重开 ~0.3s vs 总时长 ~80s 可忽略）；_ocr_num_threads 内联
  AV1 分支（先于通用分核，8 核 AV1 走 AV1 规则 ocrT=2 而非 4）。
  实测（test6 CPU+ONNX）：16 核 78.8s vs 现状 87.4s（-10%）、8 核
  81.7s vs 101.2s（-19%）、4 核持平；端到端自动分配 83.2s（±5% 波动）
  且准确率 0 错误（ocrT=4 读数与 TRT 基准一致）。GPU 解码/非 AV1
  不受影响（漏斗 0 无回归确认）。

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
- **decord v0.7.10（在 v0.7.9 的 ROI-first + 撕裂帧 + seek VFR 越位 +
  双解码器背压 + GPU 色度 siting 修复 + GPU gray 输出之上新增 YUV420
  输出，fork 仓库 D:\Repo\decord）**：
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
  - **GPU 真批量异步解码 = 已验证死路（2026-08-15 实测）**：display 回调
    去 sync + 延迟解映射 + 32 surface + 批级 SyncStream 的完整实现放在
    fork 分支 `experiment/async-batch-decode`（980 帧 A/B 与同步版逐位
    一致），但实测 **生产路径 0% 收益**：test6 灰 ROI 1734 vs 1741fps、
    端到端 decode 14.1s 不变 —— 全部测试视频均为 **NVDEC 硬件解码上限**
    （h264 ~960fps=2Gp/s、AV1 ~1734fps），display sync 与转换 kernel
    早已隐藏于硬件解码之下，软件层无法超过硬件解码器。唯一收益是全帧
    RGB 路径 +7%（739→791fps，其中 D2H 带宽为主）。勿重复攻关。
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
- **第二遍尖峰检测（v2.16，11→5→0）**：SEG_SPIKE_K=2 / THRESH=2.0 /
  MIN_FIX=2.0 / MIN_NBR=2 / **MIN_FPS=40** —— dense_correct 之后对
  "生产未改 + len=1"的段做孤立尖峰判别：修正后序列 ±2 段两侧中值**一致
  偏离**（同号）且至少一侧 ≥2，且 raw 值在邻域内**不重复**（孤立值判别
  ——真实曲线段的值通常与邻居重复，这是与"真实 1 帧 dip"区分的核心，
  排除全部 4 个同形态正确段）。修正目标 = 离 raw 更远的一侧中值。实测
  修对 6 个 2-off 单帧误读（test#73/#324/#1253、test2#228/#339/#403），
  harm=0。**帧率自适应（MIN_FPS=40）**：30fps 模拟（隔帧采样 + fps 减半，
  tools/archive/_sim30fps.py）实测相邻段真实速度变化 1-2 km/h（赛车急加速
  30-60 km/h/s），正确段孤立凸起与 2-off 误读不可区分 → 第二遍误改 9/
  修对 2 净负（thresh=3 仍误改）→ fps<40 跳过；第一遍（conf+DP）帧率
  无关（30fps 85 误读→2，与 57fps 同级，遗留 2 个 2-off 孤立 dip 为
  30fps 信息论极限）。**窗口 fps 缩放已验证无收益勿落地**：anchor_max/
  win_max/island/calib 按 fps 缩放后 30fps 结果不变（2→2）、57fps 逐段
  0 差异（scale 0.93-1.05 取整后多数不变）——固定帧数窗口在 30fps 下
  未造成可测损失，遗留错误与第二遍误改都是值域/信息论层面问题；
  seg_correction 已参数化（win_max_frames/island_min_frames*/默认=config）
  留作实验钩子。
  **"2-off 平滑偏移是信息论极限"表述修正**：孤立 2-off 尖峰（值不重复）
  可修；重复值/贴合一侧的 2-off 才不可区分。test2 truth 晋升人工复核
  log 后（v2.16）剩余错误归零（原"不可修"的 #483/#750/#923/#957 证实
  为旧 truth 系统性偏 2 的假错误，#387/#388 在新 ROI/复核值下也已修对）。

## 架构要点

- 生产路径：`run()` → `_run_pipelined()`（解码∥分段∥段值 OCR 流水线，
  有界队列背压）→ `_confidence` → `_dense_correct`（稠密格点 DP）→
  `_spike_second_pass`（第二遍尖峰检测，v2.16）→ `_build_rows`
- 职责拆分（v2.15.1）：segmentation.py（灰度/Otsu/聚类）、hybrid_decode.py
  （混合解码 worker/队列）、seg_correction.py（检测/置信度/DP 纯函数）、
  ocr_trt.py（TrtEngine）、analysis_plot.py / review_chart.py / gui_video.py /
  export_controller.py（GUI 拆分）；segment_flow.py 保留编排与兼容 re-export
- **串行方法（_decode_all/_segment/_ocr_segments/_detect/_correct）是实验参考
  路径**，只被 tools/ 与测试使用，生产不走——改动需同时考虑两侧
- 分段：raw 灰度（`_gray_seg`，SEG_GAMMA=0；RVTOL_SEG_GAMMA 可实验 gamma）+
  Otsu 阈值 + 逐帧异或 + `_cluster_win3`（C=5，纯 numpy）
- flag 语义：0 RAW / 11 DP_CORRECTED / 12 FILL_INTERP / 21 HIGH_TRUST /
  22 PINNED（20-29 高可信绿点，10-19 自动纠错红点，GUI 按区间着色）
- TRT 引擎缓存：`<程序目录>/ocr_engines/`（免安装便携），旧 LOCALAPPDATA
  只读回退；engine 文件名含 sm89（本机 RTX 4060），换卡靠"加载失败→删除→
  重建"兜底；TRT 10/11 双兼容（getattr 回退；实现已迁至 ocr_trt.TrtEngine）
- **decord fork 解码层状态（fork 仓库 D:\Repo\decord，本地可重建）**：
  - ✅ v0.7.5：CPU filter crop 先于 format；GPU kernel ROI 窗口 + ROI
    输出池；同步 D2H（撕裂帧竞态修复）。
  - ✅ v0.7.6：seek_accurate VFR 关键帧越位修复（seek 三层修复）。
  - ✅ v0.7.7：双解码器并发 raw 队列背压（内存爆炸修复）。
  - ✅ v0.7.8：GPU 色度 siting 修复（RGB 跨后端收敛 ±3）+ GPU
    output_format='gray' 直出 Y（按流 range 展开，与 CPU GRAY8
    逐位一致）。
  - ✅ v0.7.9：ROI-first/撕裂帧/seek/背压/GPU gray 生产版本（2026-08-15
    发布，CPU/GPU 解码结果与 v0.7.8 逐位一致，解码集成测试哈希已重新采集）。
  - ✅ v0.7.10：新增真正的 YUV420 输出（`output_format='yuv420'`，
    packed NV12：原始 Y + interleaved U/V；`get_color_range()` 供调用方
    按 gray 同语义展开 Y）。RaceVideoToLog 生产管线解码 YUV、分段/OCR
    只取 Y（与 gray 输出逐位一致，门禁 11 错不变），代表帧保留 YUV，
    最终检查前 `prepare_review_rgb()` 一次转成 RGB 显示。
  - **GPU 真批量异步解码 = 已验证死路（2026-08-15 实测）**：display 回调
    去 sync + 延迟解映射 + 32 surface + 批级 SyncStream 的完整实现放在
    fork 分支 `experiment/async-batch-decode`（980 帧 A/B 与同步版逐位
    一致），但实测 **生产路径 0% 收益**：test6 灰 ROI 1734 vs 1741fps、
    端到端 decode 14.1s 不变 —— 全部测试视频均为 **NVDEC 硬件解码上限**
    （h264 ~960fps=2Gp/s、AV1 ~1734fps），display sync 与转换 kernel
    早已隐藏于硬件解码之下，软件层无法超过硬件解码器。唯一收益是全帧
    RGB 路径 +7%（739→791fps，其中 D2H 带宽为主）。勿重复攻关。

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
