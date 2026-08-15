# Release Notes

## v2.15.1（2026-08-15）— 代码清理与 decord 0.7.10

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 🔧 依赖与打包

- 自建解码 fork 升级到 **decord v0.7.10**：新增真正的
  **YUV420 输出**（packed NV12 布局，Y/U/V 原始 4:2:0）——
  生产管线解码 YUV，分段/OCR 只取 Y 平面（与旧 gray 输出逐位一致，
  回归门禁 11 错误不变），代表帧保留 YUV
- PyInstaller 产物进一步瘦身（实测 ~390MB → ~348MB）：移除未使用的
  Pillow（~12.8MB）、重复的 OpenSSL x64 DLL 对（~5.6MB）、decord 的
  avdevice/ffprobe（~4MB）、cuda-python 未用绑定（nvml/nvrtc/cudla 等
  ~6MB）、非中英文 Qt 翻译（~6MB）及 numpy.random/fft、yaml 等；
  同时修正了 qframelesswindow 需要的 pywin32 依赖被误删的问题

### 🧹 代码与行为

- 最终检查恢复**彩色预览**：管线只在解码阶段保留代表帧的 YUV420
  （~1.5 字节/像素），打开最终检查前把所有代表帧一次转成 RGB
  （test5 2.5k 段 ~0.2s、test6 8.1k 段 ~0.9s），随后显示/缩放不再
  重复转换；同时修复了初始预览被放大裁边的问题
- CSV 头只写入 `ocr_backend` = **本次实际推理引擎**（onnxruntime /
  tensorrt，与老版本对齐，不再写请求值 auto）；从 CSV 导入设置会把它
  归一化回可请求参数（onnxruntime → CPU、tensorrt → TensorRT），
  CLI `--from-csv` 与 GUI「导入 CSV 设置」均生效
- 进度条/CLI 进度现在跟随真实管线：解码 3→58、OCR 58→86、纠错 88→100，
  TensorRT 首次构建引擎时会明确提示“正在构建”
- 日志级别（CLI `--log-level` / GUI 下拉）现在真正生效：normal=INFO、
  detailed=项目 DEBUG、debug=全部 DEBUG
- 清理大量无调用者代码（邻帧一致性评分、候选生成、旧 GPU 后端选择、
  StageTimer 等）并修正过时注释
- 段级算法参数（批大小、48 高、120 帧窗口上限、置信度门槛、TRT profile
  等）收敛到 `config.py`，行为不变（准确率门禁 final=11 与基线一致）

## v2.15.0（2026-08-14）— CPU+NVDEC 混合解码

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### ⚡ 实验性「CPU+NVDEC」混合解码（环境变量开关，默认关闭）

- 不新增 GUI/CLI 选项：设置环境变量 **RVTOL_HYBRID_DECODE=1** 后，
  「自动」/「NVDEC」解码内部改走 **CPU 软解 + NVDEC 硬解同时工作**
  （CPU 解前段、GPU 解后段再按序拼接），充分利用 CPU + GPU 两个资源
- **保守分法（CPU 只分 10%）**：h264 上 CPU 软解吞吐（~1260fps）与
  NVDEC（~960fps）相当，对半分才有砍半优势；但 HEVC/AV1 的 CPU 软解
  只有 NVDEC 的 1/3~1/5，大份额 CPU 段反成瓶颈。10% 分法下
  wall = max(CPU 10% 耗时, GPU 90% 耗时)，实测三种编码 **均不弱于
  纯 GPU**（decode 阶段，venv+TRT）：HEVC 2.5 vs 2.6s、h264 3.1 vs
  3.3s（test3）/ 7.2 vs 7.7s（test5）
- **AV1 自动按纯 GPU 解码**：CPU 软解 AV1 极耗核且与 GPU 段并发竞争
  反而拖慢 GPU 吞吐（实测混合 19.1s vs 纯 GPU 14.4s）→ AV1 视频不再
  打开 CPU 解码器，行为与「自动」完全一致（decode 14.4 vs 14.1s）
- 收益不确定且增加复杂度（双解码器并发、seek 对齐、背压），故默认
  关闭不暴露参数；切分比例可用 RVTOL_HYBRID_SPLIT 微调（实验）
- 依赖自建 fork **decord v0.7.7**：修复两处关键 bug ——
  ① 稀疏关键帧 VFR 视频的帧中段 seek 落到下一个关键帧（+267 帧静默错位，
  曾致混合解码 test3 433 误读）；② 双解码器并发时解码帧队列无界增长
  （内存爆炸到 ~11GB，混合解码改用背压限到 ~500MB）。两者都是混合解码
  能正确跑的前提
- 无 NVIDIA 显卡时该选项自动回退纯 CPU（行为与「CPU」一致）

### ⚡ 实验性 OCR 混合（环境变量开关，默认关闭）

- 设置环境变量 **RVTOL_HYBRID_OCR=1** 后，OCR 用 **TensorRT（GPU）+
  onnxruntime（CPU）双引擎并发**处理段批：与解码不同，OCR 无状态约束
  （结果按段索引聚合、批顺序无关），实现天然简单——共享一个批队列、
  两个推理线程各持一引擎，谁空闲谁取批
- test6 实测（GPU 解码，venv）：纯 ONNX OCR **22.9s → 混合 15.3s**
  （≈ 纯 TensorRT 的 14.7s）。ONNX 慢的主因是 16 线程与解码阶段抢核，
  一半 OCR 移上 GPU 后解码阶段 22.2→14.6s
- **结论：TensorRT 可用时「自动」已是最优**（纯解码 13.8s vs 自动
  14.7s——OCR 完全隐藏于解码阶段之下，非瓶颈），混合只在"TensorRT
  可用但强制 OCR=CPU"时有意义，故默认关闭不暴露参数
- 准确率无扰动：test2（8 错）与 test6（0 错）混合前后与基线一致

### 🎯 准确率提升（12 → 11 错误）+ 解码灰度统一（decord v0.7.8）

- **GPU 现在也直出灰度**（`output_format='gray'` 在 NVDEC 路径生效）：
  与 CPU 软解输出**逐位一致**（同一 Y 平面、按视频色彩范围展开）——
  分段不再有"GPU 转 RGB 再转回灰度"的两步舍入，CPU / NVDEC / 混合
  三种后端的分段结果完全相同
- **GPU RGB 色度修复**：NVDEC 的色度上采样在奇数像素错位（与 CPU
  swscale 不一致，彩色边缘差 40+）→ 修复后两后端 RGB 差异收敛到 ±3
- 端到端错误 **12 → 11**（test 视频 4→3，test2 保持 8）：灰度更准确
  后一个边缘误读自然消失；同时消除了后端间行为差异这个隐患
- 依赖 **decord v0.7.8**（色度 siting + GPU gray，API 与上游 0.6.0 兼容）


## v2.14.0（2026-08-13）— CPU 解码性能 +22% + 准确率提升 + 启动提速

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### ⚡ CPU 解码 + CPU 推理性能 +22%（无 GPU 机器体验大幅改善）

- 纯 CPU 场景（test5 实测）：**12.3s → 9.6s**（-22%）。解码改为**批量流水线**
  （`get_batch` + ROI 裁剪 + FFmpeg 8 帧线程并行），OCR 预处理与推理重叠，
  特征计算按批向量化
- 依赖 **decord v0.7.3**（自建 fork 新发布：预取批量解码、全方法 ROI、
  `__dlpack__` 零拷贝）。`setup_venv.bat` 自动从新 release 下载

### 🎯 准确率提升（13 → 12 错误，零误改）

- 孤立尖峰豁免：偏离曲线的中等 jerk 尖峰误读不再被锚定保留（test#74
  raw=107→truth=103 类修复），5 视频漏斗 13 → **12** 错误，误改 0

### 🚀 GUI 启动提速

- 数据分析图表（pyqtgraph）延迟到首次切换到该 Tab 时才加载：frozen 版
  窗口出现 **1.1s → 0.7s**（冷启动差距更大）

### ⚠️ 参数变更（旧 CSV 导入自动兼容）

- `--max-width` → `--force-aspect`（浮点强制宽高比，0=不启用）
- `--pad`（边缘填充）→ `--fill-width`（OCR 输入 pad 宽度下限，默认 224）
- 旧版 `--from-csv` 导入仍可用（头字段解析兼容新旧命名）

### 🔧 内部

- 打包修复：py-modules/spec 模块列表同步（补 segment_flow —— 全新安装
  曾缺生产流水线模块）
- `_preprocess_standard` 修复：force_aspect 对恰好 48 高 crop 静默失效
- 死代码清理、带宽计算去重、correction 计时补全
- 单测扩充至 32 用例（DP 纠错链/build_rows/CSV 往返/OCR 文本/预处理）
- 误读分析工具链入 tools/（jerk 带通探针、准确率漏斗、解码 profile 等）
- 新增 CLAUDE.md 项目知识（回归门禁、死路清单、性能基线）

## v2.13.2（2026-08-11）— 数据目录移入程序文件夹 + Release 附件改 7z

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 📂 完全便携（免安装不留残留）

- **不再向 `%LOCALAPPDATA%` 写任何东西**：TensorRT 引擎缓存、运行日志全部改为
  写入**程序所在文件夹**（`ocr_engines/`、`logs/`）——卸载/移动时删除整个程序
  文件夹即清理干净
- 旧版用户升级：本机已有的 LOCALAPPDATA 引擎缓存仍会被自动复用（省去首次重建），
  但新版本不会再往那里写入任何内容

### 📦 安装包更小（.zip → .7z）

- 发布附件从 `.zip` 改为 `.7z`（LZMA2 最大压缩）：**v2.13.1 的 158.7MB → 约 120MB
  （-26%）**
- Windows 11 23H2+ 原生支持解压 .7z（资源管理器直接解压）；Win10 及以下用户需
  安装 7-Zip（免费，装机率很高）

### 🔧 内部

- config 新增 `app_data_dir()`/`app_logs_dir()` 统一数据/日志根目录
- TRT 引擎缓存路径：`%LOCALAPPDATA%\RaceVideoToLog\ocr_engines` → 程序文件夹
  `ocr_engines/`；旧路径保留为只读 fallback
- release workflow 改用 windows-latest 预装的 7-Zip 打包（`-mx=9 -m0=LZMA2`），
  布局校验同步（7z l -slt）


## v2.13.1（2026-08-10）— EXE 瘦身（排除 tiny onnx + pywin32）

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 📦 体积减小

- **安装包从 v2.13.0 的 166.6MB 减到 ~160MB**（-7MB）
- 打包排除不再使用的 tiny OCR 模型（v2.13 起固定 v6_small）+ PyInstaller 构建
  依赖意外带入的 pywin32 运行时组件（win32 / pythoncom / mfc140u，项目零引用）
- 功能不变：OCR 精度、GUI、全部流程与 v2.13.0 一致（仅体积优化）

### 🔧 内部

- spec 打包过滤 tiny onnx（源码 assets 保留，实验工具不受影响）
- spec 排除 win32com/pywin32 纯模块 + 按文件名过滤 win32 binaries
  （excludes 拦不住 hook 收集的 binaries，需二次过滤）

## v2.13.0（2026-08-10）— gray+gamma OCR 正式化 + 段级纠错完善 + flag 重设计

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 🎨 OCR 预处理正式灰度+gamma

- **数字识别更稳**：OCR 前把 ROI 转灰度并做 gamma 2.0 对比度增强（放大白字黄底等
  背景色块的高段分离），5 视频原始 OCR 误读 183→152（**-17%**），test5 32→7
- **之前版本不生效**：此前的 gamma 是 RGB 逐通道实验钩子，视觉差异小、误读反而
  增多；现改为灰度版并正式化

### 🎯 端到端准确度追平并更稳

- 重调纠错提交阈值（DP 修正 |Δ|>3 才提交）：消掉正确段被微调改错的问题
  （误改 2→0），端到端错误 13（test 5 / test2 8 / test3-5-6 0）
- **剩余 11 处漏纠为 1-2 km/h 平滑区整体偏移**（整片读偏 2），物理约束（加速度/
  一致性）无法区分于真实平滑，属系统极限；基线同样存在

### 🏷️ 输出 flag 重设计（对齐段级流水线）

- CSV 第 4 列 flag 现在真实反映来源：`21` 高可信段（绝大多数，物理验证通过）、
  `11` DP 自动纠正、`12` OCR 未读出插值、`0` 未验证原始值、`22` 用户在最终
  检查里手动修正。旧 flag（重 OCR/Viterbi 时代）已删除；GUI 图表着色语义不变
  （1X 红 = 纠正段，2X 绿 = 高可信段）。

### 🔧 其他

- 移除损坏的 div 帧采样参数，恢复 buffer 队列缓冲
- 移除 OCR 模型选择（固定 v6_small），新增解码/推理后端参数（auto/cpu/tensorrt）

## v2.12.2（2026-08-07）— O(V) Viterbi 凸变换 + Phase-1 向量化（纯 Python 提速）

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### ⚡ 纠错与信号计算提速（纯 Python，精度逐位不变）

- **稠密格点 Viterbi O(V²)→O(V)**：转移代价用凸核 min-plus 卷积（下包络）替换
  每帧 O(V²) 向量化矩阵 —— test5 dense Viterbi 0.58s→0.32s（**1.8x**）
- **Phase-1 信号批量计算**：detector 从逐帧 advance 改为每 64 帧批量
  （中位数插值/带宽跨帧向量化）—— test5 phase1 0.92s→0.20s（**4.6x**）
- **精度实测逐位一致**：test5 7195 / test6 23435 / test.mp4 3505 / test4 6089
  与 v2.12.1 完全相同（O(V) 变换与批量信号均经逐位验证）
- test5 全程 ~13.3s→~12.4s（-7%）；墙钟仍由 OCR 推理（TensorRT，C++）主导

## v2.12.1（2026-08-07）— 移除手动纠错模式，只保留自动模式

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 🗑️ 单一纠错模式（简化）

- **删除模式选择**：GUI「纠错模式」卡片（自动/人工辅助）、命令行 `--mode` 均移除。自动模式成为唯一纠错模式
- **理由**：维护两种模式成本高，且自动模式准确性已很高（test6 误差 0.02%、test.mp4 1.9%、test4 1.84%），人工辅助模式的收益不足以覆盖维护成本
- **行为不变**：删除仅影响「可选手动模式」这条路径；自动模式的纠错行为与 v2.12.0 完全相同（四视频精度实测一致）
- 导出后的「最终检查」对话框保留（仍可逐帧手动钉死修正，与纠错模式无关）

## v2.12.0（2026-08-07）— 稠密格点 DP 替换离散 Viterbi + ref 修复

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 🎯 纠错算法换核：稠密格点连续 DP（更准）

- **候选瓶颈根治**：旧 Viterbi 只能在每帧的候选集（OCR 读数 + 变体）里选值，实测 97% 的修正帧真值不在候选集里——它只能挑最不坏的错误候选。新算法把状态空间换成 0..400 每 1 km/h 的稠密格点，**能合成任意整数**，不再受候选枚举限制
- **精度实测全面反超**（matched 帧 / 真值判定 |输出-真值|<0.5）：

| 视频 | 旧版 | 新版 | Δ |
| --- | --- | --- | --- |
| test6（主） | 23424 | **23435** | +11 |
| test.mp4（真实模糊） | 3486 | **3505** | +19 |
| test4（阶梯坡） | 6079 | **6089** | +10 |
| test5 | 7187 | **7195** | +8 |

- **ref 修复**：插值参考值只在偏离 raw ≥6 km/h 时才参与（此前 ≥3）——修复阶梯显示上滞后插值把正确 raw 拖偏 ±3-5 的问题（旧版 test4 回归的根源）
- **两个关键守卫**（换核必须有，否则更差）：单候选帧硬锚定（正确帧不抖动）、min-obs（raw 与参考任一接近即低代价，保护正确 raw）
- **插值锚点置信度门槛**（本轮新增）：`_local_interp` 锚点必须 Phase-1 高置信或已可信。物理自洽的误读簇（如 test5 1600 读 "7" 且 ±2 邻居一致）会污染本地插值、distant 又跨过真实速度曲线 → 修到错值；置信度门槛根治（abs 信号有更宽上下文，误读簇置信度低）。同时 **dense 钉死守卫**：单候选钉死值与参考严重矛盾（>30 km/h）时解除钉死，避免误读簇被钉死拖垮整段（test5 5265/5267 raw=11 拖垮 118 坡）
- 旧离散 Viterbi（viterbi.py）保留为历史参考；新增 viterbi_dense.py

## v2.11.0（2026-08-06）— TRT 11 采纳 + 陈旧引擎自动重建

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### ⚡ TensorRT 11 采纳（更快）

- **默认构建路线实测更快**：tiny 模型推理快 17-53%（宽度 160: 0.398 vs 0.847ms；320: 0.598 vs 0.720ms；640: 0.971 vs 1.179ms，trtexec 中位数），small 模型快 2-10%；引擎构建也更快（tiny 55s vs 73s）
- **依赖升级**：`tensorrt_cu13_bindings` 10.x → 11.x。需 TensorRT 11.x + CUDA 13.x 全栈（详见 README「GPU 加速配置」）；TRT 10 的旧引擎缓存会自动重建
- 修复 2 处 TRT 11 API 兼容问题，代码向后兼容 TRT 10
- **GPTuner（Global Performance Tuner）结论**：经源码 + 实验证实 Windows 上不可用 —— Windows DLL 的 API 面存在（`getAllBuildRoutes()` 可调用）但 knob 数据库为空（实测文档真实 knob 全部被丢弃），trtexec 调优循环被 `#if defined(_WIN32)` 编译期禁用（依赖 fork()）。官方文档未提平台限制，但所有官方 Windows 分发（安装器/ZIP/pip）均无此功能。TRT 11 默认路线的收益即是 Windows 能拿到的全部

### 🔧 陈旧引擎自动重建（不再静默回退 ONNX）

- TRT 升级后序列化版本不匹配、或换显卡后 GPU 架构不匹配的缓存引擎：自动删除并重建，不再静默回退 ONNX 后端
- 升级到本版本后首次运行会自动重建缓存引擎（约 2 分钟），之后照常复用

## v2.10.0（2026-08-06）— 永久性能日志 + 资源监测

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### 📊 性能日志与资源监测（默认开启）

- **阶段计时永久内置**：engine_load / video_open / decode / inference / phase1 / prewarm / correction / finalize 各步用时不再需要临时插桩 —— 每次导出自动写入 CSV 头 `# timing:`、`_summary.json` 与控制台汇总
- **资源监测默认开启**：内存（RSS 峰值）、CPU%、GPU 利用率 / 显存 / 温度后台采样（1s 间隔），峰值与阶段边界快照自动落入 `_summary.json`，运行摘要追加到 `%LOCALAPPDATA%\RaceVideoToLog\monitor.log`
- **可关闭**：命令行 `--no-monitor`、环境变量 `RVTOL_MONITOR=0`、GUI「资源监控」复选框 —— 关闭时零线程、零子进程
- 纠错详细级报告新增 `correction_stages`：12 个纠错子阶段（重OCR预热/候选生成/粗筛/后过滤/参考值/锚定/Viterbi/填充/平滑/对齐/强制中值/置信融合）各步用时
- `tools/bench_decoder.py` 基准 JSON 新增以上各阶段与资源峰值字段（旧键不变）

### ⚡ OCR 提速 + 重OCR 简化

- **每模型独立输入 pad 宽度**：速度数字（48 高后 78-160 宽）不再强制 pad 到 320，按内容实际宽度推理 —— 窄图推理最多 2~4 倍加速。tiny 与 small 各自最优下限（tiny=192、small=224，实测平衡精度）
- **重OCR 自动推导**：移除「重OCR」设置项，系统按主模型自动选择 —— 主模型 tiny 时用 small 做跨模型二次校验（实测错误率降至 1/4），主模型 small 时不再重OCR（同引擎重OCR 实测净效果为零，纯浪费 GPU）
- 精度实测：test6 small 错误率 0.83%→0.09%、tiny 0.47%→0.38%；test5 0.04%（保持）

## v2.9.0（2026-08-04）— v2.7.1 → v2.9.0 发布（2.7.2 / 2.8.0 未发布，合并记录）

> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。

### ⚡ 性能

- **GPU 加速（TensorRT）**：单次处理时间几乎减半 —— test4 17.1s → **9.6s**，test5 18.9s → **14.3s**
- **无 NVIDIA GPU（CPU 解码 + CPU 推理）**：test5 从 23.4s（v2.7.0）→ **14.7s**，早期因 ROI 裁切缺陷的 246s 慢速问题彻底修复（约 16 倍）
- **内存**：修复 CPU 解码时内存无限上升的问题（此前长视频会一路涨到 20GB+ 拖垮系统，现在全程稳定在 1GB 以内）
- 首次导入视频后处理不再出现长时间假死（ffprobe 探测已后台化）

### 🎨 界面

- 数据分析页图表全面升级（pyqtgraph）：缩放平移流畅不卡顿、悬停显示数值、支持框选统计区域、修饰键缩放；暗色/亮色主题下显示正确
- 最终检查（Review）窗口：更清晰的修正标记（蓝色选中、点尺寸优化）、hover 信息跟随视图、右键取消选择
- TensorRT 引擎首次构建时显示进度（不再"卡住"）
- 界面启动更快（依赖精简：移除 matplotlib / cv2 / rapidocr）

### 🎯 识别精度

- 手动纠错模式精度大幅提升（test2 d=0 92.9% → 99.1%），自动/手动模式统一纠错链
- 修复多种误纠场景：错误帧被错误标为高可信、孤岛区域被斜坡拉偏、部分数字误判
- 全量验证（test5 7223 帧）：GPU 路径 99.94%、CPU 路径 99.93% 准确率

### 🖥 命令行

- `--from-csv` 导入设置：显式指定的参数不再被 CSV 头静默覆盖（曾导致指定 tiny 模型实际跑 small 模型、速度慢 4 倍）
- `--max-width` 参数修复（此前未生效于重 OCR）
- `--roi` / `--frame-start` / `--frame-end` 等按 CSV 头导入更可靠

### 📦 安装与运行

- `setup_venv.bat` 一键重建环境（自动安装自建 decord 及 GPU 绑定）
- 依赖显著精简：移除 rapidocr / opencv / matplotlib / scipy / pyclipper / shapely（安装体积与启动时间明显下降）
- OCR 引擎（tiny / small）与 TensorRT 引擎自动缓存，重复使用不重复构建
- 主 OCR 模型建议 tiny + 重 OCR small 组合（速度/精度最佳平衡）

### 其它

- 版本号统一为 2.9.0；旧版遗留（rapidocr 时代代码、死工具、过时文档）已清理
- 更新了 `DEPENDENCIES.md`（依赖清单与已知问题）

---

## v2.9.0 (2026-08-04) — 全流程性能深度优化（本仓库 + 自建 decord）

### 性能（GPU TRT 口径，test4 6203 帧 / test5 7223 帧）

| 阶段 | test4 total | test5 total |
| --- | --- | --- |
| v2.8.0 基线 | 17.1s | 18.9s |
| 最终 | **9.6s**（-44%） | **14.3s**（-24%） |

精度全程无回退（test4 err 2.14% / test5 err 0.06%，与基线逐位一致）。

**decord 侧（自建仓库，feat/perf-deep 已推送）**：
- `next_roi()` 新 API：GPU 上只拷 ROI 矩形到主机（cudaMemcpy2D），替代
  全帧 6MB D2H + Python 裁切 —— decode 13.5→5.9s（test4）/ 17.0→6.4s（test5）
- CacheFrame 零拷贝：容错缓存改为持有池缓冲引用（refcount），消除每帧
  一次整帧同步 D2D 深拷贝；输出池 20→22
- 解码背压忙等（1ns sleep 空转核）→ condition_variable 阻塞等待
- 冒烟验证：next_roi 与 next()+crop 逐字节一致；200 帧哈希与旧 DLL 相同

**本仓库侧**：
- `_resize_norm` 等尺寸短路（省每帧一次 astype 拷贝 + zeros 双写）
- `_np_resize` 坐标映射 lru_cache（映射只依赖尺寸，主路径每帧省 ~60% 计算）
- viterbi DP 内层 C×C 循环向量化（元素级运算无归约 → 决策逐位一致，
  小候选集保留标量快路径）
- `_signal_linearity` 配对插值向量化（邻居扫描保留，中位数语义不变）
- 删除 `_auto_align_pass` 循环内 O(n) pinned_set 重建（死代码参数一并移除）
- `_ctc_decode` 批向量化（argmax/max/keep 一次归约）
- TRT `set_input_shape` 按 shape 缓存（实测每批省 ~0.5ms）
- CSV 批量写（writerows）、`parse_csv` 单趟解析

**测量基座**：`tools/bench_decoder.py` 重构为统一基准（参数化视频、
timing + 精度一并输出、JSON 记录）；新增 `tools/decord_smoke.py`
（decord 每次重建后的内容正确性冒烟）、`tools/bench_trt_fp16.py`
（FP16 vs FP32 引擎对比 —— 实测 0.97x，无收益，确认维持 FP32 默认）。

**剩余瓶颈**（记录在案）：test5 长视频墙钟 14.3s 中 GPU 利用率仅
~45%，大头为 producer/consumer 线程的 CPU 侧同步与 GIL 争抢（架构性，
无低成本解法）；decord GPU 路径 decode 硬底 ~1000fps（NVDEC 转换
流水线每帧同步受 NVDEC surface 生命周期约束，无法移除）。

### CPU 组合（decord/CPU 解码 + ONNX CPU 推理，无 NVIDIA GPU 用户）

**核心修复（性能 -95%，test5 33.1s → 13.0s）**：
- `next_frame_roi` 对 CPU reader 返回了全帧（decord 的 next_roi 对非
  CUDA 上下文回退全帧），封装未裁剪 → CPU 路径把 1080p 全帧缩略图喂给
  OCR → ~95% 帧识别失败 → 置信度崩溃 → 每帧都进 correction → test4
  全量 246s → 修复后 63s（精度 97.98%）
- **from-csv 覆盖循环误判显式参数**：`值==argparse 默认值` 被判定为
  "用户未指定" → 显式 `--ocr-model v6_tiny` 被 CSV 头的 `model=v6_small`
  静默覆盖，引擎实际是 small（CPU 3.1ms/帧 vs tiny 0.7ms/帧）→ 解释
  了"进程级 ONNX 推理慢 4 倍"谜团（非性能问题，是引擎被换）。修复后
  33.1s → 13.0s（比 v2.7.0 的 23.4s 快 43%），精度 99.93%（0.07% 误差，
  0 false_trusted）

**内存峰值优化（correction 阶段 7.3GB → ~1GB）**：批量 re-OCR 预热
一次喂 ~1000 帧 → `__call__` 内产生 (B, seq, 6906) 级中间数组
（整批 argmax int64 ~2.2GB/千帧；Windows 堆不归还 → RSS 保持高位）：
- ONNX 分片 64 → 16（实测更快 + ORT arena 峰值 920 → 300MB）
- `_ctc_decode_batch` 分块归约（每块 64 帧，峰值 ~150MB，数值一致）
- correction 批量预热分批 ≤64 帧/次调用

**配套修复**：
- ONNX 推理显式 `intra_op=cpu//2`（默认占满全部核会饿死解码器；
  与 rapidocr 时代配置一致）
- re-OCR 失败帧也写 cache 空集（避免每帧 ~8.7ms 重试推理）；
  `_multi_height_ocr` 重命名为 `_reocr_crop`（多高度早已弃用）
- 内存泄漏修复：`next_frame_roi` 视图引用全帧（3000 帧 → 18GB）
  改为 `.copy()`

**decord 侧（CPU 解码）**：
- 强制 BT.601 色彩转换（setparams）：CUDA 路径固定 BT.601 矩阵，
  FFmpeg 按流的 bt709 标志转换 → 同帧 RGB 系统性偏差（G 通道
  +7.5）→ CPU 识别失败。对齐后 CPU/GPU 像素一致（差 ≤2）
- SkipFramesImpl 改纯计数跳过（PTS 丢帧在 best_effort 时间戳不匹配
  时失效 → 帧漂移）
- SeekAccurate 恢复 seek(0) 回退（直接 keyframe seek 在 CPU 解码器
  下落点偏 2 帧）
- **NextFrameRoi CPU 分支 ROI-only 输出**（新）：原 CPU 路径返回全帧、
  asnumpy 每帧拷贝 6.2MB 再 Python 裁剪；改为 C++ 内 row-stride memcpy
  只输出 ROI 矩形（106×33 = 10KB）—— 消除每帧 ~0.6ms 全帧拷贝
  （decode 计时的 ~37%）。GPU 路径 cudaMemcpy2D 不变。
- **FFmpeg 解码线程默认 2 → 4**（16 核实测矩阵：2 线程 decode 18.1s /
  总 23.6s；4 线程 decode 11.6s / 总 16.9s；6 线程无增益且推理更慢）

**无 GPU 用户全量验证**（DECORD_FORCE_CPU=1，test5 7223 帧全范围）：
**23.6s → 14.7s（-38%）**，精度 99.93% 不变。构成：decode 9.7s
（745fps，原 18.1s）+ inference 7.4s（并行）+ correction 4.3s；
peak RSS 843MB（decode 段 ~400MB 稳定）。与 GPU 硬解路径（12.5s）
差距缩小到 ~15%（NVDEC 硬底 ~1000fps vs CPU 745fps）。

**bench_decoder.py 修复**：
- 子进程 stdout/stderr 由 PIPE 改为文件重定向 —— 原 PIPE 不读管道，
  CLI 每帧 progress flush 超 64KB 缓冲后子进程阻塞挂死（bench 超时
  10 分钟的根因）
- 显式传 --ocr-model/--reocr-model（默认 v6_tiny/v6_small）——
  原命令不传，from-csv 用 truth CSV 头的 model=v6_small 覆盖默认值，
  主 OCR 静默变 small（实测 infer 26.5s vs tiny 6.4s，同款坑第二次）
- RSS 采样覆盖 launcher 后代进程（Windows venv python.exe 是 launcher，
  原采样恒 5MB，现实测 843MB）

## v2.8.0 (2026-08-03) — 相对 v2.7.1 的完整变更（v2.7.2 未发布，合并记录）

### 算法精度提升（ground truth 验证，CPU 同口径）

手动模式 d=0 全面达到/超过自动模式（test2 手动 +6.0pp）；test2 自动 >5 误差 20→0。
关键修复：信任传播验证空洞、插值锚点物理验证、自洽帧锚定、fill 候选优先。

| 组合 | v2.7.1 d=0 | v2.8.0 d=0 | v2.7.1 >5 | v2.8.0 >5 |
| --- | --- | --- | --- | --- |
| test 自动 | 98.1% | 97.6% | 2 | 6 |
| test 手动 | 96.2% | **97.6%** | 32 | **4** |
| test2 自动 | 96.2% | **98.7%** | 20 | **3** |
| test2 手动 | 92.9% | **99.1%** | 19 | **5** |

### 性能

- 批处理 OCR 推理（-40% 推理时间；total 22.8s → 17.5s）
- 新 decord 构建（GPU NDArrayPool）：管线 decode 15.9s → 10.9s

### GUI：迁移 pyqtgraph 并全面收尾

- Matplotlib 图表全部迁移至 pyqtgraph（数据分析 Tab + 最终检查窗口）：
  高频缩放/拖动流畅，支持数千散点；Ctrl/Shift+滚轮分轴缩放、右键拖拽选范围
- 右上角统计/悬停文字：TextItem anchor 裁剪修复（此前被顶出视图不可见），
  并钉在视图角落跟随缩放/平移，不再拖后腿跳回
- 右键点击取消选区：pyqtgraph 将纯点击路由到 mouseClickEvent 而非
  mouseDragEvent → 补发 sig_drag_click；点击选区外取消并恢复提示
- `pg.mkBrush(color, alpha=…)` 静默丢弃 alpha → `make_brush()` 显式构造
  透明 QColor，修复选区/散点完全不透明
- 图表上/右框线（ViewBox.setBorder，随主题变色）；空轴零尺寸渲染被裁剪
  的坑已规避
- 审核窗口：sigRangeChangedManually 发射掩码而非 ViewBox 的崩溃修复；
  已确定点修改实时预览；选中已确定点显示蓝白描边；红/蓝特殊点尺寸 6；
  移除低置信度背景高亮（橙色散点保留标记）；区域边界改深蓝
- 图表文字/框线随主题切换变色；rebuild 断开旧回调，消除累积泄漏
- PreviewWidget 提取、主题回调泄漏修复、ReviewDialog 深副本（修复预览值泄漏）
- 清理 matplotlib 时代死代码（setup_chart_zoom_pan、HoverOverlay）

### 结构重构

- ocr_engine.py 拆分 6 模块（constants/csv_io/ocr_text/signals/video_utils）
- correction.py：ModeProfile 收敛模式差异；锚点验证/自洽锚定提升为固有机制
- 死代码清理、工具修复、文档/CI 同步

### 修复

- TRT 批处理 OCR 超出引擎优化 profile（batch 上限 6）→ 按 profile 查询
  并自动分片提交，消除 setInputShape 错误刷屏
- 模块拆分丢失的 import math / CONFUSION_MAP 归位；33/33 单测通过
- 修复 82 处 Pylance 类型错误（pyright 全项目 0 errors）；spec/pyproject
  模块清单补全，EXE 构建验证通过（headless 分析 + 完整 OCR 流程）

### 其他

- ground_truth 升级 v2.7 标准格式（实测 fps + codec + 整数行 + max_width）
- test5_ref 置信度更新
- 测试视频统一存放至 D:\Videos\racelog_test

---

## v2.7.0 (2026-08-01)

### 自建 decord：GPU 解码 + 内存修复

PyPI decord 仅提供 CPU 版本，内存占用高达 ~10 GB。v2.7.0 换用自建 decord（chr431/decord）。

**decord 源码修复**
- CPU：DECORD_FFMPEG_THREAD_COUNT env var（默认 2，原 16+ 线程），内存 ~10GB -> ~600MB
- CPU：DECORD_CPU_FRAME_QUEUE_SIZE env var（默认 32），帧队列上限防无界增长
- GPU：FindCUDA.cmake 自动搜索 Video Codec SDK 13.x
- GPU：nvcuvid 头文件更新至 SDK 13.1.15

**内存对比 (test5, div=1, 7223 帧)**
| | PyPI CPU | 自建 CPU | 自建 GPU |
|--|---------|---------|---------|
| 峰值 RSS | ~9,500 MB | ~600 MB | **~400 MB** |

**解码器选择**
- GPU 优先（NVDEC）-> CPU 自动回退
- DECORD_FORCE_CPU=1 强制 CPU 模式

### 移除 cv2

- cv2.resize -> Pillow Image.LANCZOS
- cv2.copyMakeBorder -> np.pad(mode='edge')
- cv2.VideoCapture（GUI 预览）-> decord
- cv2.cvtColor -> 直接使用（decord 返回 RGB）
- 性能变化 <7%，EXE 体积减少 ~35 MB

### GUI

- 编码字段显示真实编码（h264/hevc），通过 ffprobe.exe 检测
- 边缘填充移动到第 3 行（原视频后端位置），移除视频后端下拉框
- 进度标签：[decord/GPU + CPU] / [decord/GPU + TensorRT]

### CSV 格式 v2

- fps + codec 在 line 2；max_width 紧邻 target_h 始终写入
- video_backend 值：decord/GPU 或 decord/CPU

### 构建

- setup_venv.bat：自动检测 _decord_build/，自建 decord 优先；GPU 绑定默认安装
- build_exe.bat：移除 cv2 检查，4 步流程
- PyInstaller spec：runtime_hook.py 处理 frozen DLL 路径；排除 FFmpeg 4.x + opencv 冗余 DLL

### 配置

- 移除 DEFAULT_VIDEO_BACKEND
- opencv-python-headless 移出直接依赖
- [project.optional-dependencies] 新增 dev（pytest）和 gpu

### 性能 (7945HX + 4060M, test5, div=1)

| 模型 | CPU OCR | TRT OCR | 推荐 |
|------|--------|--------|------|
| tiny+small | 23.4s | 27.0s | CPU |
| small+small | 46.5s | 32.6s | TRT |

---

## v2.6.1 (2026-07-31)

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
