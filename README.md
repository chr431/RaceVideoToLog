# RaceVideoToLog v2.6.0

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。默认使用 cv2 视频解码 + TensorRT FP16 OCR 推理（decord 可选），支持 PySide6 Fluent Design GUI 和 CLI 两种界面。

## 安装

```bash
pip install rapidocr onnxruntime opencv-python-headless numpy matplotlib pyside6 pyside6-fluent-widgets
# 可选: GPU 视频解码
pip install decord
```

### GPU 加速（可选）

1. 安装 **CUDA Toolkit 12.x** + **TensorRT 10.x**
2. 安装 Python 绑定：
   ```bash
   pip install tensorrt-10.*-cp313-none-win_amd64.whl cuda-python
   ```
   （wheel 文件位于 TensorRT 安装目录的 `python/` 子目录）
3. 将以下目录加入系统 PATH：
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `C:\Program Files\NVIDIA\TensorRT-10.x\bin`
   - `C:\Program Files\NVIDIA\TensorRT-10.x\lib`
4. 启动程序 → 自动检测 TensorRT → 首次需构建引擎（可能需要几分钟）
5. 未检测到 GPU 环境则自动使用 CPU 推理，无需额外配置

## 使用方式

### GUI

```bash
python RaceVideoToLog.py
```

1. 导入视频文件，框选仪表盘数字区域
2. 选择 OCR 模型（v6_tiny 更快 / v6_small 更准）和纠错模式
3. 设置参数，导出 CSV

**导入设置**：可从已有 CSV 文件头一键导入所有参数（ROI、采样率、后端等）。

**人工审核**：自动纠错后展示问题段，支持手动修正（输入精确值如 `123` 或部分位数值如 `12x`）。审核窗口中图表支持滚轮缩放和右键拖拽平移。

**数据分析 Tab**：支持导入多个 CSV 进行 v-t / v-x / Δt-x 对比，拖拽选择范围自动计算积分距离/用时。

### CLI

```bash
python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 [options] -o output.csv
```

CLI 与 GUI 共用 `pipeline.py` 中的 `ProcessingPipeline`，在原生线程中运行以保证 CUDA 推理性能。

可从已有 CSV 导入设置，显式参数会覆盖 CSV 中的值：

```bash
python RaceVideoToLog.py video.mp4 --from-csv settings.csv --div 1 -o output.csv
```

### 数据分析

```bash
# 比较两个 CSV，生成 3 张对比图
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix
```

## 项目结构

```text
RaceVideoToLog/
├── RaceVideoToLog.py    # 入口: arg 解析 + GUI/CLI 调度
├── gui.py               # PySide6 主窗口 (Fluent Design)
├── gui_review.py        # 人工审核对话框
├── gui_analysis.py      # 数据分析 Tab
├── widget_utils.py      # 共享 GUI 组件
├── analysis.py          # 数据分析业务逻辑
├── pipeline.py          # 统一处理流水线 (GUI/CLI 共用)
├── correction.py        # 物理约束纠错流水线
├── ocr_engine.py        # OCR 引擎、预处理、Flag 枚举、LCS 一致性评分
├── config.py            # 集中配置（常量、颜色、默认值）
├── gpu_setup.py         # GPU DLL 加载 + TensorRT/CPU 后端选择
├── theme_manager.py     # 主题回调管理器
├── headless.py          # CLI 入口
├── RaceVideoToLog.spec  # PyInstaller 打包配置
├── DEPENDENCIES.md     # 上游依赖版本 + 已知问题
├── tools/
│   └── check_deps.py   # 检查依赖更新
└── README.md
```

## 输出格式

CSV 头包含完整处理参数和统计信息：

```csv
# RaceVideoToLog v2.6.0
# video_hash=94ac7e06b58914e7, video=test4.mp4
# roi=862,945,957,1003, format=km/h, frame_start=114, frame_end=6317
# max_speed=400.0, max_accel=70.0, div=1, target_h=48, pad=0, buffer=16
# backend=TensorRT, model=v6_tiny, reocr_model=v6_small
# stats: total=6203, trusted=6091, corrected=111
# timing: ocr=13.5s, correction=8.5s, total=22.0s
1.90,0.00,0.00,0
1.92,0.00,3.00,0
```

**Flag 定义**（定义于 `ocr_engine.Flag` 枚举）：

| Flag | 常量 | 含义 |
| ---- | ---- | ---- |
| 0 | `Flag.RAW` | 原始 OCR |
| 11 | `Flag.REOCR_AUTO` | re-OCR 自动修正 |
| 12 | `Flag.FILL_INTERP` | 物理插值填充 |
| 13 | `Flag.PARTIAL_AUTO` | 部分数字自动修正 |
| 21 | `Flag.HIGH_TRUST` | 自动高可信帧 (Viterbi+物理验证) |
| 22 | `Flag.PINNED` | 用户手动修正（绝对真值） |
| 23 | `Flag.CONFIRMED_SEG` | 人工确认段 |
| 30 | `Flag.FLAGGED_REVIEW` | 待人工审核 |

## 部分数字修正

OCR 读到的数字可能缺失部分位（如 `221` 被识别为 `21`），系统会**自动生成候选**：

- OCR 读到 1-2 位数字时，自动生成所有可能的缺位扩展（如 `"21"` → [21, 121, 221, 321]）
- 由 Viterbi 动态规划全局最优路径选择最优候选
- 人工审核时也可手动输入 `"12x"` 约束候选范围

## CLI 参数

```text
python RaceVideoToLog.py [video] [options]

位置参数:
  video                      视频文件（省略启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2          识别范围
  --format {m/s,km/h,mile/h} 速度单位 (默认: km/h)
  --div N                    采样间隔 1/N (默认: 1)
  --max-speed N              最大速度 km/h (默认: 400)
  --max-accel N              最大加速度 m/s² (默认: 50)
  --target-h N               OCR 高度 px (默认: 48)
  --pad N                    边缘填充 px (默认: 0)
  --buffer N                 缓冲队列大小 (默认: 16)
  --backend {auto,tensorrt,cpu}  OCR 后端 (默认: auto)
  --ocr-model {v6_tiny,v6_small}  OCR 模型 (默认: v6_tiny)
  --reocr-model {v6_tiny,v6_small}  重OCR 模型 (默认: v6_small)
  --from-csv PATH            从 CSV 文件头导入设置
  -o, --output PATH          输出 CSV 路径
  --frame-start N            起始帧号
  --frame-end N              结束帧号
  --analysis CSV1 CSV2       分析模式：比较两个 CSV
  --analysis-out PREFIX      分析输出前缀
```

## 性能

- TensorRT 10.x FP16 + CUDA 12.x，PP-OCRv6_small 模型
- 单帧推理 ~2.5ms（~400 fps）
- 可选 `v6_tiny` 模型：推理 ~1.2ms（~850 fps）
- decord NVDEC 硬件解码 + TensorRT 推理流水线并行

## 打包

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

生成 `dist/RaceVideoToLog/` (onedir 模式)。GPU 用户需自行安装 CUDA Toolkit 12.x + TensorRT 10.x 并加入 PATH。

也可双击 `build_exe.bat` 一键构建。

## License

MIT
