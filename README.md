# RaceVideoToLog v2.17.3

![RaceVideoToLog](docs/images/banner.png)

从赛车视频中提取速度数据，生成时间-速度-距离 CSV 文件。

![生产流水线](docs/images/pipeline.png)

## 环境要求

- Windows x64，Python 3.11–3.14
- NVIDIA 显卡 + 最新驱动（可选；无显卡时自动回退 CPU 软件解码）
- 可选：TensorRT 11.x（GPU OCR 推理，`bin` 目录加入 PATH；缺失时自动回退
  CPU OCR）。CUDA 运行时由最新驱动自带，**无需安装 CUDA Toolkit**

## 安装

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

依赖由 pip 自动安装，无需额外步骤：

- Python 包：PySide6、onnxruntime、cuda-python、TensorRT 绑定等
- 识别引擎 `video-ocr-engine`：独立仓库
  [chr431/video_ocr_engine](https://github.com/chr431/video_ocr_engine)，
  按 git tag 锁定
- 解码库 decord：自建 fork
  [chr431/decord](https://github.com/chr431/decord) 的 release wheel，
  按解释器版本以直接 URL 锁定。**请勿替换为 PyPI 官方版**——官方版不含
  `next_roi` / `get_codec`，ROI 解码优化会静默失效

版本均在 `pyproject.toml` 中锁定。本地开发引擎时改用源码直连（改代码立刻生效）：

```bash
pip install -e ..\video_ocr_engine --no-deps
```

打包 EXE（可选）：

```bash
.venv\Scripts\python -m pip install -e ".[dev,build]"
.venv\Scripts\python -m PyInstaller RaceVideoToLog.spec --noconfirm
```

产物位于 `dist/RaceVideoToLog/`。

## 使用

### GUI

```bash
.venv\Scripts\python RaceVideoToLog.py
```

### CLI

```bash
.venv\Scripts\python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 -o output.csv
```

CLI 参数：

```text
python RaceVideoToLog.py [video] [options]

位置参数:
  video                          视频文件（省略则启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2              识别范围（CLI 必需）
  --format {m/s,km/h,mile/h}     速度单位 (默认: km/h)
  --max-speed N                  最大速度 km/h (默认: 400)
  --max-accel N                  最大加速度 m/s² (默认: 50)
  --force-aspect N               强制宽高比 (默认: 0=不启用；>0 宽度=48×此值)
  --fill-width N                 预处理 pad 宽度下限 px (默认: 224)
  --buffer N                     解码∥OCR 流水线队列缓冲，段数 (默认: 128)
  --decode-backend {auto,cpu,nvdec,hybrid}  解码后端 (默认: auto 自动选 GPU；
                                 hybrid=CPU+NVDEC 混合解码，按关键帧分片路由)
  --ocr-backend {auto,cpu,tensorrt}  OCR 推理后端 (默认: auto 自动选 GPU)
  --log-level {normal,detailed,debug} 日志级别 (默认: normal)
  --frame-start N                起始帧号
  --frame-end N                  结束帧号
  --no-monitor                   禁用资源监控（内存/CPU/GPU 采样）
  --monitor-interval SEC         资源采样间隔秒 (默认: 1.0)
  --from-csv PATH                从 CSV 文件头导入设置（显式参数优先）
  -o, --output PATH              输出 CSV 路径
```

### 输出格式

```csv
# RaceVideoToLog v2.17.3
# video=test5.mp4, fps=59.767
# roi=843,993,948,1025, format=km/h, frame_start=362, frame_end=7585
# max_speed=400.0, max_accel=50.0, force_aspect=0.0, fill_width=224
# backend=decord/GPU, model=v6_small
# segments=2533, corrected=118
# timing: decode=6.4s, ocr=13.5s, correction=0.9s, total=22.0s
362,0.00,257,21
```

| Flag | 含义 |
|------|------|
| 0    | 原始 OCR 值 |
| 11   | 自动修正（DP 稠密纠正） |
| 12   | 插值填充（OCR 未读出） |
| 21   | 高可信帧（conf 通过锚定阈值） |
| 22   | 用户手动修正 |

程序在程序目录写入运行数据：`logs\`（日志）、`ocr_engines\`（TensorRT 引擎
缓存，首次运行自动构建，与 GPU 架构绑定）。

---

License: GPLv3（因依赖 PySide6-Fluent-Widgets GPLv3）—— 详见 LICENSE。
