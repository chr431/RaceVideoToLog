# RaceVideoToLog v2.13.2

从赛车视频中提取速度数据，生成时间-速度-距离 CSV 文件。

## 前置要求

- Python 3.11+
- NVIDIA 显卡 + 最新驱动（GPU 视频解码；无 GPU 自动使用 CPU 软件解码，性能差异约 15%）
- （可选）CUDA Toolkit 13.x + TensorRT 11.x（GPU OCR 推理；无则自动使用 CPU）

## 一键安装

```bash
setup_venv.bat
```

脚本自动完成：

1. 创建 `.venv` 虚拟环境
2. `pip install -e .` 安装所有 Python 依赖
3. 从 `_decord_build\` 安装自建 decord（必需，GPU 解码 + 内存修复；缺失则报错退出）
4. 安装 TensorRT / cuda-python Python 绑定

### 自建 decord（必需）

本项目**不依赖 PyPI decord**（CPU-only、无 `next_roi` / `get_codec`、CPU 解码内存溢出）。自建 fork（chr431/decord）支持 NVDEC GPU 硬解码 + CPU 软件解码，只传输识别 ROI（解码提速 ~45%，编码信息直接来自 decord），且 GPU API 运行时动态加载 —— 无 NVIDIA 设备自动回退 CPU 解码。

获取 decord 发布产物（推荐）：运行 [chr431/decord](https://github.com/chr431/decord) 的 **Release workflow**（Actions → Release → Run workflow，输入版本号如 `0.7.0`），它会构建并发布 `decord-<ver>-win64-gpu.zip`。解压到本仓库 `_decord_build\`：

```text
_decord_build\
├── decord.dll
├── avcodec-62.dll          （FFmpeg 8.x）
├── avformat-62.dll
├── avutil-60.dll
├── avfilter-11.dll
├── avdevice-62.dll
├── swresample-6.dll
├── swscale-9.dll
├── msvcp140.dll
├── vcruntime140.dll
├── vcruntime140_1.dll
└── python\decord\          （fork 的 Python 层：next_roi / get_codec）
```

然后重新运行 `setup_venv.bat`。

> 没有 `_decord_build\` 时 `setup_venv.bat` 会报错退出（**无 PyPI 回退**）—— 必须先获取自建 decord 产物。

## 使用

### GUI

```bash
.venv\Scripts\python RaceVideoToLog.py
```

### CLI

```bash
.venv\Scripts\python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 -o output.csv
```

## 输出格式

```csv
# RaceVideoToLog v2.13.2
# video_hash=..., video=test5.mp4, fps=59.767, codec=h264
# roi=843,993,948,1025, format=km/h, frame_start=362, frame_end=7585
# max_speed=400.0, max_accel=40.0, div=1, force_aspect=1.5, pad=0, buffer=16
# backend=TensorRT, model=v6_tiny, reocr_model=v6_small, video_backend=decord/GPU
# stats: total=7223, trusted=7090, corrected=118
# timing: ocr=13.5s, decode=6.4s, inference=12.7s, correction=0.9s
362,0.00,257,21
```

| Flag | 含义 |
|------|------|
| 0    | 原始 OCR 值 |
| 11   | 自动修正 |
| 12   | 插值填充 |
| 13   | 部分数字推断修正 |
| 21   | 高可信帧 |
| 22   | 用户手动修正 |

## CLI 参数

```text
python RaceVideoToLog.py [video] [options]

位置参数:
  video                          视频文件（省略则启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2              识别范围（CLI 必需）
  --format {m/s,km/h,mile/h}     速度单位 (默认: km/h)
  --div N                        采样间隔 1/N (默认: 2)
  --max-speed N                  最大速度 km/h (默认: 400)
  --max-accel N                  最大加速度 m/s² (默认: 50)
  --force-aspect N               强制宽高比 (默认: 0=自动；>0 宽度=48×此值)
  --pad N                        边缘填充 px (默认: 0)
  --buffer N                     缓冲队列大小 (默认: 16)
  --backend {auto,tensorrt,cpu}  OCR 后端 (默认: auto)
  --ocr-model {v6_tiny,v6_small} 主 OCR 模型 (默认: v6_tiny)；重 OCR 自动推导：tiny→small / small→无
  --log-level {normal,detailed,debug} 日志级别 (默认: normal)
  --frame-start N                起始帧号
  --frame-end N                  结束帧号
  --from-csv PATH                从 CSV 文件头导入设置
  -o, --output PATH              输出 CSV 路径
```

## 打包

```bash
build_exe.bat
```

生成 `dist/RaceVideoToLog/`。GPU 用户仅需 NVIDIA 驱动（NVDEC 解码）；TensorRT OCR 推理需额外安装 CUDA Toolkit + TensorRT 并加入 PATH。

## 变更记录

完整发布日志（v2.7.1 → v2.13.2）见 [release_notes.md](release_notes.md)。

## 运行时缓存（卸载时需删除）

程序会在用户目录创建以下缓存（卸载/清理时需手动删除）：

| 路径 | 内容 |
| --- | --- |
| `%LOCALAPPDATA%\RaceVideoToLog\ocr_engines\`（Windows）；`~/.cache/racevideotolog/ocr_engines/`（Linux/macOS） | TensorRT 引擎缓存：首次运行时由 ONNX 模型自动构建（约 2 分钟），之后直接复用。**与 GPU 架构绑定**（如 sm89 = RTX 40 系）—— 换显卡后旧引擎会自动失效并回退/重建。删除后下次运行会重新构建。 |
| `%LOCALAPPDATA%\RaceVideoToLog\gui_timing.log` | 调试计时日志（排查界面卡顿用），可随时删除。 |

## License

GPLv3（因依赖 PySide6-Fluent-Widgets GPLv3）。详见 LICENSE 文件。
