# RaceVideoToLog v2.7.0

从赛车视频中提取速度数据，生成时间-速度-距离 CSV 文件。

## 前置要求

- Python 3.11+
- NVIDIA 显卡 + 最新驱动（GPU 视频解码；CPU 模式也可用但内存较高）
- （可选）CUDA Toolkit + TensorRT 10.x（GPU OCR 推理；无则自动使用 CPU）

## 一键安装

```bash
setup_venv.bat
```

脚本自动完成：

1. 创建 `.venv` 虚拟环境
2. `pip install -e .` 安装所有 Python 依赖
3. 检测 `_decord_build\` 目录，若存在则安装自建 decord（GPU 解码 + 内存修复）
4. 安装 TensorRT / cuda-python Python 绑定

### 自建 decord（推荐）

PyPI 版 decord 是 CPU-only 且内存占用高达 ~10 GB。自建版本支持 NVDEC GPU 硬解码，内存仅 ~400 MB。

1. 按 [decord wiki](https://github.com/chr431/decord) 构建 decord
2. 将构建产物放入 `_decord_build\`：

```text
_decord_build\
├── decord.dll
├── avcodec-59.dll
├── avformat-59.dll
├── avutil-57.dll
├── avfilter-8.dll
├── avdevice-59.dll
├── swresample-4.dll
├── swscale-6.dll
├── postproc-56.dll
├── msvcp140.dll
├── vcruntime140.dll
├── vcruntime140_1.dll
├── ffprobe.exe         （可选，用于显示视频编码）
```

3. 重新运行 `setup_venv.bat`

如无 `_decord_build\`，安装脚本会使用 PyPI decord（CPU-only，功能正常但内存较高）。

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
# RaceVideoToLog v2.7.0
# video_hash=..., video=test5.mp4, fps=59.767, codec=h264
# roi=843,993,948,1025, format=km/h, frame_start=362, frame_end=7585
# max_speed=400.0, max_accel=40.0, div=1, target_h=48, pad=0, buffer=16
# backend=CPU, model=v6_tiny, reocr_model=v6_small, video_backend=decord/GPU
# stats: total=7223, trusted=7090, corrected=118
# timing: ocr=17.0s, decode=16.6s, inference=15.7s, correction=4.5s
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
| 23   | 人工确认段 |

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
  --target-h N                   OCR 高度 px (默认: 48)
  --max-width N                  最大宽度 px（0=不限；扁宽字体可设为 96）
  --pad N                        边缘填充 px (默认: 0)
  --buffer N                     缓冲队列大小 (默认: 16)
  --backend {auto,tensorrt,cpu}  OCR 后端 (默认: auto)
  --ocr-model {v6_tiny,v6_small} 主 OCR 模型 (默认: v6_tiny)
  --reocr-model {v6_tiny,v6_small} 重 OCR 模型 (默认: v6_small)
  --mode {auto,manual}           纠错模式 (默认: auto)
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

## License

GPLv3（因依赖 PySide6-Fluent-Widgets GPLv3）。详见 LICENSE 文件。
