# RaceVideoToLog v2.6

从赛车视频中提取速度数据，生成时间-速度-距离 CSV 文件。

## 安装

```bash
pip install rapidocr onnxruntime opencv-python-headless numpy matplotlib pyside6 pyside6-fluent-widgets
# 可选: GPU 视频解码
pip install decord
```

### GPU 加速（可选）

1. 安装 CUDA Toolkit 12.x + TensorRT 10.x
2. 安装 Python 绑定：`pip install tensorrt-10.*-cp313-none-win_amd64.whl cuda-python`
3. 将 CUDA 和 TensorRT 的 `bin/`、`lib/` 目录加入系统 PATH
4. 启动程序自动检测 TensorRT，首次需构建引擎（几分钟）
5. 未检测到 GPU 则自动使用 CPU 推理

## 使用

### GUI

```bash
python RaceVideoToLog.py
```

1. 导入视频，框选仪表盘速度数字区域
2. 选择 OCR 模型和纠错模式
3. 点击"导出 CSV"

导出完成后弹出**最终检查**窗口：全帧速度曲线图，橙色标记低置信度帧，点击任意数据点可查看原始 ROI 图像并手动修正该帧速度值。支持滚轮缩放和拖拽平移。

**纠错模式**：
- **自动**（推荐）：全自动处理，输出平滑曲线，适合大多数场景
- **人工辅助**：保守修正，减少自动干预

**导入设置**：可从已有 CSV 文件头一键导入所有参数。

### CLI

```bash
python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2 -o output.csv
```

CLI 与 GUI 共用同一处理管线。支持从 CSV 导入设置：

```bash
python RaceVideoToLog.py video.mp4 --from-csv settings.csv --div 1 -o output.csv
```

### 数据分析

```bash
# 比较两个 CSV，生成速度-时间、速度-距离、时间差-距离对比图
python RaceVideoToLog.py --analysis csv1.csv csv2.csv --analysis-out prefix
```

## 输出格式

```csv
# RaceVideoToLog v2.6.0
# video_hash=..., video=test4.mp4
# roi=862,945,957,1003, format=km/h, frame_start=114, frame_end=6317
# max_speed=400.0, max_accel=70.0, div=1, target_h=48, pad=0, buffer=16
# backend=TensorRT, model=v6_tiny, reocr_model=v6_small, video_backend=cv2
# stats: total=6203, trusted=6091, corrected=111
# timing: ocr=13.5s, correction=8.5s, total=22.0s
frame,distance,speed_kmh,flag
114,0.00,0,21
115,0.00,0,21
```

CSV 头包含完整处理参数。每行 4 列：帧号、累计距离 (m)、速度 (km/h)、flag（见下表）。

| Flag | 含义 |
|------|------|
| 0 | 原始 OCR 值 |
| 11 | 自动修正 |
| 12 | 插值填充 |
| 13 | 部分数字推断修正 |
| 21 | 高可信帧 |
| 22 | 用户手动修正 |
| 23 | 人工确认段 |
| 30 | 待审核 |

## CLI 参数

```
python RaceVideoToLog.py [video] [options]

位置参数:
  video                          视频文件（省略则启动 GUI）

可选参数:
  --roi X1 Y1 X2 Y2              识别范围（CLI 必需）
  --format {m/s,km/h,mile/h}     速度单位 (默认: km/h)
  --div N                        采样间隔 1/N (默认: 1)
  --max-speed N                  最大速度 km/h (默认: 400)
  --max-accel N                  最大加速度 m/s² (默认: 50)
  --target-h N                   OCR 高度 px (默认: 48)
  --pad N                        边缘填充 px (默认: 0)
  --buffer N                     缓冲队列大小 (默认: 16)
  --backend {auto,tensorrt,cpu}  OCR 后端 (默认: auto)
  --video-backend {cv2,decord}   视频解码器 (默认: cv2)
  --ocr-model {v6_tiny,v6_small} 主 OCR 模型 (默认: v6_tiny)
  --reocr-model {v6_tiny,v6_small} 重 OCR 模型 (默认: v6_small)
  --mode {auto,manual}           纠错模式 (默认: auto)
  --log-level {normal,detailed,debug} 日志级别 (默认: normal)
  --frame-start N                起始帧号
  --frame-end N                  结束帧号
  --from-csv PATH                从 CSV 文件头导入设置
  -o, --output PATH              输出 CSV 路径
  --analysis CSV1 CSV2           分析模式：比较两个 CSV
  --analysis-out PREFIX          分析输出前缀
```

## 打包

```bash
pip install pyinstaller
python -m PyInstaller RaceVideoToLog.spec --noconfirm
# 或双击 build_exe.bat
```

生成 `dist/RaceVideoToLog/`。GPU 用户需自行安装 CUDA Toolkit 12.x + TensorRT 10.x 并加入 PATH。

## License

MIT
