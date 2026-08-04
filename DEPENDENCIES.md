# 上游依赖跟踪（v2.9.0）

## 核心依赖

| 包 | 最低版本 | 来源 | 备注 |
| --- | --- | --- | --- |
| onnxruntime | 1.27 | PyPI | CPU 推理后端（OcrEngine 直连） |
| numpy | 2.0 | PyPI | 预处理/信号计算（纯 numpy，无 scipy） |
| PySide6-Essentials | 6.11 | PyPI | Qt 6 GUI（只装核心，省 Addons ~300MB） |
| PySide6-Fluent-Widgets | 1.11 | PyPI | Fluent Design 组件库 |
| pyqtgraph | 0.14 | PyPI | 分析/检查图表（替代 matplotlib） |
| cuda-python | — | PyPI | CUDA Python 绑定（TRT 执行 + decord GPU DLL 注册） |
| tensorrt_cu13_bindings | 10 | PyPI | TensorRT Python 绑定（~1MB） |
| decord | 自建 | 自建仓库 chr431/decord（feat/perf-deep） | NVDEC 硬解 + CPU 软件解码；FFmpeg 8.x DLL。**PyPI 版不支持 next_roi / CPU ROI 优化**，见 setup_venv.bat |

## GPU 加速（运行时，不打包）

| 组件 | 来源 | 备注 |
| --- | --- | --- |
| CUDA Toolkit | NVIDIA 官网 | cudart/cublas 等 DLL，需在 PATH |
| TensorRT | NVIDIA 官网 | nvinfer DLL，需在 PATH；首次运行自动构建引擎缓存到 `%LOCALAPPDATA%/RaceVideoToLog/ocr_engines/` |

`tensorrt` 元包与 `tensorrt_cu13_libs`（~2.2GB DLL）被有意排除 —— 运行时 DLL 从系统 PATH 加载。

## 打包工具

| 包 | PyPI |
| --- | --- |
| pyinstaller | [pyinstaller](https://pypi.org/project/pyinstaller/) |

## 已知问题

### decord（自建）
- 需与 FFmpeg 8.x DLL（avcodec-62 等）同目录；`setup_venv.bat` 自动从 `_decord_build/` 拷贝
- 无 NVIDIA GPU 时自动回退 CPU 软件解码（`DECORD_FORCE_CPU=1` 可强制）

### onnxruntime
- TRT/CUDA provider DLL 已从 EXE 排除（TRT 由 OcrEngine 直接调用，不走 ORT provider）

### tensorrt 10.x
- `find_lib()` 只搜 `os.environ["PATH"]`，不认 `os.add_dll_directory()` —— `gpu_setup` 已同时更新 PATH
- 首次构建引擎 FP32 ~80s，缓存于用户目录；FP16 构建 2.2x 慢且推理无提升，不推荐

## 检查更新

```bash
pip list --outdated
```

### 测试新版本

1. 创建分支：`git checkout -b test-upgrade-<pkg>`
2. 升级单个包：`pip install --upgrade <pkg>`
3. 运行测试：`python -m pytest tests/ -v`
4. 端到端验证：`python RaceVideoToLog.py test.mp4 --roi 862 945 957 1003 ...`
5. 若通过则合并到 dev，更新本文件中的版本号
