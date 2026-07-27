# 上游依赖跟踪

## 核心依赖

| 包 | 当前版本 | 最低版本 | PyPI | 备注 |
| --- | --- | --- | --- | --- |
| rapidocr | 3.9.2 | 3.9 | [rapidocr](https://pypi.org/project/rapidocr/) | PP-OCRv6 ONNX 模型。3.9.2 新增 `use_preprocess_img`（保持 True，TRT 内部 resize 更优） |
| onnxruntime | 1.28.0 | 1.27 | [onnxruntime](https://pypi.org/project/onnxruntime/) | CPU 推理回退；CUDA provider 已移除 |
| opencv-python-headless | 5.0.0 | 5.0 | [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) | 图像预处理 (resize, cvtColor) |
| decord | 0.6.0 | 0.6 | [decord](https://pypi.org/project/decord/) | NVDEC 硬件视频解码；⚠️ 捆绑 FFmpeg 4.x DLL |
| PySide6 | 6.11.1 | 6.11 | [PySide6](https://pypi.org/project/PySide6/) | Qt 6 GUI |
| PySide6-Fluent-Widgets | 1.11.2 | 1.11 | [PySide6-Fluent-Widgets](https://pypi.org/project/PySide6-Fluent-Widgets/) | Fluent Design 组件库 |
| numpy | 2.5.1 | 2.0 | [numpy](https://pypi.org/project/numpy/) | |
| matplotlib | 3.11.1 | 3.10 | [matplotlib](https://pypi.org/project/matplotlib/) | 数据分析图表 |
| pyclipper | 1.4.0 | — | [pyclipper](https://pypi.org/project/pyclipper/) | rapidocr 依赖 |
| shapely | 2.1.2 | — | [shapely](https://pypi.org/project/shapely/) | rapidocr 依赖 |

## GPU 加速

| 包 | 当前测试版本 | 来源 | 备注 |
| --- | --- | --- | --- |
| tensorrt | 10.16.1.11 | [tensorrt](https://pypi.org/project/tensorrt/) | **仅 10.x**；`--no-deps` 安装，DLL 走系统 PATH |
| tensorrt_cu13_bindings | 10.16.1.11 | [tensorrt_cu13_bindings](https://pypi.org/project/tensorrt_cu13_bindings/) | Python 绑定（~1MB） |
| cuda-python | 13.3.1 | [cuda-python](https://pypi.org/project/cuda-python/) | CUDA Python 绑定（~33MB） |
| CUDA Toolkit | 12.9 | [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) | 提供 cudart, cublas 等 DLL（系统 PATH） |

**注意**：`tensorrt` 通过 `setup_venv.bat` 以 `--no-deps` 安装，只装 Python 绑定。`tensorrt_cu13_libs`（~2.2GB DLL）被排除，DLL 从系统 PATH 加载。

## 打包工具

| 包 | 当前版本 | PyPI |
| --- | --- | --- |
| pyinstaller | 6.21.0 | [pyinstaller](https://pypi.org/project/pyinstaller/) |

## 已知问题

### rapidocr 3.9.1
- `_initialize()` 无条件创建 `TextDetector` / `TextClassifier`（即使 `use_det=False` / `use_cls=False`）。已通过 monkey-patch 规避 (`ocr_engine._patch_rapidocr_init`)
- 识别模型需要 BGR 输入（不能灰度），已适配 `_preprocess_standard` 和 `_re_ocr_frame`

### decord 0.6.0
- 捆绑 FFmpeg 4.x DLL（avcodec-58 等）。必须**在 ORT 之后导入**，否则 DLL 初始化失败
- 无 CPU 软件解码路径（`cpu(0)` 仍用 NVDEC），无 GPU 时回退 cv2

### onnxruntime 1.27
- `onnxruntime_providers_tensorrt.dll` 和 `onnxruntime_providers_cuda.dll` 已从 EXE 排除（TRT 直接调用 rapidocr，不走 ORT）

### tensorrt 10.x
- `find_lib()` 只搜 `os.environ["PATH"]`，不认 `os.add_dll_directory()`。已在 `gpu_setup` 中同时更新 PATH
- 首次构建引擎 ~80s (FP32)；引擎缓存于 `rapidocr/models/models/*.engine`
- FP16 构建 ~178s (2.2x slower)，推理速度无提升，不推荐
- `cuda.bindings` 需额外安装 `cuda-python`
- ⚠️ 不允许 pip 自动拉入 `tensorrt_cu13_libs`（~2.2GB DLL），使用 `--no-deps` 安装

## 检查更新

```bash
# 查看所有可更新包
pip list --outdated

# 仅检查核心依赖
pip list --outdated | grep -iE "rapidocr|onnxruntime|opencv|decord|pyside6|qfluentwidgets|numpy|matplotlib|tensorrt|cuda-python|pyinstaller"
```

### 测试新版本

1. 创建分支：`git checkout -b test-upgrade-<pkg>`
2. 升级单个包：`pip install --upgrade <pkg>`
3. 运行测试：`python -m pytest tests/ -v`
4. 端到端验证：`python RaceVideoToLog.py test.mp4 --roi 862 945 957 1003 ...`
5. 若通过则合并到 dev，更新本文件中的版本号
