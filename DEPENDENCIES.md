# 上游依赖跟踪（v2.15.2）

## 核心依赖

| 包 | 最低版本 | 来源 | 备注 |
| --- | --- | --- | --- |
| onnxruntime | ≥1.28 | PyPI | CPU 推理后端（OcrEngine 直连）；1.28 含 protobuf CVE 修复；**1.29.0（2026-08-12 PyPI 发布）实测升级安全、性能持平**：3000 帧端到端 test5 h264 -2%（4.20→4.10s）、test6 AV1 +1%（波动内）；逐帧读数与 1.28 完全一致（0 差异）；新增参数（ORT_INTRA/INTER_OP_NUM_THREADS env、parallel 执行、spin off）全部无收益，保持现状不落地 |
| numpy | 2.0 | PyPI | 预处理/信号计算（纯 numpy，无 scipy） |
| PySide6-Essentials | 6.11 | PyPI | Qt 6 GUI（只装核心，省 Addons ~300MB） |
| PySide6-Fluent-Widgets | 1.11 | PyPI | Fluent Design 组件库 |
| pyqtgraph | 0.14 | PyPI | 分析/检查图表（替代 matplotlib） |
| cuda-python | — | PyPI | CUDA Python 绑定（TRT 执行 + decord GPU DLL 注册） |
| tensorrt_cu13_bindings | 11 | PyPI | TensorRT Python 绑定（~1MB） |
| psutil | 6 | PyPI | 资源监测 RSS / CPU%（可选：缺失时降级为 None，GPU 采样不受影响） |
| decord | 0.7.10 | 自建仓库 chr431/decord | NVDEC 硬解 + CPU 软件解码；FFmpeg 8.x DLL。**PyPI 版不支持 next_roi / CPU ROI 优化 / YUV420 输出**，见 setup_venv.bat |

## GPU 加速（运行时，不打包）

| 组件 | 来源 | 备注 |
| --- | --- | --- |
| CUDA Toolkit 13.x | NVIDIA 官网 | cudart/cublas 等 DLL，需在 PATH；与 tensorrt_cu13_bindings / decord（全栈统一 CUDA 13）一致 |
| TensorRT | NVIDIA 官网 | nvinfer DLL，需在 PATH；首次运行自动构建引擎缓存到 `<程序目录>/ocr_engines/`（旧 `%LOCALAPPDATA%/RaceVideoToLog/ocr_engines/` 只读回退） |

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

### tensorrt 11.x
- `find_lib()` 只搜 `os.environ["PATH"]`，不认 `os.add_dll_directory()` —— `TrtEngine` 初始化前会调用 `gpu_setup.ensure_gpu_initialized()` 更新 PATH
- 首次构建引擎 FP32 ~1min，缓存于 `<程序目录>/ocr_engines/`；FP16 构建 2.2x 慢且推理无提升，不推荐
- TRT 引擎与**构建版本不兼容**（10 产物无法被 11 加载）—— 升级后旧缓存自动重建
  （`TrtEngine` 反序列化失败即删除重建，不会静默回退 ONNX）
- **GPTuner（Global Performance Tuner）Windows 不可用**：`--tuneBuildRoutes` /
  `--setBuildRoute` 报 "not supported on Windows (no fork())"，Python
  `config.all_build_routes` 返回空 —— 调优路线只能走 Linux 或默认路线

## 检查更新

```bash
pip list --outdated
```

### 依赖维护状态（2026-08 检查，全部活跃，无停止支持项）

| 包 | 安装版 | PyPI 最新 | 上游活跃度 | 备注 |
|---|---|---|---|---|
| onnxruntime | 1.29.0 | 1.29.0 | ✅ 活跃 | 1.29.0 升级实测见 CLAUDE.md 2026-08-18 节 |
| numpy | 2.5.1 | 2.5.2 | ✅ 活跃 | 差补丁版，无风险 |
| PySide6-Essentials | 6.11.1 | 6.11.2 | ✅ Qt 官方 | 差补丁版 |
| PySide6-Fluent-Widgets | 1.11.3 | 1.11.3 | ✅ 活跃（最后 push 2026-08-01） | qfluentwidgets.com；PyPI 版本与仓库同步 |
| pyqtgraph | 0.14.0 | 0.14.0 | ✅ 活跃（最后 push 2026-08-17） | |
| psutil | 7.2.2 | 7.2.2 | ✅ 活跃（最后 push 2026-08-17） | |
| cuda-python | 13.3.1 | 13.3.1 | ✅ NVIDIA 官方 | |
| tensorrt_cu13_bindings | 11.2.1.2 | 11.2.1.2 | ✅ NVIDIA 官方 | |
| pyinstaller | 6.21.0 | 6.22.2 | ✅ 活跃 | 打包工具，升级低优先 |
| pytest | 9.1.1 | 9.1.1 | ✅ 活跃 | dev 依赖 |
| decord（fork） | 0.7.10 (python 层) | fork v0.7.11 | ⚠️ 上游 dmlc 停更（最后 push 2024-07） | **自建 fork chr431/decord 承担维护**；venv 需从 v0.7.11 release 完整对齐（见下） |

**停止支持风险点（已化解/已知）**：
- **decord 上游 dmlc/decord 已停更 1 年+**（PyPI 0.6.0 仍 2021 行为）——本项目依赖自建 fork
  chr431/decord（v0.7.11，含 AV1 帧并行修复 / ROI-first / GPU gray / YUV420），
  fork 由本仓库维护，CI decoder-smoke 从 fork release 下载。**勿回退 PyPI 版**。
- **Python 3.13.2** 运行（requires-python >=3.11 满足；3.13 安全支持至 2029-10）。
- `pip check` 唯一红项 = PySide6-Fluent-Widgets 传递依赖 PySide6-Addons 未装——
  **有意省略**（省 ~300MB，项目只用 Essentials），非停止支持问题。

**venv 残留孤儿包（无 Required-by，历史实验遗留，可卸载省 ~710MB）**：
torch(490MB) / wandb(74MB) / polars / sentry-sdk / lightning-utilities / onnx /
shapely / pyclipper / omegaconf / ml_dtypes / hf-xet 等——源码不 import，
`pip uninstall` 可安全清理（torch 连带 networkx/sympy/mpmath/Jinja2/ninja）。

**decord 对齐注意**：site-packages 的 decord.dll（2016-08-17，0.30MB）与
`_decord_build\decord.dll`（0.40MB）为不同构建但 **AV1 帧并行功能一致**
（dcd=12 均 640+fps，语义等价）；venv python 层版本串仍报 0.7.10（release 已
0.7.11），重跑 setup_venv.bat 会以 _decord_build 覆盖 site-packages（该 DLL
同样含 AV1 修复，738fps 实测），无回归风险。建议下次按 v0.7.11 release zip
整体对齐 python 层+DLL。

### 测试新版本

1. 创建分支：`git checkout -b test-upgrade-<pkg>`
2. 升级单个包：`pip install --upgrade <pkg>`
3. 运行测试：`python -m pytest tests/ -v`
4. 端到端验证：`python RaceVideoToLog.py test.mp4 --roi 862 945 957 1003 ...`
5. 若通过则合并到 dev，更新本文件中的版本号

## 版本与发布

**单一事实源：`config.py` 的 `__version__`**（运行时写 CSV 头/控制台）。所有其它引用
（pyproject.toml、CLI docstring、README 标题/CSV 示例/变更记录区间、本文件标题、
EXE 版本资源）都由 `tools/version.py` 同步，不手工改。

### 版本号规则（SemVer）

- `MAJOR`：破坏性变更（CSV 格式不兼容、GUI 交互重做、删除用户功能）
- `MINOR`：新功能（新增设置项、新算法、性能优化）—— 大部分迭代走这里
- `PATCH`：纯修复（行为不变，只修 bug）
- 当前发布只接受纯 `X.Y.Z`，不带 `-dev`/`-rc` 后缀（如需要请扩展
  `tools/version.py` 的 `SEMVER_RE` 并同步 CI）

### 发布流程（一次发布 = 一次 `bump`）

```bash
# 1. 在 dev 分支完成改动，确认测试通过
python -m pytest tests/ -v

# 2. 升版本：同步全部引用 + 在 release_notes.md 顶部插入新节
#    （已是目标版本时只同步不一致的引用，不会重复插节）
python tools/version.py bump 2.11.0 "标题"

# 3. 填 release_notes.md 新节（bump 只生成骨架 "### 待补充"）

# 4. 校验全部引用一致（CI 的 version-check job 也跑这个，退出码 1 = 不一致）
python tools/version.py

# 5. 回归：pytest + 基准（tools/bench_decoder.py 至少跑一次）
python -m pytest tests/ -v

# 6. 构建 EXE（build_exe.bat 内置版本一致性检查，失败即中止；
#    版本号写入 EXE 文件属性 → 右键属性/详细信息可见）
build_exe.bat

# 7. 提交 + 合并到 master（master 是发布分支，不跑 tests/）
git add -A && git commit -m "release: v2.11.0 ..."
git checkout master && git merge dev && git push
```

### 一键发布（GitHub Action）

合并到 master 后，Actions → **Release** → Run workflow（默认 `ref: master`），自动完成：

1. 校验版本引用一致性（`tools/version.py`，不一致即中止）
2. 读取 `config.__version__`，确认 tag `v<版本>` 不存在
3. 下载 decord fork 发布产物 `decord-<ver>-win64-gpu.zip`（`decord-version` 输入，默认 `0.7.10`）到 `_decord_build\`
4. `setup_venv.bat --ci` + `build_exe.bat --ci` 构建 EXE（跳过 pause）
5. 打包 `RaceVideoToLog.<版本>.zip`（dist 布局与现有 release 一致）
6. 打 tag `v<版本>` + push，创建 GitHub Release（notes 取自 `release_notes.md` 对应节）

发布后如需小修：继续 `bump` 到下一个 PATCH，不回改已发布的版本号。
