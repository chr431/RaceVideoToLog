# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all

# ═══════════════════ 构建时屏蔽 CUDA / TensorRT 路径 ═══════════════════
# PyInstaller 会在"动态库搜索"阶段把 CUDA 系统 DLL 全部打包，
# 但用户机器已安装 CUDA Toolkit / TensorRT，无需重复打包。
# 临时从 PATH 中移除 CUDA/TensorRT 相关目录，避免误抓。
_SAVED_PATH = os.environ.get("PATH", "")
_PATH_BLOCKLIST = {"cuda", "cudnn", "tensorrt"}
os.environ["PATH"] = ";".join([
    p for p in _SAVED_PATH.split(";")
    if not any(b in p.lower() for b in _PATH_BLOCKLIST)
])

# ═══════════════════ 基础依赖 ═══════════════════
datas = []
binaries = []
hiddenimports = [
    'queue',
    'threading', 'concurrent.futures',
    # numpy 2.x PyInstaller 兼容性修复
    'numpy._core._multiarray_umath', 'numpy._core.multiarray',
    'numpy._core.umath', 'numpy._core._methods',
    # decord
    'decord',
    # Project modules (force inclusion; auto-discovered but explicit is safer)
    'gui', 'headless', 'segment_flow', 'config', 'constants', 'gpu_setup', 'ocr_engine',
    'analysis', 'analysis_plot', 'gui_analysis', 'gui_review',
    'gui_export', 'gui_settings', 'gui_preview', 'gui_video',
    'export_controller', 'review_chart',
    'widget_utils', 'theme_manager', 'csv_io', 'ocr_text', 'signals',
    'video_utils', 'tensorrt', 'ocr_native', 'ocr_trt', 'monitor',
    'segmentation', 'seg_correction', 'hybrid_decode', 'logging_setup',
]

# onnxruntime（CPU provider；TensorRT 由 tensorrt_bindings 直接调用）
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# qfluentwidgets (Fluent Design 组件库)
tmp_ret = collect_all('qfluentwidgets')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# cuda-python: 项目只用 cuda.bindings.runtime（TRT 执行 + monitor 显存查询）。
# 显式列出 runtime 的 Cython 传递依赖（PyInstaller 无法从 .pyx 模块分析
# import），不再 collect_submodules('cuda') 全量收集（nvml/nvrtc/cudla 等
# ~6MB 无用）。
hiddenimports += [
    'cuda.bindings',
    'cuda.bindings.runtime',
    'cuda.bindings.driver',
    'cuda.bindings._bindings',
    'cuda.bindings._bindings.cydriver',
    'cuda.bindings._bindings.cyruntime',
    'cuda.bindings._bindings.cyruntime_ptds',
    'cuda.bindings._internal',
    'cuda.bindings._internal._fast_enum',
    'cuda.bindings._version',
    'cuda.bindings.cydriver',
    'cuda.bindings.cyruntime',
    'cuda.bindings.utils',
    'cuda.bindings.utils._nvvm_utils',
    'cuda.bindings.utils._ptx_utils',
    'cuda.bindings.utils._version_check',
]
# cuda.pathfinder 是纯 Python 子包（driver 运行时经它找系统 DLL），
# cuda 是 namespace package，PyInstaller 不会自动展开 → 显式收集子模块。
try:
    from PyInstaller.utils.hooks import collect_submodules
    hiddenimports += collect_submodules('cuda.pathfinder')
except Exception:
    pass  # cuda-python not installed

try:
    import tensorrt  # noqa: F401
    hiddenimports += ['tensorrt']
except Exception:
    pass  # tensorrt not installed

# decord（NVDEC 硬件加速视频解码）
# GPU API（CUDA 驱动 / NVCUVID / NVML）已改为运行时动态加载（nv_gpu_dyn），
# decord.dll 导入表无 NVIDIA 依赖 → 无驱动设备自动回退 CPU 解码。
tmp_ret = collect_all('decord')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# PySide6 (Qt 6 GUI) — 只收集核心模块 + qfluentwidgets 依赖
# QtOpenGL 为 pyqtgraph 0.14 必需（OpenGLHelpers），显式收集
for _qt_mod in ['PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui',
                 'PySide6.QtXml', 'PySide6.QtSvg', 'PySide6.QtOpenGL']:
    _qt_ret = collect_all(_qt_mod)
    datas += _qt_ret[0]; binaries += _qt_ret[1]; hiddenimports += _qt_ret[2]

# ── 精简：移除不需要的文件 ──
# v5_server 模型 (已从 UI 移除，省 165MB)
# DirectML provider (未使用)
_EXCLUDE_FILES = {
    # Unused ONNX providers (TRT replaces CUDA; CPU uses onnxruntime.dll)
    'DirectML.dll', 'onnxruntime_providers_tensorrt.dll',
    'onnxruntime_providers_cuda.dll',
}
datas = [(s, d) for s, d in datas if os.path.basename(s) not in _EXCLUDE_FILES
         and not os.path.basename(s).endswith('.engine')]
binaries = [(s, d) for s, d in binaries if os.path.basename(s) not in _EXCLUDE_FILES
            and not os.path.basename(s).endswith('.engine')]

# 过滤 onnxruntime 非推理子目录（transformers/tools/quantization 等在 excludes 中已被跳过，
# 但若以 data 形式被 collect_all 收集则需二次过滤）
_EXCLUDE_DATAS_SUBDIRS = {
    'onnxruntime\\transformers', 'onnxruntime\\tools',
    'onnxruntime\\quantization', 'onnxruntime\\datasets',
    'onnxruntime\\backend',
}
datas = [(s, d) for s, d in datas
         if not any(e in d.replace('/', '\\') for e in _EXCLUDE_DATAS_SUBDIRS)]

# ── OCR 模型资产（onnx + 字符表）──
# TRT .engine 不随 EXE 分发（GPU 架构绑定）：首次运行时本地自动构建，
# 缓存到 <程序目录>/ocr_engines/（免安装设计；旧 %LOCALAPPDATA% 缓存只读回退）
# 只打包 v6_small（v2.13 起唯一模型，GUI/CLI 无模型选择）—— tiny onnx
# 已无用，排除省 ~4.3MB（源码 assets/ 保留，tools 实验脚本仍可用）。
for _root, _dirs, _files in os.walk('assets/ocr_models'):
    for _f in _files:
        if _f.endswith('.engine') or 'tiny' in _f:
            continue
        datas.append((os.path.join(_root, _f),
                      os.path.join('ocr_models', os.path.relpath(_root, 'assets/ocr_models'))))

# ── 精简二进制：移除不需要的 DLL ──
_NVIDIA_DLL_PREFIXES = {
    'cublas', 'cublaslt', 'cudart', 'cufft', 'curand', 'cusparse', 'cusolver',
    'npp', 'nvjpeg', 'nvrtc', 'nvblas', 'nvjitlink',
    'tensorrt', 'nvinfer', 'nvonnxparser',
    'directml', 'cudnn', 'cudnn64',
}
# Also exclude tk/tcl (replaced by PySide6)
_TK_PREFIXES = {'tcl', 'tk', 'tkinter'}
binaries = [
    (src, dst) for src, dst in binaries
    if os.path.basename(src).split('.')[0].lower()
       not in _NVIDIA_DLL_PREFIXES
    and not any(os.path.basename(src).lower().startswith(p)
                for p in _NVIDIA_DLL_PREFIXES)
    and os.path.basename(src).split('.')[0].lower().split('8')[0]
       not in _TK_PREFIXES
]


# NOTE: Run PyInstaller from repo root: pyinstaller RaceVideoToLog.spec
_PROJECT_ROOT = os.path.abspath('.')
a = Analysis(
    [os.path.join(_PROJECT_ROOT, 'RaceVideoToLog.py')],
    pathex=[_PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(_PROJECT_ROOT, 'runtime_hook.py')],
    # 排除 onnxruntime 中的非推理模块 + scipy 测试和未用子模块
    # 这些模块的 hidden imports 会产生大量 ERROR 日志（不影响功能）
    excludes=[
        'onnxruntime.transformers', 'onnxruntime.transformers.*',
        'onnxruntime.tools', 'onnxruntime.tools.*',
        'onnxruntime.quantization', 'onnxruntime.quantization.*',
        'onnxruntime.datasets', 'onnxruntime.datasets.*',
        'onnxruntime.backend',
        # scipy: 已用纯 numpy 替代 savgol_filter，完全排除
        'scipy', 'tkinter', '_tkinter',
        # PaddlePaddle (rapidocr 时代遗留；~1.1GB)
        'paddle', 'paddlepaddle', 'paddlepaddle_gpu',
        # Unused paddle deps
        'safetensors', 'opt_einsum', 'networkx',
        # Pillow：仅 qfluentwidgets 的可选 acrylic 模糊 import（运行时缺
        # colorthief/scipy 已走 fallback），项目自身零引用 → ~12.8MB 冗余
        'PIL', 'PIL.*', 'Pillow',
        # yaml 仅 numpy.__config__ 的可选 import，运行时无引用
        'yaml',
        # numpy.random / numpy.fft：项目零引用（PyInstaller 经 numpy 惰性
        # __getattr__ 误收）
        'numpy.random', 'numpy.fft',
    ],
    noarchive=False,
    optimize=2,   # 最高字节码优化：移除 docstring 和 assert
)
pyz = PYZ(a.pure)

# ── EXE 版本资源（Windows 文件属性 → 属性/详细信息）──
# 版本号来自 config.__version__（单一事实源，与 tools/version.py 一致）。
# 生成失败不阻断构建（降级为无版本资源）。
_VERSION_INFO = None
try:
    import config as _cfg
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable,
        StringStruct, VarFileInfo, VarStruct,
    )
    _ver_parts = [int(x) for x in _cfg.__version__.split('.')[:3]]
    while len(_ver_parts) < 3:
        _ver_parts.append(0)
    _VER_TUPLE = tuple(_ver_parts + [0])
    _VERSION_INFO = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=_VER_TUPLE, prodvers=_VER_TUPLE,
            mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo([
                StringTable('040904B0', [
                    StringStruct('CompanyName', 'RaceVideoToLog'),
                    StringStruct('FileDescription', 'RaceVideoToLog - 赛车视频速度 OCR 提取'),
                    StringStruct('FileVersion', _cfg.__version__),
                    StringStruct('ProductName', 'RaceVideoToLog'),
                    StringStruct('ProductVersion', _cfg.__version__),
                ]),
            ]),
            VarFileInfo([VarStruct('Translation', [1033, 1200])]),
        ],
    )
except Exception as _e:  # 版本资源是附加信息，任何失败都不该阻断构建
    _VERSION_INFO = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RaceVideoToLog',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=_VERSION_INFO,
)
# Post-Analysis 精简：移除 Analysis 重新发现的 DLL
_EXCLUDE_BINARIES = {
    'DirectML.dll', 'onnxruntime_providers_tensorrt.dll',
    'onnxruntime_providers_cuda.dll',
    'tcl86t.dll', 'tk86t.dll', '_tkinter.pyd',
    # Qt6 未使用模块（仅用 Widgets/Core/Gui）
    # Qt6OpenGL* 必须保留：pyqtgraph 0.14 OpenGLHelpers import PySide6.QtOpenGL
    'opengl32sw.dll', 'avcodec-61.dll',
    'Qt6Quick.dll', 'Qt6Qml.dll', 'Qt6Pdf.dll',
    'Qt6Network.dll', 'Qt6Multimedia.dll',
    'Qt6Sql.dll', 'Qt6Test.dll',
    'Qt6QuickWidgets.dll', 'Qt6QmlModels.dll', 'Qt6QmlWorkerScript.dll',
    'Qt6PrintSupport.dll', 'Qt6WebChannel.dll',
    'Qt6WebEngine.dll', 'Qt6WebEngineCore.dll', 'Qt6WebEngineQuick.dll',
    'Qt6Designer.dll', 'Qt6Help.dll', 'Qt6UiTools.dll',
    # PySide6-bundled FFmpeg 6.x/7.x (decord provides FFmpeg 8.x)
    'swresample-5.dll', 'swscale-8.dll', 'avformat-61.dll',
    'avutil-59.dll', 'avcodec-61.dll', 'avdevice-61.dll', 'avfilter-10.dll',
    'postproc-58.dll',
    'avformat-60.dll', 'avutil-58.dll', 'avcodec-60.dll',
    'avdevice-60.dll', 'avfilter-9.dll', 'postproc-57.dll',
    # Stale FFmpeg DLLs (PyPI decord 4.x + previous self-build 5.x)
    'avcodec-58.dll', 'avformat-58.dll', 'avutil-56.dll',
    'avfilter-7.dll', 'avdevice-58.dll', 'swresample-3.dll',
    'swscale-5.dll', 'postproc-55.dll',
    'avcodec-59.dll', 'avformat-59.dll', 'avutil-57.dll',
    'avfilter-8.dll', 'avdevice-59.dll', 'swresample-4.dll',
    'swscale-6.dll', 'postproc-56.dll',
    # decord 发布产物中运行时不需要的二进制：decord.dll 不导入 avdevice，
    # 项目也不调用 ffprobe；qdirect2d 平台插件不用；libcrypto/libssl 的
    # -x64 重复对无任何导入者（_hashlib/_ssl 用无后缀版本）
    'avdevice-62.dll', 'ffprobe.exe', 'qdirect2d.dll',
    'libcrypto-3-x64.dll', 'libssl-3-x64.dll',
}
_PIL_BINARY_PREFIXES = ('_avif', '_imaging', '_webp', '_imagingft',
                        '_imagingmath', '_imagingcms', '_imagingtk')
a.binaries = [(n, p, t) for n, p, t in a.binaries
              if os.path.basename(p) not in _EXCLUDE_BINARIES
              and not os.path.basename(p).startswith(_PIL_BINARY_PREFIXES)]


def _keep_translation(p: str) -> bool:
    """只保留 Qt 的英文与中文翻译，其余 ~6MB 与目标用户无关。"""
    name = os.path.basename(p).lower()
    if 'translations' not in p.replace('\\', '/').lower():
        return True
    return (name.startswith(('qt_en', 'qt_zh_cn', 'qt_zh_tw'))
            or name.startswith(('qtbase_en', 'qtbase_zh_cn', 'qtbase_zh_tw'))
            or name.startswith(('qt_help_en', 'qt_help_zh_cn', 'qt_help_zh_tw')))


# 移除 tk/tcl 数据文件 + 非中英文 Qt 翻译 + PIL 数据
a.datas = [(n, p, t) for n, p, t in a.datas
           if '_tcl_data' not in p and '_tk_data' not in p and 'tcl8' not in p
           and _keep_translation(p)
           and 'PIL' not in p.replace('/', '\\')]

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[
        'onnxruntime.dll',
        'onnxruntime_providers_shared.dll',
        # decord FFmpeg 8.x DLLs (UPX may corrupt)
        'avcodec-62.dll', 'avformat-62.dll', 'avutil-60.dll',
        'swresample-6.dll', 'swscale-9.dll',
    ],
    name='RaceVideoToLog',
)
