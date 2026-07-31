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
    'queue', 'PIL._tkinter_finder',
    'threading', 'concurrent.futures',
    # numpy 2.x PyInstaller 兼容性修复
    'numpy._core._multiarray_umath', 'numpy._core.multiarray',
    'numpy._core.umath', 'numpy._core._methods',
    # rapidocr 内部依赖
    'yaml',
    # decord
    'decord',
    # Project modules (force inclusion; auto-discovered but explicit is safer)
    'pipeline', 'correction', 'config', 'gpu_setup', 'ocr_engine',
    'headless', 'analysis', 'gui_analysis', 'gui_review',
    'gui_export', 'gui_settings', 'viterbi', 'error_detection',
    'widget_utils', 'theme_manager',
    # rapidocr transitive deps (may not be auto-discovered)
    'shapely', 'pyclipper', 'colorlog', 'omegaconf',
    'cv2',  # opencv (headless)
]

# rapidocr（OCR 引擎，含 ONNX 后端）
tmp_ret = collect_all('rapidocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# onnxruntime（CPU provider；TensorRT 由 rapidocr 直接调用）
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# qfluentwidgets (Fluent Design 组件库)
tmp_ret = collect_all('qfluentwidgets')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# cuda-python: collect all .py/.pyd submodules (NOT collect_all which bundles
# ~500MB of native .dll — those come from system PATH at runtime).
try:
    from PyInstaller.utils.hooks import collect_submodules
    _cuda_hidden = collect_submodules('cuda')
    hiddenimports += _cuda_hidden
    # collect_submodules may miss Cython pyd files inside subpackages;
    # run a deeper scan on cuda.bindings specifically
    _cuda_bindings = collect_submodules('cuda.bindings')
    hiddenimports += [m for m in _cuda_bindings if m not in _cuda_hidden]
except Exception:
    pass  # cuda-python not installed

try:
    import tensorrt  # noqa: F401
    hiddenimports += ['tensorrt']
except Exception:
    pass  # tensorrt not installed

# decord（NVDEC 硬件加速视频解码）
tmp_ret = collect_all('decord')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# PySide6 (Qt 6 GUI) — 只收集核心模块 + qfluentwidgets 依赖
for _qt_mod in ['PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui',
                 'PySide6.QtXml', 'PySide6.QtSvg']:
    _qt_ret = collect_all(_qt_mod)
    datas += _qt_ret[0]; binaries += _qt_ret[1]; hiddenimports += _qt_ret[2]

# ── 精简：移除不需要的文件 ──
# v5_server 模型 (已从 UI 移除，省 165MB)
# DirectML provider (未使用)
_EXCLUDE_FILES = {
    # v5 models (replaced by v6_small)
    'ch_PP-OCRv5_mobile_det_infer.onnx', 'ch_PP-OCRv5_mobile_rec_infer.onnx',
    'ch_PP-OCRv5_det_server_infer.onnx', 'ch_PP-OCRv5_rec_server_infer.onnx',
    # v3 legacy
    'ch_PP-OCRv3_det_infer.onnx', 'ch_PP-OCRv3_rec_infer.onnx',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx',
    # v6 detection models (skipped — ROI is already tightly cropped)
    'PP-OCRv6_det_tiny.onnx', 'PP-OCRv6_det_small.onnx',
    # v6 extras (medium unused)
    'PP-OCRv6_det_medium.onnx', 'PP-OCRv6_rec_medium.onnx',
    # TRT engine cache (GPU-specific, rebuilt by user if needed)
    # Use prefix match below for all .engine files
    # Unused ONNX providers (TRT replaces CUDA; CPU uses onnxruntime.dll)
    'DirectML.dll', 'onnxruntime_providers_tensorrt.dll',
    'onnxruntime_providers_cuda.dll',
}
datas = [(s, d) for s, d in datas if os.path.basename(s) not in _EXCLUDE_FILES
         and not os.path.basename(s).endswith('.engine')]
binaries = [(s, d) for s, d in binaries if os.path.basename(s) not in _EXCLUDE_FILES
            and not os.path.basename(s).endswith('.engine')]

# matplotlib（数据分析 tab）
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# 二次过滤：matplotlib 可能带回部分之前排除的文件
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
    runtime_hooks=[],
    # 排除 onnxruntime 中的非推理模块 + scipy/matplotlib 测试和未用子模块
    # 这些模块的 hidden imports 会产生大量 ERROR 日志（不影响功能）
    excludes=[
        'onnxruntime.transformers', 'onnxruntime.transformers.*',
        'onnxruntime.tools', 'onnxruntime.tools.*',
        'onnxruntime.quantization', 'onnxruntime.quantization.*',
        'onnxruntime.datasets', 'onnxruntime.datasets.*',
        'onnxruntime.backend',
        # matplotlib: 排除测试和未用后端
        'matplotlib.tests', 'matplotlib.testing',
        'matplotlib.backends.backend_gtk3', 'matplotlib.backends.backend_gtk3agg',
        'matplotlib.backends.backend_gtk3cairo', 'matplotlib.backends.backend_gtk4',
        'matplotlib.backends.backend_gtk4agg', 'matplotlib.backends.backend_gtk4cairo',
        'matplotlib.backends.backend_cairo', 'matplotlib.backends.backend_macosx',
        'matplotlib.backends.backend_nbagg', 'matplotlib.backends.backend_pgf',
        'matplotlib.backends.backend_ps', 'matplotlib.backends.backend_qt5',
        'matplotlib.backends.backend_qt5agg', 'matplotlib.backends.backend_qt5cairo',
        'matplotlib.backends.backend_svg', 'matplotlib.backends.backend_template',
        'matplotlib.backends.backend_tkcairo', 'matplotlib.backends.backend_wx',
        'matplotlib.backends.backend_wxagg', 'matplotlib.backends.backend_wxcairo',
        'matplotlib.sphinxext',
        # scipy: 已用纯 numpy 替代 savgol_filter，完全排除
        'scipy', 'tkinter', '_tkinter',
        # PaddlePaddle (only for paddlepaddle_migrate branch; ~1.1GB)
        'paddle', 'paddlepaddle', 'paddlepaddle_gpu',
        # Unused paddle deps
        'safetensors', 'opt_einsum', 'networkx',
    ],
    noarchive=False,
    optimize=2,   # 最高字节码优化：移除 docstring 和 assert
)
pyz = PYZ(a.pure)

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
)
# Post-Analysis 精简：移除 Analysis 重新发现的 DLL
_EXCLUDE_BINARIES = {
    'DirectML.dll', 'onnxruntime_providers_tensorrt.dll',
    'onnxruntime_providers_cuda.dll',
    'tcl86t.dll', 'tk86t.dll', '_tkinter.pyd',
    # Qt6 未使用模块（仅用 Widgets/Core/Gui）
    'opengl32sw.dll', 'avcodec-61.dll',
    'Qt6Quick.dll', 'Qt6Qml.dll', 'Qt6Pdf.dll',
    'Qt6Network.dll', 'Qt6Multimedia.dll',
    'Qt6Sql.dll', 'Qt6Test.dll',
    'Qt6QuickWidgets.dll', 'Qt6QmlModels.dll', 'Qt6QmlWorkerScript.dll',
    'Qt6OpenGL.dll', 'Qt6OpenGLWidgets.dll',
    'Qt6PrintSupport.dll', 'Qt6WebChannel.dll',
    'Qt6WebEngine.dll', 'Qt6WebEngineCore.dll', 'Qt6WebEngineQuick.dll',
    'Qt6Designer.dll', 'Qt6Help.dll', 'Qt6UiTools.dll',
    # 音频格式
    'swresample-5.dll', 'swscale-8.dll', 'avformat-61.dll',
    'avutil-59.dll', 'avdevice-61.dll', 'avfilter-10.dll',
    'postproc-58.dll',
}
a.binaries = [(n, p, t) for n, p, t in a.binaries
              if os.path.basename(p) not in _EXCLUDE_BINARIES]
# 移除 tk/tcl 数据文件
a.datas = [(n, p, t) for n, p, t in a.datas
           if '_tcl_data' not in p and '_tk_data' not in p and 'tcl8' not in p]

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[
        'onnxruntime.dll',
        'onnxruntime_providers_shared.dll',
        # opencv-python-headless 5.x uses cv2.pyd (no opencv_world*.dll)
        # decord FFmpeg 4.x DLLs (UPX may corrupt)
        'avcodec-58.dll', 'avformat-58.dll', 'avutil-56.dll',
        'swresample-3.dll', 'swscale-5.dll',
    ],
    name='RaceVideoToLog',
)
