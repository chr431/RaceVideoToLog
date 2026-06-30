# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all

# ═══════════════════ 构建时屏蔽 CUDA 路径 ═══════════════════
# PyInstaller 会在"动态库搜索"阶段把 CUDA 系统 DLL 全部打包，
# 但用户机器已安装 CUDA Toolkit，无需重复打包。
# 临时从 PATH 中移除 CUDA 相关目录，避免误抓。
_SAVED_PATH = os.environ.get("PATH", "")
os.environ["PATH"] = ";".join([
    p for p in _SAVED_PATH.split(";")
    if "cuda" not in p.lower()
    and "cudnn" not in p.lower()
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
]

# rapidocr_onnxruntime（OCR 引擎）
tmp_ret = collect_all('rapidocr_onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# onnxruntime（CPU / CUDA）
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# PySide6 (Qt 6 GUI) — 只收集核心模块
for _qt_mod in ['PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui']:
    _qt_ret = collect_all(_qt_mod)
    datas += _qt_ret[0]; binaries += _qt_ret[1]; hiddenimports += _qt_ret[2]

# ── 精简：移除不需要的文件 ──
# v5_server 模型 (已从 UI 移除，省 165MB)
# DirectML provider (仅 CUDA 需要)
_EXCLUDE_FILES = {
    # v5 models (replaced by v6_small)
    'ch_PP-OCRv5_mobile_det_infer.onnx', 'ch_PP-OCRv5_mobile_rec_infer.onnx',
    'ch_PP-OCRv5_det_server_infer.onnx', 'ch_PP-OCRv5_rec_server_infer.onnx',
    # v3 legacy
    'ch_PP-OCRv3_det_infer.onnx', 'ch_PP-OCRv3_rec_infer.onnx',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx',
    # v6 extras (only small needed)
    'PP-OCRv6_det_tiny.onnx', 'PP-OCRv6_rec_tiny.onnx',
    'PP-OCRv6_det_medium.onnx', 'PP-OCRv6_rec_medium.onnx',
    # Unused ONNX providers
    'DirectML.dll', 'onnxruntime_providers_tensorrt.dll',
}
datas = [(s, d) for s, d in datas if os.path.basename(s) not in _EXCLUDE_FILES]
binaries = [(s, d) for s, d in binaries if os.path.basename(s) not in _EXCLUDE_FILES]

# matplotlib（数据分析 tab）
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# 二次过滤：matplotlib 可能带回部分之前排除的文件
datas = [(s, d) for s, d in datas if os.path.basename(s) not in _EXCLUDE_FILES]
binaries = [(s, d) for s, d in binaries if os.path.basename(s) not in _EXCLUDE_FILES]

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


a = Analysis(
    ['D:\\Repo\\RaceVideoToLog\\RaceVideoToLog.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 排除 onnxruntime 中的非推理模块 + scipy/matplotlib 测试和未用子模块
    excludes=[
        'onnxruntime.transformers',
        'onnxruntime.tools',
        'onnxruntime.quantization',
        'onnxruntime.datasets',
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
    'tcl86t.dll', 'tk86t.dll', '_tkinter.pyd',
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
        'onnxruntime_providers_cuda.dll',
        'onnxruntime_providers_shared.dll',
        'opencv_world4100.dll',
    ],
    name='RaceVideoToLog',
)
