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
    'tkinter', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk',
    'threading', 'concurrent.futures',
]

# rapidocr_onnxruntime（OCR 引擎）
tmp_ret = collect_all('rapidocr_onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# onnxruntime（CPU / CUDA）
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ── 精简二进制：移除不需要的 DLL ──
_NVIDIA_DLL_PREFIXES = {
    'cublas', 'cublaslt', 'cudart', 'cufft', 'curand', 'cusparse', 'cusolver',
    'npp', 'nvjpeg', 'nvrtc', 'nvblas', 'nvjitlink',
    'tensorrt', 'nvinfer', 'nvonnxparser',
    'directml', 'cudnn', 'cudnn64',
}
binaries = [
    (src, dst) for src, dst in binaries
    if os.path.basename(src).split('.')[0].lower()
       not in _NVIDIA_DLL_PREFIXES
    and not any(os.path.basename(src).lower().startswith(p)
                for p in _NVIDIA_DLL_PREFIXES)
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
    # 排除 onnxruntime 中的非推理模块（transformers/tools/quantization/datasets）
    excludes=[
        'onnxruntime.transformers',
        'onnxruntime.tools',
        'onnxruntime.quantization',
        'onnxruntime.datasets',
        'onnxruntime.backend',
    ],
    noarchive=False,
    optimize=2,   # 最高字节码优化：移除 docstring 和 assert
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RaceVideoToLog',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # 不移除符号表（避免破坏 python313.dll）
    upx=True,
    upx_exclude=[
        'onnxruntime.dll',
        'onnxruntime_providers_cuda.dll',
        'onnxruntime_providers_shared.dll',
        'opencv_world4100.dll',
    ],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
