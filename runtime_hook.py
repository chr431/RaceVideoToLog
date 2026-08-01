"""PyInstaller runtime hook — sets up DLL search paths for decord."""
import os
import sys

if getattr(sys, 'frozen', False):
    _base = os.path.dirname(sys.executable)
    _internal = os.path.join(_base, '_internal')
    _decord_dir = os.path.join(_internal, 'decord')
    if os.path.isdir(_decord_dir):
        os.environ['DECORD_LIBRARY_PATH'] = _decord_dir
        os.environ['PATH'] = _decord_dir + os.pathsep + os.environ.get('PATH', '')
    # CUDA
    for _k, _v in os.environ.items():
        if _k.startswith("CUDA_PATH") and _v:
            _bin = os.path.join(_v, "bin")
            if os.path.isdir(_bin):
                try:
                    os.add_dll_directory(_bin)
                except Exception:
                    pass
