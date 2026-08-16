"""新旧 decord DLL 像素一致性：解码同帧序列，输出字节 sha256 对比。"""
import hashlib
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from tools.detect_eval import load_meta  # noqa: E402

N = 300


def decode_hash(dll_path: str):
    """在子进程里用指定 DLL 解码，返回 (hash, shape)。"""
    import subprocess
    code = f"""
import hashlib, sys
sys.path.insert(0, {str(PROJECT)!r})
from tools.detect_eval import load_meta
roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta('test6')
from decord import VideoReader, cpu
vr = VideoReader('D:/Videos/racelog_test/test6.mp4', ctx=cpu(0),
                 output_format='gray',
                 roi=(roi[0], roi[1], roi[2]+1, roi[3]+1), num_threads=12)
frames = list(range(f_start, f_start + {N}))
out = vr.get_batch(frames, roi=(roi[0], roi[1], roi[2]+1, roi[3]+1)).asnumpy()
h = hashlib.sha256(out.tobytes()).hexdigest()
print('HASH', h, out.shape, out.dtype)
"""
    import os as _os
    env = dict(_os.environ)
    env["DECORD_LIBRARY_PATH"] = dll_path
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, encoding="utf-8")
    for line in r.stdout.splitlines():
        if line.startswith("HASH"):
            return line.split()
    print("ERR:", r.stderr[-300:])
    return None


if __name__ == "__main__":
    site = PROJECT / ".venv" / "Lib" / "site-packages" / "decord"
    old = str(site / "decord.dll.bak")
    new = str(site / "decord.dll")
    h_old = decode_hash(old)
    h_new = decode_hash(new)
    print("旧 DLL:", h_old)
    print("新 DLL:", h_new)
    if h_old and h_new:
        print("逐位一致 ✓" if h_old[1] == h_new[1] else "不一致 ✗")
