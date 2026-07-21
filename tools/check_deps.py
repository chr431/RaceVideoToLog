"""检查上游依赖更新 — 通过 PyPI JSON API 查询最新版本。"""
import json, urllib.request, sys
from pathlib import Path

# 核心依赖（包名 → 当前版本）
DEPS = {
    "rapidocr": "3.9.1",
    "onnxruntime": "1.27.0",
    "opencv-python-headless": "5.0.0",
    "decord": "0.6.0",
    "PySide6": "6.11.1",
    "PySide6-Fluent-Widgets": "1.11.2",
    "numpy": "2.5.1",
    "matplotlib": "3.11.1",
    "pyclipper": "1.4.0",
    "shapely": "2.1.2",
}

# GPU 加速（可选，版本跟踪）
GPU_DEPS = {
    "tensorrt": "10.16.1.11",
    "cuda-python": "13.3.1",
}

# 打包
BUILD_DEPS = {
    "pyinstaller": "6.21.0",
}


def get_latest(pkg_name: str) -> str:
    """从 PyPI JSON API 获取最新版本号。"""
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception as e:
        return f"ERROR: {e}"


def check_group(label: str, deps: dict) -> int:
    updates = 0
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for pkg, current in deps.items():
        latest = get_latest(pkg)
        if latest.startswith("ERROR"):
            print(f"  {pkg:35s} current={current:12s}  {latest}")
            continue
        flag = " ← UPDATE" if latest != current else ""
        if flag:
            updates += 1
        print(f"  {pkg:35s} current={current:12s}  latest={latest:12s}{flag}")
    return updates


if __name__ == "__main__":
    total = 0
    total += check_group("核心依赖", DEPS)
    total += check_group("GPU 加速（可选）", GPU_DEPS)
    total += check_group("打包工具", BUILD_DEPS)

    print(f"\n{'='*50}")
    if total:
        print(f"  共 {total} 个包可更新")
    else:
        print(f"  所有 {len(DEPS)+len(GPU_DEPS)+len(BUILD_DEPS)} 个包均为最新")
    print(f"{'='*50}")
