"""打包完整性：根目录顶层模块必须全部列入 pyproject py-modules 清单。

防新增模块忘加清单（pip install -e . / 打包后 ImportError）。历史上
py-modules 是手工维护的 22 项列表，新增顶层模块容易漏。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT = Path(__file__).parent.parent

# 故意不在 py-modules 的顶层模块（按设计，勿随意增删）：
# - runtime_hook.py: PyInstaller runtime hook（.spec 引用，不作为库分发）
# - tensorrt.py: TRT shim —— 仅源码运行时遮蔽 PyPI tensorrt 元包；
#   pip 安装后必须让真实 tensorrt 包生效，故不入清单
WHITELIST = {"runtime_hook", "tensorrt"}


def _listed_modules() -> set:
    with open(PROJECT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return set(data["tool"]["setuptools"]["py-modules"])


def test_all_top_level_modules_listed():
    actual = {p.stem for p in PROJECT.glob("*.py")} - WHITELIST
    listed = _listed_modules()
    missing = actual - listed
    stale = listed - actual
    assert not missing, (
        f"pyproject [tool.setuptools] py-modules 缺少: {sorted(missing)}"
        " —— 新增顶层模块请加入清单（特例加入 WHITELIST）")
    assert not stale, (
        f"pyproject [tool.setuptools] py-modules 多余: {sorted(stale)}"
        " —— 已删除的模块请从清单移除")


def test_whitelist_files_exist():
    for name in WHITELIST:
        assert (PROJECT / f"{name}.py").exists(), \
            f"白名单文件 {name}.py 已不存在，请清理 WHITELIST"
