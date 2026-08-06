"""RaceVideoToLog 版本工具 — 单一事实源 + 全引用一致性与自动升版本。

版本号只维护一个事实源：`config.__version__`（运行时写 CSV 头）。本工具负责：
- `check`（默认）：校验所有版本引用与 config.py 一致（CI 调用，退出码 1 = 不一致）
- `bump <新版本> [标题]`：更新全部引用 + 在 release_notes.md 顶部插入新版本节

覆盖的引用位置（全部由 bump 自动同步）：
  config.py __version__            (事实源，运行时写 CSV 头/控制台)
  pyproject.toml version           (打包元数据)
  RaceVideoToLog.py 首行 docstring (CLI 显示)
  README.md 标题 + CSV 输出示例    (文档)
  README.md 变更记录区间右端点      (文档)
  DEPENDENCIES.md 标题             (文档)
  release_notes.md 新版本节        (变更日志，追加不改写历史)

版本号规则（SemVer）：
  MAJOR=破坏性变更 / MINOR=新功能 / PATCH=修复。当前发布只接受纯 X.Y.Z
  （不含 -dev/-rc 后缀）；若需预发布构建请扩展 SEMVER_RE 并同步 CI。

用法：
  python tools/version.py                 # 一致性检查（CI 用）
  python tools/version.py check           # 同上
  python tools/version.py bump 2.11.0 "标题"   # 升版本 + 插入变更日志节
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# ═══════════════════ 读写（保留原换行风格）═══════════════════


def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _nl(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


# ═══════════════════ 各引用位置的读取/更新 ═══════════════════

def _get_config(text: str) -> str | None:
    m = re.search(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"', text)
    return m.group(1) if m else None


def _set_config(text: str, v: str) -> str:
    new, n = re.subn(r'__version__ = "\d+\.\d+\.\d+"', f'__version__ = "{v}"', text, count=1)
    return new if n else text


def _get_pyproject(text: str) -> str | None:
    m = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', text, re.M)
    return m.group(1) if m else None


def _set_pyproject(text: str, v: str) -> str:
    new, n = re.subn(r'^version = "\d+\.\d+\.\d+"',
                     f'version = "{v}"', text, count=1, flags=re.M)
    return new if n else text


def _get_docstring(text: str) -> str | None:
    m = re.search(r"RaceVideoToLog v(\d+\.\d+\.\d+)", text)
    return m.group(1) if m else None


def _set_docstring(text: str, v: str) -> str:
    new, n = re.subn(r"RaceVideoToLog v\d+\.\d+\.\d+",
                     f"RaceVideoToLog v{v}", text, count=1)
    return new if n else text


def _get_readme_title(text: str) -> str | None:
    m = re.search(r"^# RaceVideoToLog v(\d+\.\d+\.\d+)", text, re.M)
    return m.group(1) if m else None


def _get_readme_csv(text: str) -> str | None:
    ms = re.findall(r"# RaceVideoToLog v(\d+\.\d+\.\d+)", text)
    return ms[1] if len(ms) > 1 else None


def _set_readme(text: str, v: str) -> str:
    new, n = re.subn(r"(# RaceVideoToLog v)\d+\.\d+\.\d+", rf"\g<1>{v}", text)
    return new if n else text


def _get_readme_range(text: str) -> str | None:
    m = re.search(r"（v[\d.]+ → v(\d+\.\d+\.\d+)）", text)
    return m.group(1) if m else None


def _set_readme_range(text: str, v: str) -> str:
    new, n = re.subn(r"（(v[\d.]+ → v)\d+\.\d+\.\d+）",
                     rf"（\g<1>{v}）", text, count=1)
    return new if n else text


def _get_deps(text: str) -> str | None:
    m = re.search(r"# 上游依赖跟踪（v(\d+\.\d+\.\d+)）", text)
    return m.group(1) if m else None


def _set_deps(text: str, v: str) -> str:
    new, n = re.subn(r"# 上游依赖跟踪（(v\d+\.\d+\.\d+)）",
                     rf"# 上游依赖跟踪（v{v}）", text, count=1)
    return new if n else text


# ═══════════════════ 引用清单 ═══════════════════

def _refs() -> list[tuple[str, Path, object, object]]:
    """[(label, path, getter, setter)]，setter 为 None 表示只读（release_notes）。"""
    return [
        ("config.py __version__", ROOT / "config.py", _get_config, _set_config),
        ("pyproject.toml version", ROOT / "pyproject.toml", _get_pyproject, _set_pyproject),
        ("RaceVideoToLog.py docstring", ROOT / "RaceVideoToLog.py", _get_docstring, _set_docstring),
        ("README.md 标题", ROOT / "README.md", _get_readme_title, _set_readme),
        ("README.md CSV 示例", ROOT / "README.md", _get_readme_csv, None),
        ("README.md 变更记录区间", ROOT / "README.md", _get_readme_range, _set_readme_range),
        ("DEPENDENCIES.md 标题", ROOT / "DEPENDENCIES.md", _get_deps, _set_deps),
    ]


# ═══════════════════ check ═══════════════════

def check() -> int:
    """校验所有引用一致，返回进程退出码。"""
    cfg_text = _read(ROOT / "config.py")
    canonical = _get_config(cfg_text)
    if not canonical:
        print("ERROR: 未在 config.py 找到 __version__")
        return 1

    print(f"单一事实源 config.__version__ = {canonical}")
    print(f"{'引用位置':<32}{'版本':<12}状态")
    print("-" * 60)
    ok = True
    for label, path, getter, _setter in _refs():
        found = getter(_read(path))
        if found is None:
            print(f"{label:<32}{'-':<12}⚠ 未找到")
            ok = False
        elif found == canonical:
            print(f"{label:<32}{found:<12}✓")
        else:
            print(f"{label:<32}{found:<12}✗ 不一致")
            ok = False

    # release_notes 顶部应有当前版本节
    notes = _read(ROOT / "release_notes.md")
    if re.search(rf"^## v{canonical}", notes, re.M):
        print(f"{'release_notes.md v' + canonical:<32}{'有':<12}✓")
    else:
        print(f"{'release_notes.md v' + canonical:<32}{'-':<12}✗ 缺变更日志节")
        ok = False

    print("-" * 60)
    if ok:
        print(f"OK: 全部引用一致（{canonical}）")
        return 0
    print(f"不一致！修复方式：python tools/version.py bump <新版本> 或手动同步")
    return 1


# ═══════════════════ bump ═══════════════════

def bump(new_version: str, title: str) -> int:
    """升版本：更新全部引用 + 插入 release_notes 新节。"""
    if not SEMVER_RE.match(new_version):
        print(f"ERROR: 版本号必须为纯 X.Y.Z（当前: {new_version}）")
        return 1
    cfg_text = _read(ROOT / "config.py")
    old = _get_config(cfg_text)
    if not old:
        print("ERROR: 未在 config.py 找到 __version__")
        return 1
    if old == new_version:
        print(f"提示: 已是 {new_version}，仅同步不一致的引用（release_notes 不重复插入）")

    changed = 0
    for label, path, _getter, setter in _refs():
        if setter is None:
            continue
        text = _read(path)
        new_text = setter(text, new_version)
        if new_text != text:
            _write(path, new_text)
            changed += 1
            print(f"  ✓ {label:<32}→ {new_version}")
        else:
            print(f"  - {label:<32}已是 {new_version}")

    # release_notes：顶部插入新版本节（避免重复）
    notes_path = ROOT / "release_notes.md"
    notes = _read(notes_path)
    if re.search(rf"^## v{new_version}", notes, re.M):
        print(f"  - release_notes.md v{new_version} 节已存在，不重复插入")
    else:
        today = date.today().strftime("%Y-%m-%d")
        nl = _nl(notes)
        section = (
            f"## v{new_version}（{today}）— {title}{nl}"
            f"{nl}"
            f"> 本节面向使用者：只讲你能直接感知到的变化。技术细节见下方各节。{nl}"
            f"{nl}"
            f"### 待补充{nl}"
            f"{nl}"
        )
        # 插到 "# Release Notes" 标题行之后（跳过其后的空行，保持原有空行风格）
        lines = notes.splitlines(keepends=True)
        insert_at = 0
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("# Release Notes"):
                insert_at = i + 1
                while insert_at < len(lines) and not lines[insert_at].strip():
                    insert_at += 1
                break
        lines.insert(insert_at, section)
        _write(notes_path, "".join(lines))
        print(f"  ✓ release_notes.md 插入 v{new_version} 节（请填写内容）")

    print(f"版本 {old} → {new_version}，{changed} 个引用已更新。请运行 pytest 验证。")
    return 0


# ═══════════════════ CLI ═══════════════════

def main(argv: list[str]) -> int:
    # Windows 控制台默认 GBK：✓/✗/⚠ 等符号会 UnicodeEncodeError。
    # 统一改 UTF-8（CI 是 UTF-8，重配置无害；errors=replace 防罕见字符）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not argv or argv[0] in ("check", "--check", "-c"):
        return check()
    if argv[0] in ("bump", "--bump"):
        if len(argv) < 2:
            print("用法: python tools/version.py bump <新版本> [标题]")
            return 2
        title = argv[2] if len(argv) > 2 else "版本更新"
        return bump(argv[1], title)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
