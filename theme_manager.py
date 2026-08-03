"""主题管理器 — 集中管理所有需手动更新的主题回调。

Fluent 控件自动响应 qconfig.themeChanged。
这里仅注册原生 Qt / matplotlib 等需要手动更新的 widget。
"""
from __future__ import annotations
from collections.abc import Callable


class ThemeManager:
    """单例式主题回调管理器。"""
    _callbacks: list[Callable[[bool], None]] = []

    @classmethod
    def register(cls, fn: Callable[[bool], None]) -> Callable[[bool], None]:
        cls._callbacks.append(fn)
        return fn

    @classmethod
    def unregister(cls, fn: Callable[[bool], None]) -> None:
        if fn in cls._callbacks:
            cls._callbacks.remove(fn)

    @classmethod
    def refresh(cls) -> None:
        from qfluentwidgets import isDarkTheme
        dark = isDarkTheme()
        for fn in cls._callbacks:
            try:
                fn(dark)
            except Exception:
                pass
