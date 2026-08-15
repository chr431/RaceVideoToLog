"""日志级别配置（CLI --log-level 与 GUI 日志级别下拉共用）。

级别语义：
- normal：根日志 INFO（保持 v2.14 及更早的默认行为）
- detailed：RaceVideoToLog 项目日志 DEBUG，第三方仍 INFO
- debug：根日志 DEBUG（含第三方库）
"""
from __future__ import annotations

import logging

PROJECT_LOGGER_PREFIX = "RaceVideoToLog"


def configure_logging(level: str) -> None:
    key = (level or "normal").strip().lower()
    root_level = logging.INFO
    project_level = logging.INFO
    if key in ("detailed", "debug"):
        project_level = logging.DEBUG
    if key == "debug":
        root_level = logging.DEBUG
    logging.basicConfig(
        level=root_level,
        format="%(name)s: %(message)s",
        force=True,
    )
    logging.getLogger(PROJECT_LOGGER_PREFIX).setLevel(project_level)
