import re

import config


def test_version_uses_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", config.__version__)


def test_chart_colors_match_theme():
    assert config.chart_colors(dark=True) == (config.COLOR_BG_DARK, config.COLOR_FG_DARK)
    assert config.chart_colors(dark=False) == (config.COLOR_BG_LIGHT, config.COLOR_FG_LIGHT)
