"""Unit tests for RaceVideoToLog core pure functions.

Tests are intentionally limited to pure functions — deterministic,
config-independent, no external side effects. This ensures tests
remain valid across config changes and algorithm refactoring.
"""
import pytest
import numpy as np
from pathlib import Path

from correction import expand_partial, _find_neighbor_trusted, _interp_candidate, _auto_expand_digits
from ocr_engine import (
    _savgol_filter_np, normalize_ocr_text, safe_int, safe_float, Flag,
    build_speed_candidates, SpeedObservation,
)
from analysis import parse_csv


# ═══════════════════ expand_partial ═══════════════════

class TestExpandPartial:
    def test_exact(self):
        assert expand_partial("123", 400) == [123.0]

    def test_one_x(self):
        result = expand_partial("12x", 200)
        assert 120 in result
        assert 129 in result
        assert all(v <= 200 for v in result)

    def test_all_x_returns_empty(self):
        assert expand_partial("xxx", 400) == []

    def test_exceeds_max(self):
        result = expand_partial("5xx", 400)
        assert all(v <= 400 for v in result)

    def test_two_x(self):
        result = expand_partial("1xx", 200)
        assert 100 in result
        assert 199 in result
        assert len(result) == 100


# ═══════════════════ Savitzky-Golay Filter ═══════════════════

class TestSGFilter:
    def test_basic(self):
        y = np.sin(np.linspace(0, 10, 100)) + np.random.default_rng(42).normal(0, 0.1, 100)
        sy = _savgol_filter_np(y, 11, 3)
        assert len(sy) == len(y)
        assert not np.any(np.isnan(sy))

    def test_short_input(self):
        y = np.array([1.0, 2.0, 3.0])
        sy = _savgol_filter_np(y, 11, 3)
        assert np.array_equal(sy, y)

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 200)
        y_clean = np.sin(x) * 10 + 50
        y_noisy = y_clean + rng.normal(0, 0.5, 200)
        sy = _savgol_filter_np(y_noisy, 11, 3)
        assert np.mean(np.abs(sy - y_clean)) < np.mean(np.abs(y_noisy - y_clean))


# ═══════════════════ OCR Utilities ═══════════════════

class TestOCRUtils:
    def test_normalize_ocr_text(self):
        assert normalize_ocr_text("12O") == "120"
        assert normalize_ocr_text("l23") == "123"
        assert normalize_ocr_text("S6") == "56"

    def test_safe_int(self):
        assert safe_int("123") == 123
        assert safe_int("") is None
        assert safe_int("abc") is None

    def test_safe_float(self):
        assert safe_float("3.14") == 3.14
        assert safe_float("") is None


# ═══════════════════ Flag Enum ═══════════════════

class TestFlag:
    def test_is_corrected(self):
        assert Flag.is_corrected(11)
        assert Flag.is_corrected(13)
        assert not Flag.is_corrected(0)
        assert not Flag.is_corrected(21)

    def test_is_trusted(self):
        assert Flag.is_trusted(21)
        assert Flag.is_trusted(22)
        assert Flag.is_trusted(23)
        assert not Flag.is_trusted(0)
        assert not Flag.is_trusted(11)
        assert not Flag.is_trusted(13)

    def test_is_anchor_backward_compat(self):
        """is_anchor() remains as backward-compat alias for is_trusted()."""
        assert Flag.is_anchor(21)
        assert Flag.is_anchor(22)
        assert not Flag.is_anchor(0)
        assert not Flag.is_anchor(11)


# ═══════════════════ Find Neighbor Trusted ═══════════════════

class TestFindNeighborTrusted:
    def _make_rows(self, speeds, flags=None):
        n = len(speeds)
        if flags is None:
            flags = [Flag.HIGH_TRUST] * n
        return [[float(i), 0.0, float(speeds[i]), flags[i]] for i in range(n)]

    def test_find_both(self):
        rows = self._make_rows([100, 102, 101, 103],
            [Flag.RAW, Flag.HIGH_TRUST, Flag.RAW, Flag.HIGH_TRUST])
        la, ra = _find_neighbor_trusted(2, 4, rows)
        assert la == 1
        assert ra == 3

    def test_no_left(self):
        rows = self._make_rows([100, 102, 101],
            [Flag.RAW, Flag.RAW, Flag.HIGH_TRUST])
        la, ra = _find_neighbor_trusted(0, 3, rows)
        assert la is None
        assert ra == 2

    def test_no_right(self):
        rows = self._make_rows([100, 102, 101],
            [Flag.HIGH_TRUST, Flag.RAW, Flag.RAW])
        la, ra = _find_neighbor_trusted(2, 3, rows)
        assert la == 0
        assert ra is None

    def test_skips_invalid_speed(self):
        """Trusted frame with negative speed should be skipped."""
        rows = self._make_rows([100, -1, 101],
            [Flag.RAW, Flag.HIGH_TRUST, Flag.HIGH_TRUST])
        la, ra = _find_neighbor_trusted(0, 3, rows)
        assert la is None  # frame 1 has flag>=20 but speed -1
        assert ra == 2

    def test_pinned_is_trusted(self):
        rows = self._make_rows([100, 102, 101],
            [Flag.RAW, Flag.PINNED, Flag.RAW])
        la, ra = _find_neighbor_trusted(0, 3, rows)
        assert la is None
        assert ra == 1  # PINNED counts as trusted


# ═══════════════════ Parse CSV ═══════════════════

class TestParseCSV:
    def test_parse_skips_comments(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("# header\n1.0,10.0,100.0,0\n2.0,20.0,200.0,21\n", encoding="utf-8")
        times, dists, speeds, flags = parse_csv(str(f))
        assert len(times) == 2
        assert speeds == [100.0, 200.0]
        assert flags == [0, 21]


# ═══════════════════ Build Speed Candidates ═══════════════════

class TestBuildSpeedCandidates:
    def test_normal(self):
        result = build_speed_candidates("50", 400)
        assert 50 in result
        assert all(v <= 400 for v in result)

    def test_empty(self):
        assert build_speed_candidates("", 400) == []
        assert build_speed_candidates("abc", 400) == []

    def test_suffix_expansion(self):
        """OCR reads '60' → candidates should include 60, 160, 260, 360"""
        result = build_speed_candidates("60", 400)
        for expected in [60, 160, 260, 360]:
            assert expected in result

    def test_confusion_chars(self):
        """OCR confuses '8' and '0'"""
        result = build_speed_candidates("80", 400)
        assert len(result) > 1

    def test_confusion_1_to_9(self):
        """211 → 219: '1'→'9' confusion (critical fix)."""
        result = build_speed_candidates("211", 400)
        assert 219 in result  # last digit 1→9
        assert 291 in result  # middle digit 1→9
        assert 911 not in result  # first digit 1→9 → 911 > 400

    def test_confusion_9_to_1(self):
        """219 → 211: '9'→'1' confusion (reverse direction)."""
        result = build_speed_candidates("219", 400)
        assert 211 in result  # last digit 9→1


# ═══════════════════ Auto Expand Digits ═══════════════════

class TestAutoExpandDigits:
    def test_1_digit(self):
        """Single digit should generate 1-2 digit expansions."""
        result = _auto_expand_digits("5", 400)
        assert 5 in result
        assert len(result) > 1  # should have expansions like 15, 25, ..., 95

    def test_2_digits(self):
        """Two digits should generate 2-3 digit expansions."""
        result = _auto_expand_digits("21", 400)
        assert 21 in result
        assert 121 in result
        assert 221 in result

    def test_3_digits(self):
        """Three digits: single-char replace generates candidates (e.g. x23, 1x3, 12x)."""
        result = _auto_expand_digits("123", 400)
        assert 123 in result
        assert 23 in result   # x23 → 023
        assert 223 in result  # 2x3 → 223
        assert len(result) > 10  # many 3-digit expansions

    def test_empty(self):
        assert _auto_expand_digits("", 400) == []
        assert _auto_expand_digits("abc", 400) == []

    def test_all_within_max_speed(self):
        result = _auto_expand_digits("2", 50)
        assert all(v <= 50 for v in result)


# ═══════════════════ Interp Candidate ═══════════════════

class TestInterpCandidate:
    def test_linear_interp(self):
        rows = [[0.0, 0.0, 100.0, Flag.HIGH_TRUST],
                [0.1, 0.0, 0.0, Flag.RAW],
                [0.2, 0.0, 200.0, Flag.HIGH_TRUST]]
        val = _interp_candidate(1, rows, [r[0] for r in rows], 400)
        assert val == pytest.approx(150.0)

    def test_no_neighbors(self):
        rows = [[0.0, 0.0, 100.0, Flag.RAW],
                [0.1, 0.0, 0.0, Flag.RAW]]
        val = _interp_candidate(0, rows, [r[0] for r in rows], 400)
        assert val is None
