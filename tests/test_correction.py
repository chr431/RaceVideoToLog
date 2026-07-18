"""Tests for core pure functions."""
import pytest
from correction import (
	expand_partial, _find_neighbor_anchors, _interp_candidate,
	compute_confidence, _score_candidate,
	_detect_neighbor_jump, _detect_v_shape, _detect_cliff,
	_detect_anchor_trend, _detect_isolated_spike, _detect_local_trend,
	_detect_errors,
)
from ocr_engine import (
	_savgol_filter_np, normalize_ocr_text, safe_int, safe_float, Flag,
	build_speed_candidates, SpeedObservation,
	auto_select_anchors, _anchor_adaptive_window,
	_anchor_select_center, _anchor_validate_neighbors,
)
from analysis import parse_csv
import numpy as np


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


class TestFlag:
	def test_is_corrected(self):
		assert Flag.is_corrected(11)
		assert Flag.is_corrected(13)
		assert not Flag.is_corrected(0)
		assert not Flag.is_corrected(21)

	def test_is_anchor(self):
		assert Flag.is_anchor(21)
		assert Flag.is_anchor(22)
		assert not Flag.is_anchor(0)
		assert not Flag.is_anchor(11)


class TestAnchorHelper:
	def test_find_neighbor_anchors(self):
		anchors = {0, 5, 10}
		la, ra = _find_neighbor_anchors(3, 15, anchors)
		assert la == 0
		assert ra == 5

	def test_no_left_anchor(self):
		anchors = {5, 10}
		la, ra = _find_neighbor_anchors(1, 15, anchors)
		assert la is None
		assert ra == 5

	def test_no_right_anchor(self):
		anchors = {0, 3}
		la, ra = _find_neighbor_anchors(7, 15, anchors)
		assert la == 3
		assert ra is None


class TestParseCSV:
	def test_parse_skips_comments(self, tmp_path):
		f = tmp_path / "test.csv"
		f.write_text("# header\n1.0,10.0,100.0,0\n2.0,20.0,200.0,21\n", encoding="utf-8")
		times, dists, speeds, flags = parse_csv(str(f))
		assert len(times) == 2
		assert speeds == [100.0, 200.0]
		assert flags == [0, 21]


class TestBuildSpeedCandidates:
	def test_normal(self):
		result = build_speed_candidates("50", 400)
		assert 50.0 in result
		assert all(v <= 400 for v in result)

	def test_empty(self):
		assert build_speed_candidates("", 400) == []
		assert build_speed_candidates("abc", 400) == []

	def test_suffix_expansion(self):
		"""OCR reads '60' → candidates should include 60, 160, 260, 360"""
		result = build_speed_candidates("60", 400)
		for expected in [60, 160, 260, 360]:
			assert float(expected) in result

	def test_confusion_chars(self):
		"""OCR confuses '8' and '0'"""
		result = build_speed_candidates("80", 400)
		# Should include variations: 80, 00 (invalid), 88, 08, 60, etc.
		assert len(result) > 1


class TestComputeConfidence:
	def test_all_clean(self):
		"""All frames are anchors — should have high confidence."""
		rows = [[float(i), 0.0, 100.0, 21] for i in range(10)]  # all anchors
		obs = [SpeedObservation(float(i), 100.0, "100") for i in range(10)]
		result = compute_confidence(rows, obs, 400, 50)
		assert len(result) == 10
		assert all(c["score"] >= 60 for c in result)

	def test_low_confidence(self):
		"""Frame with corrected flag gets penalty."""
		rows = [[i * 0.1, 0.0, 100.0, 0] for i in range(10)]
		rows[5][3] = 11  # auto-corrected flag → -30 penalty
		obs = [SpeedObservation(r[0], r[2], str(int(r[2]))) for r in rows]
		result = compute_confidence(rows, obs, 400, 50)
		assert result[5]["score"] < result[4]["score"]  # corrected has lower score

	def test_extreme_accel(self):
		"""Extreme acceleration → confidence drops."""
		rows = [[0.0, 0.0, 100.0, 0], [0.1, 0.0, 300.0, 0]]  # 2000 km/h/s jump
		obs = [SpeedObservation(r[0], r[2], str(int(r[2]))) for r in rows]
		result = compute_confidence(rows, obs, 400, 50)
		assert result[1]["score"] < 90


class TestErrorDetectors:
	def _make_context(self, speeds, anchors=None, dt=0.1):
		"""Helper: create detector context from speed values."""
		n = len(speeds)
		raw_vals = [float(s) for s in speeds]
		times = [i * dt for i in range(n)]
		rows = [[times[i], 0.0, raw_vals[i], 0] for i in range(n)]
		anchors_set = anchors or set()
		return raw_vals, times, rows, anchors_set, n

	def test_neighbor_jump_clean(self):
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 102, 101, 103])
		assert not _detect_neighbor_jump(1, raw_vals[1], n, raw_vals, times, 400, 50)

	def test_neighbor_jump_bad(self):
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 200, 101])  # frame 1 jumps 100 km/h → ~278 m/s²
		assert _detect_neighbor_jump(1, raw_vals[1], n, raw_vals, times, 400, 50)

	def test_v_shape(self):
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 50, 100])  # rapid decel then accel
		assert _detect_v_shape(1, raw_vals[1], n, raw_vals, times, 50)

	def test_cliff(self):
		"""One side jumps 4000→100 while the other stays flat."""
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 500, 490])  # left accel huge, right accel near zero
		assert _detect_cliff(1, raw_vals[1], n, raw_vals, times, 50)

	def test_anchor_trend(self):
		raw_vals, times, rows, anchors_set, n = self._make_context(
			[100, 20, 100], anchors={0, 2})  # frame 1 at 20 deviates 80 from interp 100, threshold ~54
		assert _detect_anchor_trend(1, raw_vals[1], n, rows, times, anchors_set, 50)

	def test_isolated_spike(self):
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 101, 200, 101, 100])  # frame 2 is a spike
		assert _detect_isolated_spike(2, raw_vals[2], n, raw_vals, times, 50)

	def test_local_trend(self):
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 101, 200, 101, 100])  # neighbors all near 100, frame 2 at 200
		assert _detect_local_trend(2, raw_vals[2], n, raw_vals, 400)

	def test_integrated_detection(self):
		"""All detectors together via _detect_errors."""
		raw_vals, times, rows, anchors, n = self._make_context(
			[100, 102, 101, 103, 250, 104, 105])  # frame 4 is bad
		errors = _detect_errors(rows, {0, 2, 6}, times, 400, 50)
		assert 4 in errors  # frame 4 should be caught


class TestAutoSelectAnchors:
	def test_adaptive_window(self):
		times = [i * 0.02 for i in range(100)]  # 50 fps
		w = _anchor_adaptive_window(times, 100)
		assert w >= 5
		assert w % 2 == 1  # odd

	def test_center_selection(self):
		"""Clean stable speeds should produce many anchors."""
		n = 50
		raw_vals = [100.0 + np.random.default_rng(0).normal(0, 1.0) for _ in range(n)]
		times = [i * 0.02 for i in range(n)]
		anchors = _anchor_select_center(raw_vals, times, n, 11, 400, 4.0)
		assert len(anchors) > n // 2

	def test_neighbor_validation(self):
		"""Isolated bad anchor should be removed."""
		raw_vals = [100.] * 10
		raw_vals[5] = 500  # outlier
		anchors = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
		filtered = _anchor_validate_neighbors(anchors, raw_vals, 10)
		assert 5 not in filtered  # outlier removed
		assert 4 in filtered  # neighbors kept


class TestScoreCandidate:
	def _make_context(self, speeds, error_set=None, anchors=None, dt=0.1):
		n = len(speeds)
		times = [i * dt for i in range(n)]
		rows = [[times[i], 0.0, float(speeds[i]), 0] for i in range(n)]
		return rows, times, anchors or set(), error_set or set(), n

	def test_perfect_match(self):
		rows, times, anchors, errors, n = self._make_context(
			[100, 102, 101, 103])
		score = _score_candidate(101.0, 2, rows, anchors, errors, times, 400, 50)
		assert 0 <= score <= 1.0

	def test_anchors_prefer(self):
		"""Candidate close to anchor interpolation should score higher."""
		rows, times, anchors, errors, n = self._make_context(
			[100, 50, 100], anchors={0, 2})
		good = _score_candidate(100.0, 1, rows, anchors, errors, times, 400, 50)
		bad = _score_candidate(50.0, 1, rows, anchors, errors, times, 400, 50)
		assert good > bad
