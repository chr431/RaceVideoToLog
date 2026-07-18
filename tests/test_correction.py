"""Tests for correction.py pure functions."""
import pytest
from correction import expand_partial, _find_neighbor_anchors, _interp_candidate
from ocr_engine import _savgol_filter_np, normalize_ocr_text, safe_int, safe_float, Flag
from analysis import parse_csv
import numpy as np
import os


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
