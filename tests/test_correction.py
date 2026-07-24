"""Tests for core pure functions — Viterbi-based correction API."""
import pytest
from correction import (
    expand_partial, _find_neighbor_trusted, _interp_candidate,
    compute_confidence, _auto_expand_digits,
    correct_with_trust,
)
from viterbi import viterbi_correct, _split_segments, _compute_median_profile
from ocr_engine import (
    _savgol_filter_np, normalize_ocr_text, safe_int, safe_float, Flag,
    build_speed_candidates, SpeedObservation,
    compute_lcs_scores, _lcs_score_for_value,
    lcs_detect_errors, compute_lcs_scores_lr,
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


class TestFindNeighborTrusted:
    def _make_rows(self, speeds, flags=None):
        """Helper: create rows with given speeds and flags."""
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
        assert len(result) > 1

    def test_confusion_1_to_9(self):
        """211 → 219: '1'→'9' confusion (critical fix)."""
        result = build_speed_candidates("211", 400)
        assert 219.0 in result  # last digit 1→9
        assert 291.0 in result  # middle digit 1→9
        assert 911.0 not in result  # first digit 1→9 → 911 > 400

    def test_confusion_9_to_1(self):
        """219 → 211: '9'→'1' confusion (reverse direction)."""
        result = build_speed_candidates("219", 400)
        assert 211.0 in result  # last digit 9→1


class TestComputeConfidence:
    def test_all_consistent(self):
        """All frames identical → high confidence."""
        rows = [[i * 0.1, 0.0, 100.0, Flag.RAW] for i in range(20)]
        obs = [SpeedObservation(r[0], r[2], "100") for r in rows]
        result = compute_confidence(rows, obs, 400, 50)
        assert len(result) == 20
        assert all(c["score"] >= 70 for c in result)

    def test_physically_inconsistent(self):
        """Frame at 300 km/h far from median → low confidence."""
        rows = [[0.0, 0.0, 100.0, Flag.RAW],
                [0.1, 0.0, 300.0, Flag.RAW],
                [0.2, 0.0, 100.0, Flag.RAW]]
        obs = [SpeedObservation(r[0], r[2], str(int(r[2]))) for r in rows]
        result = compute_confidence(rows, obs, 400, 50)
        assert result[1]["score"] < 50  # middle frame far from median

    def test_reason_labels(self):
        """Score thresholds produce correct reason labels."""
        rows = [[0.0, 0.0, 100.0, Flag.RAW],
                [0.1, 0.0, 200.0, Flag.RAW],
                [0.2, 0.0, 100.0, Flag.RAW]]
        obs = [SpeedObservation(r[0], r[2], str(int(r[2]))) for r in rows]
        result = compute_confidence(rows, obs, 400, 50)
        assert result[1]["reason"] != '正常'

    def test_out_of_range(self):
        """Speed out of range → score=0."""
        rows = [[0.0, 0.0, -1.0, Flag.RAW],
                [0.1, 0.0, 500.0, Flag.RAW]]
        obs = [SpeedObservation(r[0], r[2], "") for r in rows]
        result = compute_confidence(rows, obs, 400, 50)
        assert result[0]["score"] == 0
        assert result[0]["reason"] == '速度超出范围'
        assert result[1]["score"] == 0

    def test_pinned_boost(self):
        """Trusted frames get high confidence."""
        rows = [[0.0, 0.0, 100.0, Flag.PINNED],
                [0.02, 0.0, 300.0, Flag.RAW],
                [0.04, 0.0, 101.0, Flag.RAW]]
        obs = [SpeedObservation(r[0], r[2], str(int(r[2]))) for r in rows]
        result = compute_confidence(rows, obs, 400, 50, pinned={0})
        # Pinned frame always 100
        assert result[0]["score"] == 100.0
        # Frame 1 far from median → low
        assert result[1]["score"] < 50


class TestAutoExpandDigits:
    def test_1_digit(self):
        """Single digit should generate 1-2 digit expansions."""
        result = _auto_expand_digits("5", 400)
        assert 5 in result
        assert len(result) > 1

    def test_2_digits(self):
        """Two digits should generate 2-3 digit expansions."""
        result = _auto_expand_digits("21", 400)
        assert 21 in result
        assert 121 in result
        assert 221 in result

    def test_3_digits(self):
        """Three digits: single-char replace generates candidates."""
        result = _auto_expand_digits("123", 400)
        assert 123.0 in result
        assert 23.0 in result
        assert 223.0 in result
        assert len(result) > 10

    def test_empty(self):
        assert _auto_expand_digits("", 400) == []
        assert _auto_expand_digits("abc", 400) == []

    def test_all_within_max_speed(self):
        result = _auto_expand_digits("2", 50)
        assert all(v <= 50 for v in result)


class TestLcsScoreForValue:
    """LCS helper still exists for gui_review._check_accel."""
    def _make_rows(self, speeds, dt=0.1):
        times = [i * dt for i in range(len(speeds))]
        return [[times[i], 0.0, float(speeds[i]), Flag.RAW] for i in range(len(speeds))], times

    def test_perfect_consistency(self):
        rows, times = self._make_rows([100, 100, 100, 100, 100])
        score = _lcs_score_for_value(2, 100.0, rows, times, 400, 50)
        assert score == 1.0

    def test_outlier_low_score(self):
        rows, times = self._make_rows([100, 102, 200, 101, 103])
        score = _lcs_score_for_value(2, 200.0, rows, times, 400, 50)
        assert score < 0.5

    def test_pinned_boost(self):
        rows, times = self._make_rows([100, 200, 105])
        rows[0][3] = Flag.PINNED
        score_pinned = _lcs_score_for_value(2, 100.0, rows, times, 400, 100,
                                                high_weight={0})
        rows[0][3] = Flag.RAW
        score_no_pin = _lcs_score_for_value(2, 100.0, rows, times, 400, 100)
        assert score_pinned > score_no_pin

    def test_out_of_range_zero(self):
        rows, times = self._make_rows([100, 102, 500, 101, 103])
        score = _lcs_score_for_value(2, 500.0, rows, times, 400, 50)
        assert score == 0.0


class TestComputeLcsScores:
    def test_all_consistent(self):
        rows = [[i * 0.1, 0.0, 100.0, Flag.RAW] for i in range(20)]
        scores = compute_lcs_scores(rows, 400, 50)
        assert all(s > 0.9 for s in scores)

    def test_pinned_affects_neighbors(self):
        rows = [[i * 0.1, 0.0, 100.0 + i * 0.1, Flag.RAW] for i in range(20)]
        rows[10][3] = Flag.PINNED
        scores = compute_lcs_scores(rows, 400, 50, pinned={10})
        assert all(0 <= s <= 1.0 for s in scores)


class TestDetectErrors:
    """LCS error detection — still available via ocr_engine for backward compat."""
    def test_out_of_range(self):
        rows = [[0.0, 0.0, 100.0, Flag.RAW],
                [0.1, 0.0, -1.0, Flag.RAW],
                [0.2, 0.0, 500.0, Flag.RAW]]
        scores_l, scores_r = compute_lcs_scores_lr(rows, 400, 50)
        errors, borderline = lcs_detect_errors(scores_l, scores_r)
        # Out-of-range values score 0 in LCS
        assert 1 in errors
        assert 2 in errors

    def test_anchors_excluded(self):
        """lcs_detect_errors doesn't exclude pinned frames — caller must filter.

        Frame 1 has speed 500 > max_speed=400 → LCS score = 0.0, which is < LCS_ERROR_LOW.
        The old _detect_errors in correction.py excluded pinned from the result,
        but the raw lcs_detect_errors does not. The caller is responsible for filtering.
        """
        rows = [[0.0, 0.0, 100.0, Flag.PINNED],
                [0.1, 0.0, 500.0, Flag.PINNED]]
        scores_l, scores_r = compute_lcs_scores_lr(rows, 400, 50, pinned={0, 1})
        errors, borderline = lcs_detect_errors(scores_l, scores_r)
        # Frame 1 has out-of-range speed → LCS score 0.0 → appears in errors
        # Caller must filter pinned: errors -= pinned_set
        errors_filtered = errors - {0, 1}
        assert len(errors_filtered) == 0

    def test_returns_scores(self):
        rows = [[i * 0.1, 0.0, 100.0, Flag.RAW] for i in range(10)]
        scores_l, scores_r = compute_lcs_scores_lr(rows, 400, 50)
        assert len(scores_l) == 10
        assert len(scores_r) == 10
        assert all(0 <= s <= 1.0 for s in scores_l)
        assert all(0 <= s <= 1.0 for s in scores_r)


class TestInterpCandidate:
    def test_linear_interp(self):
        rows = [[0.0, 0.0, 100.0, Flag.HIGH_TRUST],
                [0.1, 0.0, 0.0, Flag.RAW],
                [0.2, 0.0, 200.0, Flag.HIGH_TRUST]]
        val = _interp_candidate(1, rows, set(), [r[0] for r in rows], 400)
        assert val == pytest.approx(150.0)

    def test_no_neighbors(self):
        rows = [[0.0, 0.0, 100.0, Flag.RAW],
                [0.1, 0.0, 0.0, Flag.RAW]]
        val = _interp_candidate(0, rows, set(), [r[0] for r in rows], 400)
        assert val is None


# ═══════════════════════════════════════════════════════════════
# Mock OCR for integration tests
# ═══════════════════════════════════════════════════════════════

class MockOCR:
    """模拟 RapidOCR：根据帧数据中的 frame_id 返回预设读数。"""
    def __init__(self, readings: dict[int, float] | None = None):
        self._readings = readings or {}

    def __call__(self, proc):
        fi = int(proc.flat[0]) if proc.size > 0 else -1
        if fi in self._readings:
            val = self._readings[fi]
            class FakeResult:
                txts = [str(int(val))]
                scores = [1.0]
            return FakeResult()
        class FakeEmpty:
            txts = []
            scores = []
        return FakeEmpty()


def _make_frame(frame_id: int) -> "np.ndarray":
    """创建一个携带 frame_id 的假帧图像。"""
    import numpy as np
    arr = np.zeros((24, 48, 3), dtype=np.uint8)
    arr.flat[0] = frame_id % 256
    return arr


# ═══════════════════════════════════════════════════════════════
# Viterbi DP 单元测试
# ═══════════════════════════════════════════════════════════════

class TestViterbiCorrect:
    """Unit tests for the Viterbi DP path selection."""

    def _make_rows(self, speeds, flags=None):
        n = len(speeds)
        if flags is None:
            flags = [Flag.RAW] * n
        return [[float(i), 0.0, float(speeds[i]), flags[i]] for i in range(n)]

    def test_all_consistent(self):
        """All frames identical → no corrections, high confidence."""
        rows = self._make_rows([100, 100, 100, 100, 100])
        times = [i * 0.1 for i in range(5)]
        # Each frame has only raw value as candidate (no alternatives needed)
        candidates = {i: [100.0] for i in range(5)}
        result = viterbi_correct(rows, candidates, set(), times, 400, 50)
        assert len(result['corrected']) == 0  # No corrections
        assert len(result['error_set']) == 0  # No errors
        assert all(c['score'] >= 80 for c in result['confidence'])

    def test_single_outlier(self):
        """Single outlier frame corrected by Viterbi."""
        rows = self._make_rows([100, 100, 200, 100, 100])
        times = [i * 0.1 for i in range(5)]
        # Frame 2: raw=200, but re-OCR returns correct 100
        candidates = {
            0: [100.0], 1: [100.0],
            2: [200.0, 100.0],  # raw + correct candidate
            3: [100.0], 4: [100.0],
        }
        result = viterbi_correct(rows, candidates, set(), times, 400, 50)
        # Frame 2 should be corrected to 100
        assert 2 in result['corrected']
        assert result['corrected'][2] == 100.0

    def test_consistency_island(self):
        """Two consecutive wrong frames that agree with each other → Viterbi fixes both.

        This is the KEY test: under LCS, frames 2 and 3 (both 60) would vote for
        each other, creating a consistency island. Viterbi sees the global picture —
        going through 60 requires impossible transitions at the boundaries.
        """
        rows = self._make_rows([100, 103, 60, 60, 110])
        times = [i * 0.1 for i in range(5)]
        candidates = {
            0: [100.0], 1: [103.0],
            2: [60.0, 105.0],  # re-OCR provides correct value
            3: [60.0, 108.0],  # re-OCR provides correct value
            4: [110.0],
        }
        result = viterbi_correct(rows, candidates, set(), times, 400, 50)
        # Both frames should be corrected
        assert 2 in result['corrected']
        assert 3 in result['corrected']
        assert result['corrected'][2] == 105.0
        assert result['corrected'][3] == 108.0

    def test_trusted_boundary(self):
        """Trusted anchors are never modified."""
        rows = self._make_rows([100, 100, 200, 100, 100],
                                [Flag.PINNED, Flag.RAW, Flag.RAW, Flag.RAW, Flag.PINNED])
        times = [i * 0.1 for i in range(5)]
        candidates = {
            1: [100.0], 2: [200.0, 100.0], 3: [100.0],
        }
        result = viterbi_correct(rows, candidates, {0, 4}, times, 400, 50)
        # Trusted frames never in corrected
        assert 0 not in result['corrected']
        assert 4 not in result['corrected']

    def test_accel_constraint(self):
        """Candidate that violates acceleration constraint is rejected.

        Frame 1 at speed 100, frame 3 at speed 102. Frame 2 candidate 200
        would require impossible acceleration → Viterbi picks the feasible option.
        """
        rows = self._make_rows([100, 101, 200, 103, 104])
        times = [i * 0.1 for i in range(5)]  # dt=0.1s, max_dv=18 km/h
        candidates = {
            0: [100.0], 1: [101.0],
            2: [200.0, 102.0],  # 200 is physically impossible, 102 is feasible
            3: [103.0], 4: [104.0],
        }
        result = viterbi_correct(rows, candidates, set(), times, 400, 50)
        assert 2 in result['corrected']
        assert result['corrected'][2] == 102.0

    def test_median_profile(self):
        """Median profile is correctly computed."""
        speeds = [100.0, 102.0, 200.0, 101.0, 103.0]
        profile = _compute_median_profile(speeds, half_window=2)
        # The outlier at index 2 should be suppressed in median
        assert profile[2] == pytest.approx(102.0)  # median of [100,102,200,101,103]

    def test_split_segments(self):
        """Segment splitting between trusted anchors."""
        # Frame 0 and 4 are trusted, frames 1,2,3 in between
        segments = _split_segments(5, {0, 4})
        assert len(segments) == 1
        assert segments[0] == (0, 4)

    def test_split_segments_no_anchors(self):
        """No trusted frames → one segment covering all."""
        segments = _split_segments(5, set())
        assert len(segments) == 1
        assert segments[0] == (0, 4)

    def test_split_segments_middle_anchor(self):
        """Anchor in the middle splits into two segments."""
        segments = _split_segments(6, {3})
        # Segment 1: 0→3, Segment 2: 3→5
        assert len(segments) == 2
        assert segments[0] == (0, 3)
        assert segments[1] == (3, 5)


# ═══════════════════════════════════════════════════════════════
# 集成测试：correct_with_trust (Viterbi-based)
# ═══════════════════════════════════════════════════════════════

class TestCorrectWithTrust:
    """集成测试：用 mock OCR 数据运行完整 Viterbi 管线。"""

    def _make_rows(self, speeds, flags=None):
        n = len(speeds)
        if flags is None:
            flags = [Flag.RAW] * n
        return [[float(i), 0.0, float(speeds[i]), flags[i]] for i in range(n)]

    def _make_obs(self, raw_texts):
        from ocr_engine import SpeedObservation
        return [SpeedObservation(float(i), float(t), t) for i, t in enumerate(raw_texts)]

    def test_basic_no_errors(self):
        """全是正确的值 → 无修正，标记 HIGH_TRUST。"""
        rows = self._make_rows([100, 100, 100, 100, 100])
        obs = self._make_obs(["100"] * 5)
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        result = correct_with_trust(rows, obs, raw_frames, MockOCR(),
                                     400, 50, fps=30.0)
        speeds = [r[2] for r in result]
        assert speeds == [100, 100, 100, 100, 100]
        # All should have high confidence → marked HIGH_TRUST
        flags = [r[3] for r in result]
        assert all(f == Flag.HIGH_TRUST for f in flags)

    def test_outlier_fixed_by_reocr(self):
        """中间帧异常，re-OCR 提供正确值 → Viterbi 修正。"""
        rows = self._make_rows([100, 100, 200, 100, 100])
        obs = self._make_obs(["100", "100", "200", "100", "100"])
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        mock = MockOCR({2: 100.0})  # re-OCR returns 100 for frame 2
        result = correct_with_trust(rows, obs, raw_frames, mock,
                                     400, 50, fps=30.0)
        assert result[2][2] == 100  # corrected by Viterbi
        assert result[2][3] == Flag.REOCR_AUTO

    def test_reocr_only_flag(self):
        """reocr_only=True 时管线不崩溃。"""
        rows = self._make_rows([100, 100, 200, 100, 100])
        obs = self._make_obs(["100"] * 5)
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        mock = MockOCR({2: 100.0})
        result = correct_with_trust(rows, obs, raw_frames, mock,
                                     400, 50, reocr_only=True, fps=30.0)
        assert result[2][2] == 100  # still corrected

    def test_light_mode_no_iteration(self):
        """light_mode=True 时只做 Viterbi 不做 fill。"""
        rows = self._make_rows([100, 100, 0, 100, 100])
        obs = self._make_obs(["100"] * 5)
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        mock = MockOCR({2: 100.0})
        result = correct_with_trust(rows, obs, raw_frames, mock,
                                     400, 50, light_mode=True, fps=30.0)
        assert result[2][2] == 100
        assert result[2][3] == Flag.REOCR_AUTO

    def test_skip_fill_and_reocr_only(self):
        """skip_fill + reocr_only — 短文本帧通过扩展候选修正 (is_short path)。"""
        rows = self._make_rows([100, 100, 0, 100, 100])
        obs = self._make_obs(["100", "100", "0", "100", "100"])  # short text for frame 2
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        mock = MockOCR({})  # re-OCR returns nothing
        result = correct_with_trust(rows, obs, raw_frames, mock,
                                     400, 50, skip_fill=True, reocr_only=True,
                                     fps=30.0)
        # frame 2 has raw_text="0" (1 digit) → is_short=True
        # → expansion + profile candidate both active → Viterbi can correct
        assert result[2][2] == 100
        assert result[2][3] == Flag.REOCR_AUTO

    def test_consistency_island_fixed(self):
        """连续两帧错误（一致性孤岛）→ Viterbi 同时修正两帧。

        这验证了 Viterbi 对比 LCS 的核心优势：
        LCS 下两帧互相投票无法修正，Viterbi 通过全局优化同时修正。
        """
        rows = self._make_rows([100, 103, 60, 60, 110])
        obs = self._make_obs(["100", "103", "60", "60", "110"])
        raw_frames = [(i, _make_frame(i)) for i in range(5)]
        # re-OCR returns correct values for both wrong frames
        mock = MockOCR({2: 105.0, 3: 108.0})
        result = correct_with_trust(rows, obs, raw_frames, mock,
                                     400, 50, fps=30.0)
        # Both frames should be corrected — this is the key assertion
        assert result[2][2] != 60  # no longer the wrong value
        assert result[3][2] != 60  # no longer the wrong value
        assert result[2][2] == 105.0
        assert result[3][2] == 108.0
