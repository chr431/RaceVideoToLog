"""Error Detection — Phase 1: multi-signal per-frame confidence scoring.

Computes continuous confidence scores [0, 100] for every frame using 4
independent signals. READ-ONLY: does not modify any values or flags.

Signals:
  1. OCR model confidence — from OcrEngine output
  2. Physics reachability  — both-neighbor check
  3. Local linearity       — median-of-pairs robust interpolation
  4. Acceleration spikes   — opposing spike pairs = consistency island
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


from config import (
    MPS_TO_KMH,
    ERROR_DETECT_OCR_CONF_WEIGHT,
    ERROR_DETECT_PHYSICS_WEIGHT,
    ERROR_DETECT_LINEARITY_WEIGHT,
    ERROR_DETECT_ACCEL_SPIKE_WEIGHT,
    ERROR_DETECT_FLOOR_CAP,
    PHYSICS_TIME_WINDOW, PHYSICS_DECAY_FACTOR,
    LINEARITY_DECAY_FACTOR, LINEARITY_TIME_WINDOW, LINEARITY_MAX_NEIGHBORS,
    ACCEL_SPIKE_VIOLATION_MULT, ACCEL_SPIKE_SEARCH_WINDOW,
    CONF_TIER_LOW_MAX, CONF_TIER_MEDIUM_MAX,
    ACCEL_SCORE_NORMAL, ACCEL_SCORE_NEAR_ONE, ACCEL_SCORE_SAME_DIR,
    ACCEL_SCORE_VIOLATION, ACCEL_SCORE_ISLAND_INTERIOR,
)

if TYPE_CHECKING:
    from ocr_native import OcrEngine

logger = logging.getLogger("RaceVideoToLog.error_detection")


@dataclass
class ErrorReport:
    confidence: list[dict] = field(default_factory=list)


# ═══════════════════ Signal 1: OCR model confidence ═══════════════════

def _signal_ocr_conf(observations: list, start: int = 0,
                    end: int | None = None) -> list[float]:
    """OCR 置信度信号，逐帧独立（可增量计算 [start, end) 区间）。"""
    end = len(observations) if end is None else min(end, len(observations))
    scores = []
    for i in range(start, end):
        if observations[i] is not None:
            conf = observations[i].confidence
            scores.append(round(conf * 100, 1) if conf > 0 else 50.0)
        else:
            scores.append(50.0)
    return scores


# ═══════════════════ Helper: nearest valid frame within time window ═══════════════════

def _find_nearest_valid(vals: list, times: list[float], i: int, direction: int,
                        time_window: float) -> int | None:
    """Return the nearest frame in *direction* with speed >= 0 within time_window seconds."""
    step = 1 if direction > 0 else -1
    j = i + step
    t_i = times[i]
    while 0 <= j < len(vals):
        if abs(times[j] - t_i) > time_window:
            break
        if vals[j] >= 0:
            return j
        j += step
    return None


# ═══════════════════ Signal 2: Physics reachability ═══════════════════

def _signal_physics(vals: list, times: list[float], max_accel_mps2: float,
                    start: int = 0, end: int | None = None,
                    time_window: float = PHYSICS_TIME_WINDOW) -> list[float]:
    """Physics reachability — continuous scoring, no text-length dependency.

    vals: 每帧 raw 速度值列表（原 rows 的第 2 列）。可增量计算 [start, end)
    区间（邻居搜索用全局 vals/times，区间内帧的邻居必然就绪）。

    v < 0      →   0.0  (invalid measurement)
    no neighbour →  50.0  (neutral — insufficient context)
    reachable  → 100.0  (perfect)
    excess     → 100 * exp(-excess_ratio * decay)  (continuous drop)
    """
    n = len(vals)
    end = n if end is None else min(end, n)
    scores = []
    for i in range(start, end):
        v = vals[i]
        if v < 0:
            scores.append(0.0)
            continue

        side_scores: list[float] = []
        for direction in (-1, 1):
            ni = _find_nearest_valid(vals, times, i, direction, time_window)
            if ni is None:
                continue
            nv = vals[ni]
            dt = abs(times[i] - times[ni])
            if dt <= 0:
                continue
            max_dv = max_accel_mps2 * dt * MPS_TO_KMH
            dv = abs(v - nv)
            if dv <= max_dv:
                side_scores.append(100.0)
            else:
                excess = (dv - max_dv) / max_dv
                side_scores.append(100.0 * math.exp(-excess * PHYSICS_DECAY_FACTOR))

        if not side_scores:
            scores.append(50.0)
        else:
            scores.append(round(min(side_scores), 1))

    return scores


# ═══════════════════ Signal 3: Local linearity (median-of-pairs) ═══════════════════

def _signal_linearity(vals: list, times: list[float],
                      start: int = 0, end: int | None = None,
                      time_window: float = LINEARITY_TIME_WINDOW,
                      max_neighbors: int = LINEARITY_MAX_NEIGHBORS) -> list[float]:
    """Robust linearity check using median-of-pairs interpolation.

    vals: 每帧 raw 速度值列表（原 rows 的第 2 列）。可增量计算 [start, end)。

    For each frame, finds valid-speed neighbours on each side within
    *time_window* seconds (capped at *max_neighbors* per side), linearly
    interpolates from all left×right pairs, takes the median expected
    value, and scores based on relative deviation.

    No reliability gate — the median naturally rejects outlier anchors.
    Works equally well at all speed ranges (no text-length dependency).
    """
    n = len(vals)
    end = n if end is None else min(end, n)
    scores = []
    for i in range(start, end):
        v = vals[i]
        if v < 0:
            scores.append(0.0)
            continue

        t_i = times[i]

        # Collect valid frames on each side within time_window
        left_frames: list[int] = []
        j = i - 1
        while j >= 0 and len(left_frames) < max_neighbors:
            if t_i - times[j] > time_window:
                break
            if vals[j] >= 0:
                left_frames.append(j)
            j -= 1

        right_frames: list[int] = []
        j = i + 1
        while j < n and len(right_frames) < max_neighbors:
            if times[j] - t_i > time_window:
                break
            if vals[j] >= 0:
                right_frames.append(j)
            j += 1

        if not left_frames or not right_frames:
            scores.append(50.0)
            continue

        # Expected values from all left-right pairs, median for robustness
        # （配对部分向量化：集合语义与原双层循环相同 —— span 过滤 +
        # expected>0 过滤 → 排序后中位数一致）
        lv = np.fromiter((vals[li] for li in left_frames),
                         dtype=np.float64, count=len(left_frames))
        lt = np.fromiter((times[li] for li in left_frames),
                         dtype=np.float64, count=len(left_frames))
        rv = np.fromiter((vals[ri] for ri in right_frames),
                         dtype=np.float64, count=len(right_frames))
        rt = np.fromiter((times[ri] for ri in right_frames),
                         dtype=np.float64, count=len(right_frames))
        span = rt[None, :] - lt[:, None]
        mask = span >= 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = (t_i - lt[:, None]) / span  # span==0 → nan，被 mask 排除
        expected = lv[:, None] + (rv[None, :] - lv[:, None]) * frac
        exp_flat = expected[mask]
        exp_flat = exp_flat[exp_flat > 0]

        if exp_flat.size == 0:
            scores.append(50.0)
            continue

        exp_flat.sort()
        expected = float(exp_flat[exp_flat.size // 2])  # median

        deviation = abs(v - expected) / expected
        score = 100.0 * math.exp(-deviation * LINEARITY_DECAY_FACTOR)
        scores.append(round(max(0.0, min(100.0, score)), 1))

    return scores


# ═══════════════════ Signal 4: Acceleration spike pairs ═══════════════════

def _build_accel_state(vals: list, times: list[float], fps: float,
                       max_accel_mps2: float) -> tuple[list, list, list]:
    """构建 accel 信号的前置状态（diffs / violation / violation_sign）。

    逐帧 O(1) 增量可维护（IncrementalDetector.add 复用同公式）；
    整段调用与增量维护的结果逐位一致。
    """
    n = len(vals)
    dt = times[1] - times[0] if n >= 2 else 1.0 / max(fps, 1.0)
    max_dv_frame = max_accel_mps2 * dt * MPS_TO_KMH
    threshold = max_dv_frame * ACCEL_SPIKE_VIOLATION_MULT

    diffs = [0.0] * n
    for i in range(1, n):
        vi, vp = vals[i], vals[i - 1]
        if vi > 0 and vp > 0:
            diffs[i] = vi - vp

    violation = [False] * n
    violation_sign = [0] * n
    for i in range(n):
        if abs(diffs[i]) > threshold:
            violation[i] = True
            violation_sign[i] = 1 if diffs[i] > 0 else -1
    return diffs, violation, violation_sign


def _accel_scores_range(violation: list, violation_sign: list,
                        start: int, end: int) -> list[float]:
    """accel 打分（纯区间）：scores[i] 只依赖 violation[i-15:i+16]。"""
    n = len(violation)
    look = ACCEL_SPIKE_SEARCH_WINDOW
    scores = []
    for i in range(start, end):
        if violation[i]:
            scores.append(ACCEL_SCORE_VIOLATION); continue

        left_start = max(0, i - look)
        right_end = min(n, i + look + 1)
        left_v = [violation_sign[j] for j in range(left_start, i) if violation[j]]
        right_v = [violation_sign[j] for j in range(i + 1, right_end) if violation[j]]
        has_left, has_right = len(left_v) > 0, len(right_v) > 0

        if has_left and has_right:
            if left_v[-1] != right_v[0]:
                scores.append(ACCEL_SCORE_ISLAND_INTERIOR)  # opposing spikes → island interior
            else:
                scores.append(ACCEL_SCORE_SAME_DIR)
        elif has_left or has_right:
            scores.append(ACCEL_SCORE_NEAR_ONE)  # near one spike
        else:
            scores.append(ACCEL_SCORE_NORMAL)  # normal
    return scores


def _signal_accel_spikes(rows: list, times: list[float], fps: float,
                         max_accel_mps2: float, start: int = 0,
                         end: int | None = None) -> list[float]:
    """Detect consistency islands via opposing acceleration spike pairs.

    A consistency island produces two opposing spikes:
    one at entry (correct→wrong), one at exit (wrong→correct).
    Frames between opposing spikes are inside the island.
    """
    n = len(rows)
    if n < 3:
        return [50.0] * n
    end = n if end is None else min(end, n)
    diffs, violation, violation_sign = _build_accel_state(rows, times, fps,
                                                          max_accel_mps2)
    return _accel_scores_range(violation, violation_sign, start, end)


# ═══════════════════ Main entry point ═══════════════════

def detect_errors(rows: list, observations: list, times: list[float],
                  max_accel_mps2: float, max_speed_kmh: float,
                  fps: float = 60.0) -> ErrorReport:
    n = len(rows)
    if n == 0:
        return ErrorReport()

    ocr_conf_scores = _signal_ocr_conf(observations, 0, n)
    vals = [r[2] for r in rows]
    physics_scores = _signal_physics(vals, times, max_accel_mps2)
    linearity_scores = _signal_linearity(vals, times)
    accel_scores = _signal_accel_spikes(vals, times, fps, max_accel_mps2)
    confidence = _combine_confidence(vals, max_speed_kmh,
                                     ocr_conf_scores, physics_scores,
                                     linearity_scores, accel_scores, 0)
    return ErrorReport(confidence=confidence)


# ═══════════════════ 流式 Phase 1（增量检测器）═══════════════════

class IncrementalDetector:
    """Phase 1 流式计算器：随主 OCR 批处理增量计算局部信号。

    所有信号都是 ±WINDOW 帧的局部计算（physics/linearity ±0.25s 时间窗、
    accel 尖峰搜索 15 帧）→ 帧 i 在 i+WINDOW 帧就绪后可精确计算，
    与整段 detect_errors 数值一致（同公式同输入，仅遍历区间不同）。

    主 OCR 完成后补算尾部 WINDOW 帧（此时邻居齐全），返回完整 confidence
    —— Phase 1 的墙钟时间全部并进主 OCR 阶段。
    """

    # accel 搜索 15 帧 + 差分 1 帧；physics/linearity ±0.25s（按 fps 换算）
    WINDOW = max(ACCEL_SPIKE_SEARCH_WINDOW + 1,
                 int(LINEARITY_TIME_WINDOW * 60.0) + 1,
                 int(PHYSICS_TIME_WINDOW * 60.0) + 1)

    def __init__(self, fps: float, max_accel_mps2: float,
                 max_speed_kmh: float) -> None:
        self._fps = fps
        self._max_accel = max_accel_mps2
        self._max_speed = max_speed_kmh
        self._vals: list[float] = []      # raw 速度（rows 的第 2 列）
        self._times: list[float] = []
        self._diffs: list[float] = []
        self._violation: list[bool] = []
        self._vsign: list[int] = []
        self._sig_end: int = 0            # 已计算信号区间 [0, sig_end)
        self._conf_cache: list[dict] = []  # 已计算的置信度（保持顺序）

    def add(self, timestamp: float, raw_speed_kmh: int) -> None:
        """主 OCR 每帧完成后调用（O(1)：增量维护 accel 差分/违规状态）。"""
        i = len(self._vals)
        self._vals.append(float(raw_speed_kmh))
        self._times.append(timestamp)
        d = 0.0
        if i >= 1:
            vi, vp = self._vals[i], self._vals[i - 1]
            if vi > 0 and vp > 0:
                d = vi - vp
        self._diffs.append(d)
        dt = (self._times[1] - self._times[0]
              if len(self._times) >= 2 else 1.0 / max(self._fps, 1.0))
        threshold = (self._max_accel * dt * MPS_TO_KMH
                     * ACCEL_SPIKE_VIOLATION_MULT)
        self._violation.append(abs(d) > threshold)
        self._vsign.append(1 if d > 0 else (-1 if d < 0 else 0))

    def confidence_so_far(self) -> list[dict]:
        """已完整计算的置信度（用于提前启动 re-OCR 预热）。"""
        return list(self._conf_cache)

    def advance(self, observations: list) -> None:
        """主 OCR 推进后调用：计算 [sig_end, 可用帧) 的信号与置信度。

        可用帧 = len(vals) - WINDOW（其 ±WINDOW 邻居全部就绪）。
        """
        n = len(self._vals)
        can = n - self.WINDOW
        if can <= self._sig_end or can <= 0:
            return
        start, end = self._sig_end, can
        ocr_s = _signal_ocr_conf(observations, start, end)
        phy_s = _signal_physics(self._vals, self._times, self._max_accel,
                                start, end)
        lin_s = _signal_linearity(self._vals, self._times, start, end)
        acc_s = _accel_scores_range(self._violation, self._vsign, start, end)
        self._conf_cache.extend(
            _combine_confidence(self._vals, self._max_speed,
                                ocr_s, phy_s, lin_s, acc_s, start))
        self._sig_end = end

    def finalize(self, observations: list) -> list[dict]:
        """主 OCR 完成后：补算尾部 WINDOW 帧，返回完整置信度。"""
        n = len(self._vals)
        can = n - self.WINDOW
        if can > self._sig_end:
            self.advance(observations)
        start = self._sig_end
        if start < n:
            ocr_s = _signal_ocr_conf(observations, start, n)
            phy_s = _signal_physics(self._vals, self._times, self._max_accel,
                                    start, n)
            lin_s = _signal_linearity(self._vals, self._times, start, n)
            acc_s = _accel_scores_range(self._violation, self._vsign,
                                        start, n)
            self._conf_cache.extend(
                _combine_confidence(self._vals, self._max_speed,
                                    ocr_s, phy_s, lin_s, acc_s, start))
            self._sig_end = n
        return self._conf_cache


def _combine_confidence(vals: list, max_speed_kmh: float,
                        ocr_conf_scores: list, physics_scores: list,
                        linearity_scores: list, accel_scores: list,
                        base_index: int) -> list[dict]:
    """多信号加权组合 → 单帧置信度（逐帧独立，可增量）。

    vals 为全局速度数组，base_index 为信号区间起始帧号（对应返回值的
    index 字段）。
    """
    total_w = (ERROR_DETECT_OCR_CONF_WEIGHT + ERROR_DETECT_PHYSICS_WEIGHT +
               ERROR_DETECT_LINEARITY_WEIGHT + ERROR_DETECT_ACCEL_SPIKE_WEIGHT)
    w_ocr = ERROR_DETECT_OCR_CONF_WEIGHT / total_w
    w_phy = ERROR_DETECT_PHYSICS_WEIGHT / total_w
    w_lin = ERROR_DETECT_LINEARITY_WEIGHT / total_w
    w_acc = ERROR_DETECT_ACCEL_SPIKE_WEIGHT / total_w

    confidence: list[dict] = []
    for k, i in enumerate(range(base_index, base_index + len(ocr_conf_scores))):
        score = (w_ocr * ocr_conf_scores[k] + w_phy * physics_scores[k] +
                 w_lin * linearity_scores[k] + w_acc * accel_scores[k])
        score = round(max(0.0, min(100.0, score)), 1)

        min_sig = min(physics_scores[k], linearity_scores[k], accel_scores[k])
        for threshold, cap in sorted(ERROR_DETECT_FLOOR_CAP.items()):
            if min_sig < threshold:
                score = min(score, cap)
                break

        if score < CONF_TIER_LOW_MAX: tier = "low"
        elif score < CONF_TIER_MEDIUM_MAX: tier = "medium"
        else: tier = "high"

        signals_sorted = sorted(
            [("ocr_conf", ocr_conf_scores[k]), ("physics", physics_scores[k]),
             ("linearity", linearity_scores[k]),
             ("accel", accel_scores[k])], key=lambda x: x[1])
        lowest_signal, lowest_val = signals_sorted[0]
        if score >= 70: reason = "正常"
        elif score < 30: reason = f"错误({lowest_signal}={lowest_val:.0f})"
        else: reason = f"存疑({lowest_signal}={lowest_val:.0f})"

        cur_v = vals[i]
        confidence.append({
            "index": i, "score": score, "tier": tier,
            "speed": cur_v, "reason": reason,
            "signals": {
                "ocr_conf": round(ocr_conf_scores[k], 1),
                "physics": round(physics_scores[k], 1),
                "linearity": round(linearity_scores[k], 1),
                "accel": round(accel_scores[k], 1),
            },
            "is_corrected": False,
        })
    return confidence
