"""Error Detection — Phase 1: multi-signal per-frame confidence scoring.

Computes continuous confidence scores [0, 100] for every frame using 4
independent signals. READ-ONLY: does not modify any values or flags.

Signals:
  1. OCR model confidence         — from OcrEngine output
  2. Bandwidth-norm abs residual  — median-of-pairs interpolation; absolute
     residual normalised by local bandwidth (replaces physics + linearity)
  3. Acceleration spikes          — opposing spike pairs = consistency island
  4. Frequency-domain residual    — Gaussian low-pass residual (high-freq = non-physical)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


from config import (
    MPS_TO_KMH,
    ERROR_DETECT_OCR_CONF_WEIGHT,
    ERROR_DETECT_ABS_WEIGHT,
    ERROR_DETECT_ACCEL_SPIKE_WEIGHT,
    ERROR_DETECT_FREQ_WEIGHT,
    ERROR_DETECT_FLOOR_CAP,
    FREQ_FLOOR_THRESHOLD, FREQ_FLOOR_CAP, FREQ_CORROBORATE_THRESHOLD,
    LINEARITY_TIME_WINDOW, LINEARITY_MAX_NEIGHBORS,
    ABS_RESID_FLOOR, ABS_RESID_WINDOW,
    ACCEL_SPIKE_VIOLATION_MULT, ACCEL_SPIKE_SEARCH_WINDOW,
    CONF_TIER_LOW_MAX, CONF_TIER_MEDIUM_MAX,
    ACCEL_SCORE_NORMAL, ACCEL_SCORE_NEAR_ONE, ACCEL_SCORE_SAME_DIR,
    ACCEL_SCORE_VIOLATION, ACCEL_SCORE_ISLAND_INTERIOR,
    FREQ_RESID_SIGMA, FREQ_RESID_SCALE,
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


# ═══════════════════ Signal 2: bandwidth-normalised absolute residual ═══════════════════

def _median_pairs_expected(vals: list, times: list[float], start: int, end: int,
                           time_window: float = LINEARITY_TIME_WINDOW,
                           max_neighbors: int = LINEARITY_MAX_NEIGHBORS) -> list[float]:
    """median-of-pairs 插值期望值（robust interpolation）。

    对 [start, end) 内每帧，取两侧 time_window 秒内最多 max_neighbors 个有效帧，
    所有左×右对线性插值后取中位数。无插值（单侧无邻居 / 无效值）→ NaN
    （score 50 中性）。可增量计算：邻居搜索用全局 vals/times。
    """
    n = len(vals)
    exp: list[float] = []
    for i in range(start, end):
        if vals[i] < 0:
            exp.append(float("nan"))
            continue
        t_i = times[i]

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
            exp.append(float("nan"))
            continue

        lv = np.fromiter((vals[li] for li in left_frames), dtype=np.float64,
                         count=len(left_frames))
        lt = np.fromiter((times[li] for li in left_frames), dtype=np.float64,
                         count=len(left_frames))
        rv = np.fromiter((vals[ri] for ri in right_frames), dtype=np.float64,
                         count=len(right_frames))
        rt = np.fromiter((times[ri] for ri in right_frames), dtype=np.float64,
                         count=len(right_frames))
        span = rt[None, :] - lt[:, None]
        mask = span >= 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (t_i - lt[:, None]) / span  # span==0 → nan，被 mask 排除
        expected = lv[:, None] + (rv[None, :] - lv[:, None]) * frac
        exp_flat = expected[mask]
        exp_flat = exp_flat[exp_flat > 0]
        if exp_flat.size == 0:
            exp.append(float("nan"))
            continue
        exp_flat.sort()
        exp.append(float(exp_flat[exp_flat.size // 2]))  # median
    return exp


def _signal_abs_residual(vals: list, times: list[float],
                         start: int = 0, end: int | None = None,
                         floor: float = ABS_RESID_FLOOR,
                         window: int = ABS_RESID_WINDOW,
                         time_window: float = LINEARITY_TIME_WINDOW,
                         max_neighbors: int = LINEARITY_MAX_NEIGHBORS) -> list[float]:
    """带宽归一化绝对残差信号（Signal 2，取代 physics + linearity）。

    expected = 中位数插值期望值（与 linearity 同引擎）；
    resid = |raw − expected|（绝对残差，修正 linearity 相对归一化的高速漏检）；
    bandwidth = |raw 一阶差分| 滑动均值 + floor（真实变速时带宽大 → 大偏差不罚，
    修正 physics 的起步/刹车误报；巡航时带宽小 → 小偏差即可疑）。

    score = 100 * exp(−resid / max(bw, floor))。巡航帧插值噪声 ~0.5-1.5 km/h，
    故 floor 须 ≥3 防误报。NaN 期望 → 50 中性；raw<0 → 0。
    """
    n = len(vals)
    end = n if end is None else min(end, n)
    if end <= start:
        return []
    expected = _median_pairs_expected(vals, times, start, end,
                                      time_window, max_neighbors)
    exp_arr = np.asarray(expected, dtype=np.float64)
    # 局部带宽：|diff| 滑动均值，外扩 k 帧（边缘扩展）保证边界帧上下文。
    # prepend 用 vals[lo-1]（而非 seg[0]）→ 段首 diff 与全局数组逐位一致，
    # 增量路径与整段路径的带宽相同（段边界不引入 0 伪 diff）。
    k = window // 2
    lo = max(0, start - k)
    hi = min(n, end + k)
    seg = np.asarray(vals[lo:hi], dtype=np.float64)
    prepend = vals[lo - 1] if lo > 0 else seg[0]
    rd = np.abs(np.diff(seg, prepend=prepend))
    pad = np.pad(rd, (k, k), mode="edge")
    kernel = np.ones(window) / window
    bw_all = np.convolve(pad, kernel, mode="valid")  # 长度 = hi - lo
    i0, i1 = start - lo, end - lo
    bw = np.maximum(bw_all[i0:i1], floor)
    raw = np.asarray(vals[start:end], dtype=np.float64)
    resid = np.abs(raw - exp_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 100.0 * np.exp(-resid / bw)
    scores = np.where(np.isnan(exp_arr), 50.0, scores)
    scores = np.where(raw < 0, 0.0, scores)
    return [round(float(s), 1) for s in scores]


# ═══════════════════ Signal 3: Acceleration spike pairs ═══════════════════

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


# ═══════════════════ Signal 4: 频域残差（高频内容 = 非物理） ═══════════════════

def _signal_frequency(vals: list, times: list[float],
                      start: int = 0, end: int | None = None,
                      sigma: float = FREQ_RESID_SIGMA,
                      scale: float = FREQ_RESID_SCALE) -> list[float]:
    """频域残差信号 — 中心高斯低通后取高频残差，指数衰减评分。

    真实速度信号是低带宽的（实测 99% 频谱能量 ≤ 2.5Hz，即变化至少 ~0.4s
    尺度）。对逐帧速度做短窗高斯低通（sigma=3 帧），残差 = 被滤除的高频
    内容 —— 单帧/短时的 3-9 km/h 尖峰残差大（score 低），而真实变速是
    平滑 ramp（残差小，score 高）。补 accel 尖峰信号（阈值 3×max_dv≈9
    km/h 才触发）漏掉的 3-9 km/h 中间区间。

    v < 0 → 0.0；边界帧用边缘扩展补足窗口（与增量 WINDOW 约束一致）。
    """
    n = len(vals)
    end = n if end is None else min(end, n)
    k = int(4 * sigma)
    if end <= start:
        return []
    # 区间窗口外扩 k 帧（边缘扩展），保证区间内每帧有完整 ±k 上下文
    lo = max(0, start - k)
    hi = min(n, end + k)
    seg = np.asarray(vals[lo:hi], dtype=np.float64)
    pad_seg = np.pad(seg, (k, k), mode="edge")
    x = np.arange(-k, k + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # valid 卷积：out[j] = 以 pad_seg[j+k] 为中心的低通 → 对应全局帧 lo+j
    smooth = np.convolve(pad_seg, kernel, mode="valid")
    # 区间 [start, end) 对应 seg 下标 [start-lo, end-lo)
    i0, i1 = start - lo, end - lo
    raw = np.asarray(vals[start:end], dtype=np.float64)
    resid = np.abs(raw - smooth[i0:i1])
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 100.0 * np.exp(-resid / scale)
    scores = np.where(raw < 0, 0.0, scores)
    return [round(float(s), 1) for s in scores]


# ═══════════════════ Main entry point ═══════════════════

def detect_errors(rows: list, observations: list, times: list[float],
                  max_accel_mps2: float, max_speed_kmh: float,
                  fps: float = 60.0) -> ErrorReport:
    n = len(rows)
    if n == 0:
        return ErrorReport()

    ocr_conf_scores = _signal_ocr_conf(observations, 0, n)
    vals = [r[2] for r in rows]
    abs_scores = _signal_abs_residual(vals, times)
    accel_scores = _signal_accel_spikes(vals, times, fps, max_accel_mps2)
    freq_scores = _signal_frequency(vals, times)
    confidence = _combine_confidence(vals, max_speed_kmh,
                                     ocr_conf_scores, abs_scores,
                                     accel_scores, 0,
                                     freq_scores=freq_scores)
    return ErrorReport(confidence=confidence)


# ═══════════════════ 流式 Phase 1（增量检测器）═══════════════════

class IncrementalDetector:
    """Phase 1 流式计算器：随主 OCR 批处理增量计算局部信号。

    所有信号都是 ±WINDOW 帧的局部计算（abs 期望 ±0.25s + 带宽 ±7 帧、
    accel 尖峰搜索 15 帧）→ 帧 i 在 i+WINDOW 帧就绪后可精确计算，
    与整段 detect_errors 数值一致（同公式同输入，仅遍历区间不同）。

    主 OCR 完成后补算尾部 WINDOW 帧（此时邻居齐全），返回完整 confidence
    —— Phase 1 的墙钟时间全部并进主 OCR 阶段。
    """

    # accel 搜索 15 帧 + 差分 1 帧；abs 期望 ±0.25s + 带宽 ±7 帧（按 fps 换算）
    WINDOW = max(ACCEL_SPIKE_SEARCH_WINDOW + 1,
                 int(LINEARITY_TIME_WINDOW * 60.0) + 1,
                 ABS_RESID_WINDOW // 2 + 1)

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
        abs_s = _signal_abs_residual(self._vals, self._times, start, end)
        acc_s = _accel_scores_range(self._violation, self._vsign, start, end)
        freq_s = _signal_frequency(self._vals, self._times, start, end)
        self._conf_cache.extend(
            _combine_confidence(self._vals, self._max_speed,
                                ocr_s, abs_s, acc_s, start,
                                freq_scores=freq_s))
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
            abs_s = _signal_abs_residual(self._vals, self._times, start, n)
            acc_s = _accel_scores_range(self._violation, self._vsign,
                                        start, n)
            freq_s = _signal_frequency(self._vals, self._times, start, n)
            self._conf_cache.extend(
                _combine_confidence(self._vals, self._max_speed,
                                    ocr_s, abs_s, acc_s, start,
                                    freq_scores=freq_s))
            self._sig_end = n
        return self._conf_cache


def _combine_confidence(vals: list, max_speed_kmh: float,
                        ocr_conf_scores: list, abs_scores: list,
                        accel_scores: list,
                        base_index: int,
                        freq_scores: list | None = None) -> list[dict]:
    """多信号加权组合 → 单帧置信度（逐帧独立，可增量）。

    vals 为全局速度数组，base_index 为信号区间起始帧号（对应返回值的
    index 字段）。freq_scores 缺省时按 50（中性）处理。
    """
    if freq_scores is None:
        freq_scores = [50.0] * len(ocr_conf_scores)
    total_w = (ERROR_DETECT_OCR_CONF_WEIGHT + ERROR_DETECT_ABS_WEIGHT +
               ERROR_DETECT_ACCEL_SPIKE_WEIGHT + ERROR_DETECT_FREQ_WEIGHT)
    w_ocr = ERROR_DETECT_OCR_CONF_WEIGHT / total_w
    w_abs = ERROR_DETECT_ABS_WEIGHT / total_w
    w_acc = ERROR_DETECT_ACCEL_SPIKE_WEIGHT / total_w
    w_freq = ERROR_DETECT_FREQ_WEIGHT / total_w

    confidence: list[dict] = []
    for k, i in enumerate(range(base_index, base_index + len(ocr_conf_scores))):
        score = (w_ocr * ocr_conf_scores[k] + w_abs * abs_scores[k] +
                 w_acc * accel_scores[k] + w_freq * freq_scores[k])
        score = round(max(0.0, min(100.0, score)), 1)

        min_sig = min(abs_scores[k], accel_scores[k])
        for threshold, cap in sorted(ERROR_DETECT_FLOOR_CAP.items()):
            if min_sig < threshold:
                score = min(score, cap)
                break

        # 协同频域 floor：freq 分数低仅当 min(abs,accel) 也低（确实可疑）
        # 才压顶 —— 避免真实高频变速（起步/刹车）被 freq 误伤。
        if freq_scores[k] < FREQ_FLOOR_THRESHOLD and min_sig < FREQ_CORROBORATE_THRESHOLD:
            score = min(score, FREQ_FLOOR_CAP)

        if score < CONF_TIER_LOW_MAX: tier = "low"
        elif score < CONF_TIER_MEDIUM_MAX: tier = "medium"
        else: tier = "high"

        signals_sorted = sorted(
            [("ocr_conf", ocr_conf_scores[k]), ("abs", abs_scores[k]),
             ("accel", accel_scores[k]), ("freq", freq_scores[k])],
            key=lambda x: x[1])
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
                "abs": round(abs_scores[k], 1),
                "accel": round(accel_scores[k], 1),
                "freq": round(freq_scores[k], 1),
            },
            "is_corrected": False,
        })
    return confidence
