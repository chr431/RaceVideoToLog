"""Dense-lattice continuous DP — production Viterbi replacement.

States are every integer km/h in [0, max_speed] instead of the sparse
per-frame candidate sets, so the DP can OUTPUT any value the observation /
transition model supports — not just values some OCR pass produced.  This
removes the measured 97% failure mode of the old discrete Viterbi (truth
not in candidate set → could only pick least-bad wrong candidate).

A/B verdict (v2.12) vs discrete trellis, with REF_MIN_DIFF=6:
  test6 23424 → 23434 (+10), test.mp4 3486 → 3504 (+18), test4 6079 → 6085 (+6)

Two guards are load-bearing — without them dense REGRESSES:

1. Single-candidate pin.  correction's anchor stage sets frames where raw
   agrees with interpolation to a single candidate.  The discrete trellis
   could only choose that value; dense must treat it as a hard anchor too,
   else the 401-state lattice finds near-equal-cost paths and flaps the
   correct raw ±1-2 (measured -354 frames on test6 without the pin).

2. min-obs (raw/reference either-near-is-cheap).  If the reference replaces
   raw as the zero-cost point, island interpolation that lags a correct raw
   by ±3-5 drags it off (measured 60 frames broke on test4).  min-obs keeps
   a peak at BOTH raw and ref, so correct raws are protected while true
   misreads (ref far from raw) are still pulled to the ref.

Observation/transition formulas mirror the old viterbi.py (peak at effective
raw/reference; quadratic excess beyond max_accel*dt) so the only change is
the state space plus the two guards.
"""
from __future__ import annotations

import os

import numpy as np

from config import (
    MPS_TO_KMH,
    VITERBI_OBS_WEIGHT, VITERBI_ACCEL_WEIGHT,
    VITERBI_TRUSTED_BOUNDARY_CONFIDENCE, VITERBI_FALLBACK_DT,
    VITERBI_ANCHOR_COST, VITERBI_CHANGE_THRESHOLD_KMH,
    VITERBI_OBS_COST_FALLBACK_MULT, DENSE_PIN_REF_MAX_DIFF,
)
from viterbi import _split_segments, _compute_confidence_scores

DENSE_DUMP = os.environ.get("RVTOL_DENSE_DUMP", "0") == "1"


def dense_viterbi(
    rows: list,
    candidates_by_frame: dict[int, list[float]],
    confidence_scores: list[dict],
    times: list[float],
    max_speed_kmh: float,
    max_accel_mps2: float,
    obs_weight: float = VITERBI_OBS_WEIGHT,
    accel_weight: float = VITERBI_ACCEL_WEIGHT,
    trusted_boundary_threshold: int = VITERBI_TRUSTED_BOUNDARY_CONFIDENCE,
    trusted_indices: set[int] | None = None,
    reference_values: dict[int, float] | None = None,
) -> dict:
    n = len(rows)
    if n == 0:
        return {'corrected': {}, 'confidence': [], 'error_set': set(), 'dp_cost': []}

    conf_by_idx: dict[int, float] = {c['index']: c['score'] for c in confidence_scores}
    segments = _split_segments(n, conf_by_idx, trusted_boundary_threshold, trusted_indices)

    corrected: dict[int, float] = {}
    error_set: set[int] = set()
    dp_cost: list[float] = [-1.0] * n

    for seg_start, seg_end in segments:
        _dense_segment(seg_start, seg_end, rows, candidates_by_frame, conf_by_idx,
            times, max_speed_kmh, max_accel_mps2, obs_weight, accel_weight,
            trusted_boundary_threshold, corrected, error_set, dp_cost,
            reference_values=reference_values)

    confidence = _compute_confidence_scores(n, dp_cost, rows, conf_by_idx, max_speed_kmh)
    return {'corrected': corrected, 'confidence': confidence,
            'error_set': error_set, 'dp_cost': dp_cost}


def _dense_segment(
    seg_start: int, seg_end: int, rows: list,
    candidates_by_frame: dict[int, list[float]], conf_by_idx: dict[int, float],
    times: list[float], max_speed_kmh: float, max_accel_mps2: float,
    obs_weight: float, accel_weight: float, trusted_boundary_threshold: int,
    corrected: dict[int, float], error_set: set[int], dp_cost: list[float],
    reference_values: dict[int, float] | None = None,
) -> None:
    n_seg = seg_end - seg_start + 1
    if n_seg < 2:
        return

    V = int(max_speed_kmh) + 1
    grid = np.arange(V, dtype=np.float64)

    raw_vals = [rows[seg_start + k][2] for k in range(n_seg)]
    is_anchor = [conf_by_idx.get(seg_start + k, 50) >= trusted_boundary_threshold
                 for k in range(n_seg)]
    eff_ref = [(reference_values.get(seg_start + k, 0.0) if reference_values else 0.0)
               for k in range(n_seg)]

    obs_list: list[np.ndarray] = []
    for k in range(n_seg):
        fi = seg_start + k
        if is_anchor[k]:
            o = np.full(V, np.inf, dtype=np.float64)
            v_raw = raw_vals[k]
            if 0 <= v_raw < V:
                o[int(round(v_raw))] = VITERBI_ANCHOR_COST
        elif len(candidates_by_frame.get(fi, [])) == 1:
            v_pin = candidates_by_frame[fi][0]
            ref = eff_ref[k]
            if ref > 0 and abs(ref - v_pin) > DENSE_PIN_REF_MAX_DIFF:
                # 钉死守卫 2：conf-gated 参考与钉死值严重矛盾（>30）→ 该帧
                # 是物理自洽的误读簇被 anchor 阶段误钉死（test5 5265/5267
                # raw=11 钉死，DP 被迫穿过 11 拖垮整簇 118 坡）。解除钉死
                # 交给 DP 修正；test4 阶梯坡正确帧 ref 偏 ≤18 不受影响。
                o = _dense_obs(grid, raw_vals[k], eff_ref[k], obs_weight, V)
            else:
                # 单候选钉死（守卫 1）：correction 的 anchor 阶段把「raw 与
                # 插值自洽」的帧设成单候选。离散 trellis 只能选该值；稠密若
                # 不钉死，401 状态格点会找到近等代价路径，把正确 raw 抖动
                # ±1-2（实测无此守卫 test6 -354 帧）。
                o = np.full(V, np.inf, dtype=np.float64)
                if 0 <= v_pin < V:
                    o[int(round(v_pin))] = VITERBI_ANCHOR_COST
        else:
            o = _dense_obs(grid, raw_vals[k], eff_ref[k], obs_weight, V)
        obs_list.append(o)

    dp = obs_list[0].copy()
    dp_all: list[np.ndarray] = [dp]
    back_all: list[np.ndarray] = [np.full(V, -1, dtype=np.int64)]

    for k in range(1, n_seg):
        fi = seg_start + k
        dt = times[fi] - times[fi - 1] if fi > 0 else 1.0
        if dt <= 0:
            dt = VITERBI_FALLBACK_DT
        max_dv = max_accel_mps2 * dt * MPS_TO_KMH
        # O(V) 凸变换：T[v] = min_w (dp[w] + accel_weight*max(0, |v-w|-max_dv)^2)
        # 等价于原 O(V²) 向量化（argmin 语义，ties 偏向小 w），但每帧 O(V)。
        # V=401 实测 688µs vs O(V²) numpy 1068µs。
        best_cost, best_idx = _minplus_trunc(dp, max_dv, accel_weight)
        dp = best_cost + obs_list[k]
        dp_all.append(dp)
        back_all.append(best_idx)

    best_final = int(np.argmin(dp_all[n_seg - 1]))
    path = np.zeros(n_seg, dtype=np.float64)
    cur = best_final
    for k in range(n_seg - 1, -1, -1):
        path[k] = grid[cur]
        cur = int(back_all[k][cur])

    for k in range(n_seg):
        fi = seg_start + k
        optimal = path[k]
        raw_v = rows[fi][2]
        dp_cost[fi] = float(np.min(dp_all[k]))
        if DENSE_DUMP and abs(optimal - raw_v) > 2:
            print(f"DUMP {fi}: raw={raw_v:.1f} ref={eff_ref[k]:.1f} "
                  f"conf={conf_by_idx.get(fi, 50):.0f} out={optimal:.1f}")
        if abs(optimal - raw_v) > VITERBI_CHANGE_THRESHOLD_KMH:
            corrected[fi] = float(optimal)
            error_set.add(fi)


# ═══════════════════ O(V) 凸变换（min-plus convolution，替换 O(V²) 转移）═══════════════════

def _sliding_argmin(f: list, r: int) -> tuple[list, list]:
    """窗口最小值 + argmin：win[v] = min_{w in [v-r, v+r]} f[w]，O(V) 单调队列。
    ties 偏向最小 w（与 np.argmin 语义一致）。"""
    n = len(f)
    from collections import deque
    dq: deque = deque()
    vals = [0.0] * n
    idx = [0] * n
    for i in range(min(r + 1, n)):
        while dq and f[dq[-1]] > f[i]:
            dq.pop()
        dq.append(i)
    vals[0] = f[dq[0]]; idx[0] = dq[0]
    for v in range(1, n):
        lo = v - r
        while dq and dq[0] < lo:
            dq.popleft()
        hi = v + r
        if hi < n:
            while dq and f[dq[-1]] > f[hi]:
                dq.pop()
            dq.append(hi)
        vals[v] = f[dq[0]] if dq else float("inf")
        idx[v] = dq[0] if dq else 0
    return vals, idx


def _prefix_quad_argmin(f: list, a: float, x_grid: list) -> tuple[list, list]:
    """P[x] = min_{y<=x} (f[y] + a(x-y)²)，下包络 O(V)。返回 (值, argmin y)。
    x_grid 单调递增（可为浮点）。ties 偏向小 y（下包络保留旧抛物线）。"""
    n = len(f)
    v = [0] * n
    z = [0.0] * (n + 1)
    k = -1          # hull 最后抛物线下标
    ptr = 0         # 上一求值点活动的抛物线
    added = -1
    vals = [0.0] * n
    idx = [0] * n
    inf = float("inf")
    for xi, x in enumerate(x_grid):
        while added + 1 < n and added + 1 <= x:
            q = added + 1
            added = q
            if f[q] == inf:
                continue  # 不可达状态（dp=inf）永不成为 min —— 跳过，防 inf-inf=nan
            if k < 0:
                k = 0; v[0] = q; z[0] = -inf; z[1] = inf
            else:
                fq = f[q]
                vk = v[k]
                s = ((fq + a * q * q) - (f[vk] + a * vk * vk)) / (2.0 * a * (q - vk))
                while s < z[k]:
                    k -= 1
                    if k < 0:
                        k = 0; v[0] = q; z[0] = -inf; z[1] = inf
                        break
                    vk = v[k]
                    s = ((fq + a * q * q) - (f[vk] + a * vk * vk)) / (2.0 * a * (q - vk))
                else:
                    k += 1; v[k] = q; z[k] = s; z[k + 1] = inf
        if k < 0:
            vals[xi] = inf; idx[xi] = 0
            continue
        if ptr > k:
            ptr = k
        while ptr < k and z[ptr + 1] < x:
            ptr += 1
        d = x - v[ptr]
        vals[xi] = f[v[ptr]] + a * d * d
        idx[xi] = v[ptr]
    return vals, idx


def _suffix_quad_argmin(f: list, a: float, x_grid: list) -> tuple[list, list]:
    """S[x] = min_{y>=x} (f[y] + a(y-x)²)，翻转转成 prefix。返回 (值, argmin y)。"""
    n = len(f)
    fr = f[::-1]
    xr = [(n - 1) - x for x in x_grid[::-1]]
    vals, idx = _prefix_quad_argmin(fr, a, xr)
    return vals[::-1], [n - 1 - i for i in idx[::-1]]


def _minplus_trunc(f: list, m: float, a: float) -> tuple[np.ndarray, np.ndarray]:
    """T[v] = min_w (f[w] + a*max(0, |v-w|-m)²)，O(V)。返回 (值, argmin w)。

    截断二次核拆三路：窗口平坦区（|v-w|<=m → cost 0）+ 两侧二次尾。
    窗口半径 r=floor(m)：整数 w 满足 |v-w|<=m ⟺ w ∈ [v-r, v+r]。
    二次尾：w<=v-m → a(v-m-w)²（prefix），w>=v+m → a(w-v-m)²（suffix）。
    """
    n = len(f)
    r = int(m)
    win_v, win_i = _sliding_argmin(f, r)
    p_v, p_i = _prefix_quad_argmin(f, a, [i - m for i in range(n)])
    s_v, s_i = _suffix_quad_argmin(f, a, [i + m for i in range(n)])
    vals = [0.0] * n
    idx = [0] * n
    for v in range(n):
        bv, bi = win_v[v], win_i[v]
        for cv, ci in ((p_v[v], p_i[v]), (s_v[v], s_i[v])):
            if cv < bv or (cv == bv and ci < bi):
                bv, bi = cv, ci
        vals[v] = bv
        idx[v] = bi
    return np.asarray(vals, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def _dense_obs(grid: np.ndarray, raw_v: float, ref_v: float, obs_weight: float,
               V: int) -> np.ndarray:
    """min-obs（守卫 2）：raw 与 ref 任一接近即低代价。

    若 ref 无条件替换 raw 作零代价点，滞后 ±3-5 的 island 插值会把正确
    raw 拖偏（实测 test4 60 帧破坏）。min 保留两峰：正确 raw 受保护，
    真误读帧（ref 远偏离 raw）仍被 ref+邻居拉正。
    """
    eff = ref_v if ref_v is not None and ref_v > 0 else raw_v
    if eff > 0:
        ratio = np.abs(grid - eff) / max(1.0, abs(eff))
    else:
        ratio = np.abs(grid - eff) * VITERBI_OBS_COST_FALLBACK_MULT
    np.minimum(1.0, ratio, out=ratio)
    o = obs_weight * ratio
    if ref_v is not None and ref_v > 0 and raw_v > 0:
        ratio_raw = np.abs(grid - raw_v) / max(1.0, abs(raw_v))
        np.minimum(1.0, ratio_raw, out=ratio_raw)
        np.minimum(o, obs_weight * ratio_raw, out=o)
    return o
