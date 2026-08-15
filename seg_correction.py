"""段级检测 / 置信度 / 稠密 DP 纠正的纯函数实现。

与 SegmentPipeline 解耦：所有参数显式传入（默认值来自 config），
便于单测与扫参脚本复用。生产路径 run() 通过 SegmentPipeline 的
同名方法调用本模块。
"""
from __future__ import annotations

import numpy as np

import config


def local_bandwidth(seg_vals: list, seg_times: list, win: int) -> list:
    """每段局部带宽：帧窗口内相邻差绝对值的中位数（未加 floor）。

    供 detect_segments 与 confidence_scores 共用（两者原为逐行相同的重复循环）。
    """
    n = len(seg_vals)
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(win * max(med_gap, 1.0), config.SEG_WIN_MAX_FRAMES)
    st = np.asarray(seg_times, dtype=np.float64)
    bw_raw = [0.0] * n
    for i in range(n):
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        bw_raw[i] = float(np.median(dvs)) if dvs else 0.0
    return bw_raw


def detect_segments(seg_vals: list, seg_times: list, seg_lens=None, *,
                    med_k: int = config.SEG_MED_K,
                    mult: float = config.SEG_MULT,
                    detect_floor: float = config.SEG_DETECT_FLOOR,
                    single_floor: float = config.SEG_SINGLE_FLOOR,
                    win: int = config.SEG_WIN) -> list:
    """中值滤波检测：平滑值曲线（跟随弯曲），误读=尖峰被中值剔除。

    对每段 i，smoothed = 局部非 None 值的中位数（段索引窗口 ±med_k）。
    正确段贴合中值（偏差 ≤ 局部带宽），误读尖峰偏差大。门限 =
    max(局部相邻差中位数, floor) × mult。边缘段（左右一侧无上下文）
    不 flag —— 中值在单调上升/下降区滞后，视频起止的低/高速段会被窗口
    拉偏误判。None 段恒 suspect。

    单帧段（seg_lens[i]==1）用更紧 floor：误读率 4.2% vs 多帧 0.3%
    （12.6×，80% 误读是单帧段——过渡/模糊帧难 OCR 又易成单帧段），
    平缓区门限降到 4 抓小偏差误读，弯曲区仍按实际带宽放宽。
    """
    n = len(seg_vals)
    bw_raw = local_bandwidth(seg_vals, seg_times, win)
    suspect = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
        if len(nbrs) < config.SEG_CONF_MIN_NEIGHBORS:
            suspect[i] = True
            continue
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            continue
        med = float(np.median(nbrs))
        floor = single_floor if (seg_lens and seg_lens[i] == 1) else detect_floor
        if abs(seg_vals[i] - med) > max(bw_raw[i], floor) * mult:
            suspect[i] = True
    return suspect


def correct_segments(seg_vals: list, seg_times: list, suspect: list, *,
                     anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES,
                     min_dev: float = config.SEG_MIN_DEV) -> tuple[list, int]:
    """锚点插值纠正：suspect/None 段取最近可信锚点线性插值。

    anchor_max 限锚点帧距离：近锚点（≤anchor_max 帧）才插值，防远锚点
    跨弯曲区误插值（低 min_dev 下过度纠正的回归源）。None 段恒插值。
    """
    out = list(seg_vals)
    n_corr = 0
    for i in range(len(seg_vals)):
        if seg_vals[i] is None:
            # None 段（OCR 未读出）→ 必须插值，否则帧输出 -1
            pass
        elif not suspect[i]:
            continue
        ti = seg_times[i]
        la = None
        for j in range(i - 1, -1, -1):
            if not suspect[j] and seg_vals[j] is not None:
                if ti - seg_times[j] <= anchor_max:
                    la = j
                break
        ra = None
        for j in range(i + 1, len(seg_vals)):
            if not suspect[j] and seg_vals[j] is not None:
                if seg_times[j] - ti <= anchor_max:
                    ra = j
                break
        interp = None
        if la is not None and ra is not None:
            span = seg_times[ra] - seg_times[la]
            frac = (ti - seg_times[la]) / span if span > 1e-3 else 0.5
            interp = seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac
        elif la is not None:
            interp = seg_vals[la]
        elif ra is not None:
            interp = seg_vals[ra]
        if interp is not None:
            if seg_vals[i] is None or abs(interp - seg_vals[i]) > min_dev:
                out[i] = round(interp)
                n_corr += 1
    return out, n_corr


def _consistent_run_bounds(seg_vals: list, seg_lens, i: int,
                           tol: float = 0.0) -> tuple:
    """连续近似相同值区间：返回 (l, r, 累计帧数, 区间中值)。

    tol=0 时退化为“完全相同”；tol>0 时允许区间内 max-min ≤ tol 的小波动，
    用于识别 127,128 这类内部有轻微 OCR 波动的一致性孤岛。
    seg_lens 缺省时按每段 1 帧计。
    """
    v = seg_vals[i]
    if v is None:
        return i, i, 0, None
    n = len(seg_vals)
    l = r = i
    lo = hi = v
    total = int(seg_lens[i]) if seg_lens is not None and i < len(seg_lens) else 1
    j = i - 1
    while j >= 0 and seg_vals[j] is not None:
        cand = seg_vals[j]
        new_lo = min(lo, cand)
        new_hi = max(hi, cand)
        if new_hi - new_lo <= tol:
            l = j
            lo, hi = new_lo, new_hi
            total += int(seg_lens[j]) if seg_lens is not None and j < len(seg_lens) else 1
            j -= 1
        else:
            break
    j = i + 1
    while j < n and seg_vals[j] is not None:
        cand = seg_vals[j]
        new_lo = min(lo, cand)
        new_hi = max(hi, cand)
        if new_hi - new_lo <= tol:
            r = j
            lo, hi = new_lo, new_hi
            total += int(seg_lens[j]) if seg_lens is not None and j < len(seg_lens) else 1
            j += 1
        else:
            break
    run_vals = [seg_vals[k] for k in range(l, r + 1) if seg_vals[k] is not None]
    run_med = float(np.median(run_vals)) if run_vals else None
    return l, r, total, run_med


def _consistent_run_frames(seg_vals: list, seg_lens, i: int,
                           tol: float = 0.0) -> int:
    """连续近似相同值的累计帧数（一致性孤岛长度）。

    tol=0 时只看完全相同；tol>0 时允许区间内小波动。
    """
    return _consistent_run_bounds(seg_vals, seg_lens, i, tol)[2]


def confidence_scores(seg_vals: list, seg_times: list, seg_lens=None, *,
                      med_k: int = config.SEG_MED_K,
                      detect_floor: float = config.SEG_DETECT_FLOOR,
                      conf_jerk_scale: float = config.SEG_CONF_JERK_SCALE,
                      conf_w_med: float = config.SEG_CONF_W_MED,
                      conf_w_jerk: float = config.SEG_CONF_W_JERK,
                      win: int = config.SEG_WIN,
                      island_tol: float = config.SEG_CONF_ISLAND_TOL) -> list:
    """中值偏差 + 急动度加权置信度 [0,100]（门控急动度）。

    med_score = 100·exp(-dev/bw)：贴合曲线程度。**门控**：贴合曲线
    （med_score ≥ 50）的段 conf 直接取中值分——急动度会被邻居误读污染，
    贴合曲线的正确段不应被拉低。偏离曲线的段才让急动度参与：刹车（平滑）
    高分、误读（尖锐）低分。邻居不足的短上下文段保守取 30（单测口径）；
    真正缺少左右上下文的边缘段保守 100；None 段 0（必纠正）。
    """
    n = len(seg_vals)
    bw_raw = local_bandwidth(seg_vals, seg_times, win)
    conf = [0.0] * n
    # 结构性分支（邻居不足/边缘）不参与“一致性孤岛”封顶：它们已有
    # 各自的保守/信任语义，且视频起止的短段不应被误伤。
    island_cap_eligible = [True] * n
    for i in range(n):
        if seg_vals[i] is None:
            conf[i] = 0.0
            island_cap_eligible[i] = False
            continue
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
        if len(nbrs) < config.SEG_CONF_MIN_NEIGHBORS:
            conf[i] = config.SEG_CONF_SHORT_NEIGHBOR
            island_cap_eligible[i] = False
            continue
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            conf[i] = config.SEG_CONF_EDGE
            island_cap_eligible[i] = False
            continue
        med = float(np.median(nbrs))
        dev = abs(seg_vals[i] - med)
        bw = max(bw_raw[i], detect_floor)
        med_score = 100.0 * np.exp(-dev / bw)
        # 贴合曲线 → 直接中值分（忽略被污染的急动度）
        if med_score >= config.SEG_CONF_MED_GATE:
            conf[i] = med_score
            continue
        # 偏离曲线 → 急动度分辨刹车 vs 误读
        jl = seg_vals[i - 1] if i - 1 >= 0 else None
        jr = seg_vals[i + 1] if i + 1 < n else None
        if jl is not None and jr is not None:
            jerk = abs(jr - 2 * seg_vals[i] + jl)
            jerk_score = 100.0 * np.exp(-jerk / conf_jerk_scale)
            conf[i] = (conf_w_med * med_score + conf_w_jerk * jerk_score)
        else:
            conf[i] = med_score
    # 一致性孤岛封顶：连续“近似相同”值累计帧数太少、且该值明显脱离 run
    # 外邻居的中值（局部曲线）时，即使原 conf 高分也不能作为 HIGH_TRUST /
    # DP 锚点。tol=0 时与旧逻辑完全一致；tol>0 时把 127,128 这类内部有
    # 小波动的短孤岛也识别出来，并对整个 run 一起封顶，防止互相撑腰。
    # 坡道上的正常短段（如 315 夹在 311/318 之间）偏差小，不封顶。
    for i in range(n):
        if seg_vals[i] is None or not island_cap_eligible[i]:
            continue
        l, r, run_frames, run_med = _consistent_run_bounds(
            seg_vals, seg_lens, i, island_tol)
        if run_frames >= config.SEG_CONF_MIN_CONSISTENT_FRAMES:
            continue
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        # 找出连续近似相同值区间 [l, r]，只取区间外的邻居评估“曲线支持”
        outside = [seg_vals[j] for j in range(lo, hi)
                   if seg_vals[j] is not None and (j < l or j > r)]
        if len(outside) < config.SEG_CONF_MIN_NEIGHBORS:
            continue
        outside_med = float(np.median(outside))
        dev = abs(run_med - outside_med) if run_med is not None else 0.0
        if dev > max(bw_raw[i], detect_floor) \
                * config.SEG_CONF_ISLAND_DEV_MULT:
            cap = config.SEG_CONF_SHORT_RUN_CAP
            for j in range(l, r + 1):
                if seg_vals[j] is not None and island_cap_eligible[j]:
                    conf[j] = min(conf[j], cap)
    return conf


def fill_values(seg_vals: list, seg_times: list, is_anchor: list, *,
                anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES) -> list:
    """每段的局部锚点插值（最近左右锚点的时间线性插值）。

    非锚点段（suspect，可能是误读）的观测目标用锚点插值——贴合局部曲线，
    而非错误 raw 或污染的中值窗口。
    """
    n = len(seg_vals)
    fill: list = [None] * n
    for i in range(n):
        la = None
        for j in range(i - 1, -1, -1):
            if is_anchor[j] and seg_vals[j] is not None:
                if seg_times[i] - seg_times[j] <= anchor_max:
                    la = j
                break
        ra = None
        for j in range(i + 1, n):
            if is_anchor[j] and seg_vals[j] is not None:
                if seg_times[j] - seg_times[i] <= anchor_max:
                    ra = j
                break
        if la is not None and ra is not None:
            span = seg_times[ra] - seg_times[la]
            frac = (seg_times[i] - seg_times[la]) / span if span > 1e-3 else 0.5
            fill[i] = (seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac)
        elif la is not None:
            fill[i] = seg_vals[la]
        elif ra is not None:
            fill[i] = seg_vals[ra]
    return fill


def dp_run(lo: int, hi: int, seg_vals: list, seg_times: list, is_anchor: list,
           max_speed: float, max_accel: float, fps: float, *,
           dp_obs_weight: float = config.SEG_DP_OBS_WEIGHT,
           dp_accel_weight: float = config.SEG_DP_ACCEL_WEIGHT,
           dp_max_dv_cap: float = config.SEG_DP_MAX_DV_CAP,
           dp_anchor_cost: float = config.SEG_DP_ANCHOR_COST,
           fill: list | None = None) -> np.ndarray:
    """稠密 DP：观测=罚偏离 raw（无效 raw 填向局部锚点插值），转移=加速度约束。"""
    V = int(max_speed) + 1
    grid = np.arange(V, dtype=np.float64)
    n = hi - lo + 1
    obs_list = []
    for k in range(lo, hi + 1):
        v = seg_vals[k]
        if is_anchor[k]:
            o = np.full(V, np.inf)
            if v is not None and 0 <= v < V:
                o[int(round(v))] = dp_anchor_cost
            obs_list.append(o)
        else:
            o = np.full(V, dp_obs_weight)
            # 非锚点（suspect）：填向局部锚点插值（fill）——错误 raw 不可信，
            # 观测目标应是曲线；无 fill 时回退到 raw。
            r = fill[k] if fill else None
            if r is not None and r > 0:
                ratio = np.abs(grid - r) / max(1.0, abs(r))
                np.minimum(1.0, ratio, out=ratio)
                o = dp_obs_weight * ratio
            elif v is not None and v > 0:
                ratio = np.abs(grid - v) / max(1.0, abs(v))
                np.minimum(1.0, ratio, out=ratio)
                o = dp_obs_weight * ratio
            obs_list.append(o)
    dp = obs_list[0].copy()
    back = []
    for k in range(1, n):
        fi = lo + k
        dt = (seg_times[fi] - seg_times[fi - 1]) / max(fps, 1.0)
        if dt <= 0:
            dt = 1.0 / max(fps, 1.0)
        max_dv = min(max_accel * dt * config.MPS_TO_KMH, dp_max_dv_cap)
        # O(V²) 转移：T[v] = min_w (dp[w] + accel*max(0,|v-w|-max_dv)^2)
        w = grid[:, None]
        vv = grid[None, :]
        cost = (dp_accel_weight
                * np.maximum(0.0, np.abs(vv - w) - max_dv) ** 2)
        T = dp[:, None] + cost
        best = T.min(axis=0)
        back.append(T.argmin(axis=0))
        dp = best + obs_list[k]
    path = np.zeros(n)
    cur = int(np.argmin(dp))
    for k in range(n - 1, -1, -1):
        path[k] = grid[cur]
        if k > 0:
            cur = int(back[k - 1][cur])
    return path


def dense_correct(seg_vals: list, seg_times: list, conf: list, *,
                  max_speed: float, max_accel: float, fps: float,
                  anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES,
                  dp_anchor_conf: float = config.SEG_DP_ANCHOR_CONF,
                  dp_deanchor_jerk_min: float = config.SEG_DP_DEANCHOR_JERK_MIN,
                  dp_deanchor_jerk_max: float = config.SEG_DP_DEANCHOR_JERK_MAX,
                  dp_change_threshold: float = config.SEG_DP_CHANGE_THRESHOLD,
                  **dp_kwargs) -> tuple[list, int]:
    """段级稠密格点 DP 纠正（对齐旧 viterbi_dense，无 ref）。

    锚点 = conf ≥ SEG_DP_ANCHOR_CONF 的段（门控 conf 后正确段高分可靠锚定）。
    其余段跑 DP：观测 = 纯惩罚偏离 raw（重 OCR 已删 → ref 删除；观测的意义
    是惩罚改动，防把正确的改错），转移 = 加速度约束。
    """
    n = len(seg_vals)
    out = list(seg_vals)
    n_corr = 0
    # 无效 raw（None/0）段的填充目标：局部锚点插值（非 ±10 中值，后者在
    # 加速斜坡区被污染）
    is_anchor = [c >= dp_anchor_conf and v is not None
                 for c, v in zip(conf, seg_vals)]
    # 孤立尖峰豁免（A4）：conf∈[20,50) 的锚定段若 jerk（二阶差分）中等
    # （孤立尖峰误读特征；真刹车 jerk≈0、丢位邻居污染 jerk≥80）→ 解除
    # 锚定交给 DP，防误读被锚定保留（实测 13→12 零误改，参数见 config）
    if dp_deanchor_jerk_min > 0:
        for i in range(1, n - 1):
            if not is_anchor[i] or conf[i] >= config.SEG_DP_DEANCHOR_CONF_MAX \
                    or seg_vals[i] is None:
                continue
            jl, jr = seg_vals[i - 1], seg_vals[i + 1]
            if jl is None or jr is None:
                continue
            jerk = abs(jr - 2 * seg_vals[i] + jl)
            if dp_deanchor_jerk_min <= jerk <= dp_deanchor_jerk_max:
                is_anchor[i] = False
    fill = fill_values(seg_vals, seg_times, is_anchor, anchor_max=anchor_max)
    i = 0
    while i < n:
        if is_anchor[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not is_anchor[j + 1]:
            j += 1
        lo = i - 1 if i > 0 else i
        hi = j + 1 if j + 1 < n else j
        if lo != hi or not is_anchor[lo]:
            path = dp_run(lo, hi, seg_vals, seg_times, is_anchor,
                          max_speed, max_accel, fps, fill=fill, **dp_kwargs)
            for k in range(hi - lo + 1):
                idx = lo + k
                v = seg_vals[idx]
                val = float(path[k])
                if v is None:
                    out[idx] = round(val)
                    n_corr += 1
                elif abs(val - v) > dp_change_threshold:
                    out[idx] = round(val)
                    n_corr += 1
        i = j + 1
    return out, n_corr
