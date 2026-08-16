"""实验：加速度一致性"节"重建纠错（用户提出的新后处理算法）。

思路（按用户描述原样实现）：
1. OCR 完成后对所有段分节：相邻两节首尾重合一段（如 seg 1,2,3,4 →
   1,2,3 + 3,4）；节内任意两段间加速度差 ≤ max_accel_diff；每节计算
   mean_accel。
2. 节长（实际包含帧数）≥ min_length 且 |mean_accel| ≤ max_accel 的节
   标记可信，其中所有段被锚定、后续不可修改。
3. 所有可信节按自身 mean_accel 左右延长并互相相交 → 恢复曲线轮廓；
   未被锚定的段与重建曲线比较：差距 > gap_threshold 则填入重建值，
   否则保留原值。
4. 与当前中值滤波+DP 的比较：当前算法中值窗口宽度难以确定（太小被带偏、
   太宽无法应对加减速/转折）；本算法等价于可变窗口的中值滤波。
5. 段间加速度：段视为中点一个点，Δ帧/fps 得时间差（km/h/s 单位）。

数据：tests/fixtures/seg_series/*.json —— 生产 run() 的全量段级序列
（含 raw/corr/truth/mid/len/rep 与 meta fps/max_speed/max_accel），
离线扫参，不重跑解码+OCR（一次生产 run 的中间产物，与漏斗口径一致）。

评估口径与 tools/accuracy_breakdown.py 一致：|输出-truth|>tol 计错，
raw=None 或 truth=None 的段跳过。夹具 corr 字段复现基线 11 错误
（test 3 / test2 8 / test3/5/6 0）。

用法：
  python tools/archive/proto_accel_section.py --sweep        # 扫参
  python tools/archive/proto_accel_section.py --detail       # 最优配置逐案例
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

FIXTURE_DIR = PROJECT / "tests" / "fixtures" / "seg_series"
VIDEOS = ["test", "test2", "test3", "test5", "test6"]
TOL = 1.0
MPS_TO_KMH = 3.6


# ────────────────────────────── 算法核心 ──────────────────────────────

def pair_accels(vals, mids, fps):
    """段间加速度（km/h/s）：段视为中点一个点，(Δ帧/fps) 为时间差。"""
    n = len(vals)
    a = [None] * n          # a[i] = seg i-1 → i 的加速度；a[0] 不用
    for i in range(1, n):
        v0, v1 = vals[i - 1], vals[i]
        if v0 is None or v1 is None:
            continue
        dt = (mids[i] - mids[i - 1]) / fps
        if dt <= 0:
            dt = 1.0 / fps
        a[i] = (v1 - v0) / dt
    return a


def build_sections(vals, mids, fps, max_accel_diff):
    """贪心分节：节 [l,r] 内全部相邻对加速度 max-min ≤ max_accel_diff；
    扩展失败闭合后，下一节从 r 开始（首尾重合一段，即用户例子的 1,2,3+3,4）。
    None 段断节（None 无法参与加速度计算）。返回 (sections, a)。
    """
    n = len(vals)
    a = pair_accels(vals, mids, fps)
    sections = []
    l = 0
    while l < n - 1:
        if vals[l] is None or a[l + 1] is None:
            l += 1
            continue
        r = l + 1
        lo_a = hi_a = a[l + 1]
        while r + 1 < n:
            cand = a[r + 1]
            if cand is None:
                break
            nlo, nhi = min(lo_a, cand), max(hi_a, cand)
            if nhi - nlo > max_accel_diff:
                break
            lo_a, hi_a = nlo, nhi
            r += 1
        sections.append((l, r))
        l = r               # 首尾重合一段
    return sections, a


def trusted_sections(sections, a, seg_lens, min_length, max_accel_kmhs,
                     min_pairs=1):
    """长度（实际帧数）≥ min_length 且 |mean_accel| ≤ max_accel 的节为可信。
    返回 [(l, r, mean_accel)]。
    """
    trusted = []
    for (l, r) in sections:
        n_pairs = r - l
        if n_pairs < min_pairs:
            continue
        frames = int(sum(seg_lens[l:r + 1]))
        if frames < min_length:
            continue
        ma = float(np.mean(a[l + 1:r + 1]))
        if abs(ma) > max_accel_kmhs:
            continue
        trusted.append((l, r, ma))
    return trusted


def reconstruct(trusted, n, vals, mids, fps, max_speed):
    """可信节按自身 mean_accel 左右延长并互相相交 → 恢复曲线轮廓。

    每可信节一条线 v(s) = v[l] + ma·(s − s[l])（s 为秒）。
    - 相邻可信节之间：两延长线的交点落在区间内 → 交点前用左节延长线、
      交点后用右节延长线（曲线"相交过渡"）；否则（平行/交点在区间外）
      按两线在 t 处的值线性混合（无交点时的平滑衔接）。
    - 首节之前 / 末节之后：最近节的延长线。
    - 全部 clamp 到 [0, max_speed]。
    返回 rec[i]（无任何可信节时为全 None）。
    """
    lines = [(l, r, mids[l] / fps, float(vals[l]), ma)
             for (l, r, ma) in trusted]
    rec = [None] * n
    if not lines:
        return rec
    s_pts = [m / fps for m in mids]

    def lineval(ln, t):
        return ln[3] + ln[4] * (t - ln[2])

    for i in range(n):
        t = s_pts[i]
        inside = None
        prev = next_ = None
        for ln in lines:
            l, r = ln[0], ln[1]
            if l <= i <= r:
                inside = ln
            elif r < i:
                prev = ln
            elif l > i and next_ is None:
                next_ = ln
        if inside is not None:
            rec[i] = lineval(inside, t)
        elif prev is not None and next_ is not None:
            aA, aB = prev[4], next_[4]
            if abs(aA - aB) > 1e-9:
                tx = (next_[3] - prev[3] + aA * prev[2] - aB * next_[2]) \
                    / (aA - aB)
            else:
                tx = None
            sA_end = mids[prev[1]] / fps
            sB_start = mids[next_[0]] / fps
            if tx is not None and sA_end <= tx <= sB_start:
                rec[i] = lineval(prev, t) if t <= tx else lineval(next_, t)
            else:
                denom = sB_start - sA_end
                w = (t - sA_end) / denom if denom > 1e-9 else 0.5
                rec[i] = (1.0 - w) * lineval(prev, t) + w * lineval(next_, t)
        elif prev is not None:
            rec[i] = lineval(prev, t)
        elif next_ is not None:
            rec[i] = lineval(next_, t)
    return [min(max(x, 0.0), float(max_speed)) if x is not None else None
            for x in rec]


def correct(vals, rec, is_anchor, gap_threshold):
    """锚定段不可改；未锚定段与重建曲线比较：差距 > gap_threshold（或
    None 段）→ 填入重建值，否则保留原值。返回 (out, n_changed)。"""
    out = list(vals)
    n_changed = 0
    for i in range(len(vals)):
        if is_anchor[i]:
            continue
        r = rec[i]
        if r is None:
            continue
        rv = int(round(r))
        if vals[i] is None or abs(vals[i] - rv) > gap_threshold:
            out[i] = rv
            n_changed += 1
    return out, n_changed


def run_algorithm(segs_data, meta, max_accel_diff, min_length,
                  gap_threshold, max_accel_factor=1.0, min_pairs=1):
    """对单个视频跑完整算法，返回 (out, info)。"""
    vals = [s["raw"] for s in segs_data]
    mids = [s["mid"] for s in segs_data]
    lens = [s["len"] for s in segs_data]
    fps = meta["fps"]
    max_accel_kmhs = meta["max_accel"] * MPS_TO_KMH * max_accel_factor
    max_speed = meta["max_speed"]
    sections, a = build_sections(vals, mids, fps, max_accel_diff)
    trusted = trusted_sections(sections, a, lens, min_length,
                               max_accel_kmhs, min_pairs=min_pairs)
    rec = reconstruct(trusted, len(vals), vals, mids, fps, max_speed)
    is_anchor = [False] * len(vals)
    for (l, r, _ma) in trusted:
        for k in range(l, r + 1):
            is_anchor[k] = True
    out, n_changed = correct(vals, rec, is_anchor, gap_threshold)
    info = {
        "sections": len(sections),
        "trusted": len(trusted),
        "anchored": sum(is_anchor),
        "changed": n_changed,
        "rec_none": sum(1 for x in rec if x is None),
    }
    return out, info


# ────────────────────────────── 评估（漏斗口径） ──────────────────────────────

def funnel(segs_data, out):
    """与 accuracy_breakdown 同口径：raw=None 或 truth=None 的段跳过。"""
    seg = raw = fix = fix_wrong = missed = harm = final = 0
    for s, o in zip(segs_data, out):
        t = s["truth"]
        sv = s["raw"]
        if t is None or sv is None:
            continue
        seg += 1
        raw_err = abs(sv - t) > TOL
        if o is None:
            final_err = True
        else:
            final_err = abs(o - t) > TOL
        changed = (o is not None and o != sv)
        if raw_err:
            raw += 1
            if changed:
                if final_err:
                    fix_wrong += 1
                else:
                    fix += 1
            else:
                missed += 1
        else:
            if changed and final_err:
                harm += 1
        if final_err:
            final += 1
    return {"seg": seg, "raw": raw, "fix": fix, "fix_wrong": fix_wrong,
            "missed": missed, "harm": harm, "final": final}


def run_video(v, max_accel_diff, min_length, gap_threshold,
              max_accel_factor=1.0, min_pairs=1):
    d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
    out, info = run_algorithm(d["segments"], d["meta"], max_accel_diff,
                              min_length, gap_threshold, max_accel_factor,
                              min_pairs)
    return funnel(d["segments"], out), info


def run_all(params, videos=VIDEOS):
    per = {}
    tot = {k: 0 for k in ("seg", "raw", "fix", "fix_wrong", "missed",
                           "harm", "final")}
    infos = {}
    for v in videos:
        f, info = run_video(v, *params)
        per[v] = f
        infos[v] = info
        for k in tot:
            tot[k] += f[k]
    return per, tot, infos


def baseline_of(v):
    """夹具 corr（生产 dense_correct 结果）的漏斗。"""
    d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
    return funnel(d["segments"], [s["corr"] for s in d["segments"]])


# ────────────────────────────── 扫参 ──────────────────────────────

def sweep():
    print(f"基线（夹具 corr = 生产 dense_correct）：")
    tot_b = {k: 0 for k in ("seg", "raw", "fix", "fix_wrong", "missed",
                            "harm", "final")}
    for v in VIDEOS:
        f = baseline_of(v)
        for k in tot_b:
            tot_b[k] += f[k]
        print(f"  {v:<6} final={f['final']} "
              f"(raw {f['raw']} / fix {f['fix']} / fix_wrong {f['fix_wrong']}"
              f" / missed {f['missed']} / harm {f['harm']})")
    print(f"  合计 final={tot_b['final']}\n")

    diffs = [0.5, 1.0, 2.0, 4.0, 8.0, 30.0, 60.0, 100.0, 150.0, 200.0]
    min_lens = [5, 10, 20, 40, 80]
    gaps = [2, 3, 4, 6, 8, 12]
    factors = [1.0, 0.5, 0.3, 0.1]
    min_pairss = [1, 2]
    results = []
    for diff in diffs:
        for ml in min_lens:
            for gap in gaps:
                for fac in factors:
                    for mp in min_pairss:
                        per, tot, infos = run_all((diff, ml, gap, fac, mp))
                        results.append((tot["final"], diff, ml, gap, fac, mp,
                                        tot, per))
    results.sort(key=lambda x: (x[0], -x[1], -x[2], x[3]))
    print(f"扫参 {len(diffs)}×{len(min_lens)}×{len(gaps)}×{len(factors)}×"
          f"{len(min_pairss)} = {len(results)} 组，按 total final 升序：")
    print(f"{'final':>5} {'diff':>4} {'minL':>4} {'gap':>3} {'fac':>4} "
          f"{'mp':>2}  每视频 final (test test2 test3 test5 test6)")
    for row in results[:25]:
        final, diff, ml, gap, fac, mp, tot, per = row
        line = " ".join(f"{per[v]['final']:>3}" for v in VIDEOS)
        print(f"{final:>5} {diff:>4} {ml:>4} {gap:>3} {fac:>4} {mp:>2}   "
              f"{line}")
    print()
    best = results[0]
    print(f"最优: diff={best[1]} min_length={best[2]} gap={best[3]}"
          f" max_accel_factor={best[4]} min_pairs={best[5]}"
          f" → total final={best[0]}")
    print(f"最差: {results[-1][1:6]} → total final={results[-1][0]}")
    return best


# ────────────────────────────── 最优配置逐案例 ──────────────────────────────

def detail(params, videos=VIDEOS):
    per, tot, infos = run_all(params, videos)
    diff, ml, gap, fac, mp = params
    print(f"配置 diff={diff} min_length={ml} gap={gap}"
          f" max_accel_factor={fac} min_pairs={mp}"
          f" → total final={tot['final']}（基线 11）")
    print(f"{'视频':<6} {'段':>5} {'原始误读':>6} {'纠对':>4} {'纠错':>4} "
          f"{'漏纠':>4} {'误改':>4} {'最终':>4}  {'节':>4} {'可信':>4} "
          f"{'锚定':>6} {'改动':>5} {'重建无值':>6}")
    for v in videos:
        f = per[v]
        info = infos[v]
        b = baseline_of(v)
        d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
        mark = "◀基线" if b["final"] > 0 else ""
        print(f"{v:<6} {f['seg']:>5} {f['raw']:>6} {f['fix']:>4} "
              f"{f['fix_wrong']:>4} {f['missed']:>4} {f['harm']:>4} "
              f"{f['final']:>4}  {info['sections']:>4} {info['trusted']:>4} "
              f"{info['anchored']:>6} {info['changed']:>5} "
              f"{info['rec_none']:>6}  {mark}")
        # 锚定段里的原始误读（锚定=不可改，是算法固有风险）
        vals = [s["raw"] for s in d["segments"]]
        is_anchor = [False] * len(vals)
        secs, a = build_sections(vals, [s["mid"] for s in d["segments"]],
                                 d["meta"]["fps"], params[0])
        tr = trusted_sections(secs, a, [s["len"] for s in d["segments"]],
                              params[1], d["meta"]["max_accel"] * MPS_TO_KMH
                              * params[3], min_pairs=params[4])
        for (l, r, _ma) in tr:
            for k in range(l, r + 1):
                is_anchor[k] = True
        anchored_bad = 0
        for i, s in enumerate(d["segments"]):
            if s["truth"] is None or s["raw"] is None:
                continue
            if is_anchor[i] and abs(s["raw"] - s["truth"]) > TOL:
                anchored_bad += 1
        print(f"  ↳ 锚定段中原始误读 {anchored_bad} 个（锚定=不可改）")
    print()

    # 逐案例：全部最终错误段 + 基线案例的处置
    print("── 逐案例（每视频全部最终错误段；* = 基线也错的段）──")
    for v in videos:
        d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
        segs = d["segments"]
        out, info = run_algorithm(segs, d["meta"], *params)
        base_out = [s["corr"] for s in segs]
        printed = False
        for i, (s, o, bo) in enumerate(zip(segs, out, base_out)):
            if s["truth"] is None or s["raw"] is None:
                continue
            base_err = abs(bo - s["truth"]) > TOL
            new_err = o is None or abs(o - s["truth"]) > TOL
            if not (base_err or new_err):
                continue
            if not printed:
                print(f"\n── {v} ──")
                printed = True
            mark = "*" if base_err else " "
            chg = "→" if o != s["raw"] else "="
            print(f" {mark}#{i:<5} rep={s['rep']:<6} raw={s['raw']!s:>4} "
                  f"corr={bo!s:>4} new={o!s:>4}{chg} truth={s['truth']!s:>4} "
                  f"len={s['len']:>3} mid={s['mid']}")


# ────────────────────────────── 混合变体：节锚定 + 现有 DP ──────────────────────────────

def hybrid_dp(v, max_accel_diff, min_length, max_accel_factor=1.0):
    """节锚定（可信节段 conf=100，其余 0）→ 生产 dense_correct。

    检验"节锚定作为锚点来源"是否优于生产 conf 锚定（当前算法真正的问题
    是中值窗口宽度难定 → 锚点来源；DP 转移本身保留）。
    """
    from seg_correction import dense_correct
    d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
    segs = d["segments"]
    vals = [s["raw"] for s in segs]
    mids = [s["mid"] for s in segs]
    lens = [s["len"] for s in segs]
    meta = d["meta"]
    secs, a = build_sections(vals, mids, meta["fps"], max_accel_diff)
    tr = trusted_sections(secs, a, lens, min_length,
                          meta["max_accel"] * MPS_TO_KMH * max_accel_factor)
    conf = [0.0] * len(vals)
    for (l, r, _ma) in tr:
        for k in range(l, r + 1):
            conf[k] = 100.0
    out, _n = dense_correct(vals, mids, conf,
                            max_speed=meta["max_speed"],
                            max_accel=meta["max_accel"],
                            fps=meta["fps"])
    return funnel(segs, out)


# ────────────────────────────── 邻域调试 ──────────────────────────────

def ctx_dump(v, idx, params):
    d = json.loads((FIXTURE_DIR / f"{v}.json").read_text(encoding="utf-8"))
    segs = d["segments"]
    vals = [s["raw"] for s in segs]
    mids = [s["mid"] for s in segs]
    lens = [s["len"] for s in segs]
    meta = d["meta"]
    out, info = run_algorithm(segs, meta, *params)
    secs, a = build_sections(vals, mids, meta["fps"], params[0])
    tr = trusted_sections(secs, a, lens, params[1],
                          meta["max_accel"] * MPS_TO_KMH * params[3],
                          min_pairs=params[4])
    rec = reconstruct(tr, len(vals), vals, mids, meta["fps"],
                      meta["max_speed"])
    trusted_idx = set()
    for (l, r, _ma) in tr:
        for k in range(l, r + 1):
            trusted_idx.add(k)
    lo, hi = max(0, idx - 6), min(len(vals), idx + 7)
    print(f"{v} 段#{idx} rep={segs[idx]['rep']} truth={segs[idx]['truth']} "
          f"raw={vals[idx]} new={out[idx]} rec={None if rec[idx] is None else round(rec[idx],1)}"
          f" mid={mids[idx]} len={lens[idx]}")
    print(f"  配置 diff={params[0]} minL={params[1]} gap={params[2]}"
          f" fac={params[3]} mp={params[4]}"
          f" | 可信节 {len(tr)} 锚定 {len(trusted_idx)} 改动 {info['changed']}")
    for i in range(lo, hi):
        in_tr = "A" if i in trusted_idx else "."
        ai = "" if a[i] is None else f"{a[i]:+.1f}"
        r = "" if rec[i] is None else f"{rec[i]:.1f}"
        mark = " ◀" if i == idx else ""
        print(f"  [{i}] {in_tr} raw={vals[i]!s:>4} a={ai:>7} rec={r:>7} "
              f"truth={segs[i]['truth']!s:>4} len={lens[i]:>3}{mark}")
    for k, (l, r, ma) in enumerate(tr):
        if l <= idx <= r or abs(l - idx) <= 4 or abs(r - idx) <= 4:
            lo_v = [vals[j] for j in range(l, r + 1)]
            print(f"  可信节#{k} [{l}..{r}] mean_a={ma:+.2f} "
                  f"frames={sum(lens[l:r+1])} vals={lo_v}")
    sels = [(l, r) for (l, r) in secs if l <= idx <= r or l == idx + 1
            or r == idx - 1]
    print(f"  分节（含邻域）: {sels}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="扫参")
    ap.add_argument("--detail", action="store_true", help="最优/指定配置逐案例")
    ap.add_argument("--hybrid", action="store_true",
                    help="节锚定+生产DP 变体对比")
    ap.add_argument("--ctx", nargs=2, metavar=("VIDEO", "IDX"),
                    help="邻域调试：--ctx test 74 [--diff .. --min-length ..]")
    ap.add_argument("--diff", type=float, default=None)
    ap.add_argument("--min-length", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--max-accel-factor", type=float, default=1.0)
    ap.add_argument("--min-pairs", type=int, default=1)
    args = ap.parse_args()

    if args.ctx:
        params = (args.diff if args.diff is not None else 8.0,
                  args.min_length if args.min_length is not None else 5.0,
                  args.gap if args.gap is not None else 8.0,
                  args.max_accel_factor, args.min_pairs)
        ctx_dump(args.ctx[0], int(args.ctx[1]), params)
        return

    if args.hybrid:
        for diff in (30.0, 60.0, 100.0, 150.0, 200.0):
            for ml in (5, 10, 20):
                per = {}
                tot = {k: 0 for k in ("seg", "raw", "fix", "fix_wrong",
                                      "missed", "harm", "final")}
                for v in VIDEOS:
                    f = hybrid_dp(v, diff, ml, args.max_accel_factor)
                    per[v] = f
                    for k in tot:
                        tot[k] += f[k]
                line = " ".join(f"{per[v]['final']:>3}" for v in VIDEOS)
                print(f"hybrid diff={diff:>4} minL={ml:>3} → "
                      f"total final={tot['final']:>3}  {line}")
        return

    if args.sweep or (args.diff is None and args.min_length is None
                      and args.gap is None):
        best = sweep()
        params = (best[1], best[2], best[3], best[4], best[5])
        if args.detail:
            detail(params)
    else:
        params = (args.diff, args.min_length, args.gap,
                  args.max_accel_factor, args.min_pairs)
        detail(params)


if __name__ == "__main__":
    main()
