"""conf 置信度算法实际表现分析。

对每视频（有误读的）：跑 gamma 流水线 → 每段分类 正确/误读（|raw-truth|>1），
统计：
1. conf 分布（正确 vs 误读的分位数）
2. 锚定阈值扫描：不同 anchor_conf 下 (误读被锚定=漏锚, 正确被锚定, 误读未锚定)
   → precision/recall，判断 SEG_DP_ANCHOR_CONF=20 是否最优
3. 分支表现：贴合分支（med_score≥50）vs 急动度分支（<50），各自误读率
   —— 急动度分支是否真的把「刹车正确」和「误读」分开
4. 失效模式：高 conf 误读段（应锚定却被锚）与低 conf 正确段（走 DP 被改）特征

复刻 _confidence 逻辑加 branch 标记（不改 segment_flow 签名）。

用法：python tools/_conf_analysis.py [videos...]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0


def conf_with_branch(pipe, seg_vals, seg_times):
    """复制 _confidence 逻辑，返回 (conf, branch)。

    branch ∈ fit(贴合曲线)/jerk(急动度分辨)/sparse(邻居<3)/edge(边缘)/none(None)。
    """
    n = len(seg_vals)
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(pipe._win * max(med_gap, 1.0), 120.0)
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
    conf = [0.0] * n
    branch = ["none"] * n
    for i in range(n):
        if seg_vals[i] is None:
            conf[i] = 0.0
            branch[i] = "none"
            continue
        lo = max(0, i - pipe._med_k)
        hi = min(n, i + pipe._med_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi)
                if seg_vals[j] is not None]
        if len(nbrs) < 3:
            conf[i] = 30.0
            branch[i] = "sparse"
            continue
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            conf[i] = 100.0
            branch[i] = "edge"
            continue
        med = float(np.median(nbrs))
        dev = abs(seg_vals[i] - med)
        bw = max(bw_raw[i], pipe._detect_floor)
        med_score = 100.0 * np.exp(-dev / bw)
        if med_score >= 50.0:
            conf[i] = med_score
            branch[i] = "fit"
            continue
        jl = seg_vals[i - 1] if i - 1 >= 0 else None
        jr = seg_vals[i + 1] if i + 1 < n else None
        if jl is not None and jr is not None:
            jerk = abs(jr - 2 * seg_vals[i] + jl)
            jerk_score = 100.0 * np.exp(-jerk / pipe._conf_jerk_scale)
            conf[i] = (pipe._conf_w_med * med_score
                       + pipe._conf_w_jerk * jerk_score)
            branch[i] = "jerk"
        else:
            conf[i] = med_score
            branch[i] = "fit"
    return conf, branch


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test6"])
    args = ap.parse_args()

    # 跨视频聚合
    agg = {"correct": [], "misread": [], "fit_c": 0, "fit_b": 0,
           "jerk_c": 0, "jerk_b": 0, "sparse_c": 0, "sparse_b": 0,
           "edge_c": 0, "edge_b": 0}

    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_cf_{v}.csv"))
        gv = pipe._ocr_vals
        seg_times = [s[len(s) // 2] for s in pipe._segs]
        conf, branch = conf_with_branch(pipe, gv, seg_times)

        cls = []  # True=misread, False=correct
        for i in range(len(gv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or gv[i] is None:
                cls.append(None)
            else:
                cls.append(abs(gv[i] - t) > TOL)

        c_conf = [conf[i] for i, c in enumerate(cls) if c is False]
        b_conf = [conf[i] for i, c in enumerate(cls) if c is True]
        n_c, n_b = len(c_conf), len(b_conf)

        def q(x):
            if not x:
                return (0, 0, 0, 0, 0)
            return tuple(np.percentile(x, p) for p in (10, 25, 50, 75, 90))

        print(f"\n=== {v} === 正确段 {n_c} 误读段 {n_b} "
              f"({n_b/max(n_c+n_b,1)*100:.2f}%)")
        print(f"正确 conf 分位(p10 p25 p50 p75 p90): {q(c_conf)}")
        print(f"误读 conf 分位(p10 p25 p50 p75 p90): {q(b_conf)}")

        # 分支表现
        for br in ("fit", "jerk", "sparse", "edge"):
            idxs = [i for i in range(len(cls)) if branch[i] == br
                    and cls[i] is not None]
            nc = sum(1 for i in idxs if cls[i] is False)
            nb = sum(1 for i in idxs if cls[i] is True)
            if idxs:
                rate = nb / len(idxs) * 100
                mc = np.median([conf[i] for i in idxs])
                print(f"  {br:>6}: 段 {len(idxs):>5} 误读率 {rate:5.1f}% "
                      f"(错{nb}/对{nc}) conf中位 {mc:5.1f}")
            agg[f"{br}_c"] += nc
            agg[f"{br}_b"] += nb

        # 锚定阈值扫描
        print("  anchor_conf 扫描 (误读总数 %d):" % n_b)
        for th in (10, 15, 20, 25, 30, 40, 60):
            anc_b = sum(1 for i in range(len(cls))
                        if cls[i] is True and conf[i] >= th)
            anc_c = sum(1 for i in range(len(cls))
                        if cls[i] is False and conf[i] >= th)
            miss = n_b - anc_b
            print(f"    th={th:>3}: 误读锚定 {anc_b:>3} (漏锚 {miss:>3}) | "
                  f"正确锚定 {anc_c:>4} | 锚定精度 "
                  f"{anc_b/max(anc_b+anc_c,1)*100:.1f}%")

        # 高 conf 误读 / 低 conf 正确 代表帧
        hi_bad = [(i, conf[i], gv[i], int(truth[pipe.segments[i]['rep_frame']]))
                  for i in range(len(cls))
                  if cls[i] is True and conf[i] >= config.SEG_DP_ANCHOR_CONF]
        hi_bad.sort(key=lambda x: -x[1])
        print(f"  高 conf 误读 (conf≥{config.SEG_DP_ANCHOR_CONF:.0f}, 被锚定漏纠):")
        for i, c, g, t in hi_bad[:12]:
            print(f"    #{pipe.segments[i]['rep_frame']:<7} conf={c:5.1f} "
                  f"raw={g:>3} t={t:>3}")

        for k, key in enumerate(("correct", "misread")):
            pass
        agg["correct"].extend(c_conf)
        agg["misread"].extend(b_conf)

    print("\n═══ 跨视频聚合 ═══")
    c, b = agg["correct"], agg["misread"]
    print(f"正确段 conf 分位(p10 p50 p90): "
          f"{tuple(np.percentile(c, p) for p in (10,50,90))}  n={len(c)}")
    print(f"误读段 conf 分位(p10 p50 p90): "
          f"{tuple(np.percentile(b, p) for p in (10,50,90))}  n={len(b)}")
    tot_b = sum(agg[f"{br}_b"] for br in ("fit", "jerk", "sparse", "edge"))
    print(f"误读分布: " + ", ".join(
        f"{br}={agg[br+'_b']} ({agg[br+'_b']/max(tot_b,1)*100:.0f}%)"
        for br in ("fit", "jerk", "sparse", "edge")))
    for br in ("fit", "jerk", "sparse", "edge"):
        nc, nb = agg[f"{br}_c"], agg[f"{br}_b"]
        if nc + nb:
            print(f"  {br:>6} 段 {nc+nb} 误读率 {nb/(nc+nb)*100:5.1f}%")
    # 完整锚定 ROC（跨视频）
    print("锚定阈值扫描 (聚合):")
    for th in (10, 15, 20, 25, 30, 40, 60):
        anc_b = sum(1 for x in agg["misread"] if x >= th)
        anc_c = sum(1 for x in agg["correct"] if x >= th)
        prec = anc_c / max(anc_b + anc_c, 1) * 100
        rec = (len(agg["misread"]) - anc_b) / max(len(agg["misread"]), 1) * 100
        print(f"    th={th:>3}: 误读锚定 {anc_b:>3} (漏锚率 {rec:4.1f}%) | "
              f"正确锚定 {anc_c:>4} | 误锚率 {prec:5.1f}%")


if __name__ == "__main__":
    main()
