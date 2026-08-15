"""jerk 分支锚定风险探针：统计 5 视频 conf∈[20,50) jerk 分支段的构成，
并模拟"jerk 分支 conf = min(conf, jerk_score)"后的端到端结果。

不改代码，直接用 production run() 状态重跑 _dense_correct。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
TOL = 1.0


def _components(pipe):
    """重算每段 med_score / jerk_score / 分支（与 _confidence 同逻辑）。"""
    sv = pipe._ocr_vals
    st = np.asarray([s["rep_frame"] for s in pipe.segments], dtype=np.float64)
    n = len(sv)
    med_k = pipe._med_k
    floor = pipe._detect_floor
    # _local_bandwidth 是全量 O(n·窗口) 计算 —— 必须提一次，不能在段循环内
    # 调用（否则 O(n²·窗口)，test6 8096 段 ≈ 80 亿次操作跑不完）
    bw_all = pipe._local_bandwidth(sv, st.tolist())
    comps = []
    for i in range(n):
        lo = max(0, i - med_k)
        hi = min(n, i + med_k + 1)
        nbrs = [sv[j] for j in range(lo, hi) if sv[j] is not None]
        med = float(np.median(nbrs)) if nbrs else None
        if sv[i] is None or med is None or len(nbrs) < 3:
            comps.append(("none", None, None, None))
            continue
        bw = max(bw_all[i], floor)
        dev = abs(sv[i] - med)
        med_score = 100.0 * np.exp(-dev / bw) if bw > 0 else 100.0
        jl = sv[i - 1] if i - 1 >= 0 else None
        jr = sv[i + 1] if i + 1 < n else None
        jerk = None
        if jl is not None and jr is not None:
            jerk = abs(jr - 2 * sv[i] + jl)
        if med_score >= 50.0:
            comps.append(("fit", med_score, None, jerk))
        else:
            js = (100.0 * np.exp(-jerk / pipe._conf_jerk_scale)
                  if jerk is not None else med_score)
            comps.append(("jerk", med_score, js, jerk))
    return comps


def main() -> None:
    tot = {"jerk_20_50_ok": 0, "jerk_20_50_err": 0, "jerk_all": 0,
           "jerk_err": 0, "jerk_lowjs_err": 0}
    old_final = new_final = 0
    for v in VIDEOS:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_probe_{v}.csv"))
        sv = pipe._ocr_vals
        cv = pipe._corr_vals
        conf = pipe._conf_vals
        times = [s["rep_frame"] for s in pipe.segments]
        comps = _components(pipe)
        # 旧最终错误
        e_old = e_new = 0
        for i, s in enumerate(pipe.segments):
            t = truth.get(s["rep_frame"])
            if t is None or cv[i] is None:
                continue
            if abs(cv[i] - t) > TOL:
                e_old += 1
        # 模拟 A2：jerk 分支 conf = min(conf, jerk_score)
        conf2 = list(conf)
        for i, c in enumerate(comps):
            if c[0] == "jerk" and c[2] is not None:
                conf2[i] = min(conf[i], c[2])
        corr2, _n = pipe._dense_correct(sv, times, conf2)
        for i, s in enumerate(pipe.segments):
            t = truth.get(s["rep_frame"])
            if t is None or corr2[i] is None:
                continue
            if abs(corr2[i] - t) > TOL:
                e_new += 1
        print(f"{v:<5} 旧最终 {e_old:>3}  新最终 {e_new:>3}")
        old_final += e_old
        new_final += e_new
        # jerk 分支统计
        for i, c in enumerate(comps):
            if c[0] != "jerk":
                continue
            tot["jerk_all"] += 1
            t = truth.get(pipe.segments[i]["rep_frame"])
            if t is None or sv[i] is None:
                continue
            is_err = abs(sv[i] - t) > TOL
            if is_err:
                tot["jerk_err"] += 1
                if c[2] is not None and c[2] < 50:
                    tot["jerk_lowjs_err"] += 1
            if 20 <= conf[i] < 50:
                if is_err:
                    tot["jerk_20_50_err"] += 1
                else:
                    tot["jerk_20_50_ok"] += 1

    print(f"\njerk 分支: 共 {tot['jerk_all']} 段，误读 {tot['jerk_err']}"
          f" ({tot['jerk_err']/max(tot['jerk_all'],1)*100:.1f}%)")
    print(f"jerk 分支 conf∈[20,50): 正确 {tot['jerk_20_50_ok']} / "
          f"误读 {tot['jerk_20_50_err']}")
    print(f"jerk 分支 jerk_score<50 的误读: {tot['jerk_lowjs_err']}")
    print(f"合计: 旧最终 {old_final} → 新最终 {new_final}")


if __name__ == "__main__":
    main()
