"""A3 扫描：只解锚 jerk 分支 jerk_score < θ 且 conf<50 的段。

对比：
- 基线（θ=0）：13 错误
- 每个 θ：最终错误数、解锚段数（其中正确/误读）、新增误改明细
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402
from tools._jerk_anchor_probe import _components  # noqa: E402

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
TOL = 1.0
THETAS = [0, 5, 10, 15, 20, 30, 50]


def main() -> None:
    results = {th: {"final": 0, "de": 0, "de_ok": 0, "de_err": 0,
                    "harm": [], "fix": []} for th in THETAS}
    for v in VIDEOS:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_a3_{v}.csv"))
        sv = pipe._ocr_vals
        cv = pipe._corr_vals
        conf = pipe._conf_vals
        times = [s["rep_frame"] for s in pipe.segments]
        comps = _components(pipe)
        old_final = sum(1 for i, s in enumerate(pipe.segments)
                        if (t := truth.get(s["rep_frame"])) is not None
                        and cv[i] is not None and abs(cv[i] - t) > TOL)
        for th in THETAS:
            if th == 0:
                res = results[th]
                res["final"] += old_final
                continue
            conf2 = list(conf)
            n_de = 0
            for i, c in enumerate(comps):
                if c[0] == "jerk" and c[2] is not None and conf[i] < 50 \
                        and c[2] < th:
                    conf2[i] = 19.0  # 解锚（< SEG_DP_ANCHOR_CONF=20）
                    n_de += 1
            corr2, _n = pipe._dense_correct(sv, times, conf2)
            res = results[th]
            res["de"] += n_de
            new_final = 0
            for i, s in enumerate(pipe.segments):
                t = truth.get(s["rep_frame"])
                if t is None or corr2[i] is None:
                    continue
                if abs(corr2[i] - t) > TOL:
                    new_final += 1
                    if cv[i] is not None and abs(cv[i] - t) <= TOL:
                        res["harm"].append(f"{v}#{i} raw={sv[i]} "
                                           f"corr{cv[i]}→{corr2[i]} truth={int(t)}")
            res["final"] += new_final
            # 解锚段构成
            for i, c in enumerate(comps):
                if c[0] == "jerk" and c[2] is not None and conf[i] < 50 \
                        and c[2] < th:
                    t = truth.get(pipe.segments[i]["rep_frame"])
                    if t is None or sv[i] is None:
                        continue
                    if abs(sv[i] - t) > TOL:
                        res["de_err"] += 1
                    else:
                        res["de_ok"] += 1

    print(f"{'θ':>4} {'最终':>4} {'解锚':>5} {'解锚正确':>6} {'解锚误读':>5}")
    for th in THETAS:
        r = results[th]
        print(f"{th:>4} {r['final']:>4} {r['de']:>5} {r['de_ok']:>6} "
              f"{r['de_err']:>5}")
    for th in THETAS:
        if results[th]["harm"]:
            print(f"\nθ={th} 误改明细:")
            for h in results[th]["harm"]:
                print("  ✗", h)


if __name__ == "__main__":
    main()
