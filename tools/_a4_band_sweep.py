"""A4 带通参数敏感性：下界/上界网格扫描，确认 13→12 的鲁棒性。

每个视频只 run() 一次，之后对所有 (lo, hi) 组合重算 conf2 + dense_correct。
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
LOS = [0, 3, 5, 8, 10]
HIS = [20, 30, 40, 60, 80]


def main() -> None:
    final = {(lo, hi): {"final": 0, "de": 0, "harm": 0} for lo in LOS for hi in HIS}
    for v in VIDEOS:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_a4s_{v}.csv"))
        sv, cv, conf = pipe._ocr_vals, pipe._corr_vals, pipe._conf_vals
        times = [s["rep_frame"] for s in pipe.segments]
        comps = _components(pipe)
        for lo in LOS:
            for hi in HIS:
                conf2 = list(conf)
                n_de = 0
                for i, c in enumerate(comps):
                    if c[0] == "jerk" and c[3] is not None and conf[i] < 50 \
                            and lo <= c[3] <= hi:
                        conf2[i] = 19.0
                        n_de += 1
                corr2, _n = pipe._dense_correct(sv, times, conf2)
                e = harm = 0
                for i, s in enumerate(pipe.segments):
                    t = truth.get(s["rep_frame"])
                    if t is None or corr2[i] is None:
                        continue
                    if abs(corr2[i] - t) > TOL:
                        e += 1
                        if cv[i] is not None and abs(cv[i] - t) <= TOL:
                            harm += 1
                r = final[(lo, hi)]
                r["final"] += e
                r["de"] += n_de
                r["harm"] += harm

    print(f"{'下界\\上界':>8}" + "".join(f"{hi:>6}" for hi in HIS))
    for lo in LOS:
        row = f"{lo:>8}"
        for hi in HIS:
            r = final[(lo, hi)]
            row += f"{r['final']:>6}"
        print(row)
    print(f"\n基线: 13（零解锚）")
    print(f"{'下界\\上界':>8}" + "".join(f"{hi:>6}" for hi in HIS))
    for lo in LOS:
        row = f"{lo:>8}"
        for hi in HIS:
            row += f"{final[(lo,hi)]['harm']:>6}"
        print(row)
    print("误改数矩阵（0 才可接受）")


if __name__ == "__main__":
    main()
