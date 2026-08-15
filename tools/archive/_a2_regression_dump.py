"""dump A2(min(conf,jerk_score)) 下 test6 新增错误的段特征。

目的：找 #74（想修复）与 test6 误改段的 jerk_score/conf 判别边界。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from tools.archive._jerk_anchor_probe import _components  # noqa: E402

v = sys.argv[1] if len(sys.argv) > 1 else "test6"
TOL = 1.0
roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                       fps, f_start, f_end, force_aspect=mw)
pipe.run(str(PROJECT / "outputs" / f"_a2_{v}.csv"))
sv = pipe._ocr_vals
conf = pipe._conf_vals
times = [s["rep_frame"] for s in pipe.segments]
comps = _components(pipe)
conf2 = list(conf)
for i, c in enumerate(comps):
    if c[0] == "jerk" and c[2] is not None:
        conf2[i] = min(conf[i], c[2])
corr2, n_corr = pipe._dense_correct(sv, times, conf2)
print(f"{v}: A2 dense_correct 修改 {n_corr} 段")

for i, s in enumerate(pipe.segments):
    t = truth.get(s["rep_frame"])
    if t is None:
        continue
    old_ok = pipe._corr_vals[i] is not None and abs(pipe._corr_vals[i] - t) <= TOL
    new_ok = corr2[i] is not None and abs(corr2[i] - t) <= TOL
    if old_ok and not new_ok:
        c = comps[i]
        lo = max(0, i - 2)
        hi = min(len(pipe.segments), i + 3)
        print(f"\n✗ 新增错误 #{i} rep={s['rep_frame']} len="
              f"{s['frames'][-1]-s['frames'][0]+1}"
              f" raw={sv[i]} 旧corr={pipe._corr_vals[i]} 新corr={corr2[i]}"
              f" truth={int(t)} conf={conf[i]:.1f}→{conf2[i]:.1f}"
              f" 分支={c[0]} med_score={c[1]} jerk_score={c[2]}")
        for j in range(lo, hi):
            tj = truth.get(pipe.segments[j]["rep_frame"])
            print(f"   [#{j}] raw={sv[j]} corr={corr2[j]} conf={conf2[j]:.0f}"
                  f" truth={int(tj) if tj is not None else None}")
