"""第二阶段实验：把加速度分节算法作为生产 dense_correct 之后的后处理。

输入 = 生产 corr（已修正值），跑纯算法，看能否修掉基线残留 11 个错误
而不破坏已修对的段（CLAUDE.md 约束：新纠错阶段必须先测"对已修正帧
是否净正"）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto_accel_section as P

best = None
for diff in (8.0, 30.0, 100.0):
    for ml in (5, 10, 20):
        for gap in (2, 3, 4, 6):
            tot = {k: 0 for k in ("seg", "raw", "fix", "fix_wrong",
                                  "missed", "harm", "final")}
            per = {}
            for v in P.VIDEOS:
                d = json.load(open(P.FIXTURE_DIR / f"{v}.json",
                                   encoding="utf-8"))
                segs = list(d["segments"])
                orig_raw = [s["raw"] for s in segs]
                for s, r in zip(segs, [s["corr"] for s in segs]):
                    s["raw"] = r
                out, _info = P.run_algorithm(segs, d["meta"], diff, ml, gap)
                for s, r in zip(segs, orig_raw):
                    s["raw"] = r
                f = P.funnel(segs, out)
                per[v] = f
                for k in tot:
                    tot[k] += f[k]
            key = (tot["final"], diff, ml, gap)
            if best is None or key < best[0]:
                best = (key, per, tot)

key, per, tot = best
print(f"第二阶段最优: diff={key[1]} minL={key[2]} gap={key[3]} "
      f"→ total final={key[0]}（基线 11）")
for v in P.VIDEOS:
    f = per[v]
    print(f"  {v:<6} final={f['final']} fix={f['fix']} "
          f"fix_wrong={f['fix_wrong']} missed={f['missed']} harm={f['harm']}")
