"""A4 探针：jerk 分支 conf∈[20,50) 段的 jerk 值分布（正确 vs 误读），
并模拟"jerk 带通解锚"（5 ≤ jerk ≤ 30 且 conf<50）的端到端结果。
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

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
TOL = 1.0


def main() -> None:
    ok_buckets: dict[int, int] = {}   # jerk 值桶 → 正确段数
    err_buckets: dict[int, int] = {}  # jerk 值桶 → 误读段数
    old_final = new_final = 0
    de_ok = de_err = 0
    harms = []
    fixes = []
    for v in VIDEOS:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_a4_{v}.csv"))
        sv, cv, conf = pipe.ocr_values, pipe.corrected_values, pipe.confidence_values
        times = [s["rep_frame"] for s in pipe.segments]
        comps = _components(pipe)
        for i, s in enumerate(pipe.segments):
            t = truth.get(s["rep_frame"])
            if t is None or sv[i] is None:
                continue
            is_err = abs(sv[i] - t) > TOL
            if comps[i][0] == "jerk" and 20 <= conf[i] < 50:
                jerk = comps[i][3]
                if jerk is None:
                    continue
                b = min(jerk // 10 * 10, 200)
                (err_buckets if is_err else ok_buckets)[b] = \
                    (err_buckets if is_err else ok_buckets).get(b, 0) + 1
        # ── A4：jerk 带通解锚（5 ≤ jerk ≤ 30 且 conf<50，jerk 分支）──
        conf2 = list(conf)
        for i, c in enumerate(comps):
            if c[0] == "jerk" and c[3] is not None and conf[i] < 50 \
                    and 5 <= c[3] <= 30:
                conf2[i] = 19.0
        corr2, _n = pipe._dense_correct(sv, times, conf2)
        for i, s in enumerate(pipe.segments):
            t = truth.get(s["rep_frame"])
            if t is None or corr2[i] is None:
                continue
            new_err = abs(corr2[i] - t) > TOL
            if new_err:
                new_final += 1
                if cv[i] is not None and abs(cv[i] - t) <= TOL:
                    harms.append(f"{v}#{i} raw={sv[i]} corr{cv[i]}→{corr2[i]}"
                                 f" truth={int(t)} conf={conf[i]:.0f}"
                                 f" jerk={comps[i][3]}")
            else:
                if cv[i] is not None and abs(cv[i] - t) > TOL:
                    fixes.append(f"{v}#{i} raw={sv[i]} corr{cv[i]}→{corr2[i]}"
                                 f" truth={int(t)} conf={conf[i]:.0f}"
                                 f" jerk={comps[i][3]}")
            if cv[i] is not None and t is not None and abs(cv[i] - t) > TOL:
                old_final += 1
        # 解锚构成
        for i, c in enumerate(comps):
            if c[0] == "jerk" and c[3] is not None and conf[i] < 50 \
                    and 5 <= c[3] <= 30:
                t = truth.get(pipe.segments[i]["rep_frame"])
                if t is None or sv[i] is None:
                    continue
                if abs(sv[i] - t) > TOL:
                    de_err += 1
                else:
                    de_ok += 1

    print("jerk 分支 conf∈[20,50) 段构成（jerk 值 10 一桶）:")
    print("  jerk桶  正确  误读")
    for b in sorted(set(ok_buckets) | set(err_buckets)):
        print(f"  {b:>4}-{b+9:<4} {ok_buckets.get(b,0):>5} {err_buckets.get(b,0):>4}")
    print(f"\nA4(带通 5-30) 解锚: 正确 {de_ok} / 误读 {de_err}")
    print(f"合计: 旧最终 {old_final} → 新最终 {new_final}")
    for h in harms:
        print("  ✗ 误改:", h)
    for f in fixes:
        print("  ✓ 修复:", f)


if __name__ == "__main__":
    main()
