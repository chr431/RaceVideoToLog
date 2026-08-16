"""诊断：生产检测（detect_segments / confidence_scores）在最终错误上的行为。

用夹具 raw/mid/len + meta 离线重算 suspect/conf（与生产同参数），
对每个最终错误案例（|corr-truth|>1，raw/truth 均存在）输出检测状态，
并按失败模式分类。不重跑解码+OCR。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from seg_correction import confidence_scores, detect_segments
from tools.detect_eval import load_meta

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
FIX = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "seg_series"
TOL = 1.0


def analyze(v):
    d = json.loads((FIX / f"{v}.json").read_text(encoding="utf-8"))
    segs = d["segments"]
    raw = [s["raw"] for s in segs]
    mids = [s["mid"] for s in segs]
    lens = [s["len"] for s in segs]
    conf = confidence_scores(raw, mids, lens)
    sus = detect_segments(raw, mids, lens)
    corr = [s["corr"] for s in segs]
    tru = [s["truth"] for s in segs]
    print(f"\n══ {v} ══")
    cats = {"A漏检": [], "B检测到未改对": [], "C改错": [], "H误改": []}
    for i, s in enumerate(segs):
        t, r = s["truth"], s["raw"]
        if t is None or r is None:
            continue
        final_err = corr[i] is None or abs(corr[i] - t) > TOL
        if not final_err:
            continue
        raw_err = abs(r - t) > TOL
        changed = corr[i] is not None and corr[i] != r
        if raw_err and not changed:
            cat = "A漏检" if not sus[i] else "B检测到未改对"
        elif raw_err and changed:
            cat = "C改错" if abs(corr[i] - t) > TOL else "?"
        else:
            cat = "H误改"
        cats[cat].append(i)
        jl = raw[i - 1] if i > 0 else None
        jr = raw[i + 1] if i + 1 < len(raw) else None
        jerk = abs(jr - 2 * r + jl) if (jl is not None and jr is not None) else None
        print(f" #{i:<5} rep={s['rep']:<6} raw={r!s:>4} corr={corr[i]!s:>4} "
              f"truth={t!s:>4} len={s['len']:>3} conf={conf[i]:>4.0f} "
              f"suspect={'Y' if sus[i] else 'N'} |raw-truth|={abs(r-t):>2} "
              f"jerk={jerk!s:>4} [{cat}]")
        lo, hi = max(0, i - 3), min(len(raw), i + 4)
        ctx = " ".join(f"{raw[j]}" if j != i else f"[{raw[j]}]"
                       for j in range(lo, hi))
        tc = " ".join("?" if tru[j] is None else str(int(tru[j]))
                      for j in range(lo, hi))
        print(f"       raw: {ctx}")
        print(f"       tru: {tc}")
    for k, vv in cats.items():
        print(f"  {k}: {len(vv)} {vv}")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["test", "test2"]:
        analyze(v)
