"""最终错误上下文 dump：对每个最终错误段输出 raw/corr/conf/truth 与邻域。

用 production run() 的结果（_ocr_vals/_corr_vals/_conf_vals/segments），
打印每个最终错误（|corr-truth|>1）的上下文，辅助分类：
- 平滑偏移（邻域 raw 整体偏离，局部不可区分）
- 尖峰（raw 偏离邻域）
- 填充错（raw None）
- 纠错（corrected 但改错）
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402


def dump(v: str, TOL: float = 1.0) -> None:
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                           fps, f_start, f_end, force_aspect=mw)
    pipe.run(str(PROJECT / "outputs" / f"_dump_{v}.csv"))
    sv = pipe._ocr_vals
    cv = pipe._corr_vals
    conf = pipe._conf_vals
    segs = pipe.segments
    print(f"\n════════ {v}: 段数 {len(segs)} ════════")
    n = 0
    for i, s in enumerate(segs):
        t = truth.get(s["rep_frame"])
        if t is None:
            continue
        ok = cv[i] is not None and abs(cv[i] - t) <= TOL
        if ok:
            continue
        n += 1
        lo = max(0, i - 3)
        hi = min(len(segs), i + 4)
        ctx = []
        for j in range(lo, hi):
            tj = truth.get(segs[j]["rep_frame"])
            tj = int(tj) if tj is not None else None
            mark = " ◀" if j == i else ""
            ctx.append(f"[{j}]raw={sv[j]} corr={cv[j]} "
                       f"conf={conf[j]:.0f} truth={tj}"
                       f" len={segs[j]['frames'][-1]-segs[j]['frames'][0]+1}{mark}")
        raw_err = sv[i] is not None and abs(sv[i] - t) > TOL
        print(f"\n#{i} rep={s['rep_frame']} 最终错误: corr={cv[i]} truth={int(t)}"
              f" raw={'✗' if raw_err else '✓'} conf={conf[i]:.0f}"
              f" len={s['frames'][-1]-s['frames'][0]+1} 类型="
              + ("平滑偏移" if not raw_err else
                 ("填充" if sv[i] is None else "raw误读"))
              + (" [被DP改错]" if cv[i] != sv[i] else " [未改动]"))
        for c in ctx:
            print("   " + c)
    print(f"\n{v}: 最终错误 {n} 个")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["test", "test2"]:
        dump(v)
