"""准确率漏斗分析：OCR 原始误读 → 检测 → DP 纠正 → 最终错误。

对每视频统计：
- 段数 / OCR 原始误读（|ocr-truth|>tol）
- 误读被检测（suspect）/ 未被检测
- 误读被纠正对 / 误读被纠正错 / 误读漏纠
- 正确段被误改（正确→错）
- 最终错误（|输出-truth|>tol）

口径默认 ±1（ref 有 ±1 容差）。看瓶颈在哪一级。

用法：python tools/_accuracy_breakdown.py [--tol 1] [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    agg = dict(seg=0, raw=0, det=0, fix=0, fix_wrong=0, missed=0, harm=0,
               final=0)
    print(f"{'视频':<6} {'段':>5} {'原始误读':>6} {'检出':>4} {'纠对':>4} "
          f"{'纠错':>4} {'漏纠':>4} {'误改':>4} {'最终':>4}")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_brk_{v}.csv"))
        sv = pipe._ocr_vals
        cv = pipe._corr_vals
        seg = [0] * 7  # raw, det, fix, fix_wrong, missed, harm, final
        for i in range(len(sv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or sv[i] is None:
                continue
            seg[0] += 1
            raw_err = abs(sv[i] - t) > TOL
            final_err = abs(cv[i] - t) > TOL if cv[i] is not None else True
            if raw_err:
                seg[1] += 1
                if cv[i] is not None and cv[i] != sv[i]:
                    if not final_err:
                        seg[2] += 1  # 纠对
                    else:
                        seg[3] += 1  # 纠错
                else:
                    seg[4] += 1  # 漏纠
            else:
                if cv[i] is not None and cv[i] != sv[i] and final_err:
                    seg[5] += 1  # 正确被误改
            if final_err:
                seg[6] += 1
        print(f"{v:<6} {seg[0]:>5} {seg[1]:>6} {seg[1]-seg[4]:>4} {seg[2]:>4} "
              f"{seg[3]:>4} {seg[4]:>4} {seg[5]:>4} {seg[6]:>4}")
        keys = ("seg", "raw", "fix", "fix_wrong", "missed", "harm", "final")
        for k, key in enumerate(keys):
            agg[key] += seg[k]

    det = agg["raw"] - agg["missed"]
    print(f"\n合计: 段 {agg['seg']} 原始误读 {agg['raw']} 检出 {det} "
          f"纠对 {agg['fix']} 纠错 {agg['fix_wrong']} 漏纠 {agg['missed']} "
          f"误改 {agg['harm']} 最终 {agg['final']}")
    print(f"检出率 = {det}/max({agg['raw']},1) "
          f"({det/max(agg['raw'],1)*100:.1f}%)")
    print(f"最终 = 漏纠 {agg['missed']} + 纠错 {agg['fix_wrong']} "
          f"+ 误改 {agg['harm']} = {agg['missed']+agg['fix_wrong']+agg['harm']}")


if __name__ == "__main__":
    main()
