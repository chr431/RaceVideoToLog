"""raw OCR 质量对比：正式 gray+gamma 2.0 vs 基线（纯 RGB）原始误读数。

对每视频：pipe.run() 一次（正式 gray+gamma，raw = _ocr_vals），再用
_gamma_misread_montage.run_on_reps 在相同段/代表帧上以 gamma=0 跑基线 OCR
（复用引擎缓存，不二次解码）。口径 |ocr-truth|>1（ref 有 ±1 容差）。

每段分类：
- IMPROVE：基线错 + gamma 对（gamma 修复）
- REGRESS：基线对 + gamma 错（gamma 新引入）
- BOTH：双错
输出每视频 + 合计误读数，判断 raw 是否相比基线提升。

用法：python tools/_gamma_raw_eval.py [videos...]
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools._gamma_misread_montage import run_on_reps  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    agg = dict(seg=0, b_err=0, g_err=0, improve=0, regress=0, both=0)
    print(f"{'视频':<6} {'段':>5} {'基线误读':>7} {'gamma误读':>8} "
          f"{'提升':>4} {'回归':>4} {'双错':>4}")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw)
        pipe.run(str(PROJECT / "outputs" / f"_graw_{v}.csv"))
        gv = pipe._ocr_vals                    # 正式 gray+gamma 2.0
        bv = run_on_reps(pipe, 0.0)            # 基线：纯 RGB
        assert len(gv) == len(bv) == len(pipe.segments)
        st = [0, 0, 0, 0, 0, 0]  # seg, b_err, g_err, improve, regress, both
        for i in range(len(gv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or gv[i] is None or bv[i] is None:
                continue
            st[0] += 1
            b_err = abs(bv[i] - t) > TOL
            g_err = abs(gv[i] - t) > TOL
            if b_err:
                st[1] += 1
            if g_err:
                st[2] += 1
            if b_err and not g_err:
                st[3] += 1   # improve
            elif g_err and not b_err:
                st[4] += 1   # regress
            elif b_err and g_err:
                st[5] += 1   # both
        print(f"{v:<6} {st[0]:>5} {st[1]:>7} {st[2]:>8} {st[3]:>4} "
              f"{st[4]:>4} {st[5]:>4}")
        for k, key in enumerate(("seg", "b_err", "g_err", "improve",
                                 "regress", "both")):
            agg[key] += st[k]

    print(f"\n合计: 段 {agg['seg']} | 基线误读 {agg['b_err']} → gamma误读 "
          f"{agg['g_err']} ({agg['g_err']/max(agg['b_err'],1)*100:.1f}%)")
    print(f"提升(基错→gamma对) {agg['improve']} | 回归(基对→gamma错) "
          f"{agg['regress']} | 双错 {agg['both']}")
    print(f"净提升 = {agg['improve'] - agg['regress']}")


if __name__ == "__main__":
    main()
