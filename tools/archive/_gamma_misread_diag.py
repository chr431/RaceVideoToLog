"""gamma 下纠错瓶颈诊断：dump 最终错误段的 raw/conf/邻居，定位漏纠原因。

对每视频（默认 test2/test）跑正式 gamma 流水线，对每个最终错误段
（|corr-truth|>1）打印：帧、gamma raw、baseline raw、真值、conf、是否锚定、
段帧数、±3 邻居 gamma raw。对照 gamma vs baseline 的 raw 差异，判断漏纠
是「conf 锚定」（is_anchor）还是「DP 未提交」（|dp-raw|<=threshold）。

用法：python tools/_gamma_misread_diag.py [videos...]
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import config
from segment_flow import SegmentPipeline  # noqa: E402
from tools.archive._gamma_misread_montage import run_on_reps  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

TOL = 1.0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test2", "test"])
    args = ap.parse_args()

    print(f"TOL={TOL} anchor_conf={config.SEG_DP_ANCHOR_CONF} "
          f"change_thr={config.SEG_DP_CHANGE_THRESHOLD} "
          f"mult={config.SEG_MULT} floor={config.SEG_DETECT_FLOOR} "
          f"sfloor={config.SEG_SINGLE_FLOOR} med_k={config.SEG_MED_K}")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_diag_{v}.csv"))
        gv = pipe.ocr_values
        cv = pipe.corrected_values
        bv = run_on_reps(pipe, 0.0)      # baseline raw 对照
        seg_times = [s[len(s) // 2] for s in pipe.segment_frames]
        seg_lens = [len(s) for s in pipe.segment_frames]
        conf = pipe._confidence(gv, seg_times, seg_lens)
        is_anchor = [c >= config.SEG_DP_ANCHOR_CONF and x is not None
                     for c, x in zip(conf, gv)]

        print(f"\n=== {v} ===")
        n_dump = 0
        for i in range(len(gv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or gv[i] is None:
                continue
            if abs(cv[i] - t) > TOL:
                n_dump += 1
                nbrs = [gv[j] for j in range(max(0, i - 3), i + 4)
                        if 0 <= j < len(gv)]
                kind = ("误改" if abs(gv[i] - t) <= TOL else
                        ("纠错" if cv[i] != gv[i] else "漏纠"))
                print(f"#{rep:<7} {kind} raw={gv[i]:>4} base={bv[i]:>4} "
                      f"corr={cv[i]:>4} t={int(t):>4} conf={conf[i]:>5.0f} "
                      f"anchor={is_anchor[i]!s:<5} len={seg_lens[i]:<4} "
                      f"nbr=[{','.join(str(x) if x is not None else '.' for x in nbrs)}]")
        if n_dump == 0:
            print("  无最终错误")
        else:
            # 汇总
            from collections import Counter
            kinds = Counter()
            for i in range(len(gv)):
                rep = pipe.segments[i]["rep_frame"]
                t = truth.get(rep)
                if t is None or gv[i] is None or abs(cv[i] - t) <= TOL:
                    continue
                kinds["误改" if abs(gv[i] - t) <= TOL else
                      ("纠错" if cv[i] != gv[i] else "漏纠")] += 1
            print(f"  汇总: " + ", ".join(f"{k}={v_}" for k, v_ in kinds.items()))


if __name__ == "__main__":
    main()
