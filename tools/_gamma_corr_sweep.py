"""gamma 下纠错参数扫描：找 (anchor_conf, change_thr) 组合最小化端到端错误。

纠错参数只影响 _confidence/_dense_correct（检测+DP），与 OCR 无关 → 每视频
只跑一次 pipeline（gamma raw），对每个参数组合复用 conf 重新 _dense_correct，
省去重复解码/OCR。

指标（TOL±1）：
- 最终错误（|corr-truth|>1）= 漏纠 + 纠错 + 误改
- 漏纠：raw 错且未改；纠错：raw 错改错；误改：raw 对改错

用法：python tools/_gamma_corr_sweep.py [videos...]
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0

ANCHOR_CONFS = [20.0, 30.0, 40.0]
CHANGE_THRS = [2.0, 2.5, 3.0]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test2", "test"])
    args = ap.parse_args()

    results = {}  # (ac, ct) -> [seg, final, missed, wrong, harm]
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw)
        pipe.run(str(PROJECT / "outputs" / f"_sw_{v}.csv"))
        gv = pipe._ocr_vals
        seg_times = [s[len(s) // 2] for s in pipe._segs]
        conf = pipe._confidence(gv, seg_times)
        n_seg = 0
        targets = []
        for i in range(len(gv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or gv[i] is None:
                continue
            n_seg += 1
            targets.append((i, t))
        for ac in ANCHOR_CONFS:
            for ct in CHANGE_THRS:
                pipe._dp_anchor_conf = ac
                pipe._dp_change_threshold = ct
                cv, _ = pipe._dense_correct(gv, seg_times, conf)
                key = (ac, ct)
                if key not in results:
                    results[key] = [0, 0, 0, 0, 0]
                r = results[key]
                r[0] += n_seg
                for i, t in targets:
                    raw_err = abs(gv[i] - t) > TOL
                    final_err = abs(cv[i] - t) > TOL if cv[i] is not None \
                        else True
                    if final_err:
                        r[1] += 1
                    if raw_err and cv[i] is not None and cv[i] == gv[i]:
                        r[2] += 1   # 漏纠
                    elif raw_err and final_err and cv[i] != gv[i]:
                        r[3] += 1   # 纠错
                    elif not raw_err and final_err and cv[i] != gv[i]:
                        r[4] += 1   # 误改
        print(f"{v}: raw 误读 "
              f"{sum(1 for i,t in targets if abs(gv[i]-t)>TOL)}/{n_seg}")

    print(f"\n{'anchor':>6} {'ct':>4} {'最终':>5} {'漏纠':>4} {'纠错':>4} "
          f"{'误改':>4}")
    for (ac, ct), (seg, final, missed, wrong, harm) in sorted(
            results.items(), key=lambda kv: (kv[1][1], -kv[1][4])):
        print(f"{ac:>6.0f} {ct:>4.1f} {final:>5} {missed:>4} {wrong:>4} "
              f"{harm:>4}")


if __name__ == "__main__":
    main()
