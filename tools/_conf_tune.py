"""conf 权重调参：中值偏差 + 急动度加权，找正误 conf 清晰分离的配置。

报告每个 (w_jerk, jerk_scale) 下：
- 正确段 / 错误段的 conf 分位数（分离度：错误 p90 < 正确 p10 ⇒ 清晰分离）
- 最佳阈值（max 召回且误报率 ≤ 1%）及该点召回/误报

用法：python tools/_conf_tune.py [--tol 1] [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from _conf_probe import load_meta, confidence  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    TOL = args.tol

    # 收集 (conf 正确段, conf 错误段, 总段) —— 先按默认参数算一次拿段数据，
    # 之后每个 (w,scale) 重新算 conf（便宜）
    data = []
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw, "v6_small")
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        sv, rp = pipe._ocr_segments(segs, crops, sharp)
        st = [seg[len(seg) // 2] for seg in segs]
        is_err = [False] * len(sv)
        for i, seg in enumerate(segs):
            t = truth.get(rp[i])
            ov = sv[i]
            if t is None or ov is None:
                continue
            is_err[i] = abs(ov - t) > TOL
        data.append((v, sv, st, is_err))
        print(f"加载 {v}: {len(sv)} 段, 误读 {sum(is_err)}")

    print(f"\n{'w_jerk':>6} {'scale':>5} | {'正确p10':>7} {'正确p50':>7} "
          f"{'错误p50':>7} {'错误p90':>7} | {'召回@误报≤1%':>12} {'误报':>5}")
    for w_jerk in (0.3, 0.5, 0.7, 0.8, 0.9):
        for scale in (3.0, 5.0, 8.0):
            w_med = 1.0 - w_jerk
            corr_conf, err_conf = [], []
            for v, sv, st, is_err in data:
                conf = confidence(sv, st, w_med=w_med, w_jerk=w_jerk,
                                  jerk_scale=scale)
                for i in range(len(sv)):
                    if sv[i] is None:
                        continue
                    if is_err[i]:
                        err_conf.append(conf[i])
                    else:
                        corr_conf.append(conf[i])
            corr = np.array(corr_conf)
            err = np.array(err_conf)
            # 最佳阈值：max 召回且误报率 ≤ 1%
            best_rec, best_fpr, best_t = 0, 0, 0
            for T in range(5, 100, 5):
                tp = (err < T).sum()
                fp = (corr < T).sum()
                rec = tp / max(len(err), 1)
                fpr = fp / max(len(corr), 1)
                if fpr <= 0.01 and rec > best_rec:
                    best_rec, best_fpr, best_t = rec, fpr, T
            p10 = np.percentile(corr, 10)
            p50c = np.percentile(corr, 50)
            p50e = np.percentile(err, 50)
            p90e = np.percentile(err, 90)
            gap = "✔分离" if p90e < p10 else ""
            print(f"{w_jerk:>6.1f} {scale:>5.1f} | {p10:>7.1f} {p50c:>7.1f} "
                  f"{p50e:>7.1f} {p90e:>7.1f} | "
                  f"{best_rec*100:>7.1f}%@T{best_t} {best_fpr*100:>5.2f} {gap}")


if __name__ == "__main__":
    main()
