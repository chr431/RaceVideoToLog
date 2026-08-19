"""初次 OCR 智能预处理诊断：多 gamma 预处理 + OCR 置信度选择。

固定 gray+gamma2.0 对所有帧统一处理，但不同帧最优 gamma 不同（gamma
压暗丢中间位）。本工具诊断"智能选择"是否可行：对每误读帧跑多种 gamma，
检查 **OCR 模型置信度（CTC score）能否选出正确结果**。

关键指标：
- 每预处理独立读对率
- 选"最高置信度"结果的命中率（智能选择的下限——若置信度能区分对错）
- "至少一预处理读对"的上限（若置信度完美选择）

用法：python tools/_preproc_conf_scan.py [videos...] [--gammas 0 1.0 2.0 3.0]
（gamma=0 表示纯灰度不增强）
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import ocr_native  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

TOL = 1.0
BATCH = 16

# ── 复用同一 OcrEngine ──
_ENGINE_CACHE: dict = {}
_ORIG_ENGINE = ocr_native.OcrEngine


def _engine_factory(*a, **k):
    key = a[0] if a else k.get("variant", "v6_small")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = _ORIG_ENGINE(*a, **k)
    return _ENGINE_CACHE[key]


ocr_native.OcrEngine = _engine_factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test2", "test"])
    ap.add_argument("--gammas", nargs="*", type=float,
                    default=[0.0, 1.0, 2.0, 3.0])
    args = ap.parse_args()

    eng = ocr_native.OcrEngine("v6_small", "onnxruntime")
    n_gamma = len(args.gammas)
    tot = {"mis": 0, "any_ok": 0, "best_conf_ok": 0}
    per_g = {g: 0 for g in args.gammas}
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_pcs_{v}.csv"))
        mis = []
        for i, seg in enumerate(pipe.segment_frames):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            ov = pipe.ocr_values[i]
            if t is None or ov is None:
                continue
            if abs(ov - t) > TOL:
                mis.append((rep, ov, int(t), pipe.crops.get(rep)))
        mis = [m for m in mis if m[3] is not None]
        print(f"{v}: {len(mis)} misreads")
        # 每误读帧 × 每 gamma：一次批处理拿 (值, 置信度)
        rows = []   # (rep, t, [(g, val, conf), ...])
        for rep, _ov, t, crop in mis:
            gammas = [g for g in args.gammas]
            imgs = [crop for _ in args.gammas]  # 同一 crop，preprocess 时传不同 gamma
            procs = [_preprocess_standard(c, pipe._fill_width,
                                          force_aspect=pipe._force_aspect, gamma=g)
                     for c, g in zip(imgs, gammas)]
            results = eng(procs)
            per = []
            for g, res in zip(args.gammas, results):
                sv, _rt, conf = extract_speed_value(res)
                per.append((g, int(sv) if sv is not None and sv >= 0 else None,
                            conf))
            rows.append((rep, t, per))
        # 统计
        for rep, t, per in rows:
            tot["mis"] += 1
            oks = [val is not None and abs(val - t) <= TOL for _g, val, _c in per]
            if any(oks):
                tot["any_ok"] += 1
            # 选最高置信度
            best = max(per, key=lambda x: x[2] if x[1] is not None else -1)
            if best[1] is not None and abs(best[1] - t) <= TOL:
                tot["best_conf_ok"] += 1
            for g, val, _c in per:
                if val is not None and abs(val - t) <= TOL:
                    per_g[g] += 1
        # 明细表（每帧每 gamma 值+置信度）
        print(f"  {'#fr':<7} {'t':>3} | " +
              " ".join(f"g={g:<3}(v,c)" for g in args.gammas))
        for rep, t, per in rows:
            cells = []
            for _g, val, confv in per:
                s = f"{val if val is not None else '-'},{confv:.2f}"
                ok = val is not None and abs(val - t) <= TOL
                cells.append(f"{s}{'*' if ok else ''}")
            print(f"  #{rep:<6} {t:>3} | " + "  ".join(cells))

    print(f"\n=== 汇总（{tot['mis']} 误读帧）===")
    print(f"至少一 gamma 读对（上限）: {tot['any_ok']}/{tot['mis']} "
          f"({tot['any_ok']/max(tot['mis'],1)*100:.0f}%)")
    print(f"选最高置信度命中       : {tot['best_conf_ok']}/{tot['mis']} "
          f"({tot['best_conf_ok']/max(tot['mis'],1)*100:.0f}%)")
    print(f"每 gamma 独立读对: " + ", ".join(
        f"g={g}={per_g[g]}" for g in args.gammas))
    print(f"\n结论: best_conf/total = "
          f"{tot['best_conf_ok']/max(tot['mis'],1)*100:.0f}% —— "
          f"{'置信度可作选择信号' if tot['best_conf_ok'] > tot['mis']*0.5 else '置信度不可靠'}")


if __name__ == "__main__":
    main()
