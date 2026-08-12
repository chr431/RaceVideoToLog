"""初次 OCR 固定预处理全量对比：gamma2.0 vs raw 灰度 vs 线性拉伸 vs 直方图均衡。

"智能"预处理的最简落地：不依赖选择信号（置信度/多数票/结构都被证伪），
而是找到对所有帧都更好的单一预处理。gamma2.0 压暗丢中间位（t=169 读
16/9），raw 灰度+线性映射能读出——全量验证哪种固定策略 raw 误读最少。

对全部段（不只误读帧）跑每种预处理 OCR，统计 |ocr-truth|>1 误读数。

用法：python tools/_preproc_full_compare.py [videos...]
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
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0
BATCH = 16
_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_ENGINE_CACHE: dict = {}
_ORIG_ENGINE = ocr_native.OcrEngine


def _engine_factory(*a, **k):
    key = a[0] if a else k.get("variant", "v6_small")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = _ORIG_ENGINE(*a, **k)
    return _ENGINE_CACHE[key]


ocr_native.OcrEngine = _engine_factory


def preproc(crop: np.ndarray, mode: str, target_h: int, pad: int,
            mw: int) -> np.ndarray:
    """4 种固定预处理：全走 _preprocess_standard 保证 resize/pad 一致。"""
    if mode == "gamma2":
        return _preprocess_standard(crop, target_h, pad, max_width=mw,
                                    gamma=2.0)   # 当前正式
    g = (crop.astype(np.float32) @ _GRAY_W).astype(np.uint8)
    if mode == "gray":
        g3 = np.stack([g] * 3, axis=-1)
        return _preprocess_standard(g3, target_h, pad, max_width=mw,
                                    gamma=0.0)
    if mode == "stretch":   # 线性拉伸 5%-95% 分位
        lo, hi = np.percentile(g, (5, 95))
        span = max(hi - lo, 1)
        enh = np.clip((g.astype(np.float32) - lo) * 255.0 / span,
                      0, 255).astype(np.uint8)
        return _preprocess_standard(np.stack([enh] * 3, axis=-1),
                                    target_h, pad, max_width=mw, gamma=0.0)
    if mode == "histeq":    # 直方图均衡
        hist, _ = np.histogram(g, bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
        lut = ((cdf - cdf_min) * 255.0 /
               max(cdf[-1] - cdf_min, 1)).astype(np.uint8)
        return _preprocess_standard(np.stack([lut[g]] * 3, axis=-1),
                                    target_h, pad, max_width=mw, gamma=0.0)
    if mode == "w245":      # raw 灰度 + 亮度窗口 245（20 宽±5 过渡+线性映射）
        center = 245
        d = np.abs(g.astype(np.float32) - center)
        weight = np.clip((15.0 - d) / 5.0, 0.0, 1.0)
        out = g.astype(np.float32) * weight
        nz = out[out > 0]
        if nz.size:
            lo, hi = float(nz.min()), float(nz.max())
            if hi > lo:
                out = (out - lo) * 255.0 / (hi - lo)
            out = np.clip(out, 0.0, 255.0)
        return _preprocess_standard(np.stack([out.astype(np.uint8)] * 3,
                                             axis=-1),
                                    target_h, pad, max_width=mw, gamma=0.0)
    raise ValueError(mode)


MODES = ["gamma2", "gray", "stretch", "histeq", "w245"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    default=["test", "test2", "test5", "test6"])
    args = ap.parse_args()

    eng = ocr_native.OcrEngine("v6_small", "onnxruntime")
    agg = {m: [0, 0] for m in MODES}   # mode -> [误读, 总段]
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw)
        pipe.run(str(PROJECT / "outputs" / f"_pfc_{v}.csv"))
        reps = [s["rep_frame"] for s in pipe.segments]
        crops = [pipe.crops[r] for r in reps]
        tv = [truth.get(r) for r in reps]
        print(f"{v}: {len(crops)} segments")
        for mode in MODES:
            vals = []
            for k in range(0, len(crops), BATCH):
                procs = [preproc(c, mode, pipe._target_h, pipe._pad,
                                 pipe._max_width)
                         for c in crops[k:k + BATCH]]
                for res in eng(procs):
                    sv, _rt, _c = extract_speed_value(res)
                    vals.append(int(sv) if sv is not None and sv >= 0 else None)
            err = sum(1 for i, val in enumerate(vals)
                      if tv[i] is not None and val is not None
                      and abs(val - tv[i]) > TOL)
            agg[mode][0] += err
            agg[mode][1] += len(vals)
            print(f"  {mode:>8}: raw 误读 {err:>3}")
    print(f"\n=== 汇总（{agg['gamma2'][1]} 段）===")
    for mode in MODES:
        err, tot = agg[mode]
        print(f"  {mode:>8}: {err:>3} ({err/max(tot,1)*100:.2f}%)"
              f"{'  ← 当前正式' if mode == 'gamma2' else ''}")


if __name__ == "__main__":
    main()
