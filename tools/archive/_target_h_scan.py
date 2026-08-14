"""target_h 参数必要性分析：换 target_h + 缩放 max_width，看结果是否一致。

`_preprocess_standard(target_h)` 后 OCR 引擎 `_resize_norm` 固定 48 高：
- target_h=48：一次 resize（33→48），_resize_norm 跳过
- target_h≠48：两次 resize（33→target_h→48），双线性两次插值数值不同
- max_width 是 target_h 高度下的宽度上限 → target_h 变化时应按比例缩放
  （mw_scaled = round(mw * target_h / 48)）保持相同几何

指标：
- 与 target_h=48 的**逐段值差异**（0 = 参数完全无影响，可删）
- 各 target_h 的 raw 误读（准确率差异）

用法：python tools/_target_h_scan.py [videos...] [--heights 32 40 48 64 96]
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
from video_utils import _preprocess_standard, _np_resize  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

TOL = 1.0
BATCH = 16
BASE_H = 48


def _to48(proc: np.ndarray) -> np.ndarray:
    """target_h≠48 的 proc → 48 高（模拟 _resize_norm 的二次 resize）。

    eng.__call__ 用 batch 第一张高度作统一目标，生产恒 48；实验时 target_h
    可变，先手动 resize 到 48 再喂，保持与生产（两次 resize）一致。
    """
    h, w = proc.shape[:2]
    if h == BASE_H:
        return proc
    new_w = max(1, int(round(w * BASE_H / h)))
    return _np_resize(proc, new_w, BASE_H)

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
    ap.add_argument("videos", nargs="*", default=["test2", "test5"])
    ap.add_argument("--heights", nargs="*", type=int,
                    default=[32, 40, 48, 64, 96])
    args = ap.parse_args()

    eng = ocr_native.OcrEngine("v6_small", "onnxruntime")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_th_{v}.csv"))
        reps = [s["rep_frame"] for s in pipe.segments]
        crops = [pipe.crops[r] for r in reps]
        tv = [truth.get(r) for r in reps]
        print(f"{v}: {len(crops)} segments, truth max_width={mw}")

        # 各 target_h 的 OCR 值
        vals_by_h = {}
        for h in args.heights:
            mw_s = round(mw * h / BASE_H) if mw > 0 else 0
            vals = []
            for k in range(0, len(crops), BATCH):
                procs = [_to48(_preprocess_standard(c, pipe._fill_width,
                                                    force_aspect=mw_s))
                         for c in crops[k:k + BATCH]]
                for res in eng(procs):
                    sv, _rt, _c = extract_speed_value(res)
                    vals.append(int(sv) if sv is not None and sv >= 0 else None)
            vals_by_h[h] = vals
            err = sum(1 for i, val in enumerate(vals)
                      if tv[i] is not None and val is not None
                      and abs(val - tv[i]) > TOL)
            print(f"  target_h={h:>3} (mw={mw_s:>3}): raw 误读 {err:>3}")

        # 与 target_h=48 的逐段差异
        base = vals_by_h[BASE_H]
        print(f"  与 target_h={BASE_H} 逐段差异:")
        for h in args.heights:
            if h == BASE_H:
                continue
            diff = sum(1 for a, b in zip(base, vals_by_h[h]) if a != b)
            print(f"    target_h={h:>3}: {diff} 段值不同"
                  f"（{diff/max(len(base),1)*100:.1f}%）")


if __name__ == "__main__":
    main()
