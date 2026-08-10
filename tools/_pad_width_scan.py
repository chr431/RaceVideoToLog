"""OCR pad 宽度扫描（gray+gamma 正式预处理下）：速度 + raw accuracy。

OCR 输入 pad 到 RVTOL_PAD_SMALL（v6_small 的 OCR_PAD_WIDTH_MIN=224）。灰度+
gamma 正式化后，RGB 时代的 pad 最优值（224）可能已变——降宽度可省 CPU 推理
（GPU+CPU / CPU+CPU 的瓶颈）。

对每视频：pipe.run() 一次（拿 crops/segments/truth），OCR 引擎实例复用
（onnxruntime，~1s 实例化）。对每个宽度：
- 设 RVTOL_PAD_SMALL=W → eng(procs) 批处理（B=16，与流水线一致）
- 计时（预处理一次共享，只计推理 wall）
- raw 误读 = |ocr - truth| > 1（不经纠错，看 OCR 本身）

用法：python tools/_pad_width_scan.py [--widths 224 192 160 128 96] [videos...]
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402
from ocr_native import OcrEngine  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0
BATCH = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="*", type=int,
                    default=[224, 192, 160, 128, 96])
    ap.add_argument("videos", nargs="*",
                    default=["test", "test2", "test3", "test5", "test6"])
    args = ap.parse_args()

    # 单引擎实例复用（onnxruntime，CPU；重点是 CPU 推理成本）
    eng = OcrEngine("v6_small", "onnxruntime")

    per_video = {}   # v -> (seg, crops_reps, procs, truth_vals)
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw)
        pipe.run(str(PROJECT / "outputs" / f"_padw_{v}.csv"))
        reps = [s["rep_frame"] for s in pipe.segments]
        crops = [pipe.crops[r] for r in reps]
        procs = [_preprocess_standard(c, 48, 0, max_width=mw) for c in crops]
        tv = [truth.get(r) for r in reps]
        per_video[v] = (len(reps), crops, procs, tv)
        print(f"{v}: {len(reps)} 段")

    # 每宽度：全视频推理计时 + 误读统计
    rows = {}  # W -> [wall, err, seg]
    for W in args.widths:
        os.environ["RVTOL_PAD_SMALL"] = str(W)
        wall = 0.0
        err = 0
        seg = 0
        for v, (n, _crops, procs, tv) in per_video.items():
            vals: list = []
            t0 = time.perf_counter()
            for k in range(0, n, BATCH):
                res = eng(procs[k:k + BATCH])
                for r in res:
                    sv, _rt, _c = extract_speed_value(r)
                    vals.append(int(sv) if sv is not None and sv >= 0 else None)
            wall += time.perf_counter() - t0
            for i in range(n):
                if tv[i] is None or vals[i] is None:
                    continue
                seg += 1
                if abs(vals[i] - tv[i]) > TOL:
                    err += 1
        rows[W] = (wall, err, seg)
        print(f"W={W:>3}: 推理 {wall:5.2f}s | raw 误读 {err:>4}/{seg} "
              f"({err/seg*100:.2f}%)")
        os.environ.pop("RVTOL_PAD_SMALL", None)

    print(f"\n{'width':>5} {'infer(s)':>8} {'err':>5} {'err%':>8}")
    base = rows[args.widths[0]]
    for W, (wall, err, seg) in rows.items():
        d = wall - base[0]
        sign = "-" if d < 0 else "+"
        print(f"{W:>5} {wall:8.2f} {err:>5} {err/seg*100:8.2f} "
              f"(d{sign}{abs(d):.2f}s)")


if __name__ == "__main__":
    main()
