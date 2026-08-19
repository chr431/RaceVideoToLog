"""分段流水线生产测试：精度 + 解码/OCR 计时。

用法：python tools/_segment_production_test.py [videos...]
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import segment_flow  # noqa: E402


def load_meta(video: str):
    tpath = PROJECT / f"ground_truth_csv/{video}_truth.csv"
    if not tpath.exists():
        tpath = PROJECT / f"ground_truth_csv/{video}_ref.csv"
    roi = None
    f_start = f_end = None
    fps = None
    max_speed = 400.0
    max_accel = 50.0
    max_width = 0
    for line in open(tpath, encoding="utf-8-sig"):
        m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
        if m:
            roi = tuple(int(x) for x in m.groups())
        m = re.search(r"fps=([\d.]+)", line)
        if m:
            fps = float(m.group(1))
        m = re.search(r"frame_start=(\d+)", line)
        if m:
            f_start = int(m.group(1))
        m = re.search(r"frame_end=(\d+)", line)
        if m:
            f_end = int(m.group(1))
        m = re.search(r"max_speed=([\d.]+)", line)
        if m:
            max_speed = float(m.group(1))
        m = re.search(r"max_accel=([\d.]+)", line)
        if m:
            max_accel = float(m.group(1))
        m = re.search(r"max_width=(\d+)", line)
        if m:
            max_width = int(m.group(1))
    truth = {}
    for line in open(tpath, encoding="utf-8-sig"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.strip().split(",")
        try:
            truth[int(float(p[0]))] = float(p[2])
        except (ValueError, IndexError):
            pass
    return tpath, roi, f_start, f_end, fps, max_speed, max_accel, max_width, truth


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    args = ap.parse_args()

    for v in args.videos:
        (tpath, roi, f_start, f_end, fps, ms, ma, mw,
         truth) = load_meta(v)
        video = str(PROJECT / "D:/Videos/racelog_test") if False else f"D:/Videos/racelog_test/{v}.mp4"
        t0 = time.perf_counter()
        pipe = segment_flow.SegmentPipeline(
            video, roi, ms, ma, fps, f_start, f_end,
            target_h=48, max_width=mw,
            speed_format="km/h", pad=0)
        out = str(PROJECT / "outputs" / f"_segment_{v}.csv")
        pipe.run(out)
        wall = time.perf_counter() - t0
        # 精度
        ok = err = 0
        for row in pipe.rows:
            fi, _d, spd, _fl = row
            t = truth.get(fi)
            if t is not None:
                if spd >= 0 and abs(spd - t) < 0.5:
                    ok += 1
                else:
                    err += 1
        tot = ok + err
        print(f"{v}: {len(pipe.rows)}帧 → {pipe.n_segments}段 "
              f"| 准确率 {ok}/{tot} ({ok/tot*100:.2f}%) err {err}")
        print(f"  timing: decode={pipe.timing.get('decode',0):.1f}s "
              f"ocr={pipe.timing.get('ocr',0):.1f}s total={wall:.1f}s "
              f"(decode/OCR 比 {pipe.timing.get('decode',0)/max(pipe.timing.get('ocr',0),0.01):.1f}x)")


if __name__ == "__main__":
    main()
