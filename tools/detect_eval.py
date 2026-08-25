"""段级检测/纠错评估（生产路径版）：召回率 + 误报率。

引擎七轮重构后串行参考路径（_decode_all/_segment/_ocr_segments）已从
引擎移除（dead-code 清理，见 video_ocr_engine CLAUDE.md）。本工具改为
直接评估生产 run() 的输出 CSV（逐帧 rows）对 truth 的误差：

- 对每视频：SegmentPipeline.run() 全流程 → CSV rows (frame, dist, speed, flag)
- 误读帧 = 输出 speed ≥ 0 且 |speed - truth[frame]| > tol（默认 ±1）
- flag 分布展示纠错动作（DP_CORRECTED=11 / HIGH_TRUST=21 / FILL_INTERP=12 /
  RAW=0 / PINNED=3），speed < 0（FILL_INTERP 未读出）单独计为 missing
- 召回率/误报率语义保留：TP = 触发纠错的帧中最终仍误读？——不再适用；
  本版直接给“纠错后最终错误”与缺失帧数（生产语义，串行路径已不可复现）。

用法：python tools/detect_eval.py [--tol 1] [videos...]
（load_meta 保持原签名，tools/bench_hybrid.py 仍依赖它。）
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402


def load_meta(v: str):
    tpath = PROJECT / f"ground_truth_csv/{v}_truth.csv"
    if not tpath.exists():
        tpath = PROJECT / f"ground_truth_csv/{v}_ref.csv"
    roi = f_start = f_end = fps = None
    max_speed = 400.0
    max_accel = 50.0
    force_aspect = 0.0
    for line in open(tpath, encoding="utf-8-sig"):
        import re
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
        m = re.search(r"force_aspect=([\d.]+)", line)
        if m:
            force_aspect = float(m.group(1))
        else:  # 旧头兼容
            m = re.search(r"max_width=(\d+)", line)
            if m:
                force_aspect = round(int(m.group(1)) / 48.0, 2)
    truth = {}
    for line in open(tpath, encoding="utf-8-sig"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.strip().split(",")
        try:
            truth[int(float(p[0]))] = float(p[2])
        except (ValueError, IndexError):
            pass
    return roi, f_start, f_end, fps, max_speed, max_accel, force_aspect, truth


def _read_rows(csv_path: Path) -> list[tuple[int, float, str]]:
    """输出 CSV → [(frame, speed, flag), ...]（跳过 # 头），speed<0 保留。"""
    rows = []
    for line in open(csv_path, encoding="utf-8-sig"):
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        try:
            rows.append((int(parts[0]), float(parts[2]), parts[3]))
        except ValueError:
            continue
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    default=["test", "test2", "test3", "test5", "test6"])
    ap.add_argument("--tol", type=float, default=1.0, help="±容差，默认 1")
    args = ap.parse_args()
    TOL = args.tol

    agg = {"seg": 0, "err": 0, "ok": 0, "missing": 0, "final_err": 0}
    import collections
    flag_counts: collections.Counter = collections.Counter()
    print(f"{'视频':<6} {'帧数':>6} {'误读':>5} {'正确':>5} {'缺失':>4} "
          f"{'纠错后':>6}  flag分布")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        out = PROJECT / "outputs" / f"detect_eval_{v}.csv"
        pipe = SegmentPipeline(
            f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma, fps, f_start,
            f_end, force_aspect=mw, yuv_output=True)
        pipe.run(str(out))
        rows = _read_rows(out)
        err = ok = missing = 0
        for frame, speed, flag in rows:
            t = truth.get(frame)
            flag_counts[flag] += 1
            if t is None:
                continue
            if speed < 0:
                missing += 1
            elif abs(speed - t) > TOL:
                err += 1
            else:
                ok += 1
        final = err
        flags = ",".join(sorted(flag_counts))
        print(f"{v:<6} {len(rows):>6} {err:>5} {ok:>5} {missing:>4} "
              f"{final:>6}  {flags}")
        agg["seg"] += len(rows)
        agg["err"] += err
        agg["ok"] += ok
        agg["missing"] += missing
        agg["final_err"] += final

    print(f"\n[tol=±{TOL:.0f}] 合计: 帧 {agg['seg']} "
          f"误读 {agg['err']} 正确 {agg['ok']} 缺失 {agg['missing']}")
    print(f"纠错后最终错误: {agg['final_err']} "
          f"({agg['final_err']/max(agg['seg'],1)*100:.2f}% of 帧)")
    print(f"flag 总数分布: {dict(flag_counts)}")


if __name__ == "__main__":
    main()