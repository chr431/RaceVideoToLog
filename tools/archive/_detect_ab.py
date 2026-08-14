"""段级检测/纠错参数 A/B：解码/OCR 各一次，扫门限组合，出准确率。

用法：python tools/_detect_ab.py [videos...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import segment_flow  # noqa: E402
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
        else:
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


def detect_frame_win(seg_vals, seg_times, win, mult, bw_floor):
    """帧窗口 median-of-pairs 检测（当前 _detect 的参数化版）。"""
    n = len(seg_vals)
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(win * max(med_gap, 1.0), 120.0)
    st = np.asarray(seg_times, dtype=np.float64)
    suspect = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        lefts = [j for j in range(lo, i) if seg_vals[j] is not None]
        rights = [j for j in range(i + 1, hi) if seg_vals[j] is not None]
        exps = []
        for l in lefts:
            for r in rights:
                span = seg_times[r] - seg_times[l]
                if span < 1e-3:
                    continue
                frac = (ti - seg_times[l]) / span
                exps.append(seg_vals[l] + (seg_vals[r] - seg_vals[l]) * frac)
        if not exps:
            suspect[i] = True
            continue
        exp = float(np.median(exps))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        bw = max(float(np.median(dvs)) if dvs else 0.0, bw_floor)
        if abs(seg_vals[i] - exp) > bw * mult:
            suspect[i] = True
    return suspect


def detect_phys(seg_vals, seg_times, win, mult, bw_floor):
    """物理 + 残差：残差超门限 或 与相邻段加速度超限 即 suspect。"""
    n = len(seg_vals)
    suspect = detect_frame_win(seg_vals, seg_times, win, mult, bw_floor)
    # 加速度：相邻段 |dv| 相对局部 bw 的倍率
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
        win_frames = min(win * max(med_gap, 1.0), 120.0)
        st = np.asarray(seg_times, dtype=np.float64)
        for i in range(n):
            if seg_vals[i] is None:
                continue
            ti = seg_times[i]
            lo = int(np.searchsorted(st, ti - win_frames, side="left"))
            hi = int(np.searchsorted(st, ti + win_frames, side="right"))
            dvs = [abs(seg_vals[j] - seg_vals[j - 1])
                   for j in range(lo + 1, hi)
                   if seg_vals[j] is not None and seg_vals[j - 1] is not None]
            bw = max(float(np.median(dvs)) if dvs else 0.0, 2.0)
            # 对相邻段检查
            for j in (i - 1, i + 1):
                if 0 <= j < n and seg_vals[j] is not None:
                    if abs(seg_vals[i] - seg_vals[j]) > bw * mult * 2:
                        suspect[i] = True
                        break
    return suspect


def correct_anchor(seg_vals, seg_times, suspect, min_dev):
    out = list(seg_vals)
    n_corr = 0
    for i in range(len(seg_vals)):
        if seg_vals[i] is None:
            pass
        elif not suspect[i]:
            continue
        la = None
        for j in range(i - 1, -1, -1):
            if not suspect[j] and seg_vals[j] is not None:
                la = j
                break
        ra = None
        for j in range(i + 1, len(seg_vals)):
            if not suspect[j] and seg_vals[j] is not None:
                ra = j
                break
        interp = None
        if la is not None and ra is not None:
            span = seg_times[ra] - seg_times[la]
            frac = (seg_times[i] - seg_times[la]) / span if span > 1e-3 else 0.5
            interp = seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac
        elif la is not None:
            interp = seg_vals[la]
        elif ra is not None:
            interp = seg_vals[ra]
        if interp is not None:
            if seg_vals[i] is None or abs(interp - seg_vals[i]) > min_dev:
                out[i] = round(interp)
                n_corr += 1
    return out, n_corr


def detect_new(seg_vals, seg_times, win, mult, floor):
    """残差 + 相邻跳变：|值-期望|>gate 或 与相邻段差>jump_gate 即 suspect。"""
    n = len(seg_vals)
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(win * max(med_gap, 1.0), 120.0)
    st = np.asarray(seg_times, dtype=np.float64)
    # 每段门限：局部 bw（相邻差中位数）
    gates = [0.0] * n
    for i in range(n):
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        gates[i] = max(float(np.median(dvs)) if dvs else 0.0, floor)
    suspect = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        lefts = [j for j in range(lo, i) if seg_vals[j] is not None]
        rights = [j for j in range(i + 1, hi) if seg_vals[j] is not None]
        exps = []
        for l in lefts:
            for r in rights:
                span = seg_times[r] - seg_times[l]
                if span < 1e-3:
                    continue
                frac = (ti - seg_times[l]) / span
                exps.append(seg_vals[l] + (seg_vals[r] - seg_vals[l]) * frac)
        if not exps:
            suspect[i] = True
            continue
        exp = float(np.median(exps))
        gate = gates[i] * mult
        if abs(seg_vals[i] - exp) > gate:
            suspect[i] = True
            continue
        # 相邻跳变：与任一相邻段差 > jump_gate（2× gate）
        for j in (i - 1, i + 1):
            if 0 <= j < n and seg_vals[j] is not None:
                if abs(seg_vals[i] - seg_vals[j]) > gate * 2:
                    suspect[i] = True
                    break
    return suspect


def correct_anchor_bound(seg_vals, seg_times, suspect, min_dev, anchor_max_frames):
    """锚点插值纠正 + 锚点距离上界：仅在近锚点可靠时纠正。

    anchor_max_frames：可信锚点与当前段的最大帧距离（两侧都需在此内才插值）。
    """
    out = list(seg_vals)
    n_corr = 0
    for i in range(len(seg_vals)):
        if seg_vals[i] is None:
            pass
        elif not suspect[i]:
            continue
        ti = seg_times[i]
        la = None
        for j in range(i - 1, -1, -1):
            if not suspect[j] and seg_vals[j] is not None:
                if ti - seg_times[j] <= anchor_max_frames:
                    la = j
                break
        ra = None
        for j in range(i + 1, len(seg_vals)):
            if not suspect[j] and seg_vals[j] is not None:
                if seg_times[j] - ti <= anchor_max_frames:
                    ra = j
                break
        interp = None
        if la is not None and ra is not None:
            span = seg_times[ra] - seg_times[la]
            frac = (ti - seg_times[la]) / span if span > 1e-3 else 0.5
            interp = seg_vals[la] + (seg_vals[ra] - seg_vals[la]) * frac
        elif la is not None:
            interp = seg_vals[la]
        elif ra is not None:
            interp = seg_vals[ra]
        if interp is not None:
            if seg_vals[i] is None or abs(interp - seg_vals[i]) > min_dev:
                out[i] = round(interp)
                n_corr += 1
    return out, n_corr


def detect_median(seg_vals, seg_times, win_k, mult, floor):
    """中值滤波检测：平滑值曲线（跟随弯曲），误读=尖峰被中值剔除。

    对每段 i，smoothed = 局部非 None 值的中位数（段索引窗口 ±win_k）。
    正确段贴合中值（偏差 ≤ 局部带宽），误读尖峰偏差 8+。
    门限 = max(局部相邻差中位数, floor) × mult。
    """
    n = len(seg_vals)
    # 局部带宽（相邻差中位数，帧窗口内）
    if n >= 2:
        gaps = np.diff(seg_times)
        med_gap = float(np.median(gaps)) if len(gaps) else 1.0
    else:
        med_gap = 1.0
    win_frames = min(30 * max(med_gap, 1.0), 120.0)
    st = np.asarray(seg_times, dtype=np.float64)
    bw = [0.0] * n
    for i in range(n):
        ti = seg_times[i]
        lo = int(np.searchsorted(st, ti - win_frames, side="left"))
        hi = int(np.searchsorted(st, ti + win_frames, side="right"))
        dvs = [abs(seg_vals[j] - seg_vals[j - 1])
               for j in range(lo + 1, hi)
               if seg_vals[j] is not None and seg_vals[j - 1] is not None]
        bw[i] = max(float(np.median(dvs)) if dvs else 0.0, floor)
    suspect = [False] * n
    for i in range(n):
        if seg_vals[i] is None:
            suspect[i] = True
            continue
        lo = max(0, i - win_k)
        hi = min(n, i + win_k + 1)
        nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
        if len(nbrs) < 3:
            suspect[i] = True
            continue
        # 边缘段（左右一侧无上下文）不 flag：中值在单调上升/下降区滞后，
        # 视频起止的低/高速段会被窗口拉偏误判（test2 起始 5→8→12 回归源）。
        lefts = any(seg_vals[j] is not None for j in range(lo, i))
        rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
        if not (lefts and rights):
            continue
        med = float(np.median(nbrs))
        if abs(seg_vals[i] - med) > bw[i] * mult:
            suspect[i] = True
    return suspect


def accuracy(segs, values, truth):
    err = 0
    for seg, val in zip(segs, values):
        if val is None:
            continue
        for fi in seg:
            t = truth.get(fi)
            if t is not None and abs(val - t) >= 0.5:
                err += 1
    return err


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test", "test5", "test6"])
    args = ap.parse_args()

    # 参数组合：中值滤波检测(k,floor,mult) × 纠错(min_dev, anchor_max)
    combos = []
    for floor, mult in ((3.0, 2.0), (4.0, 2.0), (3.0, 3.0), (2.0, 2.0)):
        for md in (6.0, 8.0):
            combos.append((10, floor, mult, md, 1e9))

    print(f"{'视频':<6}", end="")
    for c in combos:
        print(f" | k{c[0]} f{c[1]:.0f}x{c[2]:.0f} m{c[3]:.0f} a{'∞' if c[4]>1e8 else int(c[4])}", end="")
    print()

    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        video = f"D:/Videos/racelog_test/{v}.mp4"
        pipe = SegmentPipeline(video, roi, ms, ma, fps, f_start, f_end,
                               force_aspect=mw)
        frames, crops, grays, sharp = pipe._decode_all()
        segs = pipe._segment(frames, grays)
        seg_vals, rep_frames = pipe._ocr_segments(segs, crops, sharp)
        seg_times = [seg[len(seg) // 2] for seg in segs]
        print(f"{v:<6}", end="", flush=True)
        for win_k, floor, mult, md, am in combos:
            sus = detect_median(seg_vals, seg_times, win_k, mult, floor)
            corr, _ = correct_anchor_bound(seg_vals, seg_times, sus, md, am)
            err = accuracy(segs, corr, truth)
            print(f" | {err:3d}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
