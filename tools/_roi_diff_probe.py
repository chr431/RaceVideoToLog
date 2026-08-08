"""ROI 差分分离度探针：未变帧 vs 变帧的像素 diff 分布。

思路 2 可行性验证：display 没变 ⇒ 速度值必须相同（零变化约束）。
关键前置是信噪分离 —— 摄像头振动/传感器噪声下，未变帧的 diff 必须
干净地低于任何"真的变了"的 diff。

用 pipeline 同路径（decord next_roi 顺序读取，ROI=truth header closed+1）。
用法：python tools/_roi_diff_probe.py [video] [--max-frames N]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))


def load_truth(video_name: str):
    truth = PROJECT / f"ground_truth_csv/{video_name}_truth.csv"
    if not truth.exists():
        truth = PROJECT / f"ground_truth_csv/{video_name}_ref.csv"
    rows: dict[int, float] = {}
    roi = None
    with open(truth, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
                if m:
                    roi = tuple(int(x) for x in m.groups())
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    rows[int(float(parts[0]))] = float(parts[2])
                except ValueError:
                    pass
    return rows, f"D:/Videos/racelog_test/{video_name}.mp4", roi


def aligned_min_diff(a: np.ndarray, b: np.ndarray) -> float:
    """±1px 平移对齐后的最小 mean diff（补偿摄像头亚像素振动）。"""
    h, w = a.shape[:2]
    best = float("inf")
    af = a.astype(np.int16)
    bf = b.astype(np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                seg = abs(af - bf).mean()
            else:
                a_s = af[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
                b_s = bf[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
                if a_s.shape != b_s.shape or a_s.size == 0:
                    continue
                seg = abs(a_s - b_s).mean()
            if seg < best:
                best = seg
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default="test")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    truth, video, roi = load_truth(args.video)
    print(f"video={video}  truth frames={len(truth)}  roi={roi}")
    if roi is None:
        raise SystemExit("roi not found in truth header")
    x1, y1, x2, y2 = roi  # closed bounds → next_roi 用 x2+1,y2+1（半开，与 pipeline 一致）

    from decord import VideoReader, cpu
    vr = VideoReader(video, ctx=cpu(0))
    frames = sorted(truth)
    if args.max_frames:
        frames = frames[: args.max_frames]

    # 顺序读取（pipeline 同路径）：seek 到首帧后 next_roi 逐帧
    vr.seek_accurate(frames[0])
    prev = None
    prev_fi = None
    u_mean, u_al, u_frac, c_mean, c_al, c_frac = [], [], [], [], [], []
    change_sizes: dict[int, int] = {}
    fn_cases: list = []   # 危险侧：真变了但 diff 小的帧对
    n_pairs = 0
    for fi in frames:
        crop = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        # CPU next_roi 可能返回全帧 → 裁剪
        if crop.shape[0] != y2 - y1 + 1 or crop.shape[1] != x2 - x1 + 1:
            crop = crop[y1:y2 + 1, x1:x2 + 1]
        if prev is not None:
            n_pairs += 1
            dv = abs(truth[fi] - truth[prev_fi])
            d = np.abs(crop.astype(np.int16) - prev.astype(np.int16))
            mean_d = float(d.mean())
            frac40 = float((d > 40).mean())
            al = aligned_min_diff(prev, crop)
            if dv < 0.5:
                u_mean.append(mean_d)
                u_al.append(al)
                u_frac.append(frac40)
            else:
                c_mean.append(mean_d)
                c_al.append(al)
                c_frac.append(frac40)
                change_sizes[int(dv)] = change_sizes.get(int(dv), 0) + 1
                if al <= 2.0:
                    fn_cases.append((prev_fi, fi, truth[prev_fi], truth[fi], round(al, 3)))
        prev, prev_fi = crop, fi

    u, c = np.array(u_mean), np.array(c_mean)
    ua, ca = np.array(u_al), np.array(c_al)
    uf, cf = np.array(u_frac), np.array(c_frac)
    print(f"\npairs={n_pairs}  unchanged={len(u)}  changed={len(c)}")
    print(f"change magnitude: {dict(sorted(change_sizes.items()))}")
    print("\n--- mean pixel diff（原始）---")
    print(f"unchanged: med={np.median(u):.3f} p90={np.percentile(u,90):.3f} "
          f"p99={np.percentile(u,99):.3f} max={u.max():.3f}")
    print(f"changed:   med={np.median(c):.3f} p90={np.percentile(c,90):.3f} min={c.min():.3f}")
    print("\n--- mean pixel diff（±1px 对齐后）---")
    print(f"unchanged: med={np.median(ua):.3f} p99={np.percentile(ua,99):.3f} max={ua.max():.3f}")
    print(f"changed:   med={np.median(ca):.3f} min={ca.min():.3f}")
    print("\n--- 大 diff 像素占比 (>40) ---")
    print(f"unchanged: med={np.median(uf)*100:.3f}% p99={np.percentile(uf,99)*100:.3f}%")
    print(f"changed:   med={np.median(cf)*100:.3f}% min={cf.min()*100:.4f}%")
    if len(u) and len(c):
        print("\n--- 阈值分离度（对齐后 mean diff）---")
        print("  T  未变帧误判为'变'(FP=跳过不约束)  变帧误判为'不变'(FN=危险)")
        for T in (0.5, 1.0, 1.5, 2.0, 3.0):
            fp = (ua > T).mean()
            fn = (ca <= T).mean()
            print(f"  {T:.1f}  {fp*100:6.2f}%            {fn*100:6.2f}%")
    if fn_cases:
        print("\n--- 危险 FN 案例（真变了但对齐 diff<=2.0，会被零变化约束强制等值）---")
        for a, b, va, vb, al in fn_cases[:15]:
            print(f"  {a}->{b}: {va:.0f}->{vb:.0f} km/h  diff={al}")


if __name__ == "__main__":
    main()
