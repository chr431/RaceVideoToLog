"""实验：模拟 30fps 视频，验证当前后处理链（detect→conf→DP→第二遍尖峰）是否生效。

模拟方法：用 decord 从原始视频按 range(frame_start, frame_end, 2) 隔帧读取
（不重编码、零质量损失——真实 30fps 视频 = 采样帧序列），帧号重映射为
0..N-1，fps = 原始 fps/2（≈28.6-29.9fps，即"30fps"语义），truth 按原帧号
取子集。后处理链与生产完全一致（seg_correction 纯函数），评估口径与漏斗一致。

输出：每视频 采样段数 / 原始误读 / 无后处理错误 / 第一遍(conf+DP)错误 /
全链(+第二遍尖峰)错误 / 第二遍修复数。对比 57fps 生产基线（全部 0）。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from ocr_native import OcrEngine  # noqa: E402
from segmentation import _cluster_win3, _otsu  # noqa: E402
from seg_correction import (  # noqa: E402
    confidence_scores, dense_correct, spike_second_pass,
)
from tools.detect_eval import load_meta  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
VIDEO_DIR = "D:/Videos/racelog_test"
TOL = 1.0
B = config.OCR_BATCH_SIZE


def open_gray_reader(path, roi):
    from decord import VideoReader, gpu
    try:
        vr = VideoReader(path, ctx=gpu(0), output_format="gray",
                         roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1))
        return vr
    except Exception:
        from decord import cpu
        vr = VideoReader(path, ctx=cpu(0), output_format="gray",
                         roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1))
        return vr


def segment_grays(g, th):
    """逐帧二值 XOR + 聚类分段，与生产 _segment 一致。返回段索引列表。"""
    N = len(g)
    prev_b = g[0] > th
    edges = []
    for k in range(1, N):
        b = g[k] > th
        edges.append(_cluster_win3(prev_b != b) < config.SEG_C)
        prev_b = b
    segs = []
    s = 0
    for k in range(N - 1):
        if not edges[k]:
            segs.append(list(range(s, k + 1)))
            s = k + 1
    segs.append(list(range(s, N)))
    return segs


def run_video(v, eng, spike_k=config.SEG_SPIKE_K,
              spike_thresh=config.SEG_SPIKE_THRESH,
              spike_min_fix=config.SEG_SPIKE_MIN_FIX,
              first_only=False, scale_windows=False):
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    sample = list(range(f_start, f_end, 2))          # 原帧号隔帧
    fps2 = fps / 2.0                                  # 模拟 30fps
    # fps 自适应窗口：按 fps2/fps 缩放所有固定帧数窗口（保持时间跨度不变）
    if scale_windows:
        scale = fps2 / fps
        anchor_max = max(10.0, config.SEG_ANCHOR_MAX_FRAMES * scale)
        win_max = max(10.0, config.SEG_WIN_MAX_FRAMES * scale)
        island_min = max(1, int(round(config.SEG_CONF_MIN_CONSISTENT_FRAMES
                                      * scale)))
        island_exact = max(1, int(round(
            config.SEG_CONF_MIN_CONSISTENT_FRAMES_EXACT * scale)))
    else:
        anchor_max = config.SEG_ANCHOR_MAX_FRAMES
        win_max = config.SEG_WIN_MAX_FRAMES
        island_min = config.SEG_CONF_MIN_CONSISTENT_FRAMES
        island_exact = config.SEG_CONF_MIN_CONSISTENT_FRAMES_EXACT
    tag = f"（fps 自适应窗口: anchor={anchor_max:.0f} win={win_max:.0f} "
    tag += f"island={island_min}/{island_exact}）" if scale_windows else ""
    print(f"\n== {v}: 原始 {fps:.2f}fps → 模拟 {fps2:.2f}fps "
          f"（采样帧 {len(sample)}，原 {f_end - f_start}）{tag}")
    t0 = time.perf_counter()
    vr = open_gray_reader(f"{VIDEO_DIR}/{v}.mp4", roi)
    # 分块 get_batch 读采样帧（ROI gray）
    crops = []
    CH = 512
    for k in range(0, len(sample), CH):
        chunk = sample[k:k + CH]
        b = vr.get_batch(chunk,
                         roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)
                         ).asnumpy()
        crops.append(b[..., 0] if b.ndim == 4 else b)
    g = np.concatenate(crops, axis=0)
    del vr, crops
    N = len(g)
    # 阈值校准（前 SEG_CALIB_FRAMES 帧 Otsu 中位数；fps 自适应时按比例缩放）
    calib_n = max(8, min(config.SEG_CALIB_FRAMES
                         if not scale_windows
                         else max(8, int(round(config.SEG_CALIB_FRAMES
                                               * fps2 / fps))),
                         N))
    step = max(1, N // calib_n)
    ths = [_otsu(g[k]) for k in range(0, min(N, calib_n * step), step)]
    th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
    # 分段
    segs = segment_grays(g, th)
    # 代表帧 + OCR（批处理，与生产一致）
    sharp = g.std(axis=(1, 2))
    reps = [max(seg, key=lambda k: sharp[k]) for seg in segs]
    vals = []
    for k in range(0, len(segs), B):
        procs = [_preprocess_standard(g[r][..., None], force_aspect=mw)
                 for r in reps[k:k + B]]
        for r, res in zip(reps[k:k + B], eng(procs)):
            sv, _rt, _c = extract_speed_value(res)
            vals.append(int(sv) if sv is not None and sv >= 0 else None)
    seg_times = [seg[len(seg) // 2] for seg in segs]   # 重映射帧号
    lens = [len(seg) for seg in segs]
    # 后处理链（与生产一致；scale_windows 时窗口按 fps 缩放）
    conf = confidence_scores(vals, seg_times, lens,
                             win_max_frames=win_max,
                             island_min_frames=island_min,
                             island_min_frames_exact=island_exact)
    corr1, n1 = dense_correct(vals, seg_times, conf,
                              max_speed=ms, max_accel=ma, fps=fps2,
                              anchor_max=anchor_max)
    if first_only:
        corr2, n2, flagged = corr1, 0, []
    else:
        corr2, n2, flagged = spike_second_pass(
            vals, seg_times, corr1, lens, k=spike_k, thresh=spike_thresh,
            min_fix=spike_min_fix)
    # 评估（漏斗口径；rep 原帧号 = sample[rep_idx]）
    seg = raw = e_raw = e_p1 = e_full = 0
    for i, (seg_, r_, c1, c2) in enumerate(zip(segs, vals, corr1, corr2)):
        t = truth.get(sample[reps[i]])
        if t is None or r_ is None:
            continue
        seg += 1
        if abs(r_ - t) > TOL:
            raw += 1
        if abs(r_ - t) > TOL:
            e_raw += 1
        if abs(c1 - t) > TOL:
            e_p1 += 1
        if abs(c2 - t) > TOL:
            e_full += 1
    dt = time.perf_counter() - t0
    print(f"  段 {len(segs)}（计数 {seg}）原始误读 {raw} | 无后处理 {e_raw} "
          f"| 第一遍 {e_p1} | 全链 {e_full} | 第二遍修复 {len(flagged)}"
          f"（{dt:.0f}s）")
    # 第一遍剩余错误案例（诊断用）
    for i, (r_, c1, c2) in enumerate(zip(vals, corr1, corr2)):
        t = truth.get(sample[reps[i]])
        if t is None or r_ is None:
            continue
        if abs(c1 - t) > TOL:
            lo, hi = max(0, i - 2), min(len(vals), i + 3)
            ctx = " ".join(f"{vals[j]}" if j != i else f"[{vals[j]}]"
                           for j in range(lo, hi))
            print(f"    ✗ 第一遍遗留 #{i} raw={r_} c1={c1} truth={t} "
                  f"len={lens[i]} | {ctx}")
    # 第二遍改动案例明细（正确性判定）
    for i in flagged:
        t = truth.get(sample[reps[i]])
        ok = t is not None and abs(corr2[i] - t) <= TOL
        lo, hi = max(0, i - 2), min(len(vals), i + 3)
        ctx = " ".join(f"{vals[j]}" if j != i else f"[{vals[j]}]"
                       for j in range(lo, hi))
        print(f"    ⚑ #{i} raw={vals[i]} c1={corr1[i]} c2={corr2[i]} "
              f"truth={t} len={lens[i]} {'✓' if ok else '✗误改'} | {ctx}")
    return {"seg": seg, "raw": raw, "e_raw": e_raw, "e_p1": e_p1,
            "e_full": e_full, "n2": len(flagged)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-only", action="store_true",
                    help="禁用第二遍尖峰检测（模拟帧率自适应关闭）")
    ap.add_argument("--thresh", type=float, default=config.SEG_SPIKE_THRESH)
    ap.add_argument("--k", type=int, default=config.SEG_SPIKE_K)
    ap.add_argument("--min-fix", type=float, default=config.SEG_SPIKE_MIN_FIX)
    ap.add_argument("--scale-windows", action="store_true",
                    help="固定帧数窗口按 fps 自适应缩放（anchor_max/win/"
                         "island/calib）")
    args = ap.parse_args()

    from ocr_native import auto_ocr_thread_count
    eng = OcrEngine(config.DEFAULT_OCR_MODEL, "tensorrt",
                    fill_width=config.DEFAULT_FILL_WIDTH,
                    num_threads=auto_ocr_thread_count())
    print(f"OCR 引擎: {eng.backend_name} | spike k={args.k} thresh="
          f"{args.thresh} min_fix={args.min_fix} first_only={args.first_only}")
    tot = {"seg": 0, "raw": 0, "e_raw": 0, "e_p1": 0, "e_full": 0, "n2": 0}
    for v in VIDEOS:
        r = run_video(v, eng, spike_k=args.k, spike_thresh=args.thresh,
                      spike_min_fix=args.min_fix,
                      first_only=args.first_only,
                      scale_windows=args.scale_windows)
        for k in tot:
            tot[k] += r[k]
    print(f"\n合计: 段 {tot['seg']} 原始误读 {tot['raw']} | 无后处理 "
          f"{tot['e_raw']} | 第一遍 {tot['e_p1']} | 全链 {tot['e_full']}"
          f" | 第二遍修复 {tot['n2']}")
    print("（57fps 生产基线：全部视频 0 错误）")


if __name__ == "__main__":
    main()
