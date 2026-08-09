"""分段流水线（生产）：解码 → diff分段 → 段值OCR → 段级纠错 → CSV。

与 ProcessingPipeline 同接口（run/finalize），CLI 用 --segment 切换。
分段 OCR 大幅减少调用（36-64%），瓶颈转移到解码（读全帧算 diff/清晰度）。

算法（experiment-binary-ocr 分支验证）：
- diff 分段：聚类判别（max 3×3 窗口和 < C ⇒ 显示未变）
- 段值：每段最清晰代表帧 OCR（sharpness=灰度std）
- 段级检测：median-of-pairs abs（残差/带宽）
- 段级纠正：可信锚点插值（min_dev 门，跳过曲线正确段）
"""
from __future__ import annotations
import csv
import logging
import re
import time
from pathlib import Path

import numpy as np

import config
from constants import Flag
from monitor import STAGE
from ocr_engine import extract_speed_value

logger = logging.getLogger("RaceVideoToLog.segment_flow")

_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _gray(crop: np.ndarray) -> np.ndarray:
    return (crop.astype(np.float32) @ _GRAY_W).astype(np.uint8)


def _sharpness(crop: np.ndarray) -> float:
    return float(_gray(crop).std())


def _otsu(g: np.ndarray) -> int:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = int(g.size)
    st = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = 127
    vmax = -1.0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb = sb / wb
        mf = (st - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def _cluster_win3(diff: np.ndarray) -> float:
    """最大 3×3 窗口变化像素和 —— 聚类判别的廉价代理（纯 numpy，无 scipy）。

    原 scipy.ndimage.label 连通分量对 test6 23k 边贡献 ~2.3s；且 scipy 非
    pyproject 依赖，PyInstaller 打包会连带整个 scipy 增肥 exe。本实现用
    6 次切片错位累加求最大 3×3 窗口和（越界按 0），16µs/边，数值与
    uniform_filter 逐位一致（含边界，500 随机掩码最大差 0）。
    语义：真实数字变化必然产生 ≥5 像素连成 3×3 的密集簇（实测变帧恒=9）；
    噪声孤立像素的最大窗口和 < 5。C=5 下 test/test5/test6 0 漏检且段数更少。
    """
    if not diff.any():
        return 0.0
    s = diff.astype(np.int32)
    # 行向 3 列和（左右越界 0）
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    # 列向 3 行和（上下越界 0）
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


class SegmentPipeline:
    """生产分段流水线。"""

    def __init__(self, video_path: str, roi: tuple, max_speed_kmh: float,
                 max_accel_mps2: float, fps: float, frame_start: int | None,
                 frame_end: int | None, target_h: int, max_width: int,
                 ocr_model: str, speed_format: str = "km/h",
                 frame_div: int = 1, pad: int = 0,
                 C: float = 5.0, win: int = 30, mult: float = 3.0,
                 min_dev: float = 15.0, progress_cb=None):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        self._max_speed = max_speed_kmh
        self._max_accel = max_accel_mps2
        self._fps = fps
        self._frame_start = frame_start or 0
        self._frame_end = frame_end
        self._target_h = target_h
        self._max_width = max_width
        self._ocr_model = ocr_model
        self._speed_format = speed_format
        self._frame_div = frame_div
        self._pad = pad
        self._C = C
        self._win = win
        self._mult = mult
        self._min_dev = min_dev
        self._progress = progress_cb or (lambda m, p: None)
        self.rows: list = []
        self.timing: dict = {}

    # ── 阶段 1：解码 + 特征（diff/清晰度）──
    def _decode_all(self):
        from decord import VideoReader, gpu, cpu
        vr = None
        label = "CPU"
        try:
            from decord import gpu as _g
            vr = VideoReader(str(self._video_path), ctx=_g(0))
            label = "GPU"
        except Exception:
            vr = VideoReader(str(self._video_path), ctx=cpu(0))
        self._backend = f"decord/{label}"
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end, self._frame_div))
        crops = {}
        grays = {}
        sharp = {}
        t0 = time.perf_counter()
        for k, fi in enumerate(frames):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if c.shape[0] != y2 - y1 + 1 or c.shape[1] != x2 - x1 + 1:
                c = c[y1:y2 + 1, x1:x2 + 1]
            crops[fi] = c
            g = _gray(c)
            grays[fi] = g
            sharp[fi] = float(g.std())
            if k % 500 == 0:
                self._progress(f"[{self._backend}] 解码: {k}/{len(frames)}",
                               3 + k / max(len(frames), 1) * 70)
        self.timing["decode"] = time.perf_counter() - t0
        del vr
        return frames, crops, grays, sharp

    # ── 阶段 2：分段（聚类 diff）──
    def _segment(self, frames, grays):
        t0 = time.perf_counter()
        ths = []
        step = max(1, len(frames) // 50)
        for fi in frames[::step][:50]:
            ths.append(_otsu(grays[fi]))
        th = int(np.median(ths)) if ths else 127
        self._bin_thresh = th
        prev_b = grays[frames[0]] > th
        edges = []
        for fi in frames[1:]:
            b = grays[fi] > th
            d = prev_b != b
            edges.append(_cluster_win3(d) < self._C)
            prev_b = b
        segs = []
        s = 0
        for i in range(len(frames) - 1):
            if not edges[i]:
                segs.append(frames[s:i + 1])
                s = i + 1
        segs.append(frames[s:])
        self.timing["segment"] = time.perf_counter() - t0
        return segs

    # ── 阶段 3：段值 OCR（每段最清晰代表帧，批量）──
    def _ocr_segments(self, segs, crops, sharp):
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard
        eng = OcrEngine(self._ocr_model, "tensorrt")
        seg_vals = []
        rep_frames = []
        t0 = time.perf_counter()
        # 批量：每组 B 个代表帧一次 session.run（TRT 引擎 profile batch 上限 6，
        # 内部自动分片；B=16 摊薄预处理/launch 开销）
        B = 16
        reps = [max(seg, key=lambda fi: sharp[fi]) for seg in segs]
        for k in range(0, len(segs), B):
            chunk = segs[k:k + B]
            procs = [_preprocess_standard(crops[rep], self._target_h, self._pad,
                                          max_width=self._max_width)
                     for rep in reps[k:k + B]]
            results = eng(procs)
            for rep, res in zip(reps[k:k + B], results):
                sv, _rt, _c = extract_speed_value(res)
                seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
                rep_frames.append(rep)
            if k % 1000 == 0:
                self._progress(f"[OCR] 段: {k}/{len(segs)}",
                               73 + k / max(len(segs), 1) * 15)
        self.timing["ocr"] = time.perf_counter() - t0
        self._n_segments = len(segs)
        return seg_vals, rep_frames

    # ── 阶段 4：段级检测 + 纠正 ──
    def _detect(self, seg_vals, seg_times):
        n = len(seg_vals)
        # 窗口按帧（时间）而非段数。段索引窗口在段大小不均时失效：win3 合并
        # 噪声边界得到 300+ 帧大段，±30 段会横跨进远处减速区拉低插值期望，
        # 把正确值误判 suspect（test6 尾部 350→332 误纠错）。用中位帧间距把
        # win=30 段换算成帧窗口（≈90 帧），大段只取局部邻域；上限 120 帧防
        # 长时间巡航段把窗口撑大造成锚点稀释。
        if n >= 2:
            gaps = np.diff(seg_times)
            med_gap = float(np.median(gaps)) if len(gaps) else 1.0
        else:
            med_gap = 1.0
        win_frames = min(self._win * max(med_gap, 1.0), 120.0)
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
            bw = max(float(np.median(dvs)) if dvs else 0.0, 6.0)
            if abs(seg_vals[i] - exp) > bw * self._mult:
                suspect[i] = True
        return suspect

    def _correct(self, seg_vals, seg_times, suspect):
        out = list(seg_vals)
        n_corr = 0
        for i in range(len(seg_vals)):
            if seg_vals[i] is None:
                # None 段（OCR 未读出）→ 必须插值，否则帧输出 -1
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
                if seg_vals[i] is None or abs(interp - seg_vals[i]) > self._min_dev:
                    out[i] = round(interp)
                    n_corr += 1
        return out, n_corr

    # ── 主入口（顺序：解码 → 分段 → 段OCR → 检测纠正 → CSV）──
    def run(self, output_path):
        t_total = time.perf_counter()
        self._progress("解码...", 2.0)
        frames, crops, grays, sharp = self._decode_all()
        self._progress("分段...", 73.0)
        segs = self._segment(frames, grays)
        self._progress("段值 OCR...", 73.0)
        seg_vals, rep_frames = self._ocr_segments(segs, crops, sharp)
        seg_times = [seg[len(seg) // 2] for seg in segs]
        suspect = self._detect(seg_vals, seg_times)
        corr, self._n_corr = self._correct(seg_vals, seg_times, suspect)
        self.rows = self._build_rows(frames, segs, corr)
        self._write_csv(self.rows, output_path)
        self.timing["total"] = time.perf_counter() - t_total
        self._progress("完成", 100.0)
        return self.rows

    # ── 阶段 5：构建 rows + 写 CSV ──
    def _build_rows(self, frames, segs, corr):
        rows = []
        for seg, val in zip(segs, corr):
            for fi in seg:
                rows.append([fi, 0.0, val if val is not None else -1,
                             Flag.RAW if val is not None else Flag.RAW])
        rows.sort(key=lambda r: r[0])
        # 距离积分
        dist = 0.0
        prev = None
        for row in rows:
            if prev is not None and row[2] >= 0:
                dt = (row[0] - prev[0]) / self._fps
                dist += prev[2] / 3.6 * dt
            row[1] = round(dist, 2)
            prev = row
        return rows

    def _write_csv(self, rows, output_path):
        with Path(output_path).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow([f"# RaceVideoToLog v{config.__version__}"])
            w.writerow([f"# video={self._video_path.name}, fps={self._fps:.3f}"])
            r = self._roi
            w.writerow([f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, "
                        f"format={self._speed_format}, frame_start={self._frame_start}, "
                        f"frame_end={self._frame_end or ''}"])
            w.writerow([f"# max_speed={self._max_speed}, max_accel={self._max_accel}"])
            w.writerow([f"# backend={self._backend}, model={self._ocr_model}"])
            w.writerow([f"# segments={self._n_segments}, corrected={self._n_corr}"])
            tstr = ", ".join(f"{k}={v:.2f}" for k, v in self.timing.items())
            w.writerow([f"# timing: {tstr}"])
            for row in rows:
                w.writerow(row)
