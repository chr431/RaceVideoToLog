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
import os as _os
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
                 max_accel_mps2: float, fps: float | None, frame_start: int | None,
                 frame_end: int | None, target_h: int, max_width: int,
                 ocr_model: str, speed_format: str = "km/h",
                 frame_div: int = 1, pad: int = 0,
                 C: float = config.SEG_C, win: int = config.SEG_WIN,
                 mult: float = config.SEG_MULT,
                 min_dev: float = config.SEG_MIN_DEV,
                 med_k: int = config.SEG_MED_K,
                 detect_floor: float = config.SEG_DETECT_FLOOR,
                 anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES,
                 progress_cb=None, cancel_check=None):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        self._max_speed = max_speed_kmh
        self._max_accel = max_accel_mps2
        self._fps = fps  # None → _decode_all 里从 decord 推导
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
        self._med_k = med_k
        self._detect_floor = detect_floor
        self._anchor_max = anchor_max
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.rows: list = []
        self.timing: dict = {}
        # 段级 review / finalize 支持（run 后填充）
        self.segments: list[dict] = []   # [{start,end,value,rep_frame,rep_crop}]
        self.crops: dict = {}            # 解码的 ROI 帧（review 懒加载预览用）
        self._segs: list = []
        self._frames: list = []
        self._ocr_vals: list = []

    # ── 阶段 1：解码 + 特征（diff/清晰度）──
    def _decode_all(self):
        from decord import VideoReader, gpu, cpu
        vr = None
        label = "CPU"
        _force = _os.environ.get("DECORD_FORCE_CPU", "").strip() == "1"
        if not _force:
            try:
                from decord import gpu as _g
                vr = VideoReader(str(self._video_path), ctx=_g(0))
                label = "GPU"
            except Exception:
                vr = None
        if vr is None:
            vr = VideoReader(str(self._video_path), ctx=cpu(0))
        self._backend = f"decord/{label}"
        # fps 未指定时从解码器推导（CLI/GUI 无需传 fps）
        if self._fps is None:
            for m in ("get_avg_fps", "get_fps"):
                fn = getattr(vr, m, None)
                if fn is None:
                    continue
                try:
                    self._fps = float(fn())
                    break
                except Exception:
                    self._fps = None
            if not self._fps or self._fps <= 0:
                self._fps = 30.0
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
            if k % 100 == 0:
                self._cancel()
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
        """中值滤波检测：平滑值曲线（跟随弯曲），误读=尖峰被中值剔除。

        对每段 i，smoothed = 局部非 None 值的中位数（段索引窗口 ±med_k）。
        正确段贴合中值（偏差 ≤ 局部带宽），误读尖峰偏差大。门限 =
        max(局部相邻差中位数, detect_floor) × mult。边缘段（左右一侧无
        上下文）不 flag —— 中值在单调上升/下降区滞后，视频起止的低/高速段
        会被窗口拉偏误判（test2 起始 5→8→12 回归源）。None 段恒 suspect。
        """
        n = len(seg_vals)
        # 局部带宽（相邻差中位数，帧窗口内）
        if n >= 2:
            gaps = np.diff(seg_times)
            med_gap = float(np.median(gaps)) if len(gaps) else 1.0
        else:
            med_gap = 1.0
        win_frames = min(self._win * max(med_gap, 1.0), 120.0)
        st = np.asarray(seg_times, dtype=np.float64)
        bw = [0.0] * n
        for i in range(n):
            ti = seg_times[i]
            lo = int(np.searchsorted(st, ti - win_frames, side="left"))
            hi = int(np.searchsorted(st, ti + win_frames, side="right"))
            dvs = [abs(seg_vals[j] - seg_vals[j - 1])
                   for j in range(lo + 1, hi)
                   if seg_vals[j] is not None and seg_vals[j - 1] is not None]
            bw[i] = max(float(np.median(dvs)) if dvs else 0.0,
                        self._detect_floor)
        suspect = [False] * n
        for i in range(n):
            if seg_vals[i] is None:
                suspect[i] = True
                continue
            lo = max(0, i - self._med_k)
            hi = min(n, i + self._med_k + 1)
            nbrs = [seg_vals[j] for j in range(lo, hi) if seg_vals[j] is not None]
            if len(nbrs) < 3:
                suspect[i] = True
                continue
            lefts = any(seg_vals[j] is not None for j in range(lo, i))
            rights = any(seg_vals[j] is not None for j in range(i + 1, hi))
            if not (lefts and rights):
                continue
            med = float(np.median(nbrs))
            if abs(seg_vals[i] - med) > bw[i] * self._mult:
                suspect[i] = True
        return suspect

    def _correct(self, seg_vals, seg_times, suspect):
        """锚点插值纠正：suspect/None 段取最近可信锚点线性插值。

        anchor_max 限锚点帧距离：近锚点（≤anchor_max 帧）才插值，防远锚点
        跨弯曲区误插值（低 min_dev 下过度纠正的回归源）。None 段恒插值。
        """
        out = list(seg_vals)
        n_corr = 0
        for i in range(len(seg_vals)):
            if seg_vals[i] is None:
                # None 段（OCR 未读出）→ 必须插值，否则帧输出 -1
                pass
            elif not suspect[i]:
                continue
            ti = seg_times[i]
            la = None
            for j in range(i - 1, -1, -1):
                if not suspect[j] and seg_vals[j] is not None:
                    if ti - seg_times[j] <= self._anchor_max:
                        la = j
                    break
            ra = None
            for j in range(i + 1, len(seg_vals)):
                if not suspect[j] and seg_vals[j] is not None:
                    if seg_times[j] - ti <= self._anchor_max:
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
                if seg_vals[i] is None or abs(interp - seg_vals[i]) > self._min_dev:
                    out[i] = round(interp)
                    n_corr += 1
        return out, n_corr

    # ── 主入口（顺序：解码 → 分段 → 段OCR → 检测纠正 → CSV）──
    def run(self, output_path):
        t_total = time.perf_counter()
        self._progress("解码...", 2.0)
        frames, crops, grays, sharp = self._decode_all()
        self._cancel()
        self._progress("分段...", 73.0)
        segs = self._segment(frames, grays)
        self._cancel()
        self._progress("段值 OCR...", 73.0)
        seg_vals, rep_frames = self._ocr_segments(segs, crops, sharp)
        self._cancel()
        seg_times = [seg[len(seg) // 2] for seg in segs]
        suspect = self._detect(seg_vals, seg_times)
        corr, self._n_corr = self._correct(seg_vals, seg_times, suspect)
        self.rows = self._build_rows(frames, segs, corr)
        self._store_run_state(frames, crops, segs, seg_vals, rep_frames, corr)
        self._write_csv(self.rows, output_path)
        self.timing["total"] = time.perf_counter() - t_total
        self._progress("完成", 100.0)
        return self.rows

    def _store_run_state(self, frames, crops, segs, seg_vals, rep_frames, corr):
        """保存 run() 的中间状态，供 GUI 段级 review / finalize 使用。"""
        self._frames = frames
        self.crops = crops
        self._segs = segs
        self._ocr_vals = list(seg_vals)
        self._corr_vals = list(corr)
        self.segments = [
            {"start": seg[0], "end": seg[-1],
             "frames": list(seg),  # 该段的采样帧列表（review 逐帧绘制用）
             "value": corr[i],
             "ocr_value": seg_vals[i],
             "rep_frame": rep_frames[i],
             "rep_crop": crops[rep_frames[i]]}
            for i, seg in enumerate(segs)
        ]

    def timing_flat(self) -> dict:
        """展平 timing dict（丢弃嵌套值），兼容 headless/gui_export 调用。"""
        return {k: v for k, v in self.timing.items()
                if isinstance(v, (int, float))}

    def finalize(self, output_path, segment_values=None):
        """从（可能被用户编辑的）段值重建 rows 并重写 CSV。

        segment_values: 与 self.segments 等长的段修正值；None = 用 run() 的
        纠正结果。GUI 段级 review 改值后调用，single-pass 重写输出。
        """
        if segment_values is None:
            vals = self._corr_vals
        else:
            vals = list(segment_values)
            if len(vals) != len(self._segs):
                raise ValueError(f"段值数量 {len(vals)} ≠ 段数 {len(self._segs)}")
            # corrected 计数相对原始 OCR 值重算
            self._n_corr = sum(1 for ov, nv in zip(self._ocr_vals, vals)
                               if nv is not None and ov != nv)
            self._corr_vals = vals
        self.rows = self._build_rows(self._frames, self._segs, vals)
        self._write_csv(self.rows, output_path)
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
            r = self._roi
            # 头行用 fh.write 直接写（csv.writer 会给含逗号的注释加引号，
            # 行首变 " 导致 parse_csv_header 解析失败 —— GUI 导入 CSV 兼容性）
            fh.write(f"# RaceVideoToLog v{config.__version__}\n")
            fh.write(f"# video={self._video_path.name}, fps={self._fps:.3f}\n")
            fh.write(f"# roi={r[0]},{r[1]},{r[2]},{r[3]}, format={self._speed_format}"
                     f", frame_start={self._frame_start or ''}"
                     f", frame_end={self._frame_end or ''}\n")
            fh.write(f"# max_speed={self._max_speed}, max_accel={self._max_accel}"
                     f", div={self._frame_div}, target_h={self._target_h}"
                     f", max_width={self._max_width}, pad={self._pad}\n")
            fh.write(f"# backend={self._backend}, model={self._ocr_model}\n")
            fh.write(f"# segments={self._n_segments}, corrected={self._n_corr}\n")
            tstr = ", ".join(f"{k}={v:.2f}" for k, v in self.timing.items())
            if tstr:
                fh.write(f"# timing: {tstr}\n")
            # 数据行用 csv.writer（int 帧号/距离/速度/flag，对齐旧格式）
            w = csv.writer(fh)
            for row in rows:
                w.writerow((f"{int(row[0])}", f"{row[1]:.2f}",
                            f"{int(row[2])}", str(row[3])))
