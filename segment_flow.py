"""分段流水线（生产）：解码+分段+段值OCR 流水线化 → 段级纠错 → CSV。

分段 OCR 大幅减少调用（36-64%），解码（I/O 瓶颈）与段值 OCR 线程重叠
摊薄墙钟。算法（experiment-binary-ocr 分支验证）：
- diff 分段：聚类判别（max 3×3 窗口和 < C ⇒ 显示未变），解码循环内增量计算
- 段值：每段最清晰代表帧 OCR（sharpness=灰度std），OCR 线程批处理闭合段
- 段级检测：中值滤波（跟随弯曲，误读=尖峰被中值剔除）
- 段级纠正：可信锚点插值（锚点距离上界，跳过曲线正确段）

实现已按职责拆分：segmentation.py（灰度/Otsu/聚类）、seg_correction.py
（检测/置信度/DP）。本文件保留 SegmentPipeline 编排、串行参考路径与
CSV 输出。CPU+NVDEC 混合解码为引擎一等后端（decode_backend='hybrid'）。

注意：_decode_all/_segment/_ocr_segments/_detect/_correct 是串行参考路径
（仅 tools/ 与测试使用），生产 run() 走 _run_pipelined + _dense_correct。
"""
from __future__ import annotations
import csv
import logging
import os as _os
import threading
import time
from pathlib import Path

import numpy as np

import config
from constants import Flag
from ocr_engine import extract_speed_value
from video_ocr_engine import FieldExtractor  # 识别链（解码/分段/OCR 文本）由引擎提供
from segmentation import (  # noqa: F401 — 兼容 tools/tests 的历史导入路径
    _cluster_win3, _gray, _gray_batch, _gray_seg,
    _gray_seg_batch, _gray_seg_yuv, _gray_seg_yuv_batch, _otsu,
)
from video_utils import _nv12_luma_full, _preprocess_standard, nv12_to_rgb
from seg_correction import (
    confidence_scores, correct_segments, dense_correct, detect_segments,
    dp_run, fill_values, local_bandwidth, spike_second_pass,
)

logger = logging.getLogger("RaceVideoToLog.segment_flow")


class _ProgressGate:
    """把并行阶段（解码∥OCR）的进度回调收敛成单调、不回退的进度。

    decode 与 OCR 真正并行：OCR 可能已报到 58-86，而解码线程还在报
    3-58。若直接透传，GUI 进度条会来回跳、文字在“解码+分段”和“OCR”
    之间反复横跳。本类只允许：
      - 百分比严格前进；或
      - 进入更靠后的阶段（decode→OCR→纠错）时即使百分比相同也切换。
    同一阶段内百分比相同的重复消息会被丢弃（例如 OCR 多个末段都报
    86.0 时只显示第一条）。
    """
    def __init__(self, emit) -> None:
        self._emit = emit
        self._lock = threading.Lock()
        self._last_pct = -1.0
        self._last_phase = -1

    @staticmethod
    def _phase(msg: str, pct: float) -> int:
        # 按消息内容判断阶段，避免 58.0 这种边界值被 pct 误判：
        # 解码最后一条也是 58.0，而 OCR 第一条也是 58.0。
        if msg == "检测纠正..." or msg == "完成":
            return 2
        if msg.startswith("[OCR]"):
            return 1
        return 0

    def __call__(self, msg: str, pct: float) -> None:
        phase = self._phase(msg, pct)
        with self._lock:
            if pct < self._last_pct:
                return
            if pct == self._last_pct and phase <= self._last_phase:
                return
            self._last_pct = pct
            self._last_phase = phase
        self._emit(msg, pct)


def _ocr_batch_size() -> int:
    """OCR 批大小（段数）：OCR_BATCH 实验钩子 > config.OCR_BATCH_SIZE。"""
    _env = _os.environ.get("OCR_BATCH")
    if _env and _env.isdigit():
        return max(1, int(_env))
    return config.OCR_BATCH_SIZE


class SegmentPipeline(FieldExtractor):
    """生产分段流水线：识别链由 video_ocr_engine.FieldExtractor 继承，
    本类叠加速度后处理（检测/置信度/DP/第二遍尖峰/CSV 输出）。"""

    def __init__(self, video_path: str, roi: tuple, max_speed_kmh: float,
                 max_accel_mps2: float, fps: float | None, frame_start: int | None,
                 frame_end: int | None, force_aspect: float = 0.0,
                 speed_format: str = "km/h",
                 decode_backend: str = "auto",
                 ocr_backend: str = "auto",
                 buffer_size: int = config.DEFAULT_BUFFER_SIZE,
                 fill_width: int = config.DEFAULT_FILL_WIDTH,
                 C: float = config.SEG_C, win: int = config.SEG_WIN,
                 mult: float = config.SEG_MULT,
                 min_dev: float = config.SEG_MIN_DEV,
                 med_k: int = config.SEG_MED_K,
                 detect_floor: float = config.SEG_DETECT_FLOOR,
                 single_floor: float = config.SEG_SINGLE_FLOOR,
                 anchor_max: float = config.SEG_ANCHOR_MAX_FRAMES,
                 conf_w_med: float = config.SEG_CONF_W_MED,
                 conf_w_jerk: float = config.SEG_CONF_W_JERK,
                 conf_jerk_scale: float = config.SEG_CONF_JERK_SCALE,
                 dp_obs_weight: float = config.SEG_DP_OBS_WEIGHT,
                 dp_accel_weight: float = config.SEG_DP_ACCEL_WEIGHT,
                 dp_max_dv_cap: float = config.SEG_DP_MAX_DV_CAP,
                 dp_anchor_cost: float = config.SEG_DP_ANCHOR_COST,
                 dp_change_threshold: float = config.SEG_DP_CHANGE_THRESHOLD,
                 dp_anchor_conf: float = config.SEG_DP_ANCHOR_CONF,
                 dp_deanchor_jerk_min: float = config.SEG_DP_DEANCHOR_JERK_MIN,
                 dp_deanchor_jerk_max: float = config.SEG_DP_DEANCHOR_JERK_MAX,
                 progress_cb=None, cancel_check=None,
                 gray_output: bool = False,
                 yuv_output: bool = False,
                 merge_similar: bool = True,
                 merge_similar_threshold: float | None = None,
                 merge_text_sep: str | None = None):
        # 引擎字段（解码/分段/OCR 识别链）由 FieldExtractor.__init__ 设置
        super().__init__(
            video_path=video_path, roi=roi, frame_start=frame_start,
            frame_end=frame_end, force_aspect=force_aspect,
            decode_backend=decode_backend, ocr_backend=ocr_backend,
            buffer_size=buffer_size, fill_width=fill_width, C=C, fps=fps,
            progress_cb=progress_cb, cancel_check=cancel_check,
            gray_output=gray_output, yuv_output=yuv_output,
            # GUI review 需要代表帧预览，显式保留（引擎默认 True，这里加固）
            keep_crops=True, keep_frames=True,
            merge_similar=merge_similar,
            merge_similar_threshold=merge_similar_threshold,
            merge_text_sep=merge_text_sep)
        # ── 速度后处理与速度专属字段（应用层，不在引擎）──
        # 引擎重构后不再初始化纠错计数（应用语义），此处补默认值
        self._n_corr = 0
        self._max_speed = max_speed_kmh
        self._max_accel = max_accel_mps2
        self._speed_format = speed_format
        self._win = win
        self._mult = mult
        self._min_dev = min_dev
        self._med_k = med_k
        self._detect_floor = detect_floor
        self._single_floor = single_floor
        self._anchor_max = anchor_max
        self._conf_w_med = conf_w_med
        self._conf_w_jerk = conf_w_jerk
        self._conf_jerk_scale = conf_jerk_scale
        self._dp_obs_weight = dp_obs_weight
        self._dp_accel_weight = dp_accel_weight
        self._dp_max_dv_cap = dp_max_dv_cap
        self._dp_anchor_cost = dp_anchor_cost
        self._dp_change_threshold = dp_change_threshold
        self._dp_anchor_conf = dp_anchor_conf
        self._dp_deanchor_jerk_min = dp_deanchor_jerk_min
        self._dp_deanchor_jerk_max = dp_deanchor_jerk_max

    # ── 公共只读 API（run 后有效；tools/tests/GUI 一律走这里，
    #    不直接读 _ 前缀私有状态）──────────────────────────────
    # 引擎重构后 corrected/confidence/n_corrected 是速度领域状态，
    # 由应用层（本类）实现；frames/segment_frames/ocr_values/ocr_texts/
    # ocr_confidences/n_segments 仍由 FieldExtractor 基类提供。

    @property
    def corrected_values(self) -> list:
        """段级纠错后数值（DP + 第二遍尖峰后；run 后有效）。"""
        return self._corr_vals

    @corrected_values.setter
    def corrected_values(self, v: list) -> None:
        self._corr_vals = v

    @property
    def confidence_values(self) -> list:
        """段级置信度（供测试夹具/tools 使用）。"""
        return self._conf_vals

    @confidence_values.setter
    def confidence_values(self, v: list) -> None:
        self._conf_vals = v

    @property
    def n_corrected(self) -> int:
        """被纠正段数（DP + 第二遍尖峰）。"""
        return self._n_corr

    @n_corrected.setter
    def n_corrected(self, v: int) -> None:
        self._n_corr = v

    # ── 阶段 1：解码 + 特征（diff/清晰度）──

    # ── 阶段 2：分段（聚类 diff）──

    # ── 阶段 3：段值 OCR（每段最清晰代表帧，批量）──

    # ── 阶段 4：段级检测 + 纠正（薄封装，实现在 seg_correction.py）──
    def _local_bandwidth(self, seg_vals, seg_times):
        return local_bandwidth(seg_vals, seg_times, self._win)

    def _detect(self, seg_vals, seg_times, seg_lens=None):
        return detect_segments(seg_vals, seg_times, seg_lens,
                               med_k=self._med_k, mult=self._mult,
                               detect_floor=self._detect_floor,
                               single_floor=self._single_floor,
                               win=self._win)

    def _correct(self, seg_vals, seg_times, suspect):
        return correct_segments(seg_vals, seg_times, suspect,
                                anchor_max=self._anchor_max,
                                min_dev=self._min_dev)

    def _confidence(self, seg_vals, seg_times, seg_lens=None):
        return confidence_scores(seg_vals, seg_times, seg_lens,
                                 med_k=self._med_k,
                                 detect_floor=self._detect_floor,
                                 conf_jerk_scale=self._conf_jerk_scale,
                                 conf_w_med=self._conf_w_med,
                                 conf_w_jerk=self._conf_w_jerk,
                                 win=self._win)

    def _fill_values(self, seg_vals, seg_times, is_anchor):
        return fill_values(seg_vals, seg_times, is_anchor,
                           anchor_max=self._anchor_max)

    def _dense_correct(self, seg_vals, seg_times, conf):
        return dense_correct(
            seg_vals, seg_times, conf,
            max_speed=self._max_speed, max_accel=self._max_accel,
            fps=self._fps or config.DEFAULT_FPS_FALLBACK,
            dp_anchor_conf=self._dp_anchor_conf,
            dp_deanchor_jerk_min=self._dp_deanchor_jerk_min,
            dp_deanchor_jerk_max=self._dp_deanchor_jerk_max,
            dp_change_threshold=self._dp_change_threshold,
            dp_obs_weight=self._dp_obs_weight,
            dp_accel_weight=self._dp_accel_weight,
            dp_max_dv_cap=self._dp_max_dv_cap,
            dp_anchor_cost=self._dp_anchor_cost,
            anchor_max=self._anchor_max)

    def _spike_second_pass(self, seg_vals, seg_times, corr1, seg_lens=None):
        """第二遍尖峰检测（v2.16 实验）：第一遍去污染后补抓孤立 2-off
        单帧误读（参数 config.SEG_SPIKE_*，实测 final 11→5、harm=0）。"""
        return spike_second_pass(seg_vals, seg_times, corr1, seg_lens)

    def _dp_run(self, lo, hi, seg_vals, seg_times, is_anchor, fill=None):
        return dp_run(lo, hi, seg_vals, seg_times, is_anchor,
                      self._max_speed, self._max_accel, self._fps or 30.0,
                      dp_obs_weight=self._dp_obs_weight,
                      dp_accel_weight=self._dp_accel_weight,
                      dp_max_dv_cap=self._dp_max_dv_cap,
                      dp_anchor_cost=self._dp_anchor_cost,
                      fill=fill)

    def _ocr_segments(self, segs, crops, sharp):
        """串行参考路径：对每段代表帧 OCR，返回速度数值（应用层语义）。

        引擎提供文本识别（_run_pipelined），本方法用于 tools/测试的串行
        参考路径——为保持 `(seg_vals, rep_frames)` 返回结构不变，这里对
        识别文本做速度解析（extract_speed_value）。生产走 _run_pipelined。
        """
        from ocr_native import OcrEngine
        eng = OcrEngine(self._ocr_model, self._ocr_engine_type(),
                        fill_width=self._fill_width,
                        num_threads=self._ocr_num_threads(),
                        progress_cb=lambda msg: self._progress(msg, 2.5))
        self._ocr_backend_used = eng.backend_name
        seg_vals = []
        rep_frames = []
        texts = []
        confs = []
        t0 = time.perf_counter()
        B = _ocr_batch_size()
        reps = [max(seg, key=lambda fi: sharp[fi]) for seg in segs]
        for k in range(0, len(segs), B):
            chunk = segs[k:k + B]
            procs = [_preprocess_standard(
                _nv12_luma_full(crops[rep], self._color_range)[..., None]
                if self._yuv_output else crops[rep],
                force_aspect=self._force_aspect)
                for rep in reps[k:k + B]]
            results = eng(procs)
            for rep, res in zip(reps[k:k + B], results):
                sv, _rt, _c = extract_speed_value(res)
                seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
                rep_frames.append(rep)
                if hasattr(res, "txts"):
                    texts.append(str(res.txts[0])
                                 if res.txts and res.txts[0] else None)
                    scores = getattr(res, "scores", [])
                    confs.append(float(scores[0]) if scores else 0.0)
                else:
                    texts.append(None); confs.append(0.0)
            done = min(k + B, len(segs))
            self._progress(f"[OCR] 段: {done}/{len(segs)}",
                           73 + done / max(len(segs), 1) * 15)
        self.timing["ocr"] = time.perf_counter() - t0
        self._n_segments = len(segs)
        self._ocr_texts = texts
        self._ocr_confs = confs
        return seg_vals, rep_frames

    # ── 主入口（流水线：解码∥分段∥段OCR 重叠 → 检测纠正 → CSV）──
    def run(self, output_path):
        t_total = time.perf_counter()
        # 并行进度收敛：解码线程和 OCR 线程各自报百分比，可能互相“倒车”，
        # 这里用 gate 保证 GUI 只看到单调进度（完成后恢复原始回调，
        # 允许同一 pipeline 对象再次 run）。
        _orig_progress = self._progress
        self._progress = _ProgressGate(_orig_progress)
        try:
            self._progress("解码+分段+段值OCR...", 2.0)
            frames, segs, ocr_texts, ocr_confs, rep_frames = \
                self._run_pipelined()
            # 应用层解析：引擎输出原始文本 → 速度数值（extract_speed_value
            # 的文本直转版；识别层不感知速度语义）
            from ocr_text import _extract_speed_from_text
            seg_vals = []
            for txt, c in zip(ocr_texts, ocr_confs):
                sv, _rt, _c = _extract_speed_from_text(str(txt)
                                                       if txt else "", c)
                seg_vals.append(int(sv) if sv is not None and sv >= 0
                                else None)
            self._cancel()
            self._progress("检测纠正...", 88.0)
            t_corr = time.perf_counter()
            seg_times = [seg[len(seg) // 2] for seg in segs]
            conf = self._confidence(seg_vals, seg_times,
                                    [len(s) for s in segs])
            self._conf_vals = list(conf)       # 供 finalize/flag 判定复用
            corr, n1 = self._dense_correct(seg_vals, seg_times, conf)
            # 第二遍尖峰检测：第一遍去污染后补抓孤立 2-off 单帧误读
            # （v2.16 实验，实测 final 11→5、harm=0；参数见 config.SEG_SPIKE_*）
            # 帧率自适应：低帧率（<SEG_SPIKE_MIN_FPS，如 30fps）下相邻段
            # 真实速度变化达 1-2 km/h，正确段孤立凸起与误读不可区分，
            # 误改>修对（净负）→ 跳过（30fps 模拟实测 10→2 错误）。
            if (self._fps or 0) >= config.SEG_SPIKE_MIN_FPS:
                corr, n2, _flagged = self._spike_second_pass(
                    seg_vals, seg_times, corr, [len(s) for s in segs])
            else:
                n2 = 0
            self._n_corr = n1 + n2
            self.timing["correction"] = time.perf_counter() - t_corr
            self.rows = self._build_rows(frames, segs, corr, raw=seg_vals,
                                         conf=conf)
            self._store_run_state(frames, self.crops, segs, seg_vals,
                                  rep_frames, corr)
            self._write_csv(self.rows, output_path)
            self.timing["total"] = time.perf_counter() - t_total
            self._progress("完成", 100.0)
        finally:
            self._progress = _orig_progress
        return self.rows


    def _store_run_state(self, frames, crops, segs, seg_vals, rep_frames, corr):
        """保存 run() 的中间状态，供 GUI 段级 review / finalize 使用。

        segments[] 每项含 OCR 原始文本（"text"）与置信度（"ocr_conf"），
        供 review 展示/通用字段提取消费；"value"/"ocr_value" 保持速度语义。
        """
        self._frames = frames
        self.crops = crops
        self._segs = segs
        self._ocr_vals = list(seg_vals)
        self._corr_vals = list(corr)
        texts = getattr(self, "_ocr_texts", [])
        confs = getattr(self, "_ocr_confs", [])
        self.segments = [
            {"start": seg[0], "end": seg[-1],
             "frames": list(seg),  # 该段的采样帧列表（review 逐帧绘制用）
             "value": corr[i],
             "ocr_value": seg_vals[i],
             "text": texts[i] if i < len(texts) else None,
             "ocr_conf": confs[i] if i < len(confs) else 0.0,
             "rep_frame": rep_frames[i],
             "rep_crop": crops.get(rep_frames[i])}
            for i, seg in enumerate(segs)
        ]


    def finalize(self, output_path, segment_values=None, pinned_indices=None):
        """从（可能被用户编辑的）段值重建 rows 并重写 CSV。

        segment_values: 与 self.segments 等长的段修正值；None = 用 run() 的
        纠正结果。GUI 段级 review 改值后调用，single-pass 重写输出。
        pinned_indices: 用户手动修正的段索引集合（标 Flag.PINNED），
        覆盖 DP/插值 flag 判定。
        """
        if pinned_indices:
            self._pinned = set(pinned_indices)
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
    def _build_rows(self, frames, segs, corr, raw=None, conf=None,
                    pinned=None):
        """构建输出行 [frame, dist, speed, flag]。

        flag 按段来源推理（段内帧共享）：
        - pinned（用户手动修正）→ PINNED
        - raw None（OCR 未读出）→ FILL_INTERP
        - 值被改（corr != raw）→ DP_CORRECTED
        - 值未被改但 conf ≥ 锚定阈值（物理验证通过）→ HIGH_TRUST（绝大多数）
        - 其余（未验证的原始值）→ RAW
        """
        if raw is None:
            raw = self._ocr_vals
        if conf is None:
            conf = getattr(self, "_conf_vals", None)
        if pinned is None:
            pinned = getattr(self, "_pinned", set())
        rows = []
        for i, (seg, val) in enumerate(zip(segs, corr)):
            if i in pinned:
                flag = Flag.PINNED
            elif i < len(raw) and raw[i] is None:
                flag = Flag.FILL_INTERP
            elif i < len(raw) and val is not None and val != raw[i]:
                flag = Flag.DP_CORRECTED
            elif conf is not None and i < len(conf) \
                    and conf[i] >= self._dp_anchor_conf:
                flag = Flag.HIGH_TRUST
            else:
                flag = Flag.RAW
            for fi in seg:
                rows.append([fi, 0.0, val if val is not None else -1, flag])
        rows.sort(key=lambda r: r[0])
        # 距离积分
        dist = 0.0
        prev = None
        for row in rows:
            if prev is not None and row[2] >= 0:
                dt = (row[0] - prev[0]) / self._fps
                dist += prev[2] / config.MPS_TO_KMH * dt
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
                     f", force_aspect={self._force_aspect}, fill_width={self._fill_width}\n")
            fh.write(f"# backend={self._backend}, model={self._ocr_model}\n")
            # 与老版本对齐：只记录本次真正用于推理的 OCR 引擎，
            # 不记录用户请求的 auto 等原始参数（避免 auto 被 CSV 回灌）
            fh.write(f"# ocr_backend={self._ocr_backend_used}\n")
            fh.write(f"# segments={self._n_segments}, corrected={self._n_corr}\n")
            tstr = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, (int, float))
                else f"{k}={v}" for k, v in self.timing.items())
            if tstr:
                fh.write(f"# timing: {tstr}\n")
            # 数据行用 csv.writer（int 帧号/距离/速度/flag，对齐旧格式）
            w = csv.writer(fh)
            for row in rows:
                w.writerow((f"{int(row[0])}", f"{row[1]:.2f}",
                            f"{int(row[2])}", str(row[3])))
