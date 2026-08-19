"""重OCR 多预处理方案扫参 —— 验证"多种预处理取其一"思路（实验分支）。

思路：初次 OCR 用统一预处理（gray+gamma 2.0）稳妥识别；对**误读帧**
（模糊/背景遮挡），单独用多种预处理方法重 OCR —— 只要其中一种能清晰分离
数字，就能得到正确结果。

对每视频：pipe.run() 拿 crops + _ocr_vals（原始读数），用 truth 找出误读段
（|ocr-truth|>1）。对每个误读帧 crop 应用 N 种预处理，逐一 OCR：
- 每种方案独立统计正确数
- 关键指标：**至少一种方案读对**的误读帧占比 —— 思路成立则覆盖率高，
  全部方案都读错 = OCR 模型极限（真模糊，预处理救不了）

所有预处理产出的增强图统一走 _preprocess_standard(gamma=0)（纯 resize+pad），
预处理方案本身替代 gamma 步骤。

输出：
- 控制台：per-scheme 正确数 + at-least-one 覆盖率 + 每帧每方案 OCR 值表
- outputs/_reocr_scan_<video>.png：每行一误读帧，每列一方案（含 base），
  格子=预处理后的 crop 4x 放大，标注 OCR 值，绿=对红=错

用法：python tools/_reocr_preproc_scan.py [videos...] [--max-frames 50]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import ocr_native  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402
from ocr_engine import extract_speed_value  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

TOL = 1.0
UPSCALE = 4
BATCH = 16
_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ── 复用同一 OcrEngine（onnxruntime：TRT 引擎 profile 对输入尺寸敏感）──
_ENGINE_CACHE: dict = {}
_ORIG_ENGINE = ocr_native.OcrEngine


def _engine_factory(*a, **k):
    key = a[0] if a else k.get("variant", "v6_small")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = _ORIG_ENGINE(*a, **k)
    return _ENGINE_CACHE[key]


ocr_native.OcrEngine = _engine_factory


# ═══════════════ 预处理方案（numpy/PIL，无 cv2 依赖）═══════════════

def _gray(crop: np.ndarray) -> np.ndarray:
    return (crop.astype(np.float32) @ _GRAY_W).astype(np.uint8)


def _gamma(gray: np.ndarray, g: float) -> np.ndarray:
    return np.clip(255.0 * np.power(gray / 255.0, g), 0, 255).astype(np.uint8)


def _to3(g: np.ndarray) -> np.ndarray:
    return np.stack([g] * 3, axis=-1)


def _window_keep(gamma_gray: np.ndarray, center: int) -> np.ndarray:
    """亮度窗口：保留区 center±10（20 宽）原值，两侧 ±5 线性过渡到 0，
    随后将窗口内全部存在亮度线性映射到 0-255 完整范围。

    - 保留区（|v-c|<=10）内像素保留原灰度值（不裁剪）
    - 过渡区（10<|v-c|<=15）从原值线性衰减到 0（避免一刀切尖锐过渡）
    - 更远处全 0
    - 线性映射：对窗口内非零像素做 (v - min)*255/(max - min)，
      把窗口内相对亮度差放大到全动态范围（数字结构更清晰）

    目标是解剖数字的亮度层——gamma 后白字 ~245、黄底 ~171，区间内像素
    若独立成字则说明该亮度段是数字核心结构。
    """
    d = np.abs(gamma_gray.astype(np.float32) - center)
    # d<=10 → weight=1（保留）；10<d<=15 → 线性 1→0；d>15 → 0
    weight = np.clip((15.0 - d) / 5.0, 0.0, 1.0)
    out = gamma_gray.astype(np.float32) * weight
    # 线性映射非零像素到 0-255（映射含过渡区亮度，体现窗口内亮度差）
    nz = out[out > 0]
    if nz.size:
        lo, hi = float(nz.min()), float(nz.max())
        if hi > lo:
            out = (out - lo) * 255.0 / (hi - lo)
        out = np.clip(out, 0.0, 255.0)
    return _to3(out.astype(np.uint8))


# 方案表：(name, 标签, 处理函数)
def build_schemes(crop: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """对单个 crop 生成各方案的增强 RGB 图。

    base 列 = 正式灰度+gamma2.0（pipeline._preprocess_standard 一致）；
    其余方案 = **raw 灰度（无 gamma）** 后按亮度窗口挖取（保留 20 宽、
    两侧 ±5 线性过渡、窗口间重叠 10、截取后线性映射 0-255）→3通道，
    交 _preprocess_standard(gamma=0)。中心 255 向下步长 10 共 5 个：
    255,245,235,225,215（相邻窗口重叠 10）。

    与 v3 差异：去掉 gamma2.0 压暗，窗口直接在 raw 灰度上做 —— gamma
    会把中段压暗丢失细节，raw 灰度保留完整亮度分布。
    """
    g = _gray(crop)                                 # raw 灰度（无 gamma）
    schemes = [("base", "base", _to3(_gamma(g, 2.0)))]  # 正式路径对照
    for center in range(255, 214, -10):              # 255,245,...,215 → 5 窗口
        lo, hi = max(center - 10, 0), min(center + 10, 255)
        schemes.append((f"w{lo:03d}", f"[{lo},{hi}]",
                        _window_keep(g, center)))
    return schemes


def run_ocr(eng, crops3: list, pad: int, fa: float) -> list:
    """crops3: 3通道 RGB crop 列表 → 各方案 preprocess + OCR。"""
    outs = []
    for k in range(0, len(crops3), BATCH):
        chunk = crops3[k:k + BATCH]
        procs = [_preprocess_standard(c, pad, force_aspect=fa,
                                      gamma=0.0) for c in chunk]
        for res in eng(procs):
            sv, _rt, _c = extract_speed_value(res)
            outs.append(int(sv) if sv is not None and sv >= 0 else None)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=["test2", "test"])
    ap.add_argument("--max-frames", type=int, default=50,
                    help="每视频最多检查的误读帧数（图片尺寸上限）")
    args = ap.parse_args()

    eng = ocr_native.OcrEngine("v6_small", "onnxruntime")
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw)
        pipe.run(str(PROJECT / "outputs" / f"_rps_{v}.csv"))
        # 误读段（原始读数 |ocr-truth|>1）
        mis = []
        for i, seg in enumerate(pipe.segment_frames):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            ov = pipe.ocr_values[i]
            if t is None or ov is None:
                continue
            if abs(ov - t) > TOL:
                mis.append((rep, ov, int(t), pipe.crops.get(rep)))
        mis = [m for m in mis if m[3] is not None]
        if not mis:
            print(f"{v}: no misreads")
            continue
        if len(mis) > args.max_frames:
            mis = mis[:args.max_frames]
        print(f"{v}: {len(mis)} misreads (first {args.max_frames})")

        # 首帧确定方案集（方案集与帧无关，仅名称列表）
        schemes = build_schemes(mis[0][3])
        names = [s[0] for s in schemes]
        labels = [s[1] for s in schemes]
        # 每帧每方案的 OCR 值
        vals = {}   # (rep, name) -> int
        for rep, _ov, _t, crop in mis:
            for name, _lab, img3 in build_schemes(crop):
                vals[(rep, name)] = None
        # 逐帧逐方案 OCR（一次 preprocess 全方案）
        for rep, _ov, t, crop in mis:
            imgs3 = [img3 for _n, _lab, img3 in build_schemes(crop)]
            res = run_ocr(eng, imgs3, pipe._fill_width, pipe._force_aspect)
            for name, r in zip(names, res):
                vals[(rep, name)] = r

        # ── 汇总 ──
        scheme_ok = {n: 0 for n in names}
        at_least_one = 0
        all_fail = 0
        for rep, ov, t, crop in mis:
            oks = [vals[(rep, n)] is not None and abs(vals[(rep, n)] - t) <= TOL
                   for n in names]
            for n, ok in zip(names, oks):
                if ok:
                    scheme_ok[n] += 1
            if any(oks):
                at_least_one += 1
            else:
                all_fail += 1
        print(f"  per-scheme correct (/ {len(mis)}):")
        for n in names:
            print(f"    {n:>8}: {scheme_ok[n]:>3}")
        print(f"  [KEY] >=1 scheme correct: {at_least_one}/{len(mis)} "
              f"({at_least_one/max(len(mis),1)*100:.0f}%)")
        print(f"  all schemes fail (true blur, preproc can't help): {all_fail}")

        # ── 每帧值表 ──
        print(f"  per-frame: #fr t | " + " ".join(f"{n:>7}" for n in names))
        for rep, ov, t, crop in mis:
            row = " ".join(
                (f"{vals[(rep,n)]:>7}" if vals[(rep, n)] is not None
                 else "     -")
                for n in names)
            okm = "OK" if any(
                vals[(rep, n)] is not None and abs(vals[(rep, n)] - t) <= TOL
                for n in names) else "XX"
            print(f"  #{rep:<7} {t:>3} {okm} | {row}")

        # ── 蒙太奇：每行一误读帧，每列一方案 ──
        cell_w = max(c[3].shape[1] for c in mis) * UPSCALE
        cell_h = max(c[3].shape[0] for c in mis) * UPSCALE
        header_h = 22
        label_h = 16
        ncol = len(schemes)
        nrow = len(mis)
        W = ncol * cell_w
        H = nrow * (cell_h + label_h) + header_h
        canvas = Image.new("RGB", (W, H), (28, 28, 28))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # 顶部方案名
        for c, (_n, lab, _img) in enumerate(schemes):
            draw.text((c * cell_w + 4, 3), lab, fill=(230, 230, 230),
                      font=font)
        for r, (rep, ov, t, crop) in enumerate(mis):
            y0 = header_h + r * (cell_h + label_h)
            schemes_c = build_schemes(crop)
            for c, (name, _lab, img3) in enumerate(schemes_c):
                im = Image.fromarray(img3, "RGB")
                im = im.resize((im.width * UPSCALE, im.height * UPSCALE),
                               Image.Resampling.NEAREST)
                canvas.paste(im, (c * cell_w, y0))
                val = vals[(rep, name)]
                ok = val is not None and abs(val - t) <= TOL
                txt = f"{val}" if val is not None else "-"
                color = (120, 255, 120) if ok else (255, 110, 110)
                draw.text((c * cell_w + 4, y0 + cell_h + 2), txt,
                          fill=color, font=font)
            # 左侧帧号 + 真值
            draw.text((2, y0), f"#{rep} t={t}", fill=(255, 220, 100),
                      font=font)

        out = PROJECT / "outputs" / f"_reocr_scan_{v}.png"
        canvas.save(out)
        print(f"  montage: {out} ({W}x{H})")


if __name__ == "__main__":
    main()
