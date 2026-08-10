"""gamma2.0 误读帧蒙太奇（回归视角）：gamma 读错而基线读对的帧（原始读数）。

对 test6/test2：在同一批段/代表帧上分别以 基线（gamma=0，纯 RGB）和 gamma
跑 OCR，取 `_ocr_vals`（原始读数，未经过检测/DP 纠正）。每段分类：
- REGRESS 回归：gamma 错（|g-truth|>TOL）且基线对 ← 重点（gamma 引入的回退）
- BOTH 双错：gamma 错且基线也错（非 gamma 所致）
- IMPROVE 反转：基线错且 gamma 对

输出 outputs/_gamma_misreads.png：每行 [基线原 crop][gamma 处理 crop] 4x
最近邻放大，分节横幅标注类别，供人工检查 gamma 预处理是否把图弄糟。

预处理：gamma 正式化后，OCR 统一走 video_utils._preprocess_standard 的灰度
gamma（config.OCR_GAMMA=2.0 为正式默认）。baseline = gamma=0（纯 RGB resize，
历史基线），gamma 遍 = --gamma（默认 2.0，正式灰度+gamma）。OCR 输入全部由
_preprocess_standard(gamma=...) 生成，工具不再手动构造 crop/设 env。

实现要点：
- 灰度 gamma 在 numpy 预处理层生效、与引擎/分段无关 → 基线 run() 一次，
  gamma 遍复用它存的代表帧 crop 与同一 OcrEngine（缓存），不再二次解码/
  二次加载引擎（省 ~25s/实例）。

用法：python tools/_gamma_misread_montage.py [--videos test6 test2] [--gamma 2.0]
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
from tools._detect_eval import load_meta  # noqa: E402

TOL = 1.0
UPSCALE = 4
BATCH = 16
_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ── 复用同一 OcrEngine：gamma 在 numpy 预处理层，引擎与 gamma 无关 ──
_ENGINE_CACHE: dict = {}
_ORIG_ENGINE = ocr_native.OcrEngine


def _engine_factory(*a, **k):
    key = a[0] if a else k.get("variant", "v6_small")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = _ORIG_ENGINE(*a, **k)
    return _ENGINE_CACHE[key]


ocr_native.OcrEngine = _engine_factory  # pipeline 内 `from ocr_native import` 命中


def run_on_reps(pipe: SegmentPipeline, gamma: float) -> list:
    """在 pipe.run() 已存好的代表帧 crop 上以指定 gamma 重跑段 OCR。

    复刻 _run_pipelined 的 ocr_worker：同 B=16、同 _preprocess_standard、
    同 extract_speed_value。段/代表帧选择与预处理无关，故与整条流水线的
    _ocr_vals 逐位一致，但省一次解码。

    gamma=0 → 纯 RGB resize（历史基线）；gamma>0 → 灰度 gamma（正式路径，
    等价 config.OCR_GAMMA 默认值）。
    """
    reps = [s["rep_frame"] for s in pipe.segments]
    vals: list = []
    eng = ocr_native.OcrEngine(pipe._ocr_model, pipe._ocr_engine_type())
    for k in range(0, len(reps), BATCH):
        chunk = reps[k:k + BATCH]
        procs = [_preprocess_standard(pipe.crops[r], pipe._target_h, pipe._pad,
                                      max_width=pipe._max_width, gamma=gamma)
                 for r in chunk]
        for r, res in zip(chunk, eng(procs)):
            sv, _rt, _c = extract_speed_value(res)
            vals.append(int(sv) if sv is not None and sv >= 0 else None)
    return vals


def transform_crop(crop: np.ndarray, g: float) -> np.ndarray:
    """crop → 显示用 gray+gamma 图（与 OCR 喂入一致）；g<=0 返回原图。"""
    if g <= 0:
        return crop
    gray = (crop.astype(np.float32) @ _GRAY_W).astype(np.uint8)
    enh = 255.0 * np.power(gray / 255.0, g)
    enh = np.clip(enh, 0, 255).astype(np.uint8)
    return np.stack([enh] * 3, axis=-1)


def to_pil(crop: np.ndarray) -> Image.Image:
    img = Image.fromarray(crop, "RGB")
    return img.resize((img.width * UPSCALE, img.height * UPSCALE),
                      Image.Resampling.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=["test6", "test2"])
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()
    GAMMA = args.gamma

    rows = []  # (label, g_ocr, b_ocr, truth, crop, cat)
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"D:/Videos/racelog_test/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, 48, mw)
        pipe.run(str(PROJECT / "outputs" / f"_gb_{v}.csv"))
        bv = run_on_reps(pipe, 0.0)        # 基线：纯 RGB
        gv = run_on_reps(pipe, GAMMA)      # gamma：灰度+gamma
        assert len(gv) == len(bv) == len(pipe.segments)
        n = {"REGRESS": 0, "BOTH": 0, "IMPROVE": 0}
        for i, seg in enumerate(pipe._segs):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or gv[i] is None or bv[i] is None:
                continue
            g_err = abs(gv[i] - t) > TOL
            b_err = abs(bv[i] - t) > TOL
            if g_err and not b_err:
                cat = "REGRESS"
            elif g_err and b_err:
                cat = "BOTH"
            elif not g_err and b_err:
                cat = "IMPROVE"
            else:
                continue
            n[cat] += 1
            rows.append((f"{v}#{rep}", gv[i], bv[i], int(t),
                         pipe.crops.get(rep), cat))
        print(f"{v}: 段 {len(pipe._segs)} | gamma 错 {n['REGRESS'] + n['BOTH']}"
              f" = 回归(基对) {n['REGRESS']} + 双错 {n['BOTH']} | "
              f"反转(基错) {n['IMPROVE']}")

    # 分节排序：回归在前
    order = {"REGRESS": 0, "BOTH": 1, "IMPROVE": 2}
    rows.sort(key=lambda r: order[r[5]])

    # 数值表
    print(f"\n{'标签':<12} {'gOCR':>4} {'bOCR':>4} {'真':>4} {'类别':<8} "
          f"{'尺寸':>10}")
    for label, g, b, t, crop, cat in rows:
        print(f"{label:<12} {g:>4} {b:>4} {t:>4} {cat:<8} "
              f"{crop.shape[1]}x{crop.shape[0]:<4}")

    # 蒙太奇：每行 [基线 crop][gamma crop]，按类别分节
    max_w = max((c.shape[1] for _, _, _, _, c, _ in rows), default=1)
    max_h = max((c.shape[0] for _, _, _, _, c, _ in rows), default=1)
    cell_w = max_w * UPSCALE
    cell_h = max_h * UPSCALE
    header_h = 24
    banner_h = 26
    cols = 2
    W = cols * cell_w
    H = (banner_h * 3 + header_h + cell_h) * max(1, len(rows))
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    sections = {
        "REGRESS": f"回归 REGRESS: gamma 读错而基线读对 "
                   f"({sum(1 for r in rows if r[5]=='REGRESS')} 帧) ← 重点",
        "BOTH": f"双错 BOTH: gamma/基线都读错 "
                f"({sum(1 for r in rows if r[5]=='BOTH')} 帧)",
        "IMPROVE": f"反转 IMPROVE: 基线读错而 gamma 读对 "
                   f"({sum(1 for r in rows if r[5]=='IMPROVE')} 帧)",
    }
    colors = {"REGRESS": (255, 120, 120), "BOTH": (255, 200, 0),
              "IMPROVE": (150, 255, 150)}

    y = 0
    for cat in ("REGRESS", "BOTH", "IMPROVE"):
        cat_rows = [r for r in rows if r[5] == cat]
        draw.text((4, y + 4), sections[cat], fill=(240, 240, 240), font=font)
        y += banner_h
        for label, g, b, t, crop, _c in cat_rows:
            canvas.paste(to_pil(crop), (0, y + header_h))
            canvas.paste(to_pil(transform_crop(crop, GAMMA)),
                         (cell_w, y + header_h))
            draw.text((4, y + 2), f"{label} g={g} b={b} t={t}",
                      fill=colors[cat], font=font)
            draw.text((cell_w + 4, y + 2), f"gray+gamma{GAMMA}",
                      fill=(255, 255, 255), font=font)
            y += header_h + cell_h

    out = PROJECT / "outputs" / "_gamma_misreads.png"
    canvas.save(out)
    print(f"\n蒙太奇已保存: {out} ({W}x{H})")


if __name__ == "__main__":
    main()
