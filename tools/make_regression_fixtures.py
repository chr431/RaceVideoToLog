"""生成 CI 回归夹具（tests/fixtures/），供无视频/无 decord 的 CI 环境复跑纠错链与 OCR。

产物：
1. tests/fixtures/seg_series/<video>.json —— 生产 run() 的全量段级序列：
   每段 {rep, mid, len, raw, corr, truth}。CI 用 SegmentPipeline 重构
   _confidence + _dense_correct，断言与基线输出逐段一致、最终错误集一致。
2. tests/fixtures/ocr_frames/<video>_frame_XXXXX.npy + .png —— 最终
   错误案例的代表帧裁剪（.npy 供测试加载，.png 供人工查看）。
   YUV420 生产模式下代表帧是 packed NV12（2D）→ 存 Y 平面 (H,W,1)，
   即 OCR 实际输入（与旧 decord gray 输出语义一致）。
   manifest.json 记录每帧的期望 OCR raw 值（行为锁定基线）。

运行前提：本机 decord + 测试视频（D:/Videos/racelog_test）。仅维护用，
CI 不运行本脚本。

用法：python tools/make_regression_fixtures.py [videos...]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

VIDEO_DIR = "D:/Videos/racelog_test"
FIXTURES = PROJECT / "tests" / "fixtures"
SERIES_DIR = FIXTURES / "seg_series"
FRAMES_DIR = FIXTURES / "ocr_frames"
TOL = 1.0


def export_error_frames(v: str, pipe: SegmentPipeline, truth: dict) -> list:
    """最终错误段（|corr-truth|>TOL）的代表帧裁剪 → .npy/.png，返回案例清单。"""
    cases = []
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(pipe.segments):
        rep = s["rep_frame"]
        t = truth.get(rep)
        if t is None:
            continue
        cv = pipe.corrected_values[i]
        if cv is None or abs(cv - t) <= TOL:
            continue
        crop = pipe.crops.get(rep)
        if crop is None:
            print(f"  !! {v}#{i} rep={rep} 无裁剪帧（跳过）")
            continue
        stem = f"{v}_frame_{rep:05d}"
        # YUV420 生产模式：crops 是 packed NV12（2D）→ 存 Y 平面
        # （H,W,1，即 OCR 实际输入）；旧 gray/rgb 输出直接存
        if crop.ndim == 2:
            from video_utils import _nv12_luma_full, nv12_to_rgb
            crop_y = _nv12_luma_full(crop, pipe._color_range)[..., None]
            crop_rgb = nv12_to_rgb(crop)
        else:
            crop_y = crop
            crop_rgb = crop
        np.save(FRAMES_DIR / f"{stem}.npy",
                np.ascontiguousarray(crop_y, dtype=np.uint8))
        try:
            import cv2  # 仅本机生成用（CI 不装 cv2，测试只读 .npy）
            cv2.imwrite(str(FRAMES_DIR / f"{stem}.png"),
                        np.ascontiguousarray(crop_rgb[..., ::-1]))
        except ImportError:
            pass
        cases.append({
            "file": f"{stem}.npy",
            "video": v,
            "seg_index": i,
            "rep_frame": rep,
            "expected_raw": pipe.ocr_values[i],
            "expected_corr": cv,
            "truth": int(t),
        })
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*",
                    default=["test", "test2", "test3", "test5", "test6"])
    args = ap.parse_args()

    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for v in args.videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"{VIDEO_DIR}/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw,
                               rep_crop_format="yuv")
        print(f"== {v}: 生产管线运行中 ...")
        pipe.run(str(PROJECT / "outputs" / f"_fx_{v}.csv"))
        series = {
            "video": v,
            "version": config.__version__,
            "tol": TOL,
            "meta": {
                "roi": list(pipe._roi),
                "fps": pipe._fps,
                "max_speed": pipe._max_speed,
                "max_accel": pipe._max_accel,
                "force_aspect": pipe._force_aspect,
                "frame_start": pipe._frame_start,
                "frame_end": pipe._frame_end,
            },
            "segments": [
                {
                    "rep": s["rep_frame"],
                    "mid": s["frames"][len(s["frames"]) // 2],
                    "len": len(s["frames"]),
                    "raw": pipe.ocr_values[i],
                    "corr": pipe.corrected_values[i],
                    "truth": _fmt_truth(truth.get(s["rep_frame"])),
                }
                for i, s in enumerate(pipe.segments)
            ],
        }
        with open(SERIES_DIR / f"{v}.json", "w", encoding="utf-8") as f:
            json.dump(series, f, ensure_ascii=False)
            f.write("\n")
        n_err = sum(1 for seg in series["segments"]
                    if seg["truth"] is not None and seg["corr"] is not None
                    and abs(seg["corr"] - seg["truth"]) > TOL)
        print(f"   {v}: {len(series['segments'])} 段, 最终错误 {n_err}")
        all_cases.extend(export_error_frames(v, pipe, truth))

    manifest = {
        "version": config.__version__,
        "note": "错误案例代表帧的 OCR 行为锁定（expected_raw 是基线产出）。"
                "若 OCR/预处理/模型改动改变读数，属有意改动：先跑完整漏斗"
                "（tools/accuracy_breakdown.py）确认无回归，再重新生成本夹具。",
        "cases": all_cases,
    }
    with open(FRAMES_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    # 清理不再被 manifest 引用的旧帧文件（错误集变化后的孤儿）
    referenced = {c["file"] for c in all_cases}
    for old in FRAMES_DIR.glob("*.npy"):
        if old.name not in referenced:
            old.unlink()
            print(f"  清理孤儿帧: {old.name}")
    for old in FRAMES_DIR.glob("*.png"):
        stem = old.name.rsplit(".", 1)[0]
        if not (FRAMES_DIR / f"{stem}.npy").exists():
            old.unlink()
            print(f"  清理孤儿预览: {old.name}")
    print(f"\n夹具完成: {len(all_cases)} 个 OCR 帧案例 → {FRAMES_DIR}")
    print(f"段级序列 → {SERIES_DIR}")


def _fmt_truth(t):
    return int(t) if t is not None else None


if __name__ == "__main__":
    main()
