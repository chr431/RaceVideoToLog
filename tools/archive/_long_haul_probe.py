# -*- coding: utf-8 -*-
"""长区间诊断 + 优化验证：全量窗口下的 solo/hybrid/floor 对比。

用法：
  python tools/archive/_long_haul_probe.py --video test6 --mode solo --repeat 2
  python tools/archive/_long_haul_probe.py --video test6 --mode hybrid --onnx-threads 2
  python tools/archive/_long_haul_probe.py --video test6 --mode floor-gpu
  python tools/archive/_long_haul_probe.py --video test5 --mode dual
所有测量单跑串行。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs" / "long_haul"
_ENG = PROJECT / "third_party" / "video_ocr_engine"
for p in (str(PROJECT), str(_ENG)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_meta(v):
    spec = importlib.util.spec_from_file_location(
        "detect_eval", PROJECT / "tools" / "detect_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_meta(v)


def make_extractor(video, meta, dec="auto", ocr="auto", dual=False,
                   frames=0):
    roi, fs, fe, fps, ms, ma, mw = meta[:7]
    if frames:
        fe = fs + frames
    from video_ocr_engine.extractor import FieldExtractor
    return FieldExtractor(
        str(VIDEO_DIR / (video + ".mp4")), roi,
        frame_start=fs, frame_end=fe, force_aspect=mw,
        decode_backend=dec, ocr_backend=ocr, yuv_output=True,
        dual_pipeline=dual,
        progress_cb=lambda m_, p_: None)


def run_solo(video, meta, frames=0):
    ex = make_extractor(video, meta, frames=frames)
    t0 = time.perf_counter()
    res = ex._run_pipelined(_force_single=True)
    wall = time.perf_counter() - t0
    return pack(ex, wall, len(res[1]))


def run_hybrid(video, meta, k_onnx, frames=0):
    import video_ocr_engine.extractor  # noqa: F401 防循环导入
    from ocr_native import OcrEngine, auto_ocr_thread_count
    import engine_config as eco
    ex = make_extractor(video, meta, frames=frames)
    trt = OcrEngine(eco.DEFAULT_OCR_MODEL, "tensorrt",
                    fill_width=224,
                    num_threads=max(2, auto_ocr_thread_count()),
                    progress_cb=lambda m_: None)
    onnx = OcrEngine(eco.DEFAULT_OCR_MODEL, "onnxruntime",
                     fill_width=224, num_threads=k_onnx,
                     progress_cb=lambda m_: None)
    t0 = time.perf_counter()
    res = ex._run_pipelined(_ocr_engines=[trt, onnx])
    wall = time.perf_counter() - t0
    r = pack(ex, wall, len(res[1]))
    r["onnx_threads"] = k_onnx
    return r


def run_dual(video, meta, frames=0):
    ex = make_extractor(video, meta, dual=True, frames=frames)
    t0 = time.perf_counter()
    ex._run_pipelined()
    wall = time.perf_counter() - t0
    r = pack(ex, wall, ex._n_segments)
    for i in (1, 2):
        r["pipe%d_chunks" % i] = ex.timing.get("parallel_pipe%d_chunks" % i, 0)
        r["pipe%d_s" % i] = round(float(
            ex.timing.get("parallel_pipe%d_s" % i, 0.0)), 3)
    return r


def run_floor(video, meta, frames=0):
    roi, fs, fe, fps, ms, ma, mw = meta[:7]
    if frames:
        fe = fs + frames
    from video_ocr_engine.extractor import FieldExtractor
    ex = FieldExtractor(str(VIDEO_DIR / (video + ".mp4")), roi,
                        frame_start=fs, frame_end=fe,
                        decode_backend="auto",
                        progress_cb=lambda m, p: None)
    vr = ex._open_vr()
    x1, y1, x2, y2 = ex._roi
    frames = list(range(fs, min(fe, len(vr))))
    t0 = time.perf_counter()
    for bs in range(0, len(frames), 64):
        be = min(bs + 64, len(frames))
        vr.get_batch(frames[bs:be],
                     roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
    wall = time.perf_counter() - t0
    del vr, ex
    return {"wall_s": round(wall, 3), "fps": round(len(frames) / wall, 1),
            "frames": len(frames)}


def pack(ex, wall, nseg):
    prof = {g: dict(p) for g, p in ex.profile.items()}
    return {
        "wall_s": round(wall, 3),
        "n_segments": nseg,
        "decode_s": round(float(ex.timing.get("decode", 0.0)), 3),
        "ocr_s": round(float(ex.timing.get("ocr", 0.0)), 3),
        "ocr_tail": round(float(ex.timing.get("ocr_tail", 0.0)), 3),
        "profile": prof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test6")
    ap.add_argument("--mode", default="solo",
                    choices=["solo", "hybrid", "dual", "floor-gpu"])
    ap.add_argument("--onnx-threads", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--frames", type=int, default=0,
                    help="0=truth 全量；N=从 frame_start 截 N 帧")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.video)

    # 预热（TRT 热载 / ONNX 会话）
    if args.mode != "floor-gpu":
        print("warmup ...", flush=True)
        wex = make_extractor(args.video, meta)
        wex._frame_end = meta[1] + 600
        wex._run_pipelined(_force_single=True)
        del wex

    outs = []
    for i in range(args.repeat):
        if args.mode == "solo":
            r = run_solo(args.video, meta, args.frames)
        elif args.mode == "hybrid":
            r = run_hybrid(args.video, meta, args.onnx_threads,
                           args.frames)
        elif args.mode == "dual":
            r = run_dual(args.video, meta, args.frames)
        else:
            r = run_floor(args.video, meta, args.frames)
        outs.append(r)
        print("[%s #%d] wall=%.2fs decode=%.2f ocr=%.2f tail=%.2f seg=%d%s"
              % (args.mode, i, r["wall_s"], r.get("decode_s", -1),
                 r.get("ocr_s", -1), r.get("ocr_tail", -1),
                 r.get("n_segments", -1),
                 (" onnxT=%d" % r["onnx_threads"])
                 if "onnx_threads" in r else ""), flush=True)
        time.sleep(1.5)

    outp = OUT_DIR / ("%s_%s_%d.json"
                      % (args.video, args.mode, int(time.time())))
    outp.write_text(json.dumps(outs, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("saved -> %s" % outp)


if __name__ == "__main__":
    main()
