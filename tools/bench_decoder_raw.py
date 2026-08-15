"""纯解码吞吐微基准（NVDEC）：生产路径复刻 get_batch + ROI + asnumpy。

与 tools/bench_decoder.py 的区别：不跑 OCR/分段，直接测 decord 解码墙钟，
并输出逐帧校验和（跨 DLL 变体 A/B 逐位一致性比对用）。

用法：
    python tools/bench_decoder_raw.py --video test5 --format gray --roi truth
    python tools/bench_decoder_raw.py --video test6 --format rgb --roi full
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from tools.detect_eval import load_meta  # noqa: E402

VIDEO_DIR = Path("D:/Videos/racelog_test")


def _roi_kwargs(vr_mod, roi):
    try:
        has_api = hasattr(vr_mod, "_CAPI_VideoReaderSetRoi")
    except ImportError:
        has_api = False
    if not has_api or roi is None:
        return {}
    x1, y1, x2, y2 = roi
    return {"roi": (x1, y1, x2 + 1, y2 + 1)}


def decode_pass(video: Path, roi, fmt: str, bsz: int, nframes: int,
                no_asnumpy: bool = False) -> dict:
    """一次解码 pass，返回 (wall_s, fps, 每帧 uint8 校验和列表)。"""
    from decord import VideoReader, gpu
    import decord.video_reader as vr_mod

    kwargs = _roi_kwargs(vr_mod, roi if fmt != "rgb-full" else None)
    if fmt != "rgb-full":
        kwargs["output_format"] = "gray"
    vr = VideoReader(str(video), ctx=gpu(0), **kwargs)
    total = len(vr)
    if roi is not None and fmt != "rgb-full":
        x1, y1, x2, y2 = roi
        half = (x1, y1, x2 + 1, y2 + 1)
    else:
        half = None
    # 生产用 frame_start 起逐帧顺序解码（从 truth 头部读 start，跳过片头）
    _roi, f_start, f_end, _fps, _ms, _ma, _mw, _truth = load_meta(video.stem)
    f_start = max(0, f_start or 0)
    end = min(f_end or total, f_start + nframes, total)
    frames = list(range(f_start, end))
    checksums = []
    decode_wall = 0.0
    for bstart in range(0, len(frames), bsz):
        bend = min(bstart + bsz, len(frames))
        _t = time.perf_counter()
        if half is not None:
            arr = vr.get_batch(frames[bstart:bend], roi=half)
        else:
            arr = vr.get_batch(frames[bstart:bend])
        if not no_asnumpy:
            arr = arr.asnumpy()
        decode_wall += time.perf_counter() - _t
        if no_asnumpy:
            continue
        # 校验和计算不计入解码计时（全帧路径开销大，会污染吞吐测量）
        if arr.ndim == 4:
            for k in range(arr.shape[0]):
                checksums.append(int(arr[k].astype(np.uint64).sum()))
        else:
            raise RuntimeError(f"unexpected batch shape {arr.shape}")
    wall = decode_wall
    return {"wall_s": round(wall, 4), "fps": round(len(frames) / wall, 1),
            "frames": len(frames), "checksums": checksums,
            "sha256": hashlib.sha256(
                b"".join(f"{c:016x}".encode() for c in checksums)).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--format", default="gray", choices=["gray", "rgb-full"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=10_000_000,
                    help="解码帧数上限（默认全片）")
    ap.add_argument("--runs", type=int, default=3,
                    help="正式测量次数（另含 1 次丢弃的预热）")
    ap.add_argument("--tag", default="", help="输出文件名后缀（A/B 变体区分）")
    ap.add_argument("--no-asnumpy", action="store_true",
                    help="只取 NDArray 不做 asnumpy/D2H（隔离解码吞吐 vs 拷贝带宽）")
    args = ap.parse_args()

    video = VIDEO_DIR / f"{args.video}.mp4"
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(args.video)
    use_roi = None if args.format == "rgb-full" else roi

    record = {"video": args.video, "format": args.format,
              "batch": args.batch, "runs": []}
    # 预热（丢弃）
    decode_pass(video, use_roi, args.format, args.batch, args.frames,
                args.no_asnumpy)
    for i in range(args.runs):
        r = decode_pass(video, use_roi, args.format, args.batch,
                        args.frames, args.no_asnumpy)
        print(f"  run {i + 1}: {r['wall_s']:.3f}s {r['fps']:.1f}fps "
              f"frames={r['frames']} sha256={r['sha256'][:16]}", flush=True)
        record["runs"].append(r)
    best = min(record["runs"], key=lambda r: r["wall_s"])
    record["best"] = {k: best[k] for k in ("wall_s", "fps", "frames", "sha256")}
    print(f"  best: {best['wall_s']:.3f}s {best['fps']:.1f}fps "
          f"sha256={best['sha256'][:16]}", flush=True)
    out = PROJECT / "outputs" / \
        f"raw_{args.video}_{args.format.replace('-', '_')}" \
        f"{'_' + args.tag if args.tag else ''}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
