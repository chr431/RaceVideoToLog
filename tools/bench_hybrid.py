"""hybrid 解码端到端 benchmark：auto vs hybrid（kfe 双生产者竞争）。

引擎 v0.3（submodule e8b2637）起 `decode_backend="hybrid"` 是一等解码后端：
同一实例内 NVDEC 与 CPU 软解作为双生产者，按关键帧分片动态竞争，谁快谁多拿。
引擎 0.8.x hybrid v3/v4 实测 HEVC/AV1 与纯 NVDEC 持平后**移除了编码回退门**
（0.9.0 现役：任何编码都可 hybrid）——本工具相应校验 AV1（test6）保持
hybrid 且墙钟不明显劣于 auto（ratio<1.10 "不退化"门）。

用法：
    python tools/bench_hybrid.py                          # 3000 帧窗口 × 5 视频
    python tools/bench_hybrid.py --frames 0               # 全量帧（提交前）
    python tools/bench_hybrid.py --runs 2                 # warm 口径（默认，禁用监控采样）

所有组合单跑、串行；每组合重复 runs 次，报告最后一次（warm）。
输出 outputs/hybrid_bench/hybrid_bench.json（含 actual_backend 校验结果）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "outputs" / "hybrid_bench"

# bench_decoder 内部 import config/segment 模块需项目根在 sys.path
sys.path.insert(0, str(PROJECT))

# tools/ 无 __init__.py：用 importlib 从文件加载 bench_decoder（run/resolve）
_spec = importlib.util.spec_from_file_location(
    "bench_decoder_mod", PROJECT / "tools" / "bench_decoder.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

DEFAULT_VIDEOS = ["test", "test2", "test3", "test5", "test6"]


def run_one(video: str, mode: str, frames: int, runs: int,
            ocr_backend: str, buffer: int) -> dict:
    """跑一个 (video, decode mode) 组合，返回最后一次（warm）的 timing。"""
    video_path, truth = _bench.resolve(video)
    latest: dict | None = None
    for i in range(runs):
        out_csv = str(OUT_DIR / f"{video}_{mode}_r{i + 1}.csv")
        latest = _bench.run(video_path, truth, ocr_backend, out_csv,
                            decode_backend=mode, buffer_size=buffer,
                            no_monitor=True, frames=frames or None)
        if i < runs - 1 and latest.get("frames"):
            Path(out_csv).unlink(missing_ok=True)
    assert latest is not None
    return latest


def print_row(video: str, mode: str, t: dict) -> None:
    if "frames" not in t:
        print(f"  {video:<6} {mode:<8} ERROR: "
              f"{t.get('error', t.get('stderr', 'no timing'))[:160]}")
        return
    print(f"  {video:<6} {mode:<8} | {t.get('actual_backend', '?'):<16} "
          f"| {t.get('frames'):>6} fr | wall {t.get('wall_s', 0):5.1f}s "
          f"| decode {t.get('decode_s', 0):5.1f}s | ocr {t.get('ocr_s', 0):5.1f}s "
          f"| total {t.get('total_pipeline_s', t.get('total_s', 0)):5.1f}s "
          f"| rss {t.get('peak_rss_mb', 0):>5}MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    ap.add_argument("--frames", type=int, default=3000,
                    help="截取帧数（相对 truth frame_start；0=全量）")
    ap.add_argument("--runs", type=int, default=2, help="每组合重复次数（取最后 warm 次）")
    ap.add_argument("--ocr-backend", default="auto",
                    choices=["auto", "cpu", "tensorrt"])
    ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--json", type=str,
                    default=str(OUT_DIR / "hybrid_bench.json"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"══ hybrid vs auto · frames={args.frames or 'full'} · runs={args.runs} "
          f"· ocr={args.ocr_backend} ══")
    record = {
        "frames": args.frames or None,
        "runs": args.runs,
        "ocr_backend": args.ocr_backend,
        "av1_hybrid_check": {},
        "combos": {},
    }
    for video in args.videos:
        results: dict[str, dict] = {}
        for mode in ("auto", "hybrid"):
            print(f"Running {video} {mode} ...", end=" ", flush=True)
            t = run_one(video, mode, args.frames, args.runs,
                        args.ocr_backend, args.buffer)
            print_row(video, mode, t)
            results[mode] = t
        a, h = results["auto"], results["hybrid"]
        if "frames" in a and "frames" in h:
            dt = (h.get("total_pipeline_s", 0) - a.get("total_pipeline_s", 0))
            ratio = (h.get("total_pipeline_s", 0) / a.get("total_pipeline_s", 1))
            delta = (dt / a.get("total_pipeline_s", 1) * 100) if a.get("total_pipeline_s") else 0.0
            mark = "hybrid" if "hybrid" in str(h.get("actual_backend", "")).lower() else \
                   (f"回退({h.get('actual_backend')})" if a.get("actual_backend") == h.get("actual_backend") else f"后端变化({h.get('actual_backend')})")
            print(f"    → Δtotal {dt:+.2f}s ({delta:+.1f}%), ratio {ratio:.3f}, "
                  f"backend {a.get('actual_backend')} → {h.get('actual_backend')} [{mark}]")
        # AV1（test6）：引擎 0.8.x 起 hybrid 支持所有编码（v3 速率比例分界
        # 实测 HEVC/AV1 与纯 NVDEC 持平，回退门已删）——校验改为"保持
        # hybrid 且不明显慢于 auto"（ratio<1.10 即不退化门）。
        if video == "test6":
            hb = str(h.get("actual_backend", "")).lower()
            stays_hybrid = "hybrid" in hb
            _a = a.get("total_pipeline_s", 0) or 0
            _h = h.get("total_pipeline_s", 0) or 0
            ratio = (_h / _a) if _a else 0.0
            no_regress = bool(_a) and ratio < 1.10
            record["av1_hybrid_check"][video] = {
                "auto_backend": a.get("actual_backend"),
                "hybrid_backend": h.get("actual_backend"),
                "stays_hybrid": stays_hybrid,
                "ratio": round(ratio, 3),
                "no_regress": no_regress,
            }
            print(f"    → AV1 hybrid 校验: "
                  f"{'OK（保持 hybrid，ratio %.3f）' % ratio if stays_hybrid and no_regress else 'FAIL'}")
        record["combos"][video] = results

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"Record saved: {args.json}")

    fails = [k for k, v in record["av1_hybrid_check"].items()
             if not (v["stays_hybrid"] and v["no_regress"])]
    if fails:
        print(f"FAIL: AV1 hybrid 校验未通过（回退或退化）: {fails}")
        sys.exit(1)


if __name__ == "__main__":
    main()
