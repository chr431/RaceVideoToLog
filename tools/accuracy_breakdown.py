"""准确率漏斗分析 + 基线门禁：OCR 原始误读 → 检测 → DP 纠正 → 最终错误。

对每视频统计：
- 段数 / OCR 原始误读（|ocr-truth|>tol）
- 误读被检测（suspect）/ 未被检测
- 误读被纠正对 / 误读被纠正错 / 误读漏纠
- 正确段被误改（正确→错）
- 最终错误（|输出-truth|>tol）

基线门禁（回归测试的真正关口）：
- 默认模式：跑漏斗并把最终错误数与 tools/baseline.json 对比。
  任一视频或总量的最终错误数**增加**（回归）→ 退出码 1（CI 失败即红）。
- 有意改进/改动后：先在本机跑漏斗确认，再 --update-baseline 更新基线。
  注意：基线变化必须是有意为之（如修掉一个错误案例），且要同步更新
  tests/fixtures/ 下的回归夹具与 CLAUDE.md 的基线描述。

用法：python tools/accuracy_breakdown.py [--tol 1] [--update-baseline] [videos...]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from segment_flow import SegmentPipeline  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402

DEFAULT_VIDEOS = ["test", "test2", "test3", "test5", "test6"]
VIDEO_DIR = "D:/Videos/racelog_test"
BASELINE_PATH = PROJECT / "tools" / "baseline.json"

_METRICS = ("seg", "raw", "fix", "fix_wrong", "missed", "harm", "final")


def run_funnel(videos, tol: float = 1.0, decode_backend: str = "auto") -> dict:
    """对每个视频跑生产管线并统计漏斗指标。

    返回 {"videos": {name: {metric: value}}, "total": {metric: value}}。
    decode_backend: decord 解码后端（门禁默认 auto 不变；实验性混合
    解码对照用 env RVTOL_HYBRID_DECODE=1）。
    """
    per_video: dict = {}
    total = {k: 0 for k in _METRICS}
    print(f"{'视频':<6} {'段':>5} {'原始误读':>6} {'检出':>4} {'纠对':>4} "
          f"{'纠错':>4} {'漏纠':>4} {'误改':>4} {'最终':>4}")
    for v in videos:
        roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
        pipe = SegmentPipeline(f"{VIDEO_DIR}/{v}.mp4", roi, ms, ma,
                               fps, f_start, f_end, force_aspect=mw,
                               decode_backend=decode_backend,
                               yuv_output=True)
        pipe.run(str(PROJECT / "outputs" / f"_brk_{v}.csv"))
        sv = pipe._ocr_vals
        cv = pipe._corr_vals
        seg = {"seg": 0, "raw": 0, "fix": 0, "fix_wrong": 0, "missed": 0,
               "harm": 0, "final": 0}
        for i in range(len(sv)):
            rep = pipe.segments[i]["rep_frame"]
            t = truth.get(rep)
            if t is None or sv[i] is None:
                continue
            seg["seg"] += 1
            raw_err = abs(sv[i] - t) > tol
            final_err = abs(cv[i] - t) > tol if cv[i] is not None else True
            if raw_err:
                seg["raw"] += 1
                if cv[i] is not None and cv[i] != sv[i]:
                    if not final_err:
                        seg["fix"] += 1      # 纠对
                    else:
                        seg["fix_wrong"] += 1  # 纠错
                else:
                    seg["missed"] += 1      # 漏纠
            else:
                if cv[i] is not None and cv[i] != sv[i] and final_err:
                    seg["harm"] += 1        # 正确被误改
            if final_err:
                seg["final"] += 1
        per_video[v] = seg
        print(f"{v:<6} {seg['seg']:>5} {seg['raw']:>6} "
              f"{seg['raw']-seg['missed']:>4} {seg['fix']:>4} "
              f"{seg['fix_wrong']:>4} {seg['missed']:>4} {seg['harm']:>4} "
              f"{seg['final']:>4}")
        for k in _METRICS:
            total[k] += seg[k]

    det = total["raw"] - total["missed"]
    print(f"\n合计: 段 {total['seg']} 原始误读 {total['raw']} 检出 {det} "
          f"纠对 {total['fix']} 纠错 {total['fix_wrong']} 漏纠 {total['missed']} "
          f"误改 {total['harm']} 最终 {total['final']}")
    print(f"检出率 = {det}/{max(total['raw'], 1)} "
          f"({det/max(total['raw'], 1)*100:.1f}%)")
    print(f"最终 = 漏纠 {total['missed']} + 纠错 {total['fix_wrong']} "
          f"+ 误改 {total['harm']} = "
          f"{total['missed']+total['fix_wrong']+total['harm']}")
    return {"videos": per_video, "total": total}


def load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(path: Path, results: dict, tol: float, version: str) -> None:
    payload = {
        "tol": tol,
        "version": version,
        "updated": time.strftime("%Y-%m-%d"),
        "note": ("最终错误数是唯一门禁指标（任何算法/预处理改动不得使任一视频"
                 "或总量的最终错误数增加）。其余指标仅供诊断。"),
        "videos": results["videos"],
        "total": results["total"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def check_baseline(results: dict, baseline: dict) -> bool:
    """对比基线与本次结果。返回是否通过（无回归）。

    门禁口径：任一视频或总量的 final 错误数增加即失败；
    基线视频未全部覆盖（子集运行）也失败——防部分运行误通过。
    """
    ok = True
    print("\n── 基线对比 (tol=±{:.0f}) ──".format(baseline.get("tol", 1.0)))
    for v in baseline.get("videos", {}):
        if v not in results["videos"]:
            print(f"  {v:<6} final: 基线 {baseline['videos'][v].get('final', 0)}"
                  f" → 本次 未跑  [MISSING]")
            ok = False
            continue
        cur = results["videos"][v]
        b_final = baseline["videos"][v].get("final", 0)
        c_final = cur.get("final", 0)
        mark = "OK" if c_final <= b_final else "REGRESSION"
        if mark != "OK":
            ok = False
        print(f"  {v:<6} final: 基线 {b_final} → 本次 {c_final}  [{mark}]")
    b_tot = baseline.get("total", {}).get("final", 0)
    c_tot = results["total"].get("final", 0)
    mark = "OK" if c_tot <= b_tot else "REGRESSION"
    if mark != "OK":
        ok = False
    print(f"  {'总计':<6} final: 基线 {b_tot} → 本次 {c_tot}  [{mark}]")
    return ok


def main() -> None:
    import config
    ap = argparse.ArgumentParser(
        description="准确率漏斗分析 + 基线门禁（最终错误数回归即失败退出码 1）")
    ap.add_argument("videos", nargs="*", default=DEFAULT_VIDEOS)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--decode-backend", default="auto",
                    choices=config.DECODE_BACKEND_KEYS,
                    help="decord 解码后端（auto/cpu/nvdec；门禁默认 auto 不变；"
                         "实验性混合用 env RVTOL_HYBRID_DECODE=1）")
    ap.add_argument("--update-baseline", action="store_true",
                    help="用本次结果覆盖 tools/baseline.json（有意改动后）")
    ap.add_argument("--baseline", type=str, default=str(BASELINE_PATH),
                    help="基线文件路径")
    args = ap.parse_args()

    results = run_funnel(args.videos, tol=args.tol,
                         decode_backend=args.decode_backend)

    if args.update_baseline:
        save_baseline(Path(args.baseline), results, args.tol,
                      config.__version__)
        print(f"\n基线已更新: {args.baseline}")
        return

    bp = Path(args.baseline)
    if not bp.exists():
        print(f"\n[FAIL] 基线文件不存在: {bp}", file=sys.stderr)
        print("先跑一次漏斗并人工确认结果，然后 --update-baseline 建立基线。",
              file=sys.stderr)
        sys.exit(2)

    baseline = load_baseline(bp)
    if abs(baseline.get("tol", 1.0) - args.tol) > 1e-9:
        print(f"[FAIL] tol 不一致：基线 {baseline.get('tol')} vs 本次 "
              f"{args.tol}（基线口径是 ±{baseline.get('tol'):.0f}，勿混用）",
              file=sys.stderr)
        sys.exit(2)

    if not check_baseline(results, baseline):
        print("\n[FAIL] 准确率回归！最终错误数超出基线。", file=sys.stderr)
        print("若改动有意（修错/换参数），先确认无意外回归，再 "
              "--update-baseline 并同步更新回归夹具（tests/fixtures/）与 "
              "CLAUDE.md 基线描述。", file=sys.stderr)
        sys.exit(1)

    improved = any(results["videos"][v]["final"]
                   < baseline["videos"][v]["final"]
                   for v in baseline.get("videos", {})
                   if v in results["videos"])
    print("\n[PASS] 基线门禁通过（无回归）。")
    if improved:
        print("提示：本次结果优于基线，确认后 --update-baseline 更新基线。")
    else:
        print(f"基线口径: 最终错误 {baseline['total']['final']} 个。")


if __name__ == "__main__":
    main()
