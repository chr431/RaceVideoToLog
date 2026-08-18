"""单跑校准基线：改动后自动对比的固定口径基准。

背景（2026-08 损耗审计）：
- 并行跑多个 profile/bench 任务会互抢 CPU/GPU 制造人为差异（auto 曾被测成
  5.4s 而单跑实为 2.8s）—— 所有 A/B 必须单跑、串行。
- 本脚本固化为标准基线入口：跑一组固定组合（生产默认 auto + 备选 CPU+ONNX），
  输出 timing + 读数指纹（speed 列哈希），供改动前后对比。

用法：
  python tools/bench_baseline.py           # 跑基线（test5/test6，3000 帧），存档 latest
  python tools/bench_baseline.py --check   # 对比 latest 存档：timing 漂移 + 读数差异帧
  python tools/bench_baseline.py --full    # 全量帧（提交前门禁用）
  python tools/bench_baseline.py --quick   # 只 test5（~2 分钟快速检查）
  python tools/bench_baseline.py --runs 3  # 每组合重复次数（默认 2，取最后记录 run）

输出：outputs/baseline_latest.json（timing 数组 + 指纹 + 版本/参数元数据）。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import config  # noqa: E402

BASELINE = PROJECT / "outputs" / "baseline_latest.json"

# 固定组合：(视频, ocr_backend, decode_backend, 标签)
COMBO_AUTO = [("test5", "auto", "auto", "prod:auto"), ("test6", "auto", "auto", "prod:auto")]
COMBO_CPU = [("test5", "cpu", "cpu", "alt:cpu+cpu"), ("test6", "cpu", "cpu", "alt:cpu+cpu")]


def _read_rows(csv_path: Path) -> dict:
    """读输出 CSV 的 (frame -> speed) 映射（跳过注释行）。"""
    rows: dict = {}
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                rows[int(parts[0])] = float(parts[1])
            except (ValueError, IndexError):
                continue
    return rows


def _fingerprint(rows: dict) -> str:
    """读数指纹：speed 列规范序列哈希（改动后 0 差异帧 ⟺ 指纹相同）。"""
    blob = "\n".join(f"{k},{v:.4f}" for k, v in sorted(rows.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_combo(video: str, ocr: str, dec: str, runs: int, frames: int,
              tag: str) -> dict:
    """子进程跑 bench_decoder（单跑、无监控），返回 timing + 指纹。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("RVTOL_OCR_BATCH", None)  # 基线口径：生产默认 B=16
    json_path = PROJECT / "outputs" / f"_baseline_{video}_{ocr}_{dec}_{tag}.json"
    cmd = [sys.executable, "tools/bench_decoder.py", "--video", video,
           "--backend", ocr, "--decode-backend", dec, "--runs", str(runs),
           "--no-monitor", "--json", str(json_path)]
    if frames:
        cmd += ["--frames", str(frames)]
    r = subprocess.run(cmd, cwd=str(PROJECT), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  !! {video} {ocr}/{dec} 失败: {(r.stderr or r.stdout)[-200:]}")
        return {"error": True}
    d = json.loads(json_path.read_text(encoding="utf-8"))
    t = d.get("timing") or {}
    # 指纹：bench_decoder 保留最后 run 的 CSV outputs/bench_<v>_r<runs>.csv
    last_csv = PROJECT / "outputs" / f"bench_{video}_r{runs}.csv"
    rows = _read_rows(last_csv) if last_csv.exists() else {}
    return {
        "video": video, "ocr": ocr, "dec": dec, "tag": tag,
        "frames": t.get("frames"),
        "decode_s": t.get("decode_s"), "ocr_s": t.get("ocr_s"),
        "correction_s": t.get("correction_s"),
        "total_pipeline_s": t.get("total_pipeline_s"),
        "peak_rss_mb": t.get("peak_rss_mb"),
        "actual_backend": t.get("actual_backend"),
        "fingerprint": _fingerprint(rows) if rows else None,
        "n_rows": len(rows),
    }


def print_row(label: str, d: dict) -> None:
    if d.get("error"):
        print(f"  {label:<20} ERROR")
        return
    print(f"  {label:<20} total={d.get('total_pipeline_s', 0):>6.2f}s "
          f"decode={d.get('decode_s', 0):>6.2f}s ocr={d.get('ocr_s', 0):>6.2f}s "
          f"corr={d.get('correction_s', 0):>5.2f}s "
          f"rss={d.get('peak_rss_mb', 0):>5}MB fp={d.get('fingerprint')} "
          f"[{d.get('actual_backend')}]")


def do_check(prev: dict) -> int:
    print(f"══ 对比存档 {prev.get('saved_at')}（onnxruntime {prev.get('ort_version')}） ══")
    bad = 0
    for r in prev.get("runs", []):
        if r.get("error"):
            continue
        v, o, de = r["video"], r["ocr"], r["dec"]
        tag = f"{v} {o}/{de}"
        cur = run_combo(v, o, de, runs=1, frames=r.get("frames") or 0,
                        tag=f"check_{v}")
        if cur.get("error"):
            bad += 1
            continue
        dt = cur.get("total_pipeline_s", 0) - r.get("total_pipeline_s", 0)
        drift = dt / r.get("total_pipeline_s", 1) * 100
        same_fp = cur.get("fingerprint") == r.get("fingerprint")
        print(f"  {tag:<12} total {r.get('total_pipeline_s'):.2f} → "
              f"{cur.get('total_pipeline_s'):.2f}s ({drift:+.1f}%) "
              f"读数指纹 {'✓ 一致' if same_fp else '✗ 不同！'}")
        if abs(drift) > 10 or not same_fp:
            bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="对比存档基线（重跑各组合 1 次，报告漂移/指纹差异）")
    ap.add_argument("--full", action="store_true", help="全量帧（缺省 3000 帧）")
    ap.add_argument("--quick", action="store_true", help="只跑 test5")
    ap.add_argument("--runs", type=int, default=2, help="每组合重复次数（默认 2）")
    ap.add_argument("--frames", type=int, default=0, help="显式帧数（0=按 --full）")
    args = ap.parse_args()

    frames = 0 if args.full else (args.frames or 3000)
    combos = []
    if not args.quick:
        combos += COMBO_AUTO
    combos += COMBO_CPU
    if args.quick:
        combos = [c for c in combos if c[0] == "test5"]

    if args.check:
        if not BASELINE.exists():
            print(f"无存档基线（{BASELINE.name}）——先不带 --check 跑一次")
            return 1
        prev = json.loads(BASELINE.read_text(encoding="utf-8"))
        return do_check(prev)

    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception:
        ort_v = "?"
    import decord  # noqa: F401
    try:
        decord_v = decord.__version__
    except Exception:
        decord_v = "?"

    print(f"══ 单跑校准基线 · 3000 帧模式={frames} · runs={args.runs} "
          f"· onnxruntime {ort_v} · decord {decord_v} ══")
    runs_out = []
    for video, ocr, dec, tag in combos:
        label = f"{video} {ocr}/{dec}"
        d = run_combo(video, ocr, dec, args.runs, frames, tag)
        print_row(label, d)
        runs_out.append(d)
    rec = {
        "saved_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "ort_version": ort_v, "decord_version": decord_v,
        "config_version": config.__version__,
        "frames": frames, "runs": args.runs,
        "note": "单跑基线：A/B 前先跑本脚本存档，改动后 --check 对比。",
        "runs": runs_out,
    }
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n存档: {BASELINE.name}")


if __name__ == "__main__":
    raise SystemExit(main())