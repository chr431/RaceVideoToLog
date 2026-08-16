"""后端组合 × 核心数性能矩阵（本机测量 + CPU 亲和性模拟少核）。

组合：decode {auto=GPU/NVDEC, cpu} × OCR {auto=TRT, cpu=ONNX}
核心数：全核（本机 auto 预算）；少核用 psutil cpu_affinity 限制前 N 个
逻辑核 + RVTOL_OCR_THREADS=N 模拟（子进程继承亲和性）。

用法：
  python tools/archive/_bench_matrix.py            # 全矩阵（test5/test6 全核 + test5 少核）
  python tools/archive/_bench_matrix.py --quick    # 只跑 test5 全核 4 组合
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable


def run_bench(video: str, ocr: str, dec: str, n_cores: int | None = None,
              ocr_threads: int | None = None, tag: str = "",
              extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if n_cores:
        import psutil
        psutil.Process().cpu_affinity(list(range(n_cores)))
    if ocr_threads:
        env["RVTOL_OCR_THREADS"] = str(ocr_threads)
    elif n_cores and ocr_threads is None:
        env["RVTOL_OCR_THREADS"] = str(n_cores)
    else:
        env.pop("RVTOL_OCR_THREADS", None)
    if extra_env:
        env.update(extra_env)
    name = f"bench_{video}_{ocr}_{dec}_{n_cores or 'full'}{tag}"
    json_path = PROJECT / "outputs" / f"{name}.json"
    cmd = [PY, "tools/bench_decoder.py", "--video", video, "--backend", ocr,
           "--decode-backend", dec, "--runs", "2", "--no-monitor",
           "--json", str(json_path)]
    r = subprocess.run(cmd, cwd=str(PROJECT), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  !! {name} 失败: {(r.stderr or r.stdout)[-200:]}")
        return {"error": True}
    d = json.loads(json_path.read_text(encoding="utf-8"))
    return d


def row(label: str, d: dict) -> None:
    if d.get("error"):
        print(f"  {label:<28} ERROR")
        return
    t = d.get("timing") or {}
    rss = t.get("peak_rss_mb", 0)
    print(f"  {label:<28} decode={t.get('decode_s', 0):>6.2f}s "
          f"ocr={t.get('ocr_s', 0):>6.2f}s "
          f"corr={t.get('correction_s', 0):>5.2f}s "
          f"total={t.get('total_pipeline_s', 0):>6.2f}s "
          f"rss={rss:>6}MB  backend={t.get('actual_backend', '?')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # 第一轮：全核矩阵（test5 h264 + test6 AV1）
    print("══ 全核矩阵（本机物理核预算，OCR 线程=物理核） ══")
    for v in (["test5"] if args.quick else ["test5", "test6"]):
        print(f"── {v} ──")
        for dec in ("auto", "cpu"):
            for ocr in ("auto", "cpu"):
                label = f"{v} dec={dec} ocr={ocr}"
                row(label, run_bench(v, ocr, dec, None))
    if args.quick:
        return

    # 第二轮：核心数模拟（test5，亲和性前 N 逻辑核 + OCR 线程=N）
    print("\n══ 核心数模拟（test5，affinity 前 N 核 + RVTOL_OCR_THREADS=N） ══")
    for n in (4, 8):
        for dec, ocr, tag in (("cpu", "cpu", ""),
                              ("cpu", "auto", ""),
                              ("auto", "cpu", ""),
                              ("auto", "auto", "")):
            label = f"test5 {n}核 dec={dec} ocr={ocr}"
            row(label, run_bench("test5", ocr, dec, n, tag))


    # 第三轮：少核下 OCR 线程预算扫描（CPU+ONNX 组合对线程数敏感）
    print("\n══ OCR 线程预算扫描（test5，affinity N 核 + ONNX） ══")
    for n in (4, 8):
        for ocr_threads in (max(1, n // 2), n - 1, n):
            label = f"test5 {n}核 dec=cpu ocr=cpu ocrT={ocr_threads}"
            row(label, run_bench("test5", "cpu", "cpu", n, ocr_threads,
                                 tag=f"_ot{ocr_threads}"))
            label = f"test5 {n}核 dec=auto ocr=cpu ocrT={ocr_threads}"
            row(label, run_bench("test5", "cpu", "auto", n, ocr_threads,
                                 tag=f"_ot{ocr_threads}"))


if __name__ == "__main__":
    main()
