"""Phase 1 信号体系离线评估 — 检出率 / 误报率。

从 stage report（含 5 信号 + raw_val）与 ground truth（精确到帧）加载数据，
按 frame 对齐。对任意信号设计计算：

  - 检出率：错误帧（|raw − true| ≥ 0.5）中被 flag（conf < threshold）的比例
  - 误报率：正确帧（raw == true）中被误 flag 的比例
  - 总 flag 率：所有被 flag 的帧比例（含 raw<0 等无效帧，需单独处理）

用法：python tools/eval_phase1.py [threshold]
"""
from __future__ import annotations
import csv
import re
import sys

import numpy as np

# ── 数据加载 ──────────────────────────────────────────────

def _parse_header(path: str) -> dict:
    """解析 stage report / truth 的 # 头行，提取 fps 等参数。"""
    info: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            for key in ("fps", "max_accel", "max_speed", "frame_start",
                        "frame_end", "model"):
                m = re.search(rf"\b{key}=([0-9.]+)", line)
                if m:
                    info[key] = m.group(1)
    return info


def load_stage_report(path: str) -> tuple[dict, dict]:
    """返回 (rows, header)。rows: frame → {raw_val, signals...}

    新报告含 sig_abs；旧报告（sig_physics/sig_linearity）仍兼容读取。
    """
    header = _parse_header(path)
    rows: dict[int, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fr = int(r["frame"])
            row = {
                "raw": float(r["raw_val"]),
                "ocr_conf": float(r["sig_ocr_conf"]),
                "accel": float(r["sig_accel"]),
                "freq": float(r["sig_freq"]),
                "combined": float(r["combined_conf"]),
            }
            if r.get("sig_abs"):
                row["abs"] = float(r["sig_abs"])
            if r.get("sig_physics"):
                row["physics"] = float(r["sig_physics"])
            if r.get("sig_linearity"):
                row["linearity"] = float(r["sig_linearity"])
            rows[fr] = row
    return rows, header


def load_truth(path: str) -> dict[int, float]:
    """返回 frame → 正确显示速度（浮点）。

    truth 文件格式：`frame, 位移(m), 显示速度(km/h), flag` —— 第 3 列（下标 2）
    是人工校对的"OCR 应读出的数字"，作为误差对比基准。
    """
    truth: dict[int, float] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            fr = int(row[0])
            truth[fr] = float(row[2])
    return truth


def build_arrays(stage_path: str, truth_path: str) -> dict:
    """合并 stage report 与 truth，构造逐帧评估数组。"""
    rows, _ = load_stage_report(stage_path)
    truth = load_truth(truth_path)
    header = _parse_header(truth_path)  # fps/max_accel 等参数在 truth 头
    fps = float(header.get("fps", 60.0))

    frames = sorted(set(rows) & set(truth))
    vals = np.array([rows[fr]["raw"] for fr in frames], dtype=np.float64)
    true = np.array([truth[fr] for fr in frames], dtype=np.float64)
    times = np.array([fr / fps for fr in frames], dtype=np.float64)
    ocr_conf = np.array([rows[fr]["ocr_conf"] for fr in frames])
    sig_keys = ("abs", "accel", "freq", "combined")
    signals: dict[str, np.ndarray] = {}
    for k in sig_keys:
        if k in rows[frames[0]]:
            signals[k] = np.array([rows[fr][k] for fr in frames], dtype=np.float64)
    # 兼容旧报告（无 sig_abs，保留 physics/linearity）
    for k in ("physics", "linearity"):
        if k in rows[frames[0]]:
            signals[k] = np.array([rows[fr][k] for fr in frames], dtype=np.float64)

    error = np.abs(vals - true) >= 0.5
    valid = vals >= 0
    return {
        "frames": frames, "vals": vals, "true": true, "times": times,
        "ocr_conf": ocr_conf, "signals": signals,
        "error": error, "valid": valid, "n": len(frames), "fps": fps,
        "max_accel": float(header.get("max_accel", 50.0)),
        "max_speed": float(header.get("max_speed", 400.0)),
    }


# ── 评估 ──────────────────────────────────────────────────

def evaluate(scores: np.ndarray, data: dict, threshold: float = 70.0) -> dict:
    """对逐帧分数数组计算检出率 / 误报率。

    scores: 每帧 [0,100] 置信度（低 = 可疑）。NaN → 中性 50。
    """
    err = data["error"]
    valid = data["valid"]
    n = data["n"]

    scores = np.nan_to_num(scores, nan=50.0)
    flagged = scores < threshold

    # 错误帧（有效且 raw≠true）
    err_mask = err & valid
    # 正确帧（有效且 raw==true）
    ok_mask = valid & ~err

    det = flagged & err_mask
    fp = flagged & ok_mask

    return {
        "n": n,
        "n_err": int(err_mask.sum()),
        "n_ok": int(ok_mask.sum()),
        "n_flagged": int(flagged.sum()),
        "detect": 100.0 * det.sum() / err_mask.sum() if err_mask.sum() else float("nan"),
        "fp": 100.0 * fp.sum() / ok_mask.sum() if ok_mask.sum() else float("nan"),
        "flag_rate": 100.0 * flagged.sum() / max(1, n),
    }


def fmt(e: dict) -> str:
    return (f"检出 {e['detect']:5.1f}%  误报 {e['fp']:5.1f}%  "
            f"(flag {e['flag_rate']:4.1f}%  err={e['n_err']} ok={e['n_ok']})")


# ── 候选新信号（离线原型）──────────────────────────────

def _median_smooth(vals: np.ndarray, window: int) -> np.ndarray:
    """滑动窗口中位数（边缘扩展）。window 奇数。"""
    half = window // 2
    v = np.pad(vals, (half, half), mode="edge")
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        out[i] = np.median(v[i:i + window])
    return out


def _local_bandwidth(smooth: np.ndarray, window: int, floor: float = 1.0) -> np.ndarray:
    """局部真实带宽：平滑信号一阶差分绝对值的滑动中位数（+floor）。

    真实变速（ramp）时带宽大 → 大偏差不罚；巡航时带宽小 → 小偏差即可疑。
    """
    d = np.abs(np.diff(smooth, prepend=smooth[0]))
    half = window // 2
    dp = np.pad(d, (half, half), mode="edge")
    n = len(d)
    out = np.zeros(n)
    for i in range(n):
        out[i] = max(float(np.median(dp[i:i + window])), floor)
    return out


def _sig_median_residual(vals: np.ndarray, window: int = 15,
                         bw_floor: float = 1.0) -> np.ndarray:
    """中位数平滑残差，按局部带宽归一化。

    残差 = |raw − median_smooth|；score = 100*exp(-resid / max(bw, floor))。
    单尖峰/短孤岛：中位数不受污染 → 残差大 → 检出；真实 ramp：带宽大 → 不误报。
    """
    smooth = _median_smooth(vals, window)
    resid = np.abs(vals - smooth)
    bw = _local_bandwidth(smooth, window, bw_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 100.0 * np.exp(-resid / np.maximum(bw, 1e-9))
    scores = np.where(vals < 0, 0.0, scores)
    return np.round(scores, 1)


def candidate_signal_report(data: dict, threshold: float = 70.0) -> None:
    """原型信号 + 组合评估。"""
    vals = data["vals"]
    print(f"── 候选信号（threshold={threshold}）──")
    mw15 = _sig_median_residual(vals, 15, 1.0)
    mw5 = _sig_median_residual(vals, 5, 1.0)
    print(f"  {'med15':9s} {fmt(evaluate(mw15, data, threshold))}")
    print(f"  {'med5':9s} {fmt(evaluate(mw5, data, threshold))}")


# ── 当前 5 信号逐一评估 ────────────────────────────────────

def individual_signal_report(data: dict, threshold: float = 70.0) -> None:
    """每个信号单独作为 conf 时的检出/误报（信号分数 < thr 即 flag）。"""
    print(f"── 各信号单独判别（threshold={threshold}）──")
    for name in ("ocr_conf", "abs", "physics", "linearity", "accel", "freq", "combined"):
        if name == "ocr_conf":
            scores = data["ocr_conf"]
        elif name in data["signals"]:
            scores = data["signals"][name]
        else:
            continue
        print(f"  {name:9s} {fmt(evaluate(scores, data, threshold))}")


def error_breakdown(data: dict) -> None:
    """错误帧幅值分布 —— 说明各信号的适用误差量级。"""
    err = data["error"] & data["valid"]
    diff = np.abs(data["vals"] - data["true"])[err]
    hist: dict[str, int] = {}
    for d in diff:
        if d < 1.5: k = "±1"
        elif d < 2.5: k = "±2"
        elif d < 5: k = "±3-4"
        elif d < 10: k = "±5-9"
        elif d < 20: k = "±10-19"
        else: k = "±20+"
        hist[k] = hist.get(k, 0) + 1
    total = sum(hist.values())
    parts = "  ".join(f"{k}:{v}({100*v/total:.0f}%)" for k, v in sorted(hist.items()))
    print(f"  错误帧幅值分布 (n={total}): {parts}")


def main() -> None:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    tests = [
        ("test4", "outputs/bench_test4_r1_stage_report.csv",
         "ground_truth_csv/test4_truth.csv"),
        ("test5", "outputs/bench_test5_r1_stage_report.csv",
         "ground_truth_csv/test5_ref.csv"),
        ("test6", "outputs/bench_test6_r1_stage_report.csv",
         "ground_truth_csv/test6_ref.csv"),
    ]
    for name, sp, tp in tests:
        data = build_arrays(sp, tp)
        print(f"\n════════ {name} (fps={data['fps']:.1f}, max_accel={data['max_accel']:.0f}, n={data['n']}) ════════")
        error_breakdown(data)
        individual_signal_report(data, threshold)
        candidate_signal_report(data, threshold)


if __name__ == "__main__":
    main()
