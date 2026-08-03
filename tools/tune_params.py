"""参数调优工具 — 运行完整管线并对比 ground truth。"""
import subprocess, sys, os, csv, json, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
CONFIG = PROJECT / "config.py"
TESTS = [
    {
        "name": "test4",
        "video": "test4.mp4",
        "roi": "862 945 957 1003",
        "max_speed": 400, "max_accel": 70,
        "start": 114, "end": 6317,
    },
    {
        "name": "test",
        "video": "test.mp4",
        "roi": "877 935 961 986",
        "max_speed": 400, "max_accel": 50,
        "start": 687, "end": 4260,
    },
]

def load_speeds(path):
    speeds = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) >= 4:
                speeds[int(float(parts[0]))] = (float(parts[2]), int(parts[3]))
    return speeds

def evaluate(auto_path, truth_path):
    truth = load_speeds(truth_path)
    auto = load_speeds(auto_path)
    common = sorted(set(truth) & set(auto))
    diffs = sorted(abs(truth[fi][0] - auto[fi][0]) for fi in common)
    n = len(common)
    errors = sum(1 for d in diffs if d > 0.5)
    return {
        "frames": n,
        "mean": sum(diffs) / n,
        "median": diffs[n // 2],
        "max": max(diffs),
        "errors_05": errors,
        "errors_05_pct": 100 * errors / n,
        "errors_5": sum(1 for d in diffs if d > 5),
        "exact_pct": 100 * sum(1 for d in diffs if d < 0.5) / n,
    }

def run_test(test, params_override=None):
    """Run pipeline on one test, return evaluation results."""
    name = test["name"]
    out_csv = PROJECT / f"{name}_auto.csv"

    cmd = [
        sys.executable, str(PROJECT / "RaceVideoToLog.py"),
        test["video"],
        "--roi"] + test["roi"].split() + [
        "--max-speed", str(test["max_speed"]),
        "--max-accel", str(test["max_accel"]),
        "--div", "1",
        "--target-h", "48", "--pad", "0",
        "--frame-start", str(test["start"]),
        "--frame-end", str(test["end"]),
        "--ocr-model", "v6_tiny", "--reocr-model", "v6_small",
        "-o", str(out_csv),
        "--backend", "auto",
    ]

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    elapsed = time.time() - t0

    if r.returncode != 0:
        return {"error": r.stderr[-500:]}

    truth = PROJECT / f"ground_truth_csv/{name}_truth.csv"
    result = evaluate(out_csv, truth)
    result["time"] = elapsed
    result["name"] = name
    return result

def print_result(r):
    if "error" in r:
        print(f"  ERROR: {r['error'][:200]}")
        return
    print(f"  {r['name']}: {r['frames']} frames, {r['time']:.1f}s")
    print(f"  Mean={r['mean']:.2f}  Median={r['median']:.2f}  Max={r['max']:.0f} km/h")
    print(f"  Exact: {r['exact_pct']:.1f}%  Errors>0.5: {r['errors_05']} ({r['errors_05_pct']:.1f}%)  Errors>5: {r['errors_5']}")

def update_config(**kwargs):
    """Temporarily update config.py constants. Unmatched keys warn (no silent no-op)."""
    with open(CONFIG, 'r', encoding='utf-8') as f:
        original = f.read()
    content = original
    import re
    for key, val in kwargs.items():
        new_content, n = re.subn(rf'^{key}:.*$', f'{key}: {val}', content, flags=re.MULTILINE)
        if n == 0:
            print(f'[WARN] config key not found (no-op): {key}')
        content = new_content
    with open(CONFIG, 'w', encoding='utf-8') as f:
        f.write(content)
    return original

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        print("=== Baseline ===")
        for test in TESTS:
            r = run_test(test)
            print_result(r)
    elif len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Example: python tools/tune_params.py --sweep
        param_sets = [
            {"VITERBI_OBS_WEIGHT": "float = 0.2", "VITERBI_ACCEL_WEIGHT": "float = 0.8"},
            {"VITERBI_OBS_WEIGHT": "float = 0.4", "VITERBI_ACCEL_WEIGHT": "float = 1.2"},
            {"ERROR_DETECT_ACCEL_SPIKE_WEIGHT": "float = 0.30"},
            {"ERROR_DETECT_ACCEL_SPIKE_WEIGHT": "float = 0.50"},
            {"ERROR_DETECT_LINEARITY_WEIGHT": "float = 0.10"},
            {"ERROR_DETECT_LINEARITY_WEIGHT": "float = 0.20"},
            {"AUTO_CORRECT_THRESHOLD": "int = 70"},
            {"AUTO_CORRECT_THRESHOLD": "int = 90"},
        ]
        for i, params in enumerate(param_sets):
            print(f"\n=== Sweep {i+1}: {params} ===")
            original = update_config(**params)
            try:
                for test in TESTS:
                    r = run_test(test)
                    print_result(r)
            finally:
                with open(CONFIG, 'w', encoding='utf-8') as f:
                    f.write(original)
