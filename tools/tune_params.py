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
            if line.startswith('#'): continue
            parts = line.strip().split(',')
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
    """Temporarily update config.py LCS constants."""
    with open(CONFIG, 'r', encoding='utf-8') as f:
        original = f.read()
    content = original
    for key, val in kwargs.items():
        import re
        content = re.sub(rf'^{key}:.*$', f'{key}: {val}', content, flags=re.MULTILINE)
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
            {"LCS_ERROR_LOW": "float = 0.25", "LCS_TRUST_HIGH": "float = 0.75"},
            {"LCS_ERROR_LOW": "float = 0.35", "LCS_TRUST_HIGH": "float = 0.65"},
            {"LCS_TRUST_HIGH": "float = 0.80"},
            {"LCS_CANDIDATE_ACCEPT": "float = 0.80"},
            {"LCS_TIME_WINDOW": "float = 0.3"},
            {"LCS_TAU": "float = 0.04"},
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
