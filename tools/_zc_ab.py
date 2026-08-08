"""Zero-change constraint A/B across four videos."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent


def run(v, out_csv):
    video = f"D:/Videos/racelog_test/{v}.mp4"
    truth = PROJECT / f"ground_truth_csv/{v}_truth.csv"
    if not truth.exists():
        truth = PROJECT / f"ground_truth_csv/{v}_ref.csv"
    with open(out_csv + ".log", "w", encoding="utf-8") as fo:
        subprocess.run(
            [sys.executable, str(PROJECT / "RaceVideoToLog.py"),
             video, "--from-csv", str(truth), "--backend", "tensorrt",
             "--log-level", "detailed", "-o", out_csv],
            cwd=str(PROJECT), stdout=fo, stderr=fo, timeout=900, check=True)
    sys.path.insert(0, str(PROJECT))
    from tools.verify_accuracy import compare, load_truth
    return compare(load_truth(str(truth)), out_csv)


def main():
    # expected (without constraint): test5 7195, test6 23435, test.mp4 3505, test4 6089
    for v in ["test5", "test6", "test", "test4"]:
        out = str(PROJECT / "outputs" / f"_zcab_{v}.csv")
        s = run(v, out)
        t = s["matched"] + s["wrong"]
        print(f"{v:6s}: matched {s['matched']}/{t} err {s['error_rate']*100:.2f}% "
              f"falseT {s['false_trusted_count']}")
        Path(out).unlink(missing_ok=True)
        Path(out + ".log").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
