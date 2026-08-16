"""对比 test2_truth.csv（旧 ground truth）与 test2_log.csv（人工复核 log）。

按帧号对齐速度列，列出全部差异帧，并与剩余 5 个错误的 rep 帧对照。
"""
from pathlib import Path


def load(p):
    d = {}
    for line in Path(p).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(",")
        try:
            d[int(float(parts[0]))] = (float(parts[2]), parts[3])
        except (ValueError, IndexError):
            pass
    return d


truth = load("ground_truth_csv/test2_truth.csv")
log = load("test2_log.csv")
diffs = sorted(f for f in (set(truth) & set(log))
               if abs(truth[f][0] - log[f][0]) > 1e-9)
print(f"{'帧':>6} {'truth':>6} {'log':>6} {'log-truth':>9}  "
      f"{'truth_flag':>10} {'log_flag':>7}")
for f in diffs:
    print(f"{f:>6} {truth[f][0]:>6.0f} {log[f][0]:>6.0f} "
          f"{log[f][0] - truth[f][0]:>+9.0f}  "
          f"{truth[f][1]:>10} {log[f][1]:>7}")

print()
print("剩余 5 个错误的 rep 帧对照：")
cases = [("#387", 1596), ("#483", 1841), ("#750", 2466),
         ("#923", 2871), ("#957", 3095)]
for name, rep in cases:
    in_d = rep in diffs
    tag = "★差异帧" if in_d else "  无差异"
    if rep in truth and rep in log:
        print(f"  {name} rep={rep}: {tag}  truth={truth[rep][0]:.0f} "
              f"log={log[rep][0]:.0f}")
    else:
        print(f"  {name} rep={rep}: {tag}  (无 truth/log)")
