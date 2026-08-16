"""离线重估：以 test2_log.csv（人工复核）为 truth 时 test2 的最终错误。

用夹具 seg_series/test2.json 的 raw/corr，把 truth 换成 log 对应 rep 帧值。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.archive._truth_vs_log import load  # noqa: E402

log = {f: v[0] for f, v in load("test2_log.csv").items()}  # 只取速度列
fx = json.load(open("tests/fixtures/seg_series/test2.json", encoding="utf-8"))
TOL = 1.0
old_err = new_err = no_log = 0
cases = []
for i, s in enumerate(fx["segments"]):
    rep = s["rep"]
    t_old = s["truth"]
    t_new = log.get(rep)
    o = s["corr"]
    if s["raw"] is None:
        continue
    if t_old is not None and o is not None and abs(o - t_old) > TOL:
        old_err += 1
        cases.append((i, rep, s["raw"], o, t_old, t_new))
    if t_new is None:
        no_log += 1
    elif o is not None and abs(o - t_new) > TOL:
        new_err += 1
print(f"旧 truth 口径: test2 final = {old_err}")
print(f"log 口径:     test2 final = {new_err}（无 log 值的段 {no_log}）")
print()
print(f"{'#':>5} {'rep':>6} {'raw':>4} {'corr':>4} {'old_t':>5} "
      f"{'log':>4}  verdict")
for i, rep, r, o, t_old, t_new in cases:
    if t_new is None:
        v = "无 log"
    else:
        v = "OK 生产输出正确" if abs(o - t_new) <= TOL else "ERR 仍错误"
    print(f"{i:>5} {rep:>6} {r!s:>4} {o!s:>4} {t_old!s:>5} {t_new!s:>4}  {v}")
