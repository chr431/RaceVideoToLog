"""实验：两遍扫描尖峰检测（在修正后序列上做双侧中值一致偏离判别）。

第一遍：生产 detect+conf+dense_correct（夹具 corr 即其结果）——去污染。
第二遍：对 corr1==raw（生产未改）的 len=1 段：
  - left_med / right_med：修正后序列上左右窗口（段索引 ±k，不含自身）中值
  - 尖峰 = 两侧一致偏离（raw 同时高于两侧或同时低于两侧）且 min(dev) ≥ thresh
  - 修正 target = round(mean(left_med, right_med))，|raw - target| ≥ min_fix 才改
安全性：正确段（|raw-truth|≤1）被误 flag/误改的数量必须为 0。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

VIDEOS = ["test", "test2", "test3", "test5", "test6"]
FIX = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "seg_series"
TOL = 1.0


def load(v):
    d = json.loads((FIX / f"{v}.json").read_text(encoding="utf-8"))
    return d["segments"], d["meta"]


def side_meds(vals, i, k):
    """左右窗口中值（不含自身；None 跳过）。窗口按段索引。"""
    n = len(vals)
    left = [vals[j] for j in range(max(0, i - k), i) if vals[j] is not None]
    right = [vals[j] for j in range(i + 1, min(n, i + k + 1))
             if vals[j] is not None]
    lm = float(np.median(left)) if left else None
    rm = float(np.median(right)) if right else None
    return lm, rm, len(left), len(right)


def spike_pass(segs, k=3, thresh=2.0, min_fix=2.0, min_nbr=2):
    """第二遍：返回 (out, flags, info)。out 为修正后列表。"""
    raw = [s["raw"] for s in segs]
    corr1 = [s["corr"] for s in segs]
    lens = [s["len"] for s in segs]
    # 第二遍在修正后序列上做判别，但只允许修改生产未改动的段
    base = [corr1[i] if corr1[i] is not None else raw[i] for i in range(len(raw))]
    out = list(base)
    flags = [False] * len(raw)
    info = {"flagged": 0, "changed": 0, "missed_ok": 0, "harm": 0}
    for i in range(len(raw)):
        if raw[i] is None or corr1[i] != raw[i]:
            continue  # 生产已改/无值 → 跳过
        if lens[i] != 1:
            continue  # 只处理 len=1 单帧段
        lm, rm, nl, nr = side_meds(base, i, k)
        if lm is None or rm is None:
            continue
        if nl < min_nbr or nr < min_nbr:
            continue
        d_left = raw[i] - lm
        d_right = raw[i] - rm
        # 两侧一致偏离（同号）且至少一侧偏离 ≥ thresh
        if d_left * d_right <= 0:
            continue
        if max(abs(d_left), abs(d_right)) < thresh:
            continue
        # 孤立值判别：raw 值在 ±k 邻域内不重复（真实曲线段的值通常与
        # 邻居重复出现；孤立值 = 更可能是 OCR 误读）
        if any(base[j] == raw[i] for j in range(max(0, i - k), i)) or \
           any(base[j] == raw[i] for j in range(i + 1, min(len(raw), i + k + 1))):
            continue
        # 修正目标：两侧中值中离 raw 更远的一侧（尖峰拉回远离方向）
        if abs(d_left) >= abs(d_right):
            target = int(round(lm))
        else:
            target = int(round(rm))
        flags[i] = True
        info["flagged"] += 1
        if abs(raw[i] - target) >= min_fix and target != raw[i]:
            out[i] = target
            info["changed"] += 1
            # 正确性核查（用 truth，仅诊断用）
            t = segs[i]["truth"]
            if t is not None and abs(raw[i] - t) <= TOL and abs(target - t) > TOL:
                info["harm"] += 1
        else:
            t = segs[i]["truth"]
            if t is not None and abs(raw[i] - t) > TOL and abs(target - t) > TOL:
                info["missed_ok"] += 1
    return out, flags, info


def funnel(segs, out):
    seg = raw = fix = fix_wrong = missed = harm = final = 0
    for s, o in zip(segs, out):
        t, sv = s["truth"], s["raw"]
        if t is None or sv is None:
            continue
        seg += 1
        raw_err = abs(sv - t) > TOL
        final_err = o is None or abs(o - t) > TOL
        changed = o is not None and o != sv
        if raw_err:
            raw += 1
            if changed:
                if final_err:
                    fix_wrong += 1
                else:
                    fix += 1
            else:
                missed += 1
        else:
            if changed and final_err:
                harm += 1
        if final_err:
            final += 1
    return {"seg": seg, "raw": raw, "fix": fix, "fix_wrong": fix_wrong,
            "missed": missed, "harm": harm, "final": final}


def main():
    best = None
    all_rows = []
    for k in (2, 3, 5, 8):
        for thresh in (1.5, 2.0, 2.5, 3.0):
            for min_fix in (1.0, 2.0, 3.0):
                tot = {kk: 0 for kk in ("seg", "raw", "fix", "fix_wrong",
                                        "missed", "harm", "final")}
                per = {}
                flags_tot = changed_tot = harm_tot = 0
                for v in VIDEOS:
                    segs, meta = load(v)
                    out, flags, info = spike_pass(segs, k=k, thresh=thresh,
                                                  min_fix=min_fix)
                    f = funnel(segs, out)
                    per[v] = f
                    for kk in tot:
                        tot[kk] += f[kk]
                    flags_tot += info["flagged"]
                    changed_tot += info["changed"]
                    harm_tot += info["harm"]
                key = (tot["final"], -tot["fix"], tot["harm"], k, thresh,
                       min_fix)
                all_rows.append((tot["final"], k, thresh, min_fix,
                                 tot["harm"], per))
                if best is None or key < best[0]:
                    best = (key, per, tot, flags_tot, changed_tot, harm_tot,
                            k, thresh, min_fix)
    print("全部组合（final 升序，前 15）：")
    all_rows.sort(key=lambda r: (r[0], r[4], r[1], r[2], r[3]))
    for r in all_rows[:15]:
        line = " ".join(f"{r[5][v]['final']}" for v in VIDEOS)
        print(f"  final={r[0]:>3} k={r[1]} thresh={r[2]} min_fix={r[3]} "
              f"harm={r[4]}   {line}")
    print()
    _, per, tot, ft, ct, ht, k, th, mf = best
    print(f"最优: k={k} thresh={th} min_fix={mf} → total final={tot['final']}"
          f"（基线 11，夹具 corr=11）")
    print(f"  第二遍 flag={ft} 实际改动={ct} 误改={ht}")
    for v in VIDEOS:
        f = per[v]
        print(f"  {v:<6} final={f['final']} fix={f['fix']} "
              f"fix_wrong={f['fix_wrong']} missed={f['missed']} "
              f"harm={f['harm']}")
    # 展示最优配置下全部被改动的段
    print("\n── 最优配置逐案例（改动段）──")
    for v in VIDEOS:
        segs, meta = load(v)
        out, flags, info = spike_pass(segs, k=k, thresh=th, min_fix=mf)
        for i, (s, o) in enumerate(zip(segs, out)):
            if o != s["corr"]:
                mark = "✓" if (s["truth"] is not None
                               and abs(o - s["truth"]) <= TOL) else "✗"
                print(f" {v}#{i:<5} raw={s['raw']!s:>4} corr={s['corr']!s:>4} "
                      f"new={o!s:>4} truth={s['truth']!s:>4} len={s['len']}"
                      f" [{mark}]")


if __name__ == "__main__":
    main()
