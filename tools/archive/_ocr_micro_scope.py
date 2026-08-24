# -*- coding: utf-8 -*-
"""OCR 阶段微观探针：TRT 单批延迟分布 × 线程级资源记账 × 隔离策略矩阵。

目的：
  1) SMT 拓扑实测校准（兄弟逻辑核吞吐判别），构造真正的物理核分区；
  2) 主进程只跑 TRT OCR 批循环，记录每次调用延迟（p50/p90/p99/max）；
  3) 100ms 线程级 CPU 记账：主进程内哪个线程在烧、对端进程烧多少、
     每逻辑核占用率；
  4) 对端矩阵：onnx(8T/2T) x {自由核, 物理分区, 分区+BELOW_NORMAL} x
     dav1d x sleep-burner(控制组)。若分区/降级能恢复独跑延迟 -> 调度/
     核竞争机制实锤；若不能 -> 共享硬件资源（LLC/Fabric）回桌。
  5) --mode floor-gpu / floor-cpu：NVDEC/dav1d 全窗口纯解码下限。

用法：
  python tools/archive/_ocr_micro_scope.py                 # 全矩阵
  python tools/archive/_ocr_micro_scope.py --mode floor-gpu
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs" / "dual_micro"
_ENG = PROJECT / "third_party" / "video_ocr_engine"
for p in (str(PROJECT), str(_ENG)):
    if p not in sys.path:
        sys.path.insert(0, p)

BATCH = 16
IMG_W, IMG_H = 159, 48   # test6 ROI 宽高比 109:33 -> 48 高时宽 ~159


def parse_cores(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def apply_env_limits():
    cores = os.environ.get("MICRO_CORES", "")
    if cores:
        try:
            psutil.Process().cpu_affinity(parse_cores(cores))
        except Exception as e:
            print("[peer] affinity failed:", e)
    prio = os.environ.get("MICRO_PRIO", "")
    try:
        if prio == "below":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif prio == "idle":
            psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:
        print("[peer] nice failed:", e)


def make_engine(kind, threads):
    import video_ocr_engine.extractor  # noqa: F401  先完成包初始化防循环导入
    import engine_config as eco
    from ocr_native import OcrEngine
    return OcrEngine(eco.DEFAULT_OCR_MODEL, kind, fill_width=224,
                     num_threads=threads,
                     progress_cb=lambda m, p: None)


def peer_main(args):
    apply_env_limits()
    mode = os.environ["MICRO_MODE"]
    dur = float(os.environ.get("MICRO_DUR", "12"))
    thr = int(os.environ.get("MICRO_THREADS", "8"))
    prog = OUT_DIR / "_micro_peer_progress.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def beat(counter):
        tmp = prog.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"mode": mode, "count": counter,
             "t": round(time.perf_counter(), 2)}), encoding="utf-8")

    if mode == "burn":
        a = np.random.rand(384, 384).astype(np.float32)
        w0 = time.perf_counter()
        c0 = time.process_time()
        end = w0 + dur
        n = 0
        while time.perf_counter() < end:
            a = a @ a * np.float32(0.001)
            a = np.clip(a, -1e3, 1e3)
            n += 1
        out_path = os.environ.get("MICRO_OUT", "")
        if out_path:
            Path(out_path).write_text(json.dumps(
                {"cpu": round(time.process_time() - c0, 3),
                 "wall": round(time.perf_counter() - w0, 3),
                 "iters": n}), encoding="utf-8")
        return

    if mode == "memcopy":
        mb = int(os.environ.get("MICRO_MB", "64"))
        nbytes = mb * 1024 * 1024
        src = np.random.rand(nbytes // 4).astype(np.float32)
        dst = np.empty_like(src)
        end = time.perf_counter() + dur
        total = 0
        while time.perf_counter() < end:
            np.copyto(dst, src)
            np.copyto(src, dst)
            total += 4 * nbytes   # 读+写各一遍 x2 方向
            beat(total)
        beat(total)
        return

    if mode == "sleep":
        end = time.perf_counter() + dur
        ev = threading.Event()

        def sleeper():
            while not ev.is_set() and time.perf_counter() < end:
                time.sleep(0.001)
        ts = [threading.Thread(target=sleeper) for _ in range(thr)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return

    if mode == "onnx":
        eng = make_engine("onnxruntime", thr)
        batch = [np.random.rand(IMG_H, IMG_W, 3).astype(np.float32)
                 for _ in range(BATCH)]
        eng(batch)  # warm: 让 ORT 线程池起来
        end = time.perf_counter() + dur
        segs = 0
        while time.perf_counter() < end:
            eng(batch)
            segs += BATCH
            beat(segs)
        beat(segs)
        return

    if mode == "dav1d":
        f0 = int(os.environ.get("MICRO_F0", "1639"))
        f1 = int(os.environ.get("MICRO_F1", "4639"))
        from video_ocr_engine.extractor import FieldExtractor
        roi = (841, 994, 949, 1026)
        ex = FieldExtractor(str(VIDEO_DIR / "test6.mp4"), roi,
                            frame_start=f0, frame_end=f1,
                            decode_backend="cpu",
                            progress_cb=lambda m, p: None)
        vr = ex._open_vr()
        x1, y1, x2, y2 = ex._roi
        frames = list(range(f0, min(f1, len(vr))))
        end = time.perf_counter() + dur
        frames_done = 0
        while time.perf_counter() < end:
            for bs in range(0, len(frames), 64):
                be = min(bs + 64, len(frames))
                vr.get_batch(frames[bs:be],
                             roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
                frames_done += be - bs
                beat(frames_done)
                if time.perf_counter() >= end:
                    break
        return

    raise SystemExit("unknown MICRO_MODE: " + mode)


def pair_rate(a, b):
    """两个单核 burner 钉到逻辑核 a/b；用子进程自报的稳态 cpu/wall 判兄弟。"""
    procs = []
    outs = []
    for i, c in enumerate((a, b)):
        op = OUT_DIR / ("_cal_%d_%d.json" % (i, c))
        env = dict(os.environ, MICRO_MODE="burn", MICRO_DUR="2.5",
                   MICRO_CORES=str(c), MICRO_THREADS="1",
                   MICRO_OUT=str(op))
        procs.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--peer"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        outs.append(op)
    for p in procs:
        p.wait()
    tot_cpu = 0.0
    max_wall = 0.0
    ok = 0
    for op in outs:
        try:
            d = json.loads(op.read_text(encoding="utf-8"))
            tot_cpu += d["cpu"]
            max_wall = max(max_wall, d["wall"])
            ok += 1
        except Exception:
            pass
    if ok < 2 or max_wall <= 0:
        return float("nan")
    return tot_cpu / max_wall


def smt_mapping():
    n = psutil.cpu_count(logical=True)
    half = n // 2
    rs = [pair_rate(0, half), pair_rate(0, 1)]
    r_sib, r_adj = rs[0], rs[1]
    mapping = "i,i+N/2" if r_sib < r_adj - 0.12 else "2i,2i+1"
    print("[smt] rate(0,%d)=%.2f  rate(0,1)=%.2f  -> mapping %s"
          % (half, r_sib, r_adj, mapping), flush=True)
    return mapping, n


def expand_physical(phys_ids, mapping, n):
    half = n // 2
    out = []
    for i in phys_ids:
        out.append(i)
        if mapping == "i,i+N/2":
            out.append(i + half)
        else:
            out.append(2 * i + 1)
    return out


class Scope:
    def __init__(self, peer_pids):
        self.peer_pids = peer_pids or []
        self.rows = []
        self._stop = threading.Event()
        self._thr = None

    def _snap_threads(self, proc):
        try:
            with proc.oneshot():
                return {t.id: t.user + t.system
                        for t in proc.threads()}
        except Exception:
            return {}

    def start(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        me = psutil.Process()
        peers = []
        for pid in self.peer_pids:
            try:
                peers.append((pid, psutil.Process(pid)))
            except Exception:
                pass
        prev_me = self._snap_threads(me)
        prev_peers = [(pid, self._snap_threads(p)) for pid, p in peers]
        prev_core = psutil.cpu_times(percpu=True)
        t_prev = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(0.1)
            t_now = time.perf_counter()
            dt = t_now - t_prev
            row = {"t": round(t_now, 3), "dt": round(dt, 3)}
            cur_me = self._snap_threads(me)
            dme = [(tid, v - prev_me.get(tid, 0.0))
                   for tid, v in cur_me.items()]
            dme = sorted(dme, key=lambda x: -x[1])[:6]
            row["me_top"] = [(tid, round(d / dt, 2)) for tid, d in dme
                             if d > 0.005]
            prev_me = cur_me
            pc = []
            for i, (pid, p) in enumerate(peers):
                cur = self._snap_threads(p)
                prev = dict(prev_peers[i][1]) if i < len(prev_peers) else {}
                dv = sum(cur.values()) - sum(prev.values())
                pc.append({"pid": pid, "cpu": round(dv / dt, 2),
                           "nthreads": len(cur)})
                prev_peers[i] = (pid, cur)
            row["peers"] = pc
            ct = psutil.cpu_times(percpu=True)
            busy = []
            for c_now, c_prev in zip(ct, prev_core):
                dtot = ((c_now.user - c_prev.user)
                        + (c_now.system - c_prev.system))
                dall = sum(a - b for a, b in zip(c_now, c_prev)) or 1e-6
                busy.append(round(min(1.0, max(dtot, 0.0) / dall), 2))
            row["cores"] = busy
            prev_core = ct
            self.rows.append(row)
            t_prev = t_now

    def stop_and_summarize(self):
        self._stop.set()
        if self._thr:
            self._thr.join(3)
        rows = self.rows
        if not rows:
            return {}
        ncore = len(rows[0]["cores"])
        avg = [round(sum(r["cores"][i] for r in rows) / len(rows), 2)
               for i in range(ncore)]
        return {
            "samples": len(rows),
            "avg_core_busy": avg,
            "avg_busy_cores_gt20": sum(1 for b in avg if b > 0.2),
        }


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


class Canary:
    """subject 进程内的固定负载线程：matmul 迭代速率 = 宿主吞吐代理。"""

    def __init__(self):
        self._stop = threading.Event()
        self.iters = 0
        self._thr = None

    def _loop(self):
        a = np.random.rand(224, 224).astype(np.float32)
        while not self._stop.is_set():
            a = a @ a * np.float32(0.001)
            a = np.clip(a, -1e3, 1e3)
            self.iters += 1

    def __enter__(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


def run_condition(name, spec, eng, batch, dur):
    proc = None
    prog_file = OUT_DIR / "_micro_peer_progress.json"
    start_count = 0
    canary = Canary()
    canary.__enter__()
    canary._stop.clear()
    c_start = canary.iters
    if spec:
        env = dict(os.environ,
                   MICRO_MODE=spec["mode"],
                   MICRO_THREADS=str(spec.get("threads", 8)),
                   MICRO_DUR=str(dur + 8),
                   MICRO_CORES=spec.get("cores", ""),
                   MICRO_PRIO=spec.get("prio", ""))
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--peer"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        if prog_file.exists():
            try:
                start_count = json.loads(
                    prog_file.read_text(encoding="utf-8")).get("count", 0)
            except Exception:
                start_count = 0
    scope = Scope([proc.pid] if proc else [])
    lat = []
    scope.start()
    t_end = time.perf_counter() + dur
    submit_tid = threading.get_ident()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        eng(batch)
        lat.append(time.perf_counter() - t0)
    time.sleep(0.2)
    c_iters = canary.iters - c_start
    canary.__exit__()
    summ = scope.stop_and_summarize()
    peer_rate = None
    if proc is not None:
        proc.kill()
        proc.wait(10)
        if prog_file.exists():
            try:
                end_count = json.loads(
                    prog_file.read_text(encoding="utf-8")).get("count", 0)
                peer_rate = round((end_count - start_count) / dur, 1)
            except Exception:
                pass
    res = {
        "name": name, "spec": spec,
        "n_calls": len(lat),
        "ms_mean": round(1000 * sum(lat) / len(lat), 2),
        "ms_p50": round(1000 * pct(lat, 0.50), 2),
        "ms_p90": round(1000 * pct(lat, 0.90), 2),
        "ms_p99": round(1000 * pct(lat, 0.99), 2),
        "ms_max": round(1000 * max(lat), 2),
        "batches_per_s": round(len(lat) / dur, 1),
        "seg_per_s": round(len(lat) * BATCH / dur, 1),
        "submit_tid": submit_tid,
        "peer_rate": peer_rate,
        "canary_per_s": round(c_iters / dur, 1),
        "scope": summ,
    }
    print("[%s] mean=%.2fms p50=%.2f p90=%.2f p99=%.2f max=%.2f (%.0f seg/s)%s"
          % (name, res["ms_mean"], res["ms_p50"], res["ms_p90"],
             res["ms_p99"], res["ms_max"], res["seg_per_s"],
             ("  peer=%s/s" % peer_rate) if peer_rate else ""
             ) + "  canary=%s/s" % res["canary_per_s"], flush=True)
    return res


def floors(f0, f1):
    """NVDEC / dav1d 纯解码下限（无分段无 OCR）。"""
    from video_ocr_engine.extractor import FieldExtractor
    roi = (841, 994, 949, 1026)
    out = {}
    for tag, backend in (("gpu", "auto"), ("cpu", "cpu")):
        ex = FieldExtractor(str(VIDEO_DIR / "test6.mp4"), roi,
                            frame_start=f0, frame_end=f1,
                            decode_backend=backend,
                            progress_cb=lambda m, p: None)
        vr = ex._open_vr()
        x1, y1, x2, y2 = ex._roi
        frames = list(range(f0, min(f1, len(vr))))
        t0 = time.perf_counter()
        for bs in range(0, len(frames), 64):
            be = min(bs + 64, len(frames))
            vr.get_batch(frames[bs:be],
                         roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
        wall = time.perf_counter() - t0
        out[tag] = {"wall_s": round(wall, 3),
                    "fps": round(len(frames) / wall, 1)}
        print("[floor-%s] %.3fs  %.0f fps (%d frames)"
              % (tag, wall, len(frames) / wall, len(frames)), flush=True)
        del vr, ex
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", action="store_true")
    ap.add_argument("--mode", default="")
    ap.add_argument("--f0", type=int, default=139)
    ap.add_argument("--f1", type=int, default=3139)
    ap.add_argument("--dur", type=float, default=5.0)
    args = ap.parse_args()

    if args.peer:
        peer_main(args)
        return
    if args.mode.startswith("floor"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        r = {"floors": floors(args.f0, args.f1)}
        (OUT_DIR / "floors.json").write_text(json.dumps(r, indent=1),
                                             encoding="utf-8")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # 可选：把 Windows 定时器分辨率钉到 1ms（MICRO_TBP=1）
    if os.environ.get("MICRO_TBP") == "1":
        import ctypes
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            print("[tbp] timeBeginPeriod(1) active", flush=True)
        except Exception as e:
            print("[tbp] failed:", e)

    # 频率遥测：typeperf 每核 % Processor Performance @200ms
    freq_csv = OUT_DIR / "_freq.csv"
    if freq_csv.exists():
        freq_csv.unlink()
    freq_proc = None
    try:
        freq_proc = subprocess.Popen(
            ["typeperf",
             r"\Processor Information(*)\% Processor Performance",
             "-si", "00:00:00.20", "-f", "csv", "-o", str(freq_csv)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        ok = freq_csv.exists() and freq_csv.stat().st_size > 0
        print("[freq] typeperf started, header_ok=%s" % ok, flush=True)
        if not ok:
            freq_proc.kill()
            freq_proc = None
    except Exception as e:
        print("[freq] unavailable:", e)
    wall_clock_0 = datetime.datetime.now()
    perf_clock_0 = time.perf_counter()

    def cond_window_enter():
        return {"perf_t": round(time.perf_counter(), 3)}

    mapping, n_logical = smt_mapping()
    n_phys = n_logical // 2
    subj_phys = list(range(0, min(10, n_phys)))
    peer_phys = list(range(min(10, n_phys), n_phys))
    subj_cores = expand_physical(subj_phys, mapping, n_logical)
    peer_cores = expand_physical(peer_phys, mapping, n_logical)
    print("[split] subject=%d logicals(phys %s) peer=%d logicals(phys %s)"
          % (len(subj_cores), subj_phys, len(peer_cores), peer_phys),
          flush=True)

    eng = make_engine("tensorrt", 2)
    batch = [np.random.rand(IMG_H, IMG_W, 3).astype(np.float32)
             for _ in range(BATCH)]
    for _ in range(30):
        eng(batch)

    D = args.dur
    conds = [
        ("T0_alone_tbp", None),
        ("T1_onnx8_tbp", {"mode": "onnx", "threads": 8}),
        ("T0b_alone_tbp", None),
    ]
    windows = {}
    for name, spec in conds:
        w = cond_window_enter()
        results[name] = run_condition(name, spec, eng, batch, D)
        results[name]["smt_mapping"] = mapping
        w["t_end"] = round(time.perf_counter(), 3)
        windows[name] = w
        time.sleep(1.0)
    results["_windows"] = windows
    results["_clock_map"] = {
        "wall_epoch": wall_clock_0.strftime("%Y-%m-%d %H:%M:%S"),
        "perf_t": round(perf_clock_0, 3)}

    m0a = results["T0_alone_tbp"]
    m0b = results.get("T0b_alone_tbp") or {}
    m0 = (m0a["ms_p50"] + m0b.get("ms_p50", m0a["ms_p50"])) / 2
    c0 = (m0a["canary_per_s"] + m0b.get("canary_per_s",
                                         m0a["canary_per_s"])) / 2
    print("")
    print("=== inflation vs M0 (p50 / canary basis) ===")
    for name, r in results.items():
        if name.startswith("_") or "ms_p50" not in r:
            continue
        print("%-22s p50=%.2fms x%.2f  canary=%.0f/s (%.2fx)  peer=%s" % (
            name, r["ms_p50"], r["ms_p50"] / m0,
            r["canary_per_s"], r["canary_per_s"] / c0,
            r.get("peer_rate")))

    if freq_proc is not None:
        freq_proc.kill()
        freq_proc.wait(5)

    # 频率分析：各条件窗口内 subject 核(0..15) 与 peer 核(16..31) 的
    # % Processor Performance 平均值
    if freq_csv.exists():
        try:
            import csv as _csv
            with freq_csv.open(encoding="utf-8", errors="replace") as fh:
                rows = list(_csv.reader(fh))
            header = rows[0]
            col_core = []
            for i, h in enumerate(header):
                if "(" in h and ")" in h:
                    inst = h.split("(")[1].split(")")[0]
                    try:
                        col_core.append((i, int(inst)))
                    except ValueError:
                        pass
            subj_set = set(subj_cores)
            peer_set = set(peer_cores)
            samples = []
            for rowv in rows[1:]:
                if len(rowv) < len(header):
                    continue
                ts = rowv[0]
                try:
                    t_wall = datetime.datetime.strptime(
                        ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                rel = (t_wall - wall_clock_0).total_seconds()
                vals = {}
                for i, core in col_core:
                    try:
                        vals[core] = float(rowv[i])
                    except (ValueError, IndexError):
                        pass
                samples.append((rel, vals))
            freq_summary = {}
            for name, w in windows.items():
                lo = w["perf_t"] - perf_clock_0
                hi = w["t_end"] - perf_clock_0
                ss = [v for rel, v in samples if lo <= rel <= hi]
                if not ss:
                    continue
                def avg_over(coreset):
                    tot = []
                    for v in ss:
                        for c in coreset:
                            if c in v:
                                tot.append(v[c])
                    return round(sum(tot) / len(tot), 1) if tot else None
                freq_summary[name] = {
                    "subj_perf_pct": avg_over(subj_set),
                    "peer_perf_pct": avg_over(peer_set),
                    "samples": len(ss)}
            results["_freq_summary"] = freq_summary
            print("")
            print("=== CPU frequency telemetry (% Processor Performance) ===")
            for k, v in freq_summary.items():
                print("%-22s subj=%s%% peer=%s%% (n=%s)" % (
                    k, v["subj_perf_pct"], v["peer_perf_pct"], v["samples"]))
        except Exception as e:
            print("[freq] analysis failed:", e)

    outp = OUT_DIR / ("micro_%d.json" % int(time.time()))
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("saved -> %s" % outp)


if __name__ == "__main__":
    main()
