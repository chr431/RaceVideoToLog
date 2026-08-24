'''Dual-pipeline co-run slowdown attribution probe.
Candidates: A) in-process coupling (GIL/CUDA ctx/locks)
B) competition machinery (seek jumps / INFLIGHT drain gate / yield lag)
C) backend pairing (CPU path presence) D) same-file effects
E) hw resource contention (cores/mem bw)
Matrix:
  A1/A2 solo baselines (auto / cpu)          -- per-phase ms/frame reference
  B1    engine dual default (kfe + inflight gate + yield)
  B2    dual with INFLIGHT=0, SLOW_RATIO=0
  B3    dual explicit (auto,auto)x2 (drop CPU pairing)
  C1    same-process two threads static halves (no queue/competition/jump seeks)
  D1    two processes, same video halves (removes GIL/shared ctx)
  D2    two processes, different videos (subtitle-extractor scenario)
  B1r   B1 repeat (variance reference)
All runs serial; psutil per-core + nvidia-smi sampling alongside.
Usage:
  python tools/archive/_dual_phase_probe.py --video test6 --frames 3000
'''
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path("D:/Videos/racelog_test")
OUT_DIR = PROJECT / "outputs" / "dual_phase_probe"

sys.path.insert(0, str(PROJECT))
_ENG = PROJECT / "third_party" / "video_ocr_engine"
if _ENG.exists():
    sys.path.insert(0, str(_ENG))

os.environ.setdefault("ENGINE_PROFILE", "1")

import engine_config as eco  # noqa: E402
from video_ocr_engine.extractor import FieldExtractor  # noqa: E402


def load_meta(v):
    spec = importlib.util.spec_from_file_location(
        "detect_eval", PROJECT / "tools" / "detect_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_meta(v)


def metup(meta, video):
    return (meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], meta[6],
            meta[7], video)


class Sampler:
    def __init__(self):
        self.rows = []
        self._stop = threading.Event()
        self._thr = None

    def _loop(self):
        import psutil
        psutil.cpu_percent(percpu=True)
        it = 0
        while not self._stop.is_set():
            row = {"t": round(time.perf_counter(), 3),
                   "cores": psutil.cpu_percent(percpu=True)}
            if it % 3 == 0:
                try:
                    out = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,utilization.memory,"
                         "memory.used",
                         "--format=csv,noheader,nounits", "-i", "0"],
                        capture_output=True, text=True, timeout=5)
                    vals = [v.strip() for v in out.stdout.split(",")]
                    row["gpu_util"] = int(vals[0])
                    row["gpu_memutil"] = int(vals[1])
                    row["mem_mb"] = int(vals[2])
                except Exception:
                    pass
            self.rows.append(row)
            it += 1
            self._stop.wait(0.25)

    def __enter__(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thr:
            self._thr.join(3)

    def summary(self):
        if not self.rows:
            return {}
        n = len(self.rows)
        act = [sum(1 for c in r["cores"] if c > 20.0) for r in self.rows]
        mx = [max(r["cores"]) for r in self.rows]
        gu = [r["gpu_util"] for r in self.rows if "gpu_util" in r]
        gm = [r["mem_mb"] for r in self.rows if "mem_mb" in r]
        srt = sorted(gu) if gu else [0]
        p90i = min(len(srt) - 1, int(len(srt) * 0.9))
        return {"samples": n,
                "avg_active_cores": round(sum(act) / n, 1),
                "avg_max_core_pct": round(sum(mx) / n),
                "gpu_util_mean": round(sum(gu) / len(gu)) if gu else None,
                "gpu_util_p90": srt[p90i] if gu else None,
                "sys_mem_max_mb": max(gm) if gm else None}


CUR = {"video": "test6", "frames": 0}
METAS = {}
RES = {}


def _mk(video, dec, ocr, f0, f1, dual=False, backends=None):
    roi, fs, fe, fps, ms, ma, mw, _t8, _t9 = METAS[video]
    return FieldExtractor(
        str(VIDEO_DIR / (video + ".mp4")), roi,
        frame_start=f0, frame_end=f1, force_aspect=mw,
        decode_backend=dec, ocr_backend=ocr, yuv_output=True,
        dual_pipeline=dual, dual_backends=backends,
        progress_cb=lambda m_, p_: None)


def solo_run(video, dec, ocr, f0, f1):
    ex = _mk(video, dec, ocr, f0, f1)
    t0 = time.perf_counter()
    res = ex._run_pipelined(_force_single=True)
    wall = time.perf_counter() - t0
    n_fr = len(res[0]) if res and res[0] else 0
    return {"wall_s": round(wall, 3), "n_frames": n_fr,
            "n_segments": len(res[1]) if len(res) > 1 else -1,
            "profile": {g: dict(p) for g, p in ex.profile.items()},
            "timing": {"decode": ex.timing.get("decode"),
                       "ocr": ex.timing.get("ocr")}}


def engine_dual_run(video, backends=None, label="dual",
                    no_fallback=False):
    old_nfb = None
    if no_fallback:
        old_nfb = os.environ.pop(
            eco.DUAL_PIPELINE_NO_CODEC_FALLBACK_ENV, None)
        os.environ[eco.DUAL_PIPELINE_NO_CODEC_FALLBACK_ENV] = "1"
    try:
        return _engine_dual_inner(video, backends, label)
    finally:
        os.environ.pop(eco.DUAL_PIPELINE_NO_CODEC_FALLBACK_ENV, None)
        if old_nfb is not None:
            os.environ[eco.DUAL_PIPELINE_NO_CODEC_FALLBACK_ENV] = old_nfb


def _engine_dual_inner(video, backends, label):
    roi, fs, fe, fps, ms, ma, mw, _t8, _t9 = METAS[video]
    end = fs + CUR["frames"]
    ex = FieldExtractor(
        str(VIDEO_DIR / (video + ".mp4")), roi, frame_start=fs,
        frame_end=end, force_aspect=mw, decode_backend="auto",
        ocr_backend="auto", yuv_output=True, dual_pipeline=True,
        dual_backends=backends, progress_cb=lambda m_, p_: None)
    t0 = time.perf_counter()
    ex._run_pipelined()
    wall = time.perf_counter() - t0
    t = ex.timing
    pipes = {}
    for i in (1, 2):
        tag = "pipe" + str(i)
        tl = t.get("parallel_" + tag + "_timeline") or []
        pipes[tag] = {
            "backend": t.get("parallel_" + tag + "_backend", ""),
            "ocr": t.get("parallel_" + tag + "_ocr", ""),
            "chunks": t.get("parallel_" + tag + "_chunks", 0),
            "busy_wall": t.get("parallel_" + tag + "_s"),
            "drain": t.get("parallel_" + tag + "_drain"),
            "yielded": bool(t.get("parallel_" + tag + "_yield")),
            "frames": sum(int(e[3]) for e in tl),
            "gap_sums": round(sum(float(e[4]) for e in tl), 3),
            "timeline": tl,
        }
        pk = "producer:" + tag
        ok = "ocr:" + tag
        if pk in ex.profile or ok in ex.profile:
            pipes[tag]["profile"] = {"producer": dict(ex.profile.get(pk, {})),
                                     "ocr": dict(ex.profile.get(ok, {}))}
    return {"wall_s": round(wall, 3), "label": label,
            "stitched": t.get("parallel_stitched", 0),
            "reserve_skew": t.get("parallel_reserve_skew"),
            "profile": {g: dict(p) for g, p in ex.profile.items()},
            "pipes": pipes}


def inproc_static_halves(video, mid_f, f_end):
    fs = METAS[video][1]
    out = {}

    def work(tag, dec, ocr, a, b):
        out[tag] = solo_run(video, dec, ocr, a, b)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=work,
                            args=("pipe1", "auto", "auto", fs, mid_f)),
           threading.Thread(target=work,
                            args=("pipe2", "cpu", "cpu", mid_f, f_end))]
    for x in ths:
        x.start()
    for x in ths:
        x.join()
    return {"wall_s": round(time.perf_counter() - t0, 3),
            "mode": "inproc_static_halves", "pipes": out}


def decode_loop(video, f0, f1, secs):
    """纯 dav1d 解码循环（无 OCR）：作为对端负载。"""
    ex = _mk(video, "cpu", "cpu", f0, f1)
    vr = ex._open_vr()
    x1, y1, x2, y2 = ex._roi
    frames = list(range(f0, min(f1, len(vr))))
    t_end = time.perf_counter() + secs
    batches = 0
    while time.perf_counter() < t_end:
        for bs in range(0, len(frames), 64):
            be = min(bs + 64, len(frames))
            vr.get_batch(frames[bs:be],
                         roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
            batches += 1
            if time.perf_counter() >= t_end:
                break
    return {"mode": "decode_only", "batches": batches,
            "wall_s": round(secs, 1)}


def child_main(args):
    METAS[args.video] = metup(load_meta(args.video), args.video)
    if args.mode == "decode":
        r = decode_loop(args.video, args.f0, args.f1, args.secs)
    else:
        r = solo_run(args.video, args.dec, args.ocr, args.f0, args.f1)
    r.update({"video": args.video, "dec": args.dec, "ocr": args.ocr,
              "mode": args.mode})
    Path(args.out).write_text(json.dumps(r), encoding="utf-8")


def two_proc(jobs):
    outs = []
    procs = []
    t0 = time.perf_counter()
    for i, j in enumerate(jobs):
        op = OUT_DIR / ("_child_%d_%s_%s.json" % (i, j["video"], j["dec"]))
        if op.exists():
            op.unlink()
        outs.append(op)
        procs.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child",
             "--video", j["video"], "--dec", j["dec"], "--ocr", j["ocr"],
             "--f0", str(j["f0"]), "--f1", str(j["f1"]),
             "--out", str(op)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _, e = p.communicate()
        if p.returncode != 0:
            errs.append((e or b"").decode(errors="replace")[-600:])
    wall = time.perf_counter() - t0
    res = []
    for op in outs:
        res.append(json.loads(op.read_text(encoding="utf-8"))
                   if op.exists() else {"error": "missing"})
    for e in errs:
        print("[child stderr]", e)
    return wall, res


_PROD_KEYS = ["seek_accurate", "decode_batch", "gray_batch", "sharp_batch",
              "bin_batch", "segmentation"]


def phase_row(profile, n_fr):
    prod = profile.get("producer", {})
    ocrp = profile.get("ocr", {})
    den = max(1, n_fr)
    row = {}
    for k in _PROD_KEYS:
        row[k] = round(prod.get(k, 0.0) * 1000 / den, 2)
    for k in ("preprocess", "infer", "ctc_decode", "q_get_wait"):
        row[k] = round(ocrp.get(k, 0.0) * 1000 / den, 2)
    row["_prod_sum_s"] = round(sum(prod.get(k, 0.0)
                                   for k in _PROD_KEYS), 3)
    row["_ocr_act_s"] = round(sum(ocrp.get(k, 0.0) for k in
                                  ("preprocess", "infer", "ctc_decode")), 3)
    return row


def pipe_of(name, tag):
    d = RES.get(name) or {}
    pd = (d.get("pipes") or {}).get(tag) or {}
    pr = pd.get("profile")
    if not pr:
        return None
    fr = pd.get("frames") or pd.get("n_frames") or 0
    return phase_row(pr, fr)


def print_analysis(ref_gpu, ref_cpu):
    keys = _PROD_KEYS + ["infer", "ctc_decode", "preprocess"]
    hdr = ("%26s%6s%7s%6s" % ("config", "pipe", "wall", "fr")
           + "".join("%10s" % k[:9] for k in keys))
    lines = [hdr]
    for name, d in RES.items():
        pipes = d.get("pipes") or {}
        if pipes:
            for tag in sorted(pipes):
                pd = pipes[tag]
                pr = pd.get("profile") or {}
                fr = pd.get("frames") or pd.get("n_frames") or 0
                r = phase_row(pr, fr)
                w = pd.get("wall_s") or pd.get("busy_wall") or 0
                dec = pd.get("dec", "")
                lines.append("%26s%6s%7.2f%6d" % (
                    (name[:18] + "/" + dec)[:25], tag, w, fr)
                    + "".join("%10.2f" % r.get(k, 0) for k in keys))
        else:
            r = phase_row(d.get("profile") or {}, d.get("n_frames") or 0)
            lines.append("%26s%6s%7.2f%6d" % (
                name[:25], "-", d["wall_s"], d.get("n_frames") or 0)
                + "".join("%10.2f" % r.get(k, 0) for k in keys))
    print("\n".join(lines))

    rg = phase_row(ref_gpu["profile"], ref_gpu["n_frames"])
    rc = phase_row(ref_cpu["profile"], ref_cpu["n_frames"])
    print("\n=== inflation vs solo baseline (x times) ===")

    def ratios(name):
        out = []
        for tag, base, lab in (("pipe1", rg, "gpu"), ("pipe2", rc, "cpu")):
            row = pipe_of(name, tag)
            if row is None:
                continue
            parts = []
            for k in ("decode_batch", "segmentation", "infer", "preprocess"):
                parts.append("%sx%.2f" % (k[:6], row[k] / max(base[k], .01)))
            out.append("%s[%s]" % (lab, " ".join(parts)))
        return "  ".join(out)

    for name in ("B1c_dual_nofb", "B2c_dual_nofb_nogate",
                 "B3_dual_gpu_gpu", "C1_inproc_static",
                 "D1_proc_same_video", "D2_proc_diff_video",
                 "PA_gpu_vs_dav1d_peer", "B3r_gpu_gpu_rep",
                 "D2r_diff_video_rep", "B1c_rep_dual_nofb"):
        r = ratios(name)
        if r:
            print("%-24s %s" % (name, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test6")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--dec", default="auto")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="pipeline",
                    choices=["pipeline", "decode"])
    ap.add_argument("--secs", type=float, default=30.0)
    args = ap.parse_args()

    if args.child:
        child_main(args)
        return

    CUR["video"] = args.video
    CUR["frames"] = args.frames
    METAS[args.video] = metup(load_meta(args.video), args.video)
    if args.video != "test5":
        METAS["test5"] = metup(load_meta("test5"), "test5")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs0 = METAS[args.video][1]
    f_end = fs0 + args.frames
    mid = fs0 + args.frames // 2

    def run_cfg(name, fn, *a):
        gc.collect()
        time.sleep(1.5)
        print("\n>>> %s ..." % name, flush=True)
        with Sampler() as smp:
            r = fn(*a)
        RES[name] = r
        s = smp.summary()
        RES[name]["_sys"] = s
        dump = {"results": dict(RES), "meta": {
            "video": CUR["video"], "frames": CUR["frames"]}}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "_probe_latest.json").write_text(
            json.dumps(dump, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("    wall=%ss cores_active=%s gpu=%s%% mem=%sMB" % (
            r["wall_s"], s.get("avg_active_cores"),
            s.get("gpu_util_mean"), s.get("sys_mem_max_mb")))

    print("warmup ...", flush=True)
    solo_run(args.video, "auto", "auto", fs0, fs0 + 600)
    solo_run(args.video, "cpu", "cpu", fs0, fs0 + 600)

    run_cfg("A1a_solo_gpu", solo_run, args.video, "auto", "auto", fs0, f_end)
    ref_gpu = json.loads(json.dumps(RES["A1a_solo_gpu"]))
    run_cfg("A1b_solo_gpu_rep", solo_run, args.video, "auto", "auto",
            fs0, f_end)
    run_cfg("A1c_solo_gpu_rep", solo_run, args.video, "auto", "auto",
            fs0, f_end)
    run_cfg("A2_solo_cpu", solo_run, args.video, "cpu", "cpu", fs0, f_end)
    ref_cpu = json.loads(json.dumps(RES["A2_solo_cpu"]))
    m5 = METAS["test5"]
    run_cfg("A3_solo_cpu_test5", solo_run, "test5", "cpu", "cpu",
            m5[1], m5[1] + args.frames)

    run_cfg("B1c_dual_nofb", engine_dual_run, args.video, None,
            "nofb", True)

    old_inf = os.environ.pop(eco.DUAL_PIPELINE_INFLIGHT_ENV, None)
    old_ratio = os.environ.pop(eco.DUAL_PIPELINE_SLOW_RATIO_ENV, None)
    os.environ[eco.DUAL_PIPELINE_INFLIGHT_ENV] = "0"
    os.environ[eco.DUAL_PIPELINE_SLOW_RATIO_ENV] = "0"
    run_cfg("B2c_dual_nofb_nogate", engine_dual_run, args.video, None,
            "nofb_nogate", True)
    os.environ.pop(eco.DUAL_PIPELINE_INFLIGHT_ENV, None)
    os.environ.pop(eco.DUAL_PIPELINE_SLOW_RATIO_ENV, None)
    if old_inf is not None:
        os.environ[eco.DUAL_PIPELINE_INFLIGHT_ENV] = old_inf
    if old_ratio is not None:
        os.environ[eco.DUAL_PIPELINE_SLOW_RATIO_ENV] = old_ratio

    run_cfg("B3_dual_gpu_gpu", engine_dual_run, args.video,
            [("auto", "auto"), ("auto", "auto")])
    run_cfg("B3r_gpu_gpu_rep", engine_dual_run, args.video,
            [("auto", "auto"), ("auto", "auto")])

    def pa():
        pr = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child",
             "--mode", "decode", "--video", args.video,
             "--dec", "cpu", "--ocr", "cpu",
             "--f0", str(mid), "--f1", str(mid + args.frames),
             "--out", str(OUT_DIR / "_peer_decode.json"),
             "--secs", "45"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3.0)
        try:
            r = solo_run(args.video, "auto", "auto", fs0, f_end)
        finally:
            pr.terminate()
            pr.wait(15)
        r["mode"] = "gpu_solo_vs_dav1d_peer"
        return r
    run_cfg("PA_gpu_vs_dav1d_peer", pa)

    run_cfg("C1_inproc_static", inproc_static_halves, args.video, mid, f_end)

    def d1():
        wall, kids = two_proc([
            {"video": args.video, "dec": "auto", "ocr": "auto",
             "f0": fs0, "f1": mid},
            {"video": args.video, "dec": "cpu", "ocr": "cpu",
             "f0": mid, "f1": f_end}])
        return {"wall_s": round(wall, 3), "mode": "proc_halves",
                "pipes": {"pipe1": {"dec": "auto(test6)",
                                    "wall_s": kids[0]["wall_s"],
                                    "n_frames": kids[0].get("n_frames"),
                                    "profile": kids[0]["profile"]},
                          "pipe2": {"dec": "cpu(test6)",
                                    "wall_s": kids[1]["wall_s"],
                                    "n_frames": kids[1].get("n_frames"),
                                    "profile": kids[1]["profile"]}}}
    run_cfg("D1_proc_same_video", d1)

    def d2():
        m5 = METAS["test5"]
        wall, kids = two_proc([
            {"video": args.video, "dec": "auto", "ocr": "auto",
             "f0": fs0, "f1": mid},
            {"video": "test5", "dec": "cpu", "ocr": "cpu",
             "f0": m5[1], "f1": m5[1] + args.frames}])
        return {"wall_s": round(wall, 3), "mode": "proc_diff",
                "pipes": {"pipe1": {"dec": "auto(test6)",
                                    "wall_s": kids[0]["wall_s"],
                                    "n_frames": kids[0].get("n_frames"),
                                    "profile": kids[0]["profile"]},
                          "pipe2": {"dec": "cpu(test5)",
                                    "wall_s": kids[1]["wall_s"],
                                    "n_frames": kids[1].get("n_frames"),
                                    "profile": kids[1]["profile"]}}}
    run_cfg("D2_proc_diff_video", d2)
    run_cfg("D2r_diff_video_rep", d2)

    run_cfg("B1c_rep_dual_nofb", engine_dual_run, args.video, None,
            "nofb_rep", True)

    print("\n=== walls ===")
    a1w = RES["A1a_solo_gpu"]["wall_s"]
    for name, d in RES.items():
        extra = ""
        if d.get("pipes"):
            parts = []
            for tag in sorted(d["pipes"]):
                pd = d["pipes"][tag]
                w = pd.get("wall_s") or pd.get("busy_wall") or 0
                ch = pd.get("chunks")
                parts.append("%s=%.2fs%s" % (
                    tag, w, ("/%d" % ch) if ch else ""))
            extra = "  [" + ", ".join(parts) + "]"
        print("%-24s %8.2fs  agg=%4.0ffps  vs_A1=%.2fx%s" % (
            name, d["wall_s"], args.frames / max(d["wall_s"], 1e-6),
            d["wall_s"] / a1w, extra))

    print_analysis(ref_gpu, ref_cpu)

    dump = {"results": dict(RES), "meta": {
        "video": args.video, "frames": args.frames}}
    outp = OUT_DIR / ("probe_%s_%d.json" % (args.video, int(time.time())))
    outp.write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nsaved -> %s" % outp)


if __name__ == "__main__":
    main()
