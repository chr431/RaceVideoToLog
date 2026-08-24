"""D1 intervention: force OCR_THREADS=2 globally (TRT unaffected;
peer ONNX starved) -> if gpu-pipe infer recovers, ONNX spin-burn is the
poison. Single-run, warm."""
import importlib.util
import json
import os
import sys

os.environ["OCR_THREADS"] = "2"

spec = importlib.util.spec_from_file_location(
    "probe", r"D:/Repo/RaceVideoToLog/tools/archive/_dual_phase_probe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.OUT_DIR.mkdir(parents=True, exist_ok=True)
m.METAS["test6"] = m.metup(m.load_meta("test6"), "test6")
m.METAS["test5"] = m.metup(m.load_meta("test5"), "test5")

m.solo_run("test6", "auto", "auto", 139, 739)   # warm

with m.Sampler() as s:
    wall, kids = m.two_proc([
        {"video": "test6", "dec": "auto", "ocr": "auto",
         "f0": 139, "f1": 1639},
        {"video": "test6", "dec": "cpu", "ocr": "cpu",
         "f0": 1639, "f1": 3139}])
print("D1-throttled wall =", round(wall, 3), " sys:", s.summary())
for k in kids:
    prof = k["profile"]
    inf = (prof.get("ocr") or {}).get("infer")
    dec = (prof.get("producer") or {}).get("decode_batch")
    nf = k.get("n_frames") or 1
    print(" ", k["dec"], "wall=%s frames=%s infer_s=%s decode=%.2fms/f" % (
        k["wall_s"], k.get("n_frames"),
        round(inf, 3) if inf else None,
        dec * 1000 / nf if dec else -1))
