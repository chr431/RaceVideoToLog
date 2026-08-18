"""实验：onnxruntime 1.29.0 新增参数扫描（纯 infer 吞吐，段/s）。

1.29.0 新增（release notes #29688）：
  - ORT_INTRA_OP_NUM_THREADS / ORT_INTER_OP_NUM_THREADS 环境变量可设默认
    线程池大小（显式 SessionOptions 设置仍优先，0 保留机器默认）。
1.29.0 其它 CPU 侧：MLAS 小 MatMul 批量分配减少（#29085/#29690）、
  常量折叠确定性化（#29617）、初始化路径优化（#29880/#31964）。

本脚本实验维度：
  - intra：None(env 生效) / 4 / 8 / 16
  - inter：None(env 生效) / 1 / 2 / 8 / 16
  - parallel：False(顺序执行) / True(ORT_PARALLEL 并行节点)
  - spin_off：False / True（session.*.allow_spinning=0）
  - 单实例 vs 双实例（4+4 / 8+8，模拟生产 dual-ONNX）

比较基准（1.28.0 实测）：单16=346、单8=238、双8+8=401、双4+4=267 段/s。
"""
from __future__ import annotations
import os
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
import numpy as np  # noqa: E402
import config  # noqa: E402
from ocr_native import _models_dir  # noqa: E402
from tools.detect_eval import load_meta  # noqa: E402
from video_utils import _preprocess_standard  # noqa: E402

MODEL = _models_dir() / f"PP-OCRv6_rec_{config.DEFAULT_OCR_MODEL.replace('v6_', '')}.onnx"


def make_batch(v="test5", n=100, w_pad=config.DEFAULT_FILL_WIDTH):
    """真实预处理段 + 统一 pad 到 w_pad（与生产 __call__ 同形状语义）。"""
    from decord import VideoReader, cpu
    roi, f_start, f_end, fps, ms, ma, mw, truth = load_meta(v)
    vr = VideoReader(f"D:/Videos/racelog_test/{v}.mp4", ctx=cpu(0),
                     output_format="gray",
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1),
                     num_threads=4)
    g = vr.get_batch(list(range(f_start, f_start + 600)),
                     roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)
                     ).asnumpy()[..., 0]
    sharp = g.std(axis=(1, 2))
    reps = np.argsort(sharp)[-n:]
    h = config.OCR_TARGET_H  # 48
    out = np.zeros((n, 3, h, w_pad), dtype=np.float32)
    for k, idx in enumerate(reps):
        im = _preprocess_standard(g[idx][..., None], force_aspect=mw)
        imw = min(im.shape[1], w_pad)
        img = im[:, :imw]
        img = img.transpose((2, 0, 1)) / 255.0
        img = (img - 0.5) / 0.5
        out[k, :, :, :imw] = img
    return out


def make_session(intra=None, inter=None, parallel=False, spin_off=False,
                 env_intra=None, env_inter=None):
    import onnxruntime as ort
    if env_intra is not None:
        os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(env_intra)
    if env_inter is not None:
        os.environ["ORT_INTER_OP_NUM_THREADS"] = str(env_inter)
    so = ort.SessionOptions()
    if intra is not None:
        so.intra_op_num_threads = intra
    if inter is not None:
        so.inter_op_num_threads = inter
    if parallel:
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    if spin_off:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    sess = ort.InferenceSession(str(MODEL), sess_options=so,
                                providers=["CPUExecutionProvider"])
    return sess


def run_rounds(sess, batch, rounds=30):
    for _ in range(3):
        sess.run(None, {"x": batch})
    t0 = time.perf_counter()
    for _ in range(rounds):
        sess.run(None, {"x": batch})
    return rounds * len(batch) / (time.perf_counter() - t0)


def single(label, batch, **kw):
    sess = make_session(**kw)
    rate = run_rounds(sess, batch)
    print(f"  {label:<46} {rate:6.0f} 段/s")
    return rate


def dual(label, batch, **kw):
    """两个独立实例各处理半批（生产 dual-ONNX 模式）。"""
    def worker(sess, chunk, rounds):
        for _ in range(rounds):
            sess.run(None, {"x": chunk})
    engs = [make_session(**kw) for _ in range(2)]
    half = len(batch) // 2
    chunks = [batch[:half], batch[half:]]
    for sess, ch in zip(engs, chunks):
        worker(sess, ch, 3)
    rounds = 30
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(s, c, rounds))
          for s, c in zip(engs, chunks)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rate = rounds * len(batch) / (time.perf_counter() - t0)
    print(f"  {label:<46} {rate:6.0f} 段/s")
    return rate


if __name__ == "__main__":
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}  @ {MODEL.name}  "
          f"批={100} 段, 30 轮\n")
    batch = make_batch()

    print("== 单实例 ==")
    single("intra=16 inter=2（生产现状等价）", batch, intra=16, inter=2)
    single("intra=16 inter=2 + spin off", batch, intra=16, inter=2, spin_off=True)
    single("intra=16 inter=8", batch, intra=16, inter=8)
    single("intra=16 inter=16", batch, intra=16, inter=16)
    single("intra=16 inter=16 + parallel", batch, intra=16, inter=16, parallel=True)
    single("env INTRA=16（不显式设）", batch, env_intra=16)
    single("env INTRA=16 INTER=16", batch, env_intra=16, env_inter=16)
    single("env INTRA=16 INTER=16 + parallel", batch,
           env_intra=16, env_inter=16, parallel=True)

    print("\n== 双实例（dual-ONNX 模式）==")
    dual("8+8 线程", batch, intra=8)
    dual("8+8 线程 + spin off", batch, intra=8, spin_off=True)
    dual("4+4 线程", batch, intra=4)
    dual("8+8 线程 inter=2", batch, intra=8, inter=2)
    dual("env INTRA=8 双实例", batch, env_intra=8)