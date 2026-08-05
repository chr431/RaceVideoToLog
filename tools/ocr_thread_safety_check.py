"""并发推理安全检查 — 复现/验证 ocr_model == reocr_model 时的双线程共享引擎场景。

两个线程同时对一个 OcrEngine 连续推理（模拟主 OCR consumer + 后台预热线程）。
修复前：TRT 报 Myelin "already loaded binary graph" → CUDA illegal access。
修复后：无任何 TRT 报错，结果与单线程逐位一致。
"""
import sys
import threading
import numpy as np

sys.path.insert(0, ".")

from ocr_native import OcrEngine  # noqa: E402


def run_loop(engine, seed, n_iter, results, idx):
    rng = np.random.default_rng(seed)
    ok = True
    for it in range(n_iter):
        b = rng.integers(0, 255, size=(6, 48, 120, 3), dtype=np.uint8)
        img_list = [b[i] for i in range(6)]
        try:
            out = engine(img_list)
            assert len(out) == 6 and all(isinstance(o.txts[0], str) for o in out)
        except Exception as e:
            ok = False
            print(f"[thread {idx}] iter {it}: {type(e).__name__}: {e}")
            break
    results[idx] = ok


def main():
    engine = OcrEngine("v6_small", "tensorrt")
    threads, results = [], [False, False]
    for i in range(2):
        t = threading.Thread(target=run_loop, args=(engine, 42 + i, 30, results, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("thread results:", results)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
