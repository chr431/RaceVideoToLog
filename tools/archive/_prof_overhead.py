"""RVTOL_PROFILE 内部锁开销验证（3000 帧剖面显示 +1.1s 测量污染）。

现状 _prof_end：每次调用 time.perf_counter() + 获取全局 threading.Lock
（与 OCR 线程/生产者互斥竞争）。producer 每帧 1 次 segmentation +
每批 gray/sharp/bin，ocr 每批 infer/ctc，3000 帧共 ~1.2 万次调用。
对比：a) 现状（锁+dict 写入） b) 无锁本地累计（最后合并）。
只测纯函数开销，n 次调用耗时。
"""
from __future__ import annotations
import threading
import time


class Locked:
    def __init__(self):
        self.profile = {}
        self._lock = threading.Lock()

    def _prof_end(self, group, key, t0):
        with self._lock:
            d = self.profile.setdefault(group, {})
            d[key] = d.get(key, 0.0) + (time.perf_counter() - t0)


class Local:
    """无锁：每组本地累计（生产者/OCR 各自单线程追加，无竞争）。"""

    def __init__(self):
        self._acc = {}

    def _prof_end(self, group, key, t0):
        d = self._acc.setdefault(group, {})
        d[key] = d.get(key, 0.0) + (time.perf_counter() - t0)


def bench(cls, n_calls=50000):
    p = cls()
    t0 = time.perf_counter()
    for i in range(n_calls):
        p._prof_end("producer", "segmentation", t0 + i * 1e-9)
    return time.perf_counter() - t0


if __name__ == "__main__":
    for n in (10000, 50000):
        a = bench(Locked, n)
        b = bench(Local, n)
        print(f"calls={n:>6}: 现状锁版={a:.3f}s  无锁本地={b:.3f}s  "
              f"节省={a - b:.3f}s ({(a - b) / a * 100:.0f}%)")