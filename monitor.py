"""性能日志 + 资源监测 — 永久埋点基础设施。

两个模块级单例：
- STAGE（StageTimer）：阶段计时，永久常开，开销 ~100ns/次。这是"测试时无需再插桩"
  的核心：所有阶段的用时一直在统计，序列化进 CSV 头 / _summary.json / 控制台汇总。
- monitor（ResourceMonitor）：内存 / CPU / GPU 采样，默认开启，可通过
  --no-monitor / RVTOL_MONITOR=0 / GUI 复选框关闭。关闭时零线程、零子进程。

设计约束：
- psutil 缺失时 RSS/CPU 字段为 None（一次性警告），GPU 采样不受影响；
- 无 nvidia-smi 时自动降级到 cuda.bindings 的 cuMemGetInfo（仅显存），再失败则跳过 GPU；
- 阶段计时线程安全（threading.local 每线程栈 + 单锁累加器），各阶段不跨线程。
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("RaceVideoToLog.monitor")

# psutil 可选：缺失时 RSS/CPU 降级为 None（不阻塞其他功能）。
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

_CREATE_NO_WINDOW = 0x08000000  # nvidia-smi 子进程不闪控制台窗口


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


class _StageCtx:
    """stage() 的上下文管理器：exit 时把墙钟段累加/替换进 StageTimer。"""

    __slots__ = ("timer", "name", "accumulate", "t0", "elapsed")

    def __init__(self, timer: "StageTimer", name: str, accumulate: bool) -> None:
        self.timer = timer
        self.name = name
        self.accumulate = accumulate
        self.elapsed = 0.0
        self.t0 = time.perf_counter()

    def __enter__(self) -> "_StageCtx":
        return self

    def __exit__(self, *exc) -> bool:
        self.elapsed = time.perf_counter() - self.t0
        self.timer._exit(self.name, self.accumulate, self.elapsed)
        return False


class StageTimer:
    """阶段用时统计（永久常开）。

    多入口阶段（decode/inference/phase1）用 accumulate=True 累加；
    单入口阶段（engine_load/video_open/correction/finalize_*）默认替换
    —— GUI 二次 finalize 等重入场景不重复计数。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._acc: dict[str, float] = {}
        self._tl = threading.local()

    def reset(self) -> None:
        with self._lock:
            self._acc.clear()
        try:
            self._tl.stack.clear()
        except AttributeError:
            pass

    def stage(self, name: str, accumulate: bool = False) -> _StageCtx:
        return _StageCtx(self, name, accumulate)

    def accumulate(self, name: str, seconds: float) -> None:
        """手动两点计时（phase1、prewarm），累加。"""
        with self._lock:
            self._acc[name] = self._acc.get(name, 0.0) + seconds

    def _exit(self, name: str, accumulate: bool, elapsed: float) -> None:
        with self._lock:
            if accumulate:
                self._acc[name] = self._acc.get(name, 0.0) + elapsed
            else:
                self._acc[name] = elapsed
        # 阶段边界资源快照只对单入口阶段打（累加阶段每帧触发，会刷屏写入）。
        if not accumulate:
            monitor.stage_snapshot(name)

    def result(self) -> dict[str, float]:
        with self._lock:
            return dict(self._acc)


class ResourceMonitor:
    """内存 / CPU / GPU 后台采样。

    采样线程 daemon，间隔默认 1s。维护每指标的 last/peak/avg 与阶段边界快照。
    start()/stop() 幂等；stop() 返回冻结统计（GUI 二次 finalize 读取用）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_ev = threading.Event()
        self._interval = 1.0
        self._with_gpu = True
        self._proc = None  # psutil.Process（复用实例以计算 cpu_percent 差值）
        self._samples: list[dict] = []
        self._stats: dict | None = None
        self._stage_snapshots: dict[str, dict] = {}
        self._snap_idx: dict[str, int] = {}
        self._gpu_name = ""
        self._gpu_backend = "none"  # none | smi | cuda
        self._gpu_missing = False
        self._psutil_warned = False

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, interval_s: float = 1.0, with_gpu: bool = True) -> None:
        if self.active:
            return
        with self._lock:
            self._interval = max(0.2, float(interval_s))
            self._with_gpu = bool(with_gpu)
            self._samples = []
            self._stats = None
            self._stage_snapshots = {}
            self._snap_idx = {}
            self._stop_ev.clear()
            if psutil is not None:
                try:
                    self._proc = psutil.Process()
                    self._proc.cpu_percent(None)  # 预热差值基线，首个样本即有真实 CPU%
                except Exception:
                    self._proc = None
        self._gpu_name = self._probe_gpu_name()
        self._thread = threading.Thread(target=self._run, name="rvtol-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._sample_once()
        except Exception:
            pass
        while not self._stop_ev.wait(self._interval):
            try:
                self._sample_once()
            except Exception:
                pass

    def stop(self) -> dict | None:
        th = self._thread
        if th is None:
            return self._stats
        self._stop_ev.set()
        th.join(timeout=max(2.0, self._interval * 2 + 1.0))
        self._thread = None
        self._finalize_stats()
        return self._stats

    # ── 采样 ──────────────────────────────────────────────────────

    def _sample_once(self) -> None:
        s: dict = {"t": time.perf_counter()}
        if psutil is not None and self._proc is not None:
            try:
                s["rss_mb"] = self._proc.memory_info().rss / 1048576.0
                cpu = self._proc.cpu_percent(None)
                s["cpu_pct"] = 0.0 if cpu is None else float(cpu)
            except Exception:
                s["rss_mb"] = None
                s["cpu_pct"] = None
        else:
            if not self._psutil_warned:
                self._psutil_warned = True
                logger.warning("psutil 未安装，RSS/CPU 监测不可用（GPU 采样不受影响）")
            s["rss_mb"] = None
            s["cpu_pct"] = None
        if self._with_gpu:
            s.update(self._sample_gpu())
        else:
            s.update({"util_pct": None, "vram_mb": None, "gpu_temp_c": None})
        with self._lock:
            self._samples.append(s)

    def _probe_gpu_name(self) -> str:
        if not self._with_gpu:
            return ""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, creationflags=_CREATE_NO_WINDOW)
            name = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
            if name:
                self._gpu_backend = "smi"
            return name
        except Exception:
            return ""

    def _sample_gpu(self) -> dict:
        none_all = {"util_pct": None, "vram_mb": None, "gpu_temp_c": None}
        if self._gpu_missing:
            return none_all
        if self._gpu_backend == "none":  # 无 nvidia-smi → 尝试 cudaMemGetInfo
            self._gpu_backend = "cuda"
        if self._gpu_backend == "smi":
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, creationflags=_CREATE_NO_WINDOW)
                line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
                if not line:
                    raise ValueError("empty nvidia-smi output")
                util, used, _total, temp = (p.strip() for p in line.split(","))
                return {"util_pct": _to_float(util), "vram_mb": _to_float(used),
                        "gpu_temp_c": _to_float(temp)}
            except Exception:
                self._gpu_backend = "cuda"  # 永久降级到 cudaMemGetInfo
        try:
            from cuda.bindings import runtime as cuda_runtime
            free, total = cuda_runtime.cudaMemGetInfo()
            return {"util_pct": None, "vram_mb": (total - free) / 1048576.0, "gpu_temp_c": None}
        except Exception:
            self._gpu_missing = True
            return none_all

    # ── 统计读取 ──────────────────────────────────────────────────

    def stage_snapshot(self, name: str) -> None:
        """记录阶段边界时刻的最新采样值（每个采样周期每阶段至多一次）。"""
        if not self.active:
            return
        with self._lock:
            if not self._samples:
                return
            idx = len(self._samples) - 1
            if self._snap_idx.get(name) == idx:
                return
            last = self._samples[idx]
            self._stage_snapshots[name] = {
                "rss_mb": last.get("rss_mb"),
                "cpu_pct": last.get("cpu_pct"),
                "gpu_util_pct": last.get("util_pct"),
                "vram_used_mb": last.get("vram_mb"),
                "gpu_temp_c": last.get("gpu_temp_c"),
            }
            self._snap_idx[name] = idx

    def _compute_stats(self, samples: list[dict], snaps: dict) -> dict:
        if not samples:
            return {"samples": 0, "elapsed_s": 0.0, "gpu_name": self._gpu_name,
                    "stage_snapshots": snaps}

        def agg(key: str) -> dict:
            vals = [s[key] for s in samples if s.get(key) is not None]
            if not vals:
                return {"last": None, "peak": None, "avg": None}
            return {"last": vals[-1], "peak": max(vals), "avg": sum(vals) / len(vals)}

        return {
            "samples": len(samples),
            "elapsed_s": samples[-1]["t"] - samples[0]["t"],
            "gpu_name": self._gpu_name,
            "rss_mb": agg("rss_mb"),
            "cpu_pct": agg("cpu_pct"),
            "gpu_util_pct": agg("util_pct"),
            "vram_used_mb": agg("vram_mb"),
            "gpu_temp_c": agg("gpu_temp_c"),
            "stage_snapshots": snaps,
        }

    def _finalize_stats(self) -> None:
        with self._lock:
            samples = list(self._samples)
            snaps = dict(self._stage_snapshots)
        self._stats = self._compute_stats(samples, snaps)

    def read_snapshot(self) -> dict:
        """当前统计：运行中取实时值，已停止返回冻结统计；未开启/无样本返回 {}。"""
        if self._stats is not None:
            return self._stats
        with self._lock:
            samples = list(self._samples)
            snaps = dict(self._stage_snapshots)
        if not samples:
            return {}
        return self._compute_stats(samples, snaps)

    def peak_fields(self) -> dict:
        """峰值字段（CSV 头用）。未开启或全空时返回 {}。"""
        if self._stats is not None:
            st = self._stats
            pf = {
                "peak_rss_mb": st["rss_mb"]["peak"],
                "peak_cpu_pct": st["cpu_pct"]["peak"],
                "peak_gpu_util_pct": st["gpu_util_pct"]["peak"],
                "peak_vram_mb": st["vram_used_mb"]["peak"],
                "peak_gpu_temp_c": st["gpu_temp_c"]["peak"],
            }
        else:
            with self._lock:
                samples = list(self._samples)
            if not samples:
                return {}
            pf = {
                "peak_rss_mb": max((s["rss_mb"] for s in samples if s.get("rss_mb") is not None),
                                   default=None),
                "peak_cpu_pct": max((s["cpu_pct"] for s in samples if s.get("cpu_pct") is not None),
                                    default=None),
                "peak_gpu_util_pct": max((s["util_pct"] for s in samples if s.get("util_pct") is not None),
                                         default=None),
                "peak_vram_mb": max((s["vram_mb"] for s in samples if s.get("vram_mb") is not None),
                                    default=None),
                "peak_gpu_temp_c": max((s["gpu_temp_c"] for s in samples if s.get("gpu_temp_c") is not None),
                                       default=None),
            }
        out = {k: v for k, v in pf.items() if v is not None}
        if self._gpu_name:
            out["gpu_name"] = self._gpu_name
        return out


STAGE = StageTimer()
monitor = ResourceMonitor()


# ── 模块级委托函数（对外 API） ──────────────────────────────────────

def start(interval_s: float | None = None, with_gpu: bool | None = None) -> None:
    monitor.start(interval_s if interval_s is not None else 1.0,
                  with_gpu if with_gpu is not None else True)


def stop() -> dict | None:
    return monitor.stop()


def read_snapshot() -> dict:
    return monitor.read_snapshot()


def stage_snapshot(name: str) -> None:
    monitor.stage_snapshot(name)


def peak_fields() -> dict:
    return monitor.peak_fields()


def active() -> bool:
    return monitor.active


def _fmt_stats(stats: dict) -> str:
    def p(d: dict, key: str) -> str:
        v = d.get(key)
        return "-" if v is None else f"{v:.1f}"
    parts = [
        f"RSS peak {p(stats.get('rss_mb') or {}, 'peak')}MB",
        f"CPU peak {p(stats.get('cpu_pct') or {}, 'peak')}%",
    ]
    gpu = stats.get("gpu_util_pct") or {}
    if gpu.get("peak") is not None:
        parts.append(f"GPU util peak {p(gpu, 'peak')}%")
        vram = stats.get("vram_used_mb") or {}
        parts.append(f"VRAM peak {p(vram, 'peak')}MB")
        temp = stats.get("gpu_temp_c") or {}
        if temp.get("peak") is not None:
            parts.append(f"GPU temp peak {p(temp, 'peak')}°C")
    if stats.get("gpu_name"):
        parts.append(stats["gpu_name"])
    if stats.get("samples"):
        parts.append(f"{stats['samples']} samples")
    return " | ".join(parts)


def format_stats(stats: dict) -> str:
    """人类可读的资源汇总（无 GPU 时自动省略 GPU 字段）。"""
    return _fmt_stats(stats)


def log_run(label: str, stats: dict | None, timing: dict | None = None) -> None:
    """把一次运行写入 %LOCALAPPDATA%\\RaceVideoToLog\\monitor.log（追加，失败静默）。"""
    try:
        log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "RaceVideoToLog"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "monitor.log"
        lines = [f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} === {label}"]
        if stats:
            lines.append("  " + _fmt_stats(stats))
        if timing:
            flat = {k: v for k, v in timing.items() if isinstance(v, (int, float))}
            if flat:
                lines.append("  " + " ".join(f"{k}={v:.1f}s" for k, v in sorted(flat.items())))
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def gui_mark(mark: str) -> None:
    """GUI 时间标记（原 gui.py _t() 的实体）：追加到 gui_timing.log。"""
    try:
        log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "RaceVideoToLog"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "gui_timing.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.perf_counter():.3f} {mark}\n")
    except Exception:
        pass
