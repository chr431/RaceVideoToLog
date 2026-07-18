"""GPU 加速配置 — CUDA/cuDNN DLL 加载 + ONNX 后端选择。

从 ocr_engine.py 提取，消除 import 时的副作用（原代码在模块导入时
即扫描文件系统和加载 DLL），改为延迟初始化。
"""
from __future__ import annotations
import logging
import os as _os

import config

logger = logging.getLogger("RaceVideoToLog.gpu_setup")

# ═══════════════════ 内部状态 ═══════════════════
_gpu_initialized: bool = False
_gpu_backend: str = "CPU"
_gpu_patched: bool = False

# 后端优先级：用户选择 → 回退链
_BACKEND_FALLBACK: dict[str, list[str]] = {
	"auto": ["CUDA", "CPU"],
	"cuda": ["CUDA", "CPU"],
	"cpu":  ["CPU"],
}

_BACKEND_PROVIDER_MAP: dict[str, tuple[str, dict]] = {
	"CUDA": ("CUDAExecutionProvider", {
		"device_id": 0,
		"arena_extend_strategy": "kNextPowerOfTwo",
		"cudnn_conv_algo_search": "EXHAUSTIVE",
		"do_copy_in_default_stream": True,
	}),
	"CPU": ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
}

# CUDA 版本扫描列表（按优先级降序）
_CUDA_VERSIONS: list[str] = [
	"v13.3", "v13.2", "v13.1", "v13.0",
	"v12.9", "v12.8", "v12.7", "v12.6", "v12.5", "v12.4",
	"v12.3", "v12.2", "v12.1", "v12.0",
	"v11.8", "v11.7", "v11.6",
]

_CUDA_DLL_PREFIXES: tuple[str, ...] = (
	"cublas", "cufft", "curand", "cusparse", "cusolver",
	"npp", "nvjpeg", "nvrtc", "nvblas", "nvjitlink", "zlibwapi",
)


def get_gpu_backend() -> str:
	"""返回当前实际使用的 GPU 后端名称（CUDA 或 CPU）。"""
	return _gpu_backend


def _ensure_gpu_initialized() -> None:
	"""延迟初始化 GPU：首次调用时扫描并加载 CUDA/cuDNN DLL。"""
	global _gpu_initialized
	if not _gpu_initialized:
		_gpu_initialized = True
		_register_gpu_dlls()


def _register_gpu_dlls() -> None:
	"""将 CUDA 和 cuDNN DLL 按依赖顺序预加载到进程内存。

	扫描 C:\\Program Files\\NVIDIA 下的 CUDA Toolkit 和 cuDNN 安装。
	加载失败不抛异常，记录日志后继续（CPU 回退）。
	"""
	import ctypes as _ct

	_cuda_bin: str | None = None
	_cudnn_dir: str | None = None
	_cudnn_dlls: list[str] = []

	# ── 1. 定位 CUDA Toolkit bin 目录 ──
	_cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
	for _ver in _CUDA_VERSIONS:
		_cb = _os.path.join(_cuda_base, _ver, "bin")
		if _os.path.isdir(_cb):
			_cuda_bin = _cb
			break
	if not _cuda_bin:
		for _env in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
			_val = _os.environ.get(_env, "")
			if _val:
				_cb = _os.path.join(_val, "bin")
				if _os.path.isdir(_cb):
					_cuda_bin = _cb
					break

	# ── 2. 定位 cuDNN DLL（匹配 CUDA 版本）──
	_cuda_major = ""
	if _cuda_bin:
		import re as _re
		_m = _re.search(r"v(\d+\.\d+)", _cuda_bin.replace("\\", "/"))
		if _m:
			_cuda_major = _m.group(1)

	_cudnn_base = r"C:\Program Files\NVIDIA\CUDNN"
	if _os.path.isdir(_cudnn_base):
		_candidates: list[tuple[str, str]] = []
		for _root, _dirs, _files in _os.walk(_cudnn_base):
			for _f in _files:
				if _f.lower().startswith("cudnn") and _f.endswith(".dll"):
					_candidates.append((_root, _f))
		if _cuda_major and _candidates:
			_matched = [(r, f) for r, f in _candidates
			            if _cuda_major in r.replace("\\", "/")]
			if _matched:
				_candidates = _matched
		for _root, _f in _candidates:
			if _cudnn_dir is None:
				_cudnn_dir = _root
			_cudnn_dlls.append(_os.path.join(_root, _f))

	# ── 3. 按依赖顺序预加载 DLL ──
	_loaded = 0
	_failed: list[str] = []

	def _load_dll(_path: str) -> bool:
		nonlocal _loaded
		try:
			_ct.CDLL(_path)
			_loaded += 1
			return True
		except OSError as _e:
			_failed.append(f"{_os.path.basename(_path)}: {_e}")
			return False
		except Exception:
			return False

	if _cuda_bin:
		for _prefix in ("cudart64_", "cudart32_"):
			for _f in _os.listdir(_cuda_bin):
				if _f.lower().startswith(_prefix) and _f.endswith(".dll"):
					_load_dll(_os.path.join(_cuda_bin, _f))
		for _f in sorted(_os.listdir(_cuda_bin)):
			_fl = _f.lower()
			if _fl.endswith(".dll") and not _fl.startswith("cudart"):
				if any(_fl.startswith(p) for p in _CUDA_DLL_PREFIXES):
					_load_dll(_os.path.join(_cuda_bin, _f))

	for _dll_path in _cudnn_dlls:
		_load_dll(_dll_path)

	# ── 4. 更新 PATH ──
	_path_extra: list[str] = []
	if _cuda_bin:
		_path_extra.append(_cuda_bin)
	if _cudnn_dir:
		_path_extra.append(_cudnn_dir)
	if _path_extra:
		_existing = _os.environ.get("PATH", "")
		_os.environ["PATH"] = ";".join(_path_extra) + \
			(";" + _existing if _existing else "")

	if _cuda_bin:
		logger.info("CUDA: %s", _cuda_bin)
	else:
		logger.info("CUDA: 未找到 CUDA Toolkit 安装")
	if _cudnn_dlls:
		logger.info("cuDNN: %d 个 DLL 在 %s", len(_cudnn_dlls), _cudnn_dir)
	else:
		logger.info("cuDNN: 未找到")
	logger.info("GPU DLL 预加载: %d 个成功", _loaded)
	if _failed:
		logger.warning("GPU DLL 预加载失败 (%d 个): %s", len(_failed),
		               "; ".join(_failed[:5]))


def select_backend(preferred: str = "auto") -> str:
	"""按用户偏好选择 OCR 后端，不可用时自动回退。

	首次调用时触发 GPU DLL 延迟初始化。
	preferred: "auto" | "cuda" | "cpu"
	返回实际使用的后端名称: "CUDA" | "CPU"
	"""
	global _gpu_patched, _gpu_backend

	_ensure_gpu_initialized()

	if _gpu_patched:
		return _gpu_backend
	_gpu_patched = True

	try:
		import onnxruntime as ort
	except Exception:
		logger.warning("无法导入 onnxruntime，使用 CPU 后端")
		_gpu_backend = "CPU"
		return _gpu_backend

	available = set(ort.get_available_providers())

	chain = _BACKEND_FALLBACK.get(preferred.lower(), _BACKEND_FALLBACK["auto"])
	chosen: str | None = None

	for candidate in chain:
		ep_name = _BACKEND_PROVIDER_MAP[candidate][0]
		if ep_name in available:
			chosen = candidate
			break
	if chosen is None:
		chosen = "CPU"

	# Monkey-patch OrtInferSession 以使用选定后端
	from rapidocr_onnxruntime.utils import OrtInferSession

	ep_name, ep_opts = _BACKEND_PROVIDER_MAP[chosen]
	cpu_ep_name, cpu_opts = _BACKEND_PROVIDER_MAP["CPU"]

	def _patched_init(self, config):  # type: ignore[no-untyped-def]
		from onnxruntime import (
			SessionOptions, InferenceSession, GraphOptimizationLevel,
		)
		sess_opt = SessionOptions()
		sess_opt.log_severity_level = 4
		sess_opt.enable_cpu_mem_arena = False
		sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

		EP_list: list = [(ep_name, ep_opts)] if ep_name != cpu_ep_name else []
		EP_list.append((cpu_ep_name, cpu_opts))
		self._verify_model(config['model_path'])
		self.session = InferenceSession(
			config['model_path'], sess_options=sess_opt, providers=EP_list,
		)

	OrtInferSession.__init__ = _patched_init  # type: ignore[method-assign]

	_gpu_backend = chosen
	config._gpu_backend = chosen
	logger.info("OCR 后端已选择: %s", chosen)
	return _gpu_backend


def reset_backend() -> None:
	"""重置后端选择状态，允许用户在运行时切换后端。"""
	global _gpu_patched, _gpu_backend
	_gpu_patched = False
	_gpu_backend = "CPU"
	config._gpu_backend = "CPU"
