"""GPU 加速配置 — CUDA/cuDNN DLL 加载 + 后端选择。

提取自 ocr_engine.py，改为延迟初始化。适配 rapidocr 3.x 的 params-based 配置。
"""
from __future__ import annotations
import logging
import os as _os

import config

logger = logging.getLogger("RaceVideoToLog.gpu_setup")

# ═══════════════════ 内部状态 ═══════════════════
_gpu_initialized: bool = False
_gpu_backend: str = "CPU"
_gpu_params: dict = {}

# 后端优先级：用户选择 → 回退链
_BACKEND_FALLBACK: dict[str, list[str]] = {
	"auto": ["TensorRT", "CUDA", "CPU"],
	"cuda": ["CUDA", "CPU"],
	"tensorrt": ["TensorRT", "CUDA", "CPU"],
	"cpu":  ["CPU"],
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
	"""返回当前实际使用的 GPU 后端名称（TensorRT / CUDA / CPU）。"""
	return _gpu_backend


def get_engine_params() -> dict:
	"""返回用于 RapidOCR params 的引擎配置片段。"""
	return dict(_gpu_params)

def get_engine_type() -> str:
	"""返回引擎类型字符串 ('onnxruntime' | 'tensorrt' | 'paddle')。"""
	return "tensorrt" if _gpu_backend == "TensorRT" else "onnxruntime"


def _has_nvidia_gpu() -> bool:
	"""检测是否存在 NVIDIA GPU（通过驱动）。不依赖 CUDA Toolkit。"""
	try:
		import ctypes as _ct
		_ct.CDLL("nvcuda.dll")
		return True
	except (OSError, Exception):
		pass
	# Fallback: 检查 WMI
	try:
		import subprocess as _sp
		_result = _sp.run(
			["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
			capture_output=True, text=True, timeout=10,
			creationflags=0x08000000 if _os.name == "nt" else 0)
		return _result.returncode == 0 and _result.stdout.strip()
	except Exception:
		pass
	return False


def get_setup_advice() -> str | None:
	"""返回面向用户的 GPU 加速配置建议。None 表示已在使用 GPU。"""
	if _gpu_backend != "CPU":
		return None
	if not _has_nvidia_gpu():
		return "未检测到 NVIDIA 显卡，OCR 将使用 CPU 推理。"
	return (
		"检测到 NVIDIA 显卡，但未安装 CUDA Toolkit 或 cuDNN，"
		"OCR 回退至 CPU 推理。\n"
		"安装 CUDA Toolkit 12.x + cuDNN 9.x 可启用 GPU 加速（约 4-6x 提速）。\n"
		"详见 README 的「GPU 加速配置」章节。"
	)


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

	# ── 1. 定位 CUDA Toolkit bin 目录（含 x64 子目录，CUDA 13+ 开始使用）──
	_cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
	for _ver in _CUDA_VERSIONS:
		for _sub in ("bin", r"bin\x64"):
			_cb = _os.path.join(_cuda_base, _ver, _sub)
			if _os.path.isdir(_cb) and any(f.endswith(".dll") for f in _os.listdir(_cb)):
				_cuda_bin = _cb
				break
		if _cuda_bin:
			break
	if not _cuda_bin:
		for _env in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
			_val = _os.environ.get(_env, "")
			if _val:
				for _sub in ("bin", r"bin\x64"):
					_cb = _os.path.join(_val, _sub)
					if _os.path.isdir(_cb) and any(f.endswith(".dll") for f in _os.listdir(_cb)):
						_cuda_bin = _cb
						break
			if _cuda_bin:
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

	# ── 2b. 定位 TensorRT 安装目录（扫描任意版本）──
	_trt_base: str | None = None
	_trt_bin: str | None = None
	_trt_lib: str | None = None
	_trt_search = r"C:\Program Files\NVIDIA"
	if _os.path.isdir(_trt_search):
		for _entry in _os.listdir(_trt_search):
			if _entry.lower().startswith("tensorrt-"):
				_candidate_bin = _os.path.join(_trt_search, _entry, "bin")
				if _os.path.isdir(_candidate_bin):
					_trt_base = _os.path.join(_trt_search, _entry)
					_trt_bin = _candidate_bin
					_trt_lib = _os.path.join(_trt_base, "lib")
					break

	# ── 4. 注册 DLL 搜索目录（Windows 8.1+ 推荐方式）──
	if _cuda_bin:
		try:
			_os.add_dll_directory(_cuda_bin)
		except AttributeError:
			pass  # Python <3.8
	# 也加入 CUDA 12.x bin（ORT 1.27 需要 CUDA 12 DLL）
	for _ver in ("v12.9", "v12.8", "v12.7", "v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0"):
		_cb12 = _os.path.join(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", _ver, "bin")
		if _os.path.isdir(_cb12) and _cb12 != _cuda_bin:
			try:
				_os.add_dll_directory(_cb12)
			except AttributeError:
				pass
			break
	if _cudnn_dir:
		try:
			_os.add_dll_directory(_cudnn_dir)
		except AttributeError:
			pass
	# 同时更新 PATH（兼容旧版加载逻辑）
	_path_extra: list[str] = []
	if _trt_bin:
		try:
			_os.add_dll_directory(_trt_bin)
		except AttributeError:
			pass
		_path_extra.append(_trt_bin)
	if _trt_lib:
		try:
			_os.add_dll_directory(_trt_lib)
		except AttributeError:
			pass
		_path_extra.append(_trt_lib)
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
	if _trt_bin:
		logger.info("TensorRT: %s", _trt_base)
	logger.info("GPU DLL 预加载: %d 个成功", _loaded)
	if _failed:
		logger.warning("GPU DLL 预加载失败 (%d 个): %s", len(_failed),
		               "; ".join(_failed[:5]))


def select_backend(preferred: str = "auto") -> str:
	"""按用户偏好选择 OCR 后端，不可用时自动回退。

	首次调用时触发 GPU DLL 延迟初始化。返回后端名称字符串。
	同时设置 _gpu_params 供 RapidOCR 引擎配置。
	"""
	global _gpu_backend, _gpu_params

	_ensure_gpu_initialized()

	chain = _BACKEND_FALLBACK.get(preferred.lower(), _BACKEND_FALLBACK["auto"])
	chosen: str | None = None

	for candidate in chain:
		if candidate == "TensorRT":
			try:
				import tensorrt as _trt_check  # noqa: F401
				chosen = "TensorRT"
				break
			except Exception:
				continue
		elif candidate == "CUDA":
			try:
				import onnxruntime as ort
				if "CUDAExecutionProvider" in set(ort.get_available_providers()):
					chosen = "CUDA"
					break
			except Exception:
				continue
		else:
			chosen = "CPU"
			break
	if chosen is None:
		chosen = "CPU"

	if chosen == "TensorRT":
		_gpu_params = {
			"EngineConfig.tensorrt.device_id": 0,
			"EngineConfig.tensorrt.use_fp16": True,
			"EngineConfig.tensorrt.workspace_size": 1073741824,  # 1 GB
		}
	elif chosen == "CUDA":
		_gpu_params = {
			"EngineConfig.onnxruntime.use_cuda": True,
			"EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
			"EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kNextPowerOfTwo",
			"EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
			"EngineConfig.onnxruntime.cuda_ep_cfg.do_copy_in_default_stream": True,
		}
	else:
		_gpu_params = {"EngineConfig.onnxruntime.use_cuda": False}

	_gpu_backend = chosen
	config._gpu_backend = chosen
	logger.info("OCR 后端已选择: %s", chosen)
	return _gpu_backend


def reset_backend() -> None:
	"""重置后端选择状态，允许用户在运行时切换后端。"""
	global _gpu_backend, _gpu_params
	_gpu_backend = "CPU"
	_gpu_params = {}
	config._gpu_backend = "CPU"
