"""GPU 加速配置 — 后端选择 + DLL 搜索路径注册。

从 PATH 扫描 CUDA / TensorRT 目录，注册到 Windows DLL 搜索路径，
并提供自动后端选择（TensorRT → CPU 回退链）。
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
_dll_dir_cookies: list = []  # 保持 os.add_dll_directory() 返回值存活

# 后端优先级：用户选择 → 回退链
_BACKEND_FALLBACK: dict[str, list[str]] = {
	"auto": ["TensorRT", "CPU"],
	"tensorrt": ["TensorRT", "CPU"],
	"cuda": ["CUDA", "CPU"],
	"cpu":  ["CPU"],
}

def get_gpu_backend() -> str:
	"""返回当前实际使用的 GPU 后端名称（TensorRT / CUDA / CPU）。"""
	return _gpu_backend


def get_engine_params() -> dict:
	"""返回用于 RapidOCR params 的引擎配置片段。"""
	return dict(_gpu_params)

def get_engine_type() -> str:
	"""返回引擎类型字符串 ('tensorrt' | 'onnxruntime')。"""
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
		"检测到 NVIDIA 显卡，但 PATH 中未找到 TensorRT DLL。\n"
		"将 CUDA Toolkit 12.x 和 TensorRT 10.x 的 bin 目录加入 PATH 即可。\n"
		"详见 README 的「GPU 加速配置」章节。"
	)


def _ensure_gpu_initialized() -> None:
	"""延迟初始化 GPU：首次调用时扫描并加载 CUDA/cuDNN DLL。"""
	global _gpu_initialized
	if not _gpu_initialized:
		_gpu_initialized = True
		_register_gpu_dlls()


def _register_gpu_dlls() -> None:
	"""扫描 PATH 中的 CUDA / cuDNN / TensorRT DLL 目录并注册到搜索路径。

	用户只需将对应 bin 目录加入 PATH，无需特定安装位置。
	例如：C:\\Program Files\\NVIDIA\\TensorRT-10.x\\bin
	"""
	# DLL 特征文件名（用于识别目录类型）
	_TRT_MARKERS = ("nvinfer",)
	_CUDA_MARKERS = ("cudart64_", "cudart32_", "cublas64_")
	_CUDNN_MARKERS = ("cudnn64_", "cudnn_ops64_")

	_found_trt: list[str] = []
	_found_cuda: list[str] = []
	_found_cudnn: list[str] = []
	_other_dirs: list[str] = []

	# ── 扫描 PATH 中所有目录 ──
	# Windows: os.environ 只在进程启动时读取合并后的 PATH。
	# 注册表修改后未注销重登的会话中，需直接读注册表补充。
	_path_raw = _os.environ.get("PATH", "")
	if _os.name == "nt":
		try:
			import winreg as _wr
			_reg_paths: list[str] = []
			for _hive, _subkey in [(_wr.HKEY_CURRENT_USER, "Environment"),
			                        (_wr.HKEY_LOCAL_MACHINE,
			                         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]:
				try:
					_k = _wr.OpenKey(_hive, _subkey, 0, _wr.KEY_READ)
					_val, _ = _wr.QueryValueEx(_k, "Path")
					_wr.CloseKey(_k)
					_reg_paths.extend(_val.split(";"))
				except OSError:
					pass
			# 追加注册表中独有的条目（去重）
			_env_set = {_os.path.normpath(p) for p in _path_raw.split(_os.pathsep) if p.strip()}
			_reg_extra = [p for p in _reg_paths
			              if p.strip() and _os.path.normpath(p) not in _env_set]
			if _reg_extra:
				_path_raw += _os.pathsep + _os.pathsep.join(_reg_extra)
				logger.info("PATH 补充注册表条目: %d 个", len(_reg_extra))
		except Exception:
			pass

	_seen: set[str] = set()
	_path_entries = _path_raw.split(_os.pathsep)
	# 截断过长的 PATH（仅日志用）
	_path_preview = _path_raw[:500] + ("..." if len(_path_raw) > 500 else "")
	logger.info("PATH 扫描: %d 个条目, 前500字符: %s", len(_path_entries), _path_preview)
	for _entry in _path_entries:
		_entry = _os.path.normpath(_entry.strip())
		if not _entry or _entry in _seen:
			continue
		_seen.add(_entry)
		if not _os.path.isdir(_entry):
			continue

		# 检查目录中的 DLL 类型
		try:
			_contents = _os.listdir(_entry)
		except OSError:
			continue

		_lower_contents = [f.lower() for f in _contents]
		if any(f.startswith(m) for f in _lower_contents for m in _TRT_MARKERS):
			_found_trt.append(_entry)
		elif any(f.startswith(m) for f in _lower_contents for m in _CUDNN_MARKERS):
			_found_cudnn.append(_entry)
		elif any(f.startswith(m) for f in _lower_contents for m in _CUDA_MARKERS):
			_found_cuda.append(_entry)
		elif any(f.endswith(".dll") for f in _lower_contents):
			_other_dirs.append(_entry)

	# ── 注册 DLL 搜索目录 ──
	# os.add_dll_directory() 用于 Windows 原生 LoadLibrary，
	# 但 tensorrt 包的 find_lib() 只搜 PATH，所以也要更新 os.environ。
	global _dll_dir_cookies
	_dll_dir_cookies.clear()
	_registered = 0
	_path_new: list[str] = []
	for _label, _dirs in [("CUDA", _found_cuda), ("cuDNN", _found_cudnn),
	                       ("TensorRT", _found_trt), ("DLL", _other_dirs)]:
		if _dirs:
			logger.info("%s: %s", _label, ", ".join(_dirs[:3]))
		for _d in _dirs:
			try:
				_dll_dir_cookies.append(_os.add_dll_directory(_d))
				_path_new.append(_d)
				_registered += 1
			except (AttributeError, OSError):
				pass

	# 将找到的目录前置到 PATH（tensorrt 包依赖此路径）
	if _path_new:
		_existing = _os.environ.get("PATH", "")
		_os.environ["PATH"] = _os.pathsep.join(_path_new) + \
			(_os.pathsep + _existing if _existing else "")

	if not _found_trt:
		logger.info("TensorRT DLL 未在 PATH 中找到 (搜索了 %d 个目录)", len(_path_entries))
	logger.info("GPU DLL 搜索路径注册: %d 个目录 (TRT:%d CUDA:%d)",
		_registered, len(_found_trt), len(_found_cuda))


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
			except Exception as _trt_err:
				logger.info("TensorRT import failed (%s), falling back", _trt_err)
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
