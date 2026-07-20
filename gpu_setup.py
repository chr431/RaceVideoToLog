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
		"检测到 NVIDIA 显卡，但 PATH 中未找到 TensorRT DLL。\n"
		"将 CUDA Toolkit 和 TensorRT 的 bin 目录加入 PATH 即可启用 GPU 加速。\n"
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
	_TRT_MARKERS = ("nvinfer", "nvinfer_builder_resource")
	_CUDA_MARKERS = ("cudart64_", "cudart32_", "cublas64_")
	_CUDNN_MARKERS = ("cudnn64_", "cudnn_ops64_")

	_found_trt: list[str] = []
	_found_cuda: list[str] = []
	_found_cudnn: list[str] = []
	_other_dirs: list[str] = []

	# ── 扫描 PATH 中所有目录 ──
	_seen: set[str] = set()
	for _entry in _os.environ.get("PATH", "").split(_os.pathsep):
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

	# ── 注册 DLL 搜索目录（遵循依赖顺序：CUDA → cuDNN → TRT → 其他）──
	_registered = 0
	for _label, _dirs in [("CUDA", _found_cuda), ("cuDNN", _found_cudnn),
	                       ("TensorRT", _found_trt), ("DLL", _other_dirs)]:
		if _dirs:
			logger.info("%s: %s", _label, ", ".join(_dirs[:3]))
		for _d in _dirs:
			try:
				_os.add_dll_directory(_d)
				_registered += 1
			except (AttributeError, OSError):
				pass

	logger.info("GPU DLL 搜索路径注册: %d 个目录", _registered)


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
