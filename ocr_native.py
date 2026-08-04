"""原生 OCR 识别引擎 — 绕过 rapidocr，直接使用 ONNX Runtime / TensorRT。

与 rapidocr 的 TextRecognizer 输出逐字节对齐（预处理/CTC 后处理复刻）：
- 预处理: resize 到 48 高 + (x/255 - 0.5) / 0.5 归一化 + pad 到 batch 最大宽
- 推理:   ONNX (onnxruntime, 动态 batch) / TensorRT (.engine, batch <= profile 上限)
- 后处理: argmax(axis=2) + max(axis=2) + CTC 去重 + blank(0) 过滤 + 字符映射
- 字符表: rapidocr models/ppocrv6_dict.txt（6904 字符 + 末尾空格 + 开头 blank = 6906）

输出对象兼容 extract_speed_value（.txts / .scores）。
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


def _models_dir() -> Path:
    """模型资产目录（源码: 项目 assets/ocr_models；frozen: _internal/ocr_models）。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "ocr_models"
    return Path(__file__).resolve().parent / "assets" / "ocr_models"


class RecOut:
    """兼容 extract_speed_value 的输出对象（txts/scores）。"""

    __slots__ = ("txts", "scores")

    def __init__(self, txt: str, score: float) -> None:
        self.txts = (txt,)
        self.scores = [float(score)]


class OcrEngine:
    """PP-OCRv6 rec 原生引擎（ONNX / TensorRT 双后端）。

    Args:
        variant: "v6_tiny" | "v6_small"
        engine_type: "onnxruntime" | "tensorrt"
    """

    def __init__(self, variant: str = "v6_tiny",
                 engine_type: str = "onnxruntime",
                 progress_cb: "object | None" = None) -> None:
        """progress_cb: 构建引擎等耗时阶段的进度消息回调 (str)。"""
        self._variant = variant
        self._progress_cb = progress_cb
        size = variant.replace("v6_", "")
        models = _models_dir()

        # ── 字符表（与 rapidocr CTCLabelDecode.get_character 一致）──
        dict_name = "ppocrv6_tiny_dict.txt" if size == "tiny" else "ppocrv6_dict.txt"
        with open(models / dict_name, "rb") as f:
            chars = [ln.decode("utf-8").strip("\n").strip("\r\n")
                     for ln in f.readlines()]
        chars.append(" ")          # 末尾插入空格
        chars.insert(0, "blank")   # 开头插入 blank（CTC 空白，索引 0）
        self._chars = chars

        # ── 模型 ──
        if engine_type == "tensorrt":
            self._init_trt(models, size)
        else:
            self._init_onnx(models, size)

    # ═══════════════ 后端初始化 ═══════════════

    def _init_onnx(self, models: Path, size: str) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(models / f"PP-OCRv6_rec_{size}.onnx"),
            providers=["CPUExecutionProvider"])
        self._trt = False
        self._max_batch = None  # ONNX 动态 batch，不分片

    @staticmethod
    def _engine_candidates(size: str) -> list[Path]:
        """engine 查找顺序：模型目录（本机构建）→ 用户缓存（自动构建产物）。"""
        name = f"multi_PP-OCRv6_rec_{size}_sm89_fp32_tf32unset.engine"
        cands = [_models_dir() / "models" / name]
        if os.name == "nt":
            cache = Path(os.environ.get("LOCALAPPDATA",
                                        str(Path.home()))) / "RaceVideoToLog" / "ocr_engines"
        else:
            cache = Path.home() / ".cache" / "racevideotolog" / "ocr_engines"
        cands.append(cache / name)
        return cands

    def _init_trt(self, models: Path, size: str) -> None:
        """加载或构建 TRT 引擎；任何失败回退 ONNX。

        - engine 不存在 → 本地自动构建（首次运行，几分钟）并缓存到用户目录
        - 引擎与 GPU 架构绑定（如 sm89 = RTX 40 系）—— 架构不匹配时
          deserialize 失败 → 回退 ONNX 并提示
        """
        import logging
        log = logging.getLogger(__name__)
        engine_path: Path | None = None
        for cand in self._engine_candidates(size):
            if cand.exists():
                engine_path = cand
                break
        try:
            if engine_path is None:
                engine_path = self._engine_candidates(size)[-1]  # 缓存目录
                if self._progress_cb:
                    self._progress_cb("TensorRT 引擎不存在，开始本地构建（首次运行，约 2 分钟）...")
                log.info("TensorRT 引擎不存在，开始本地构建（首次运行，约几分钟）...")
                self._build_engine(models, size, engine_path)
                log.info("TensorRT 引擎已构建: %s", engine_path)
                if self._progress_cb:
                    self._progress_cb("TensorRT 引擎构建完成")
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
            with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:  # type: ignore[attr-defined]
                self._trt_engine = rt.deserialize_cuda_engine(f.read())
            self._trt_ctx = self._trt_engine.create_execution_context()  # type: ignore[attr-defined]
            in_name = self._trt_engine.get_tensor_name(0)
            out_name = self._trt_engine.get_tensor_name(1)
            prof_in = self._trt_engine.get_tensor_profile_shape(in_name, 0)
            prof_out = self._trt_engine.get_tensor_profile_shape(out_name, 0)
            self._trt_in_name = in_name
            self._trt_out_name = out_name
            self._max_batch = int(prof_in[2][0])  # profile 的 batch 上限（如 6）
            self._max_in_shape = tuple(int(v) for v in prof_in[2])
            self._max_out_shape = tuple(int(v) for v in prof_out[2])
            self._buffers: tuple | None = None  # (dev_in, dev_out, host_in, host_out)
            self._trt = True
        except Exception as e:
            log.warning("TensorRT 引擎不可用 (%s)，回退 ONNX 后端。", e)
            self._init_onnx(models, size)

    @staticmethod
    def _build_engine(models: Path, size: str, engine_path: Path) -> None:
        """从 ONNX 构建 TRT 引擎（复用 rapidocr 的 rec profile 配置）。"""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
        builder = trt.Builder(logger)  # type: ignore[attr-defined]
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[attr-defined]
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)  # type: ignore[attr-defined]
        onnx_path = models / f"PP-OCRv6_rec_{size}.onnx"
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError(f"ONNX 解析失败: {onnx_path}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # type: ignore[attr-defined]  # 1 GB
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name,
                          min=(1, 3, 48, 32), opt=(6, 3, 48, 320),
                          max=(6, 3, 48, 2048))
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine 构建失败")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

    # ═══════════════ 预处理（复刻 rapidocr resize_norm_img）═══════════════

    @staticmethod
    def _resize_norm(img: np.ndarray, max_wh_ratio: float,
                     height: int = 48) -> np.ndarray:
        """resize 到 48 高 + (x/255-0.5)/0.5 归一化 + pad 到 batch 最大宽。"""
        from video_utils import _np_resize
        img_width = int(height * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(height * ratio))
        resized = _np_resize(img, resized_w, height).transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        pad = np.zeros((3, height, img_width), dtype=np.float32)
        pad[:, :, :resized_w] = resized
        return pad

    # ═══════════════ 推理 ═══════════════

    def _infer(self, batch_np: np.ndarray) -> np.ndarray:
        if self._trt:
            assert self._max_batch is not None  # TRT 初始化时已设置
            outs = []
            for i in range(0, len(batch_np), self._max_batch):
                outs.append(self._trt_execute(batch_np[i:i + self._max_batch]))
            return np.concatenate(outs, axis=0)
        out = self._session.run(None, {"x": batch_np})[0]
        return np.asarray(out, dtype=np.float32)

    def _trt_execute(self, x: np.ndarray) -> np.ndarray:
        from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
        self._trt_ctx.set_input_shape(self._trt_in_name, x.shape)
        out_shape = tuple(self._trt_ctx.get_tensor_shape(self._trt_out_name))
        # 输入 buffer：max profile 形状预分配并复用
        if self._buffers is None:
            size_in = int(np.prod(self._max_in_shape)) * 4
            _, dev_in = cudart.cudaMalloc(size_in)
            host_in = np.zeros(self._max_in_shape, dtype=np.float32)
            self._buffers = (dev_in, host_in)
            self._dev_out: int | None = None
            self._out_nbytes = 0
        dev_in, host_in = self._buffers
        # 平铺拷贝（max-shape buffer 的前 x.size 个连续元素 = x 的连续内存）
        host_in.reshape(-1)[:x.size] = x.reshape(-1)
        cudart.cudaMemcpy(dev_in, host_in.ctypes.data, x.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # 输出 device buffer 按需增长复用（cudaMalloc 每次 ~ms，避免每片分配）
        out_nbytes = int(np.prod(out_shape)) * 4
        if self._dev_out is None or out_nbytes > self._out_nbytes:
            if self._dev_out is not None:
                cudart.cudaFree(self._dev_out)
            _, self._dev_out = cudart.cudaMalloc(out_nbytes)
            self._out_nbytes = out_nbytes
        dev_out = self._dev_out
        self._trt_ctx.execute_v2([dev_in, dev_out])
        host_out = np.empty(out_shape, dtype=np.float32)
        cudart.cudaMemcpy(host_out.ctypes.data, dev_out, out_nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return host_out

    # ═══════════════ 后处理（复刻 CTCLabelDecode）═══════════════

    def _ctc_decode(self, pred: np.ndarray) -> RecOut:
        """单帧 (seq, 6906) → (文本, 置信度)。

        CTC：argmax → 相邻去重 → 移除 blank(0) → 字符映射。
        置信度 = 选中帧概率均值（round 5，与 rapidocr 一致）。
        """
        idx = pred.argmax(axis=1)
        prob = pred.max(axis=1)
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0  # blank
        if keep.any():
            text = "".join(self._chars[i] for i in idx[keep])
            # 与 rapidocr 一致：每帧概率先 round(5) 再取均值，最后 round(5)
            confs = [round(float(p), 5) for p in prob[keep]]
            conf = round(float(np.mean(confs)), 5)
        else:
            text, conf = "", 0.0
        return RecOut(text, conf)

    # ═══════════════ 批处理入口 ═══════════════

    def __call__(self, img_list: list) -> list:
        """批识别：与 rapidocr text_rec 同语义，按输入顺序返回结果。"""
        if not img_list:
            return []
        heights = [im.shape[0] for im in img_list]
        h0 = heights[0]
        # 按宽度排序（rapidocr 的加速策略；结果映射回原顺序）
        order = np.argsort([im.shape[1] for im in img_list])
        # 与 rapidocr 对齐：max_wh_ratio 起点为 rec_image_shape 的 imgW/imgH
        # （v6 配置 [3, 48, 320] → 320/48），取批内最大宽高比
        max_wh = max(320.0 / 48.0,
                     *(float(im.shape[1]) / im.shape[0] for im in img_list))
        batch_np = np.stack([self._resize_norm(img_list[i], max_wh, h0)
                             for i in order])
        preds = self._infer(batch_np)
        results: list = [None] * len(img_list)
        for k, idx in enumerate(order):
            results[idx] = self._ctc_decode(preds[k])
        return results
