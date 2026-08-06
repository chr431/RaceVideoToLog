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
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

import config


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
                 progress_cb: "Callable[[str], None] | None" = None) -> None:
        """progress_cb: 构建引擎等耗时阶段的进度消息回调 (str)。"""
        self._variant = variant
        self._progress_cb = progress_cb
        # 推理锁：主 OCR 线程 + 后台重 OCR 预热线程并发调用同一引擎。
        # 注：重 OCR 自动推导（tiny→small / small→无）后主/重引擎必不同，
        # 但保留此锁以防未来同引擎并发路径复现 Myelin 崩溃
        # （TRT IExecutionContext 非线程安全 —— execute_v2 报 "already
        # loaded binary graph" 并级联 CUDA illegal access）。
        self._lock = threading.Lock()
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
        # 线程数必须显式限制：默认（=全部逻辑核）会让 ONNX 推理占满 CPU
        # 并与解码器抢核；且 os.cpu_count() 是逻辑核（7945HX 16核32线程），
        # 逻辑核/2=16 线程对物理 16 核超配（实测 16 线程 0.823ms/帧 vs
        # 8 线程 0.598ms/帧 —— 超线程核上的线程开销）。用物理核数/2。
        so = ort.SessionOptions()
        try:
            import psutil  # type: ignore[import-not-found]
            physical = psutil.cpu_count(logical=False)
        except ImportError:
            physical = None
        if not physical:
            physical = (int(os.cpu_count() or 8) // 2)  # 假设 2 线程/核
        n = max(2, physical // 2)
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(models / f"PP-OCRv6_rec_{size}.onnx"),
            sess_options=so, providers=["CPUExecutionProvider"])
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
            self._last_in_shape: tuple | None = None
            self._out_shape: tuple | None = None
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
        """resize 到 48 高 + (x/255-0.5)/0.5 归一化 + pad 到 batch 最大宽。

        输入已是 target 高度 float32（pipeline._preprocess_standard 输出）
        时跳过 _np_resize —— 其等尺寸路径的 astype 拷贝是无谓开销。
        数值路径不变（省略的是同一 float32 数据的整块拷贝），逐位一致。
        """
        from video_utils import _np_resize
        img_width = int(height * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(height * ratio))
        if resized_w == w and h == height and img.dtype == np.float32:
            resized = img
        else:
            resized = _np_resize(img, resized_w, height)
        resized = resized.transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        pad = np.empty((3, height, img_width), dtype=np.float32)
        pad[:, :, :resized_w] = resized
        if resized_w < img_width:
            # np.empty 未初始化：尾部必须显式置 0（原 np.zeros 语义）
            pad[:, :, resized_w:] = 0.0
        return pad

    # ═══════════════ 推理 ═══════════════

    def _infer(self, batch_np: np.ndarray) -> np.ndarray:
        # 整段持锁：TRT 路径的 ctx/buffers 是实例共享可变状态，预热线程与
        # 主 OCR 线程必须串行（GPU 单上下文本就不能并行，串行不损失吞吐）。
        with self._lock:
            return self._infer_locked(batch_np)

    def _infer_locked(self, batch_np: np.ndarray) -> np.ndarray:
        if self._trt:
            assert self._max_batch is not None  # TRT 初始化时已设置
            outs = []
            for i in range(0, len(batch_np), self._max_batch):
                outs.append(self._trt_execute(batch_np[i:i + self._max_batch]))
            return np.concatenate(outs, axis=0)
        # ONNX 动态 batch 无上限：re-OCR 预热可能一次喂数千帧（test4 5942
        # 帧）→ 中间激活内存爆炸（MaxPool bad allocation）。分片限制单批
        # 帧数，输出形状不变。分片 16（原 64）：小片实测更快（64 片有线程
        # 同步/带宽瓶颈）且 ORT arena 峰值更低（64: 920MB vs 16: 300MB，
        # (3,48,320) small 模型 992 帧实测）。
        onnx_max = 16
        if len(batch_np) <= onnx_max:
            return np.asarray(self._session.run(None, {"x": batch_np})[0],
                              dtype=np.float32)
        outs = []
        for i in range(0, len(batch_np), onnx_max):
            outs.append(np.asarray(
                self._session.run(None, {"x": batch_np[i:i + onnx_max]})[0],
                dtype=np.float32))
        return np.concatenate(outs, axis=0)

    def _trt_execute(self, x: np.ndarray) -> np.ndarray:
        from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
        # 主路径 shape 恒定（batch 6, 320 宽）：set_input_shape 实测每批
        # 开销 ~0.5ms（TRT context 重配置），只在 shape 变化时调用
        if self._last_in_shape != x.shape:
            self._trt_ctx.set_input_shape(self._trt_in_name, x.shape)
            self._last_in_shape = x.shape
            self._out_shape = tuple(self._trt_ctx.get_tensor_shape(self._trt_out_name))
        out_shape = self._out_shape
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

    def _ctc_decode_batch(self, preds: np.ndarray) -> list:
        """批 CTC decode：(B, seq, C) → list[RecOut]。

        分块归约（每块 64 帧）：整批 argmax 在 C=6906 时产生 (B, seq)
        int64，~1000 帧一次归约峰值 ~2.2GB（Windows 堆不归还 → RSS 保持
        高位）。分块后峰值 ~150MB。逐行归约与整批数值一致。
        """
        out: list = []
        for s0 in range(0, len(preds), 64):
            chunk = preds[s0:s0 + 64]
            idx = chunk.argmax(axis=2)  # (B, seq) int64
            prob = chunk.max(axis=2)
            keep = np.ones_like(idx, dtype=bool)
            keep[:, 1:] = idx[:, 1:] != idx[:, :-1]
            keep &= idx != 0  # blank
            for b in range(len(chunk)):
                kb = keep[b]
                if kb.any():
                    text = "".join(self._chars[i] for i in idx[b][kb])
                    # 与 rapidocr 一致：每帧概率先 round(5) 再取均值，最后 round(5)
                    confs = [round(float(p), 5) for p in prob[b][kb]]
                    conf = round(float(np.mean(confs)), 5)
                else:
                    text, conf = "", 0.0
                out.append(RecOut(text, conf))
        return out

    # ═══════════════ 批处理入口 ═══════════════

    def __call__(self, img_list: list) -> list:
        """批识别：与 rapidocr text_rec 同语义，按输入顺序返回结果。"""
        if not img_list:
            return []
        heights = [im.shape[0] for im in img_list]
        h0 = heights[0]
        # 按宽度排序（rapidocr 的加速策略；结果映射回原顺序）
        order = np.argsort([im.shape[1] for im in img_list])
        # pad 宽度 = max(批内最大宽高比, 本模型下限/48)。旧代码强制 320/48
        # 下限：速度数字是窄图（48 高后 78-160 宽），pad 到 320 让 GPU 白算
        # 2~4 倍宽度；但 v6 tiny 对输入宽度敏感（test5 max_width=72 在 72 宽
        # 下精度 0.07%→0.54%），不能无下限。每模型下限查 OCR_PAD_WIDTH_MIN_
        # BY_MODEL（实测平衡点），测试可用 RVTOL_PAD_TINY/SMALL 覆盖。
        _floor = config.OCR_PAD_WIDTH_MIN_BY_MODEL.get(
            self._variant, config.OCR_PAD_WIDTH_MIN)
        _env = os.environ.get("RVTOL_PAD_TINY" if "tiny" in self._variant
                              else "RVTOL_PAD_SMALL")
        if _env and _env.isdigit():
            _floor = int(_env)
        max_wh = max(_floor / 48.0,
                     *(float(im.shape[1]) / im.shape[0] for im in img_list))
        batch_np = np.stack([self._resize_norm(img_list[i], max_wh, h0)
                             for i in order])
        preds = self._infer(batch_np)
        results: list = [None] * len(img_list)
        # 批向量化 decode：argmax/max/keep 一次归约（与逐帧 _ctc_decode
        # 数值相同 —— 同一归约按行应用）；text 拼接保持逐帧
        if preds.ndim == 3:
            batch_results = self._ctc_decode_batch(preds)
            for k, idx in enumerate(order):
                results[idx] = batch_results[k]
        else:
            for k, idx in enumerate(order):
                results[idx] = self._ctc_decode(preds[k])
        return results
