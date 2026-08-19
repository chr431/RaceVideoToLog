"""生成 video_ocr_engine/extractor.py：骨架（Python 字面量）+ 缩进后的方法体。"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent / "video_ocr_engine"
body = (root / "_methods_body.py").read_text(encoding="utf-8")
lines = body.splitlines()
start = 1 if lines and lines[0].lstrip().startswith('"""') else 0
# 顶层函数 → 类内成员：每行加 4 空格。注意只能用 strip('\n') 去首尾空行，
# 若用 strip() 会把第一行 @property 的前导缩进剥掉（运算符优先级 + strip 坑）。
indented = "\n".join("    " + l if l.strip() else l
                     for l in lines[start:]).strip("\n")

SKELETON = '''"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（RaceVideoToLog 的 SegmentPipeline 继承本类并叠加后处理）。

方法体由 tools/archive/_gen_engine_extractor.py 从 segment_flow.py 抽取
（见 video_ocr_engine/_methods_body.py）。引擎目标是独立可发布仓库。
"""
import csv
import logging
import os as _os
import threading
import time
from pathlib import Path

import numpy as np

import engine_config as config  # 识别链只用引擎域常量
from constants import Flag
from segmentation import (
    _apply_gamma, _cluster_win3, _gray, _gray_batch, _gray_seg,
    _gray_seg_batch, _gray_seg_yuv, _gray_seg_yuv_batch, _otsu, _seg_gamma,
)
from hybrid_decode import (
    HYBRID_BACKEND_ALIASES, _decode_range_worker, _drain_queue, _hybrid_ranges,
)
from ocr_native import OcrEngine, auto_ocr_thread_count
from ocr_engine import extract_speed_value  # 待引擎 _run_pipelined 移除速度解析后删

logger = logging.getLogger(__name__)


def _ocr_batch_size() -> int:
    _env = _os.environ.get("RVTOL_OCR_BATCH")
    if _env and _env.isdigit():
        return max(1, int(_env))
    return config.OCR_BATCH_SIZE


class FieldExtractor:
    """从视频固定区域提取文本的通用引擎（识别链：解码∥分段∥OCR）。

    构造参数（识别链用）：video_path / roi / frame_start / frame_end /
    force_aspect / decode_backend / ocr_backend / buffer_size / fill_width /
    progress_cb / cancel_check / gray_output / yuv_output。
    分段阈值 C 取自 engine_config.SEG_C；引擎不含速度后处理参数。
    """

    def __init__(self, video_path: str, roi: tuple, *, frame_start=None,
                 frame_end=None, force_aspect: float = 0.0,
                 decode_backend: str = "auto", ocr_backend: str = "auto",
                 buffer_size: int | None = None, fill_width: int | None = None,
                 progress_cb=None, cancel_check=None, gray_output: bool = False,
                 yuv_output: bool = False):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        self._fps = None  # run 时从 decoder 推导
        self._frame_start = frame_start or 0
        self._frame_end = frame_end
        self._force_aspect = force_aspect
        self._decode_backend = decode_backend
        self._ocr_backend = ocr_backend
        self._ocr_backend_used = ""    # run 后填实际引擎（供 CSV 头输出）
        self._buffer_size = (buffer_size if buffer_size is not None
                             else config.DEFAULT_BUFFER_SIZE)
        self._fill_width = (fill_width if fill_width is not None
                            else config.DEFAULT_FILL_WIDTH)
        self._C = config.SEG_C           # 分段聚类阈值
        self._gray_output = gray_output
        self._yuv_output = yuv_output
        self._color_range = 0            # run 时从 decoder get_color_range 读取
        self._codec = ""                 # run 时从 decoder get_codec 探测
        self._hybrid_codec = ""
        self._backend = ""
        self._bin_thresh = 0
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.rows: list = []
        self.timing: dict = {}
        self.segments: list[dict] = []
        self.crops: dict = {}
        self._segs: list = []
        self._frames: list = []
        self._ocr_vals: list = []
        self._ocr_texts: list = []
        self._ocr_confs: list = []
        self._corr_vals: list = []
        self._conf_vals: list = []
        self._pinned: set = set()
        self._n_segments = 0
        self._n_corr = 0
        self._profile_enabled = _os.environ.get(
            "RVTOL_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
        self.profile: dict = {}
        self._prof_lock = None
        if self._profile_enabled:
            self._prof_lock = threading.Lock()
        # 后处理参数由子类（SegmentPipeline）在构造时设置；引擎识别链不读。

    def extract(self):
        """通用文本提取入口（待精修：解码∥分段∥OCR → 每段 text/conf 结果）。"""
        raise NotImplementedError

%METHODS%
'''

out_text = SKELETON.replace("%METHODS%", indented) + "\n"
(root / "extractor.py").write_text(out_text, encoding="utf-8")
print("extractor.py rebuilt:", len(out_text.splitlines()), "lines")
