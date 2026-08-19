"""RaceVideoToLog v2.15.2 — 赛车视频速度 OCR 提取工具。

从车载视频中实时 OCR 识别速度数字，支持 TensorRT / CPU 两种后端（自动选择），
输出时间-速度-距离 CSV 文件。

用法:
    python RaceVideoToLog.py                          # GUI 模式
    python RaceVideoToLog.py video.mp4 --roi X1 Y1 X2 Y2  # CLI 模式
"""
from __future__ import annotations

import sys
import io

# ── 强制 UTF-8 输出，解决 Windows 终端中文乱码 ──
# PyInstaller 打包后 console=False 时 stdout/stderr 为 None
for _stream_name in ('stdout', 'stderr'):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and getattr(_stream, 'encoding', 'utf-8') != 'utf-8':
        try:
            setattr(sys, _stream_name,
                    io.TextIOWrapper(_stream.buffer, encoding='utf-8', errors='replace'))
        except (AttributeError, ValueError):
            pass

import argparse

# ── 引擎子模块路径引导（必须在任何 import engine_config/引擎模块之前）──
# 识别链由 git submodule third_party/video_ocr_engine 提供（自拆仓起）。
from engine_bootstrap import ensure_engine_path  # noqa: E402
ensure_engine_path()


def apply_csv_settings(args, defaults: dict, argv=None) -> "object":
    """从 args.from_csv 的 CSV 头导入设置；命令行显式参数优先。

    合并规则（CLI/GUI 共用语义）：
    - 命令行显式写出的参数（即使等于默认值）优先于 CSV —— 旧逻辑误判
      "值==默认值"为"未指定"，被 CSV 静默覆盖（如 --ocr-model v6_tiny
      被 CSV 的 model=v6_small 覆盖，引擎静默换模型）。
    - CSV 中的只读字段（fps/codec 等无 argparse dest 的）跳过。
    - 解析失败的字段静默跳过（parse_csv_setting 返回 None）。
    返回 args（原地修改）。argv=None 时取 sys.argv（CLI 场景），
    测试可显式传入 argv 列表。
    """
    if not getattr(args, "from_csv", None):
        return args
    from ocr_engine import (parse_csv_header, parse_csv_setting,
                            csv_field_dest, normalize_ocr_backend)
    argv = sys.argv if argv is None else argv
    # 命令行显式写出的参数（即使等于默认值）优先于 CSV。
    _explicit = {
        a[2:].split("=", 1)[0].replace("-", "_")
        for a in argv[1:] if a.startswith("--")
    }
    csv_settings = parse_csv_header(args.from_csv)
    for key, val in csv_settings.items():
        dest = csv_field_dest(key)
        if dest is None or not hasattr(args, dest):
            continue  # read-only fields (fps/codec) or unknown — skip
        if dest in _explicit:
            continue  # 命令行显式指定 — 不被 CSV 覆盖
        cur = getattr(args, dest)
        if cur != defaults.get(dest):
            continue  # 已非默认值 — 跳过
        if key == "ocr_backend":
            # CSV 记录的是实际引擎（onnxruntime/tensorrt/…），需归一化
            # 到 CLI 可请求值（auto/cpu/tensorrt）再回填
            parsed = normalize_ocr_backend(val)
        else:
            parsed = parse_csv_setting(key, val)
        if parsed is not None:
            setattr(args, dest, parsed)
    return args


def main() -> None:
    import config
    parser = argparse.ArgumentParser(description="RaceVideoToLog - 视频速度提取工具")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"), help="识别范围")
    parser.add_argument("--format", choices=list(config.SOURCE_TO_KMH),
                        default=config.DEFAULT_SPEED_FORMAT)
    parser.add_argument("--buffer", type=int, default=config.DEFAULT_BUFFER_SIZE,
        help="解码∥OCR 流水线队列缓冲（段数）")
    parser.add_argument("--max-speed", type=float, default=config.DEFAULT_MAX_SPEED)
    parser.add_argument("--max-accel", type=float, default=config.DEFAULT_MAX_ACCEL)
    parser.add_argument("--fill-width", type=int, default=config.DEFAULT_FILL_WIDTH,
        help="预处理填充宽度下限 px（pad 到该总宽，速度窄图更准）")
    parser.add_argument("--force-aspect", type=float, default=None,
        help="强制横向宽高比（0=不启用；>0 时宽度=48×此值）。扁宽字体设 1.5-2.0 可改善识别")
    parser.add_argument("--decode-backend", choices=config.DECODE_BACKEND_KEYS,
        default=config.DEFAULT_DECODE_BACKEND,
        help="解码后端（auto/cpu/nvdec，默认 auto 自动选 GPU；实验性 "
             "CPU+NVDEC 混合解码可用环境变量 "
             + config.HYBRID_DECODE_ENV + "=1 开启）")
    parser.add_argument("--ocr-backend", choices=config.OCR_BACKEND_KEYS,
        default=config.DEFAULT_OCR_BACKEND,
        help="OCR 推理后端（auto/cpu/tensorrt，默认 auto 自动选 GPU）")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--frame-start", type=int, metavar="N")
    parser.add_argument("--frame-end", type=int, metavar="N")
    parser.add_argument("--log-level", choices=["normal","detailed","debug"],
        default=config.DEFAULT_LOG_LEVEL, help="日志级别 (默认 normal)")
    parser.add_argument("--no-monitor", action="store_true",
        help="禁用资源监控（内存/CPU/GPU 采样；默认启用，RVTOL_MONITOR=0 等效）")
    parser.add_argument("--monitor-interval", type=float, default=None,
        metavar="SEC", help="资源采样间隔秒（默认 1.0，RVTOL_MONITOR_INTERVAL 等效）")
    parser.add_argument("--from-csv", type=str, metavar="PATH",
        help="从已有 CSV 文件头导入设置（可被显式参数覆盖）")
    args = parser.parse_args()

    # ── 从 CSV 导入设置 ──
    _defaults = {a.dest: a.default
                 for a in parser._actions if a.dest != "help"}
    apply_csv_settings(args, _defaults)

    # None = 未指定：归一化为配置默认值（显式传 0 必须保留，不能被 CSV 覆盖）
    if getattr(args, "force_aspect", None) is None:
        args.force_aspect = config.DEFAULT_FORCE_ASPECT

    from logging_setup import configure_logging
    configure_logging(args.log_level)

    if args.video:
        from headless import run_headless
        run_headless(args)
    else:
        from PySide6.QtWidgets import QApplication
        import io, sys as _sys
        _saved = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            from qfluentwidgets import setTheme, Theme
        finally:
            _sys.stdout = _saved
        from gui import RaceVideoToLogApp
        app = QApplication(sys.argv)
        setTheme(Theme.AUTO)
        window = RaceVideoToLogApp()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
