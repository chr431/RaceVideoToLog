"""CLI / headless mode for RaceVideoToLog."""
from __future__ import annotations
import argparse, csv, math, os, re, sys
from pathlib import Path
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from ocr_engine import *
from ocr_engine import _reset_backend, _select_backend, _get_model_kwargs, _savgol_filter_np, ocr_digital_fallback, compute_video_hash

def run_headless(args: argparse.Namespace) -> None:
	"""命令行无头模式：不启动 GUI，直接分析并输出 CSV。"""
	if not args.roi:
		print("错误: 命令行模式需要 --roi X1 Y1 X2 Y2")
		sys.exit(1)

	video_path = Path(args.video)
	if not video_path.exists():
		print(f"错误: 找不到文件 {video_path}")
		sys.exit(1)

	output_path = Path(args.output) if args.output else video_path.with_suffix(".csv")
	region = (args.roi[0], args.roi[1], args.roi[2], args.roi[3])

	print(f"视频: {video_path}")
	print(f"识别范围: {region}")
	print(f"采样间隔: 1/{args.div}")
	print(f"最大速度: {args.max_speed} km/h, 最大加速度: {args.max_accel} m/s^2")
	print(f"OCR 后端选择: {args.backend}")

	# 初始化 OCR
	_reset_backend()
	backend_actual = _select_backend(args.backend)
	print(f"OCR 后端: {backend_actual}, 模型: {args.ocr_model}")
	model_kwargs = _get_model_kwargs(args.ocr_model)
	if model_kwargs is None:
		print(f"警告: {args.ocr_model} 模型文件不存在")
	ocr = RapidOCR(**(model_kwargs or {}))

	# 读取视频信息
	cap = cv2.VideoCapture(str(video_path))
	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	duration = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps) if fps > 0 else 0.0
	print(f"分辨率: {width}x{height}, 帧率: {fps:.2f}, 时长: {format_duration(duration)}")

	# 读取帧
	x1, y1, x2, y2 = clamp_region(*region, width, height)
	frame_step = max(1, args.div)

	raw_frames: list[tuple[float, np.ndarray]] = []
	fi = 0
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			break
		if args.frame_end is not None and fi >= args.frame_end:
			break
		if args.frame_start is not None and fi < args.frame_start:
			fi += 1
			continue
		if fi % frame_step != 0:
			fi += 1
			continue
		ts = fi / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
		crop = frame[y1:y2 + 1, x1:x2 + 1].copy()  # .copy() 断开对整帧的引用
		raw_frames.append((ts, crop))
		fi += 1
	cap.release()

	total = len(raw_frames)
	print(f"采样帧: {total}")
	if total == 0:
		print("错误: 未读取到帧")
		sys.exit(1)

	# OCR
	observations: list[SpeedObservation] = []
	for idx, (ts, crop) in enumerate(raw_frames):
		proc = _preprocess_headless(crop, args.target_h, args.pad)
		ocr_result, _ = ocr(proc)
		sv, rt = extract_speed_value(ocr_result)
		if sv is None:
			proc2 = _preprocess_headless_fallback(crop, args.target_h, args.pad)
			ocr_result, _ = ocr(proc2)
			sv, rt = extract_speed_value(ocr_result)
		if sv is None:
			# 数字仪表后备链：use_det=False → EasyOCR
			sv, rt = ocr_digital_fallback(ocr, crop, args.max_speed)
		if sv is not None and rt is not None:
			observations.append(SpeedObservation(
				timestamp=ts,
				raw_speed_kmh=sv * SOURCE_TO_KMH[args.format],
				raw_text=rt,
			))
		if (idx + 1) % 10 == 0:
			print(f"\r  OCR 进度: {idx + 1}/{total} 已识别: {len(observations)}", end="", flush=True)

	if observations:
		print(f"\r  OCR 完成: {total} 帧, 识别 {len(observations)} 条" + " " * 10)

	if not observations:
		print("错误: 未识别到速度数据")
		sys.exit(1)

	# ── 自动锚点 + Correction B（与 GUI 自动锚点模式共用后端）──
	print(f"识别: {len(observations)} 条, 正在自动选择锚点...")
	anchor_indices = auto_select_anchors(observations, args.max_speed)
	print(f"  锚点: {len(anchor_indices)} 帧 ({100*len(anchor_indices)/len(observations):.1f}%)")

	# 构建 rows
	rows_data = []
	for i, obs in enumerate(observations):
		if i in anchor_indices:
			rows_data.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 2])
		else:
			rows_data.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 0])

	# Correction（与 GUI 共享同一实现）
	from correction import correct_with_anchors
	rows_data = correct_with_anchors(
		rows_data, observations, raw_frames, ocr,
		args.max_speed, args.max_accel, anchor_indices)

	# 积分距离
	dist = 0.0; prev_t, prev_v = None, None
	for r in rows_data:
		v = r[2] / 3.6
		if prev_t is not None and prev_v is not None:
			dt = r[0] - prev_t
			if dt > 0: dist += (prev_v + v) * 0.5 * dt
		prev_t, prev_v = r[0], v; r[1] = dist

	# 写出 CSV
	print(f"导出: {output_path}")
	with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
		vhash = compute_video_hash(video_path)
		fh.write(f"# RaceVideoToLog\n")
		fh.write(f"# video_hash={vhash}, video={video_path.name}\n")
		fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format={args.format}\n")
		fh.write(f"# max_speed={args.max_speed}, max_accel={args.max_accel}, div={args.div}, target_h={args.target_h}, pad={args.pad}, backend={backend_actual}, model={args.ocr_model}, workers={args.workers}, frame_start={args.frame_start or ''}, frame_end={args.frame_end or ''}\n")
		w = csv.writer(fh)
		for r in rows_data:
			w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

	_corrected = sum(1 for r in rows_data if r[3] >= 1)
	print(f"共 {len(rows_data)} 条, 纠错 {_corrected} 条 (准确率 {100 - _corrected/len(rows_data)*100:.1f}%)")


def _preprocess_headless(crop, target_h, pad):
	"""无头模式预处理：灰度化 + 缩放。"""
	gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	return _finish_preprocess(gray, target_h, pad)


def _preprocess_headless_fallback(crop, target_h, pad):
	"""无头模式备选预处理：OTSU 二值化。"""
	gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return _finish_preprocess(gray, target_h, pad)


def _finish_preprocess(gray, target_h, pad):
	"""统一的缩放+填充+转BGR。"""
	h, w = gray.shape[:2]
	th = max(8.0, float(target_h))
	scale = th / h if h > 0 else 1.0
	if abs(scale - 1.0) > 0.02:
		gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)), interpolation=cv2.INTER_LINEAR)
	pad_int = int(pad)
	if pad_int > 0:
		gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
	return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)