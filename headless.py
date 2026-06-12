"""CLI / headless mode for RaceVideoToLog."""
from __future__ import annotations
import argparse, csv, math, os, re, sys
from pathlib import Path
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from ocr_engine import *
from ocr_engine import _reset_backend, _select_backend, _get_model_kwargs, _savgol_filter_np

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
	if model_kwargs is None and args.ocr_model != "v3":
		print(f"警告: {args.ocr_model} 模型文件不存在，回退到默认 v3")
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

	# 初次纠错 + 物理超限帧重试 OCR
	print(f"识别: {len(observations)} 条, 正在进行物理约束纠错...")
	corrected = correct_speed_series(observations, args.max_speed, args.max_accel)
	observations, corrected = _retry_suspect_frames(
		observations, corrected, raw_frames, ocr, args
	)
	if corrected is None:
		corrected = correct_speed_series(observations, args.max_speed, args.max_accel)

	# 积分 + 写出
	rows: list[tuple[float, float, float, int]] = []
	dist = 0.0
	prev_t, prev_v, prev_cspd = None, None, None
	for obs, cspd in zip(observations, corrected):
		v = cspd / 3.6
		if prev_t is not None and prev_v is not None:
			dt = obs.timestamp - prev_t
			if dt > 0:
				dist += (prev_v + v) * 0.5 * dt
		prev_t, prev_v = obs.timestamp, v
		flag = 1 if abs(obs.raw_speed_kmh - cspd) > 0.01 else 0
		# 纠正后仍超物理极限的也标记
		if not flag and prev_cspd is not None:
			dt_phys = obs.timestamp - (rows[-1][0] if rows else 0)
			if dt_phys > 0 and abs(cspd - prev_cspd) / (dt_phys * 3.6) > args.max_accel:
				flag = 1
		prev_cspd = cspd
		rows.append((obs.timestamp, dist, cspd, flag))

	with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
		w = csv.writer(fh)
		for t, d, s, fl in rows:
			w.writerow([f"{t:.2f}", f"{d:.2f}", f"{s:.2f}", str(fl)])

	corrected_count = sum(r[3] for r in rows)
	print(f"导出完成: {output_path}")
	print(f"共 {len(rows)} 条, 纠错 {corrected_count} 条 (准确率 {100 - corrected_count/len(rows)*100:.1f}%)")


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


def _retry_suspect_frames(
	observations: list,
	corrected_first: list[float],
	raw_frames: list,
	ocr,
	args,
) -> tuple[list, list[float] | None]:
	"""对物理超限帧用备选预处理重试 OCR，选最接近邻帧期望值的读数。"""
	if args.max_accel <= 0:
		return observations, None
	max_delta_kmh = args.max_accel * (1.0 / 57.0) * 3.6  # 假设 ~17.5ms 帧间隔
	_suspects = []
	for i in range(1, len(corrected_first)):
		dv = abs(corrected_first[i] - corrected_first[i - 1])
		if dv > max_delta_kmh * 2 and abs(observations[i].raw_speed_kmh - corrected_first[i]) < 0.5:
			# 纠正没改动但跳变超物理极限
			expected = corrected_first[i - 1]  # 期望接近前一帧
			_suspects.append((i, expected))

	if not _suspects:
		return observations, None

	# 对可疑帧重试 OCR
	improved = 0
	new_obs = list(observations)
	for idx, expected in _suspects:
		ts, crop = raw_frames[idx]
		best_speed = new_obs[idx].raw_speed_kmh
		best_text = new_obs[idx].raw_text
		best_diff = abs(best_speed - expected)

		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

		# 变体1: 不缩放，直接灰度图
		proc1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc1, expected, best_speed, best_text, best_diff, args)

		# 变体2: OTSU 二值化 + 缩放
		_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		h2, w2 = th2.shape[:2]
		th = max(8.0, float(args.target_h))
		sc2 = th / h2 if h2 > 0 else 1.0
		if abs(sc2 - 1.0) > 0.02:
			th2 = cv2.resize(th2, (max(1, int(w2 * sc2)), int(th)), interpolation=cv2.INTER_LINEAR)
		proc2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc2, expected, best_speed, best_text, best_diff, args)

		# 变体3: 固定阈值 120（浅色数字）
		_, th3 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
		proc3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc3, expected, best_speed, best_text, best_diff, args)

		# 变体4: 互补引擎 — v3 模型（不同网络权重，错误模式互补）
		best_speed, best_text, best_diff = _ocr_retry_v3(
			crop, expected, best_speed, best_text, best_diff, args)

		# 变体5: 互补引擎 — 反向二值化（黑底白字→白底黑字）
		th_inv = cv2.bitwise_not(th2) if th2.shape == gray.shape else cv2.bitwise_not(
			cv2.resize(th2, (gray.shape[1], gray.shape[0])))
		h5, w5 = th_inv.shape[:2]
		sc5 = max(8.0, float(args.target_h)) / h5 if h5 > 0 else 1.0
		if abs(sc5 - 1.0) > 0.02:
			th_inv = cv2.resize(th_inv, (max(1, int(w5 * sc5)), int(max(8.0, float(args.target_h)))), interpolation=cv2.INTER_LINEAR)
		proc5 = cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR)
		best_speed, best_text, best_diff = _ocr_retry(
			ocr, proc5, expected, best_speed, best_text, best_diff, args)

		if best_diff < abs(new_obs[idx].raw_speed_kmh - expected):
			new_obs[idx] = SpeedObservation(
				timestamp=new_obs[idx].timestamp,
				raw_speed_kmh=best_speed,
				raw_text=best_text,
			)
			improved += 1

	if improved > 0:
		print(f"  重试 OCR 改善 {improved} 帧, 重新纠错...")
		return new_obs, None  # 需要重新跑 DP
	return observations, corrected_first


def _ocr_retry(ocr, proc, expected, best_speed, best_text, best_diff, args):
	"""单次备选 OCR 尝试，返回（可能更新后的）best_* 值。"""
	ocr_result, _ = ocr(proc)
	sv, rt = extract_speed_value(ocr_result)
	if sv is not None and rt is not None:
		spd = sv * SOURCE_TO_KMH[args.format]
		diff = abs(spd - expected)
		if diff < best_diff:
			return spd, rt, diff
	return best_speed, best_text, best_diff


def _ocr_retry_v3(crop, expected, best_speed, best_text, best_diff, args):
	"""互补引擎重试：用 RapidOCR v3 模型（与 v5_mobile 不同网络权重，错误模式互补）。

	v3 和 v5_mobile 使用不同的检测/识别网络，错误模式互补：
	v3 极端错误少但中等偏差多，v5m 反之。
	"""
	try:
		v3_ocr = _get_v3_ocr()
		if v3_ocr is not None:
			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray = clahe.apply(gray)
			_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			h2, w2 = gray.shape[:2]
			th = max(8.0, float(args.target_h))
			sc = th / h2 if h2 > 0 else 1.0
			gray = cv2.resize(gray, (max(1, int(w2 * sc)), int(th)))
			proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
			ocr_result, _ = v3_ocr(proc)
			sv, rt = extract_speed_value(ocr_result)
			if sv is not None and rt is not None:
				spd = sv * SOURCE_TO_KMH[args.format]
				diff = abs(spd - expected)
				if diff < best_diff:
					return spd, rt, diff
	except Exception:
		pass
	return best_speed, best_text, best_diff


_v3_ocr_cache = None


def _get_v3_ocr():
	"""懒加载 v3 模型 OCR 引擎（全局单例）。"""
	global _v3_ocr_cache
	if _v3_ocr_cache is None:
		try:
			from rapidocr_onnxruntime import RapidOCR
			_v3_ocr_cache = RapidOCR()  # 默认 v3
		except Exception:
			_v3_ocr_cache = False  # 标记失败，不再重试
	return _v3_ocr_cache if _v3_ocr_cache is not False else None


def run_analysis_headless(args: argparse.Namespace) -> None:
	"""无头数据分析：从两个 CSV 导出 v-t、v-x、Δt-x 三张 PNG。"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	csv1, csv2 = Path(args.analysis[0]), Path(args.analysis[1])
	if not csv1.exists() or not csv2.exists():
		print("错误: CSV 文件不存在")
		sys.exit(1)

	out_prefix = Path(args.analysis_out) if args.analysis_out else csv1.parent / "分析"
	out_prefix.parent.mkdir(parents=True, exist_ok=True)

	# 解析 CSV
	def _read_csv(p: Path):
		times, dists, speeds, flags = [], [], [], []
		with open(p, "r", encoding="utf-8-sig") as f:
			for line in f:
				parts = line.strip().split(",")
				if len(parts) >= 3:
					times.append(float(parts[0]))
					dists.append(float(parts[1]))
					speeds.append(float(parts[2]))
					flags.append(int(parts[3]) if len(parts) > 3 else 0)
		# 去开头静止帧
		start = 0
		for i, s in enumerate(speeds):
			if s > 0:
				start = i
				break
		if start > 0:
			times = times[start:]
			speeds = speeds[start:]
			base_dist = dists[start]
			dists = [d - base_dist for d in dists[start:]]
			flags = flags[start:]
			base_time = times[0]
			times = [t - base_time for t in times]
		return times, dists, speeds, flags

	t1, d1, s1, f1 = _read_csv(csv1)
	t2, d2, s2, f2 = _read_csv(csv2)
	name1, name2 = csv1.stem, csv2.stem

	from scipy.signal import savgol_filter
	def _smooth(yv, strength=25):
		if len(yv) < 5:
			return yv
		win = int(len(yv) * strength / 100.0 * 0.0175)
		win = max(5, min(win, len(yv) - 2))
		if win % 2 == 0:
			win += 1
		return _savgol_filter_np(np.array(yv, dtype=float), win, min(3, win - 1))

	# ── v-t ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, name, c in [(s1, name1, "#2196F3"), (s2, name2, "#FF5722")]:
		ax.plot(t1 if data is s1 else t2, _smooth(data), color=c, linewidth=0.8, label=name)
	ax.set_xlabel("时间 (s)"); ax.set_ylabel("速度 (km/h)")
	ax.set_title("速度-时间曲线"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_v-t.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"v-t: {out_prefix}_v-t.png")

	# ── v-x ──
	fig, ax = plt.subplots(figsize=(10, 6))
	for data, name, c in [(s1, name1, "#2196F3"), (s2, name2, "#FF5722")]:
		ax.plot(d1 if data is s1 else d2, _smooth(data), color=c, linewidth=0.8, label=name)
	ax.set_xlabel("距离 (m)"); ax.set_ylabel("速度 (km/h)")
	ax.set_title("速度-距离曲线"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_v-x.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"v-x: {out_prefix}_v-x.png")

	# ── Δt-x ──
	fig, ax = plt.subplots(figsize=(10, 6))
	t2_interp = np.interp(d1, d2, t2)
	dt = np.array(t1) - t2_interp
	ax.plot(d1, _smooth(dt), color="#2196F3", linewidth=0.8, label=f"{name1} - {name2}")
	ax.axhline(y=0, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7)
	ax.set_xlabel("距离 (m)"); ax.set_ylabel("Δt (s)")
	ax.set_title(f"时间差-距离 ({name1} vs {name2})"); ax.legend(); ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_prefix.with_name(f"{out_prefix.name}_Δt-x.png"), dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Δt-x: {out_prefix}_Δt-x.png")

	print("分析完成。")


if __name__ == "__main__":
	main()
