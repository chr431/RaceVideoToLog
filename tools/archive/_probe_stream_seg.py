"""探针：编码码流（帧级包大小/帧型）能否预示 ROI 像素变化？ vs 解码 XOR 真值。

背景动机：设想"不解码、只看码流参考/差值信息判断两帧 ROI 是否相同"。
本探针先用**最廉价的码流信息**做第一道实证门槛：
  - ffprobe -show_frames：pkt_size（该帧编码字节数）+ pict_type（I/P/B），
    只需解封装+解析帧头，**不用重建像素**；
  - 解码（decord）后对 ROI 做生产同款 binarize + XOR + _cluster_win3，
    得到逐帧"ROI 是否变化"的精确真值；
  - 统计 P 帧编码代价在有/无 ROI 变化下的分离度（精度/召回/分位重叠）。

结论取向：若连"整帧编码字节数"这种最粗信号都在移动背景片源上毫无可分性，
则更细的逐块 skip 信号（需手写熵解码/fork 解码器，代价大得多）同样不会颠覆
"流量信息无法低成本替代解码分段"的判断。

用法：
    python tools/archive/_probe_stream_seg.py \
        --video D:/Videos/text_video_test/text_test.mp4 --roi 266 989 1569 1058 --frames 1500
（ROI 用 test_params 或 detect_eval.load_meta 的值；默认全片前 --frames 帧。）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decord import VideoReader, cpu  # noqa: E402
from segmentation import _cluster_win3, _otsu  # noqa: E402

FFPROBE_FALLBACKS = [
    ROOT / "_decord_build" / "ffprobe.exe",
    ROOT / ".venv" / "Lib" / "site-packages" / "decord" / "ffprobe.exe",
]


def _find_ffprobe() -> str:
    for p in FFPROBE_FALLBACKS:
        if p.exists():
            return str(p)
    raise RuntimeError("未找到 ffprobe.exe（_decord_build\\ffprobe.exe 或 venv decord\\ffprobe.exe）")


def read_frames_meta(ffprobe: str, video: str) -> list[dict]:
    """ffprobe -show_frames：逐帧 (pkt_size, pict_type)。"""
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pict_type,pkt_size", "-of",
         "default=noprint_wrappers=1", video],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {r.stderr[-500:]}")
    frames: list[dict] = []
    cur: dict = {}
    for line in r.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        cur[k.strip()] = v.strip()
        if k.strip() == "pict_type":
            frames.append(cur)
            cur = {}
    return frames


def decode_change_labels(video: str, roi: tuple, n: int) -> np.ndarray:
    """流式解码前 n 帧的 ROI 条带 → 生产同款分段语义的逐帧变化标签（i>=1）。

    用 next_roi 一次性流式拿 ROI（不逐帧 seek），缓存 ROI 灰度后统一做
    Otsu 阈值 + XOR + _cluster_win3（与生产分段一致的"变化"判定）。
    """
    x1, y1, x2, y2 = roi
    h, w = y2 - y1 + 1, x2 - x1 + 1
    vr = VideoReader(video, ctx=cpu(0), output_format="gray")
    n = min(n, len(vr))
    grays: list[np.ndarray] = []
    for _ in range(n):
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        g = c.squeeze()
        if g.shape[0] != h or g.shape[1] != w:  # 旧路径整帧回退裁剪
            g = g[y1:y2 + 1, x1:x2 + 1]
        grays.append(g)
    del vr

    step = max(1, n // 50)
    ths = [_otsu(g) for g in grays[::step][:50]]
    th = int(np.median(ths)) if ths else 127

    C = 5.0
    labels = np.zeros(n, dtype=bool)
    prev_b = None
    for i, g in enumerate(grays):
        b = g > th
        if prev_b is not None:
            labels[i] = _cluster_win3(prev_b != b) >= C
        prev_b = b
    return labels


def separation_metrics(p_sizes: np.ndarray, changed: np.ndarray) -> dict:
    """P 帧编码字节数按 变化/未变 分组的分离度。"""
    if changed.size == 0 or changed.sum() == 0 or (changed == 0).sum() == 0:
        return {"err": "无变化样本，无法统计"}
    c_med = float(np.median(p_sizes[changed]))
    u_med = float(np.median(p_sizes[~changed]))
    p90_u = float(np.percentile(p_sizes[~changed], 90))
    # 简单预测器：pkt_size > p90(未变) => 预测变化
    thr = p90_u
    pred = p_sizes > thr
    tp = int((pred & changed).sum())
    fp = int((pred & ~changed).sum())
    fn = int((~pred & changed).sum())
    prec = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    # "跨类重叠"率：变化帧里有多少即使帧越大于未变帧 90 分位
    sep = float(changed[p_sizes > thr].mean()) if changed.sum() else 0.0
    return {
        "changed_frame_frac": float(changed.mean()),
        "p_med_changed": round(c_med, 1),
        "p_med_unchanged": round(u_med, 1),
        "ratio_med": round(c_med / max(1.0, u_med), 3),
        "p90_unchanged": round(thr, 1),
        "pred_precision": round(prec, 3),
        "pred_recall": round(recall, 3),
        "changed_sep_gt_p90u": round(sep, 3),
        "tp/fp/fn": (tp, fp, fn),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", nargs=4, type=int, required=True)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--only-p", action="store_true")
    args = ap.parse_args()

    ffprobe = _find_ffprobe()
    meta = read_frames_meta(ffprobe, args.video)
    n = min(args.frames, len(meta))
    meta = meta[:n]
    print(f"视频: {args.video}")
    print(f"帧数: {n}  帧型分布: "
          + ", ".join(f"{t}={sum(f.get('pict_type') == t for f in meta)}" for t in "IPB"))
    labels = decode_change_labels(args.video, tuple(args.roi), n)
    if len(labels) != n:
        # ffprobe 与解码帧数不一致（容器 over-report）→ 截短到较短者
        m = min(len(labels), n)
        labels, meta = labels[:m], meta[:m]
    print(f"ROI 变化帧占比: {labels.mean():.3f}")

    # 只统计 P 帧（帧间预测主体；I 帧是关键帧重置，B 帧参考未来）
    p_idx = np.array([i for i, f in enumerate(meta) if f.get("pict_type") == "P"])
    if args.only_p and p_idx.size:
        sel = p_idx
    else:
        sel = np.arange(len(meta))
    p_sizes = np.array([int(meta[i].get("pkt_size", 0)) for i in sel], dtype=float)
    ch = labels[sel]
    print("\n=== 帧级编码字节数 vs ROI 变化（{strip}）===".format(
        strip="仅 P 帧" if args.only_p and p_idx.size else "全部帧型"))
    m = separation_metrics(p_sizes, ch)
    for k, v in m.items():
        print(f"  {k:<22}: {v}")

    print("\n说明：ratio_med=变化/未变中位包大小比；pred_* 是'包大小>未变帧P90=>判变化'的"
          "精度/召回；changed_sep_gt_p90u=变化帧中高于未变帧P90的比例（≈1 才可分）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
