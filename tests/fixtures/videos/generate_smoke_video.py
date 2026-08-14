"""参考脚本：生成 CI 解码集成测试用的迷你视频 tests/fixtures/videos/smoke_speedo.mp4。

模拟速度表：暗底 + 白色数字（cv2.putText）+ 移动竖条。2 秒 30fps 160x100。
仅供 decord 解码冒烟测试（open/next_roi/get_batch/gray 输出），不做 OCR。

注意：已提交的 smoke_speedo.mp4 字节是权威基准（tests/test_decoder_integration.py
的首帧 ROI 哈希 4045c7a10a945e95 绑定它）。用本脚本重新生成（不同 cv2 版本
编码输出可能不同）后必须重跑解码集成测试并更新参考哈希。
生成需本机装 opencv-python（cv2，非项目依赖，仅生成用）。
"""
import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)
path = str(OUT / "smoke_speedo.mp4")

W, H, FPS, N = 160, 100, 30, 60
# 先试 H.264 (avc1)，失败回退 mp4v
for fourcc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
    if vw.isOpened():
        break
if not vw.isOpened():
    raise SystemExit("无法创建 VideoWriter")

rng = np.random.default_rng(7)
for i in range(N):
    # 暗底 + 轻微噪声
    frame = np.full((H, W, 3), 16, dtype=np.uint8)
    noise = rng.integers(0, 8, (H, W, 1), dtype=np.uint8)
    frame = cv2.add(frame, np.repeat(noise, 3, axis=2))
    # 白色三位速度数字（随时间线性变化）
    speed = 60 + i  # 60..119 km/h
    cv2.putText(frame, f"{speed:03d}", (18, 62), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3, cv2.LINE_AA)
    # 移动竖条（提供逐帧变化）
    x = (i * 2) % (W - 6)
    cv2.rectangle(frame, (x, 10), (x + 5, 90), (90, 90, 90), -1)
    vw.write(frame)
vw.release()
print(f"{path}  {Path(path).stat().st_size} bytes")
