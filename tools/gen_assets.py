"""从 SVG 源生成应用图标与 README 图片。

产物：
  assets/app_icon_256.png        GUI 窗口图标（QIcon 用）
  assets/app.ico                 frozen exe 图标（16-256 六尺寸，PNG 条目容器）
  docs/images/banner.png         README 首页横幅（2x）
  docs/images/pipeline.png       流水线流程图（2x）

用法：python tools/gen_assets.py（SVG 源改动后重跑）
依赖：PySide6（QtSvg 渲染）。ICO 为手工组装的 PNG 条目容器，无需 Pillow。
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from PySide6.QtCore import QBuffer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ICO_SIZES = (16, 32, 48, 64, 128, 256)
JOBS = [  # (svg 源, png 产物, 宽, 高, 放大倍数)
    ("assets/app_icon.svg", "assets/app_icon_256.png", 256, 256, 1),
    ("docs/images/banner.svg", "docs/images/banner.png", 1200, 320, 2),
    ("docs/images/pipeline.svg", "docs/images/pipeline.png", 1360, 760, 2),
]


def render_png(svg_path: Path, w: int, h: int, scale: float = 1.0) -> bytes:
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"SVG 无效: {svg_path}")
    img = QImage(int(w * scale), int(h * scale), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    buf = QBuffer()
    buf.open(QBuffer.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def build_ico(entries: list[tuple[int, bytes]]) -> bytes:
    """ICO 容器（PNG 条目，Vista+ 标准）。entries: [(边长px, png字节)]。"""
    head = struct.pack("<HHH", 0, 1, len(entries))
    directory = b""
    offset = 6 + 16 * len(entries)
    blobs = []
    for size, data in entries:
        side = 0 if size >= 256 else size  # 256 在 ICO 目录里编码为 0
        directory += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32,
                                 len(data), offset)
        blobs.append(data)
        offset += len(data)
    return head + directory + b"".join(blobs)


def main() -> None:
    _app = QGuiApplication(sys.argv)  # 字体/绘制初始化
    for svg_rel, png_rel, w, h, scale in JOBS:
        data = render_png(ROOT / svg_rel, w, h, scale)
        out = ROOT / png_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"✓ {png_rel}  {w*scale:.0f}x{h*scale:.0f}  {len(data)/1024:.0f} KB")
    entries = [(s, render_png(ROOT / "assets/app_icon.svg", s, s))
               for s in ICO_SIZES]
    ico = ROOT / "assets/app.ico"
    ico.write_bytes(build_ico(entries))
    print(f"✓ assets/app.ico  {len(ICO_SIZES)} 尺寸  "
          f"{ico.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
