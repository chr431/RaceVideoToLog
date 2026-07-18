"""人工审核对话框 — 聚焦问题段审核，含速度曲线 + 原始图像预览。

在自动纠错后展示置信度低的问题段，人工审核关键帧后重新纠错。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
	QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
	QListWidget, QListWidgetItem, QMessageBox, QSplitter, QLineEdit,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor
from qfluentwidgets import (BodyLabel, StrongBodyLabel, CaptionLabel,
	PrimaryPushButton, PushButton, isDarkTheme)
from theme_manager import ThemeManager
from widget_utils import make_static_card, setup_chart_zoom_pan

import cv2

class ReviewDialog(QDialog):
	"""人工审核对话框 — 左侧问题段列表，右侧速度曲线 + 图像 + 修正控件。"""

	def __init__(self, parent: QWidget, rows: list, observations: list,
				 raw_frames: list, confidences: list[dict],
				 segments: list[dict], max_speed: float) -> None:
		super().__init__(parent)
		self.setWindowTitle("人工审核 — 聚焦问题段 (← → 逐帧导航)")
		self.resize(1200, 750)
		self.setMinimumSize(1000, 600)
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

		self._rows = rows
		self._observations = observations
		self._raw_frames = raw_frames  # [(timestamp, np.ndarray), ...]
		self._confidences = confidences
		self._segments = segments
		self._max_speed = max_speed
		self._corrections: dict[int, float] = {}
		self._partial_corrections: dict[int, str] = {}
		self._current_frame: int = 0
		self._last_selected_seg: dict | None = None  # 记住上次选择的段

		self._build_ui()
		self._register_theme_callbacks()

	def _register_theme_callbacks(self) -> None:
		def _update(dark: bool) -> None:
			bg = QColor("#1f1f1f" if dark else "#f5f5f5")
			fg = QColor("#f0f0f0" if dark else "#000000")
			btn_bg = QColor("#3a3a3a" if dark else "#e8e8e8")
			img_bg = "#111" if dark else "#e0e0e0"
			p = self.palette()
			for role, color in [(QPalette.ColorRole.Window, bg), (QPalette.ColorRole.Base, btn_bg),
								(QPalette.ColorRole.WindowText, fg), (QPalette.ColorRole.Text, fg),
								(QPalette.ColorRole.Button, btn_bg), (QPalette.ColorRole.ButtonText, fg)]:
				p.setColor(role, color)
			self.setPalette(p)
			self._list.setPalette(p)
			self._img_label.setStyleSheet(f"background-color: {img_bg}; border-radius: 4px;")
			if hasattr(self, '_figure'): self._redraw_chart()
		ThemeManager.register(_update)
		_update(isDarkTheme())

	def _build_ui(self) -> None:
		root = QVBoxLayout(self)
		root.setContentsMargins(16, 16, 16, 12)
		root.setSpacing(8)

		# Header
		header = QHBoxLayout()
		header.addWidget(StrongBodyLabel("聚焦人工审核"))
		header.addStretch()
		total = sum(s['count'] for s in self._segments)
		header.addWidget(CaptionLabel(f"发现 {len(self._segments)} 个问题段，共 {total} 帧待审核"))
		root.addLayout(header)

		# ── 主内容：左右分栏 ──
		splitter = QSplitter(Qt.Orientation.Horizontal)
		self._splitter = splitter

		# 左侧：问题段列表
		left = QWidget()
		ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 8, 0); ll.setSpacing(4)
		ll.addWidget(CaptionLabel("问题段"))
		self._list = QListWidget()
		self._list.setFixedWidth(290)
		self._list.currentRowChanged.connect(self._on_select)
		ll.addWidget(self._list, 1)
		for seg in self._segments:
			self._add_segment_item(seg)
		splitter.addWidget(left)

		# 右侧：图表 + 原始图像 + 控件
		right = QWidget()
		rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

		# 速度曲线
		chart_card = make_static_card()
		cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
		cl.addWidget(CaptionLabel("速度曲线（当前段加粗高亮，红色=问题段，绿色=已确认，蓝点=已修正）"))
		self._figure, self._ax, self._canvas = self._create_chart()
		cl.addWidget(self._canvas, 1)
		rl.addWidget(chart_card, 2)

		# 原始图像 + 修正控件（水平排列）
		bottom_row = QHBoxLayout(); bottom_row.setSpacing(8)

		# 原始图像预览（较小宽度，匹配 ROI 裁剪实际比例）
		img_card = make_static_card()
		il = QVBoxLayout(img_card); il.setContentsMargins(8, 8, 8, 4)
		il.addWidget(CaptionLabel("当前帧原始图像（ROI 裁剪区域）"))
		self._img_label = QLabel("选择帧后显示")
		self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self._img_label.setMinimumSize(120, 80)
		self._img_label.setStyleSheet("background-color: #111; border-radius: 4px;")
		il.addWidget(self._img_label, 1)
		bottom_row.addWidget(img_card, 1)

		# 修正控件
		ctrl_card = make_static_card()
		ctrl = QVBoxLayout(ctrl_card); ctrl.setContentsMargins(8, 8, 8, 4)

		cf_row = QHBoxLayout()
		cf_row.addWidget(BodyLabel("当前帧: "))
		self._frame_label = BodyLabel("—")
		cf_row.addWidget(self._frame_label)
		cf_row.addStretch()
		ctrl.addLayout(cf_row)

		self._suggested_widget = QWidget()
		sl = QHBoxLayout(self._suggested_widget)
		sl.setContentsMargins(0, 0, 0, 0)
		sl.addWidget(CaptionLabel("建议审核帧: "))
		self._suggested_btns: list[PushButton] = []
		sl.addStretch()
		ctrl.addWidget(self._suggested_widget)
		self._suggested_widget.hide()

		ctrl.addSpacing(4)
		cr = QHBoxLayout()
		cr.addWidget(BodyLabel("修正速度"))
		self._speed_edit = QLineEdit()
		self._speed_edit.setFixedWidth(90)
		self._speed_edit.setPlaceholderText("ex: 123 or 12x")
		cr.addWidget(self._speed_edit)
		cr.addWidget(BodyLabel("km/h"))
		cr.addStretch()
		ctrl.addLayout(cr)

		btn_row = QHBoxLayout()
		btn_add = PrimaryPushButton("添加修正")
		btn_add.setFixedWidth(100)
		btn_add.clicked.connect(self._add_correction)
		btn_row.addWidget(btn_add)
		self._btn_delete = PushButton("删除修正")
		self._btn_delete.setFixedWidth(100)
		self._btn_delete.setEnabled(False)
		self._btn_delete.clicked.connect(self._delete_correction)
		btn_row.addWidget(self._btn_delete)
		btn_row.addStretch()
		ctrl.addLayout(btn_row)

		bottom_row.addWidget(ctrl_card, 2)
		rl.addLayout(bottom_row, 1)

		# 底部完成按钮
		btn_finish = PrimaryPushButton("完成审核，重新纠错")
		btn_finish.setFixedWidth(200)
		finish_row = QHBoxLayout()
		finish_row.addStretch()
		finish_row.addWidget(btn_finish)
		btn_finish.clicked.connect(self._finish)
		rl.addLayout(finish_row)

		splitter.addWidget(right)
		splitter.setStretchFactor(1, 1)
		root.addWidget(splitter, 1)

		if self._segments:
			QTimer.singleShot(200, lambda: self._list.setCurrentRow(0))

	def _create_chart(self):
		"""创建 matplotlib 图表并附加缩放/平移交互。"""
		from matplotlib.figure import Figure
		from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

		fig = Figure(figsize=(8, 3.5), dpi=100, layout='none')
		fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
		ax = fig.add_subplot(111)
		dark = isDarkTheme()
		bg = "#2a2a2a" if dark else "#ffffff"
		fg = "#e0e0e0" if dark else "#333333"
		fig.set_facecolor(bg)
		ax.set_facecolor(bg)
		self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}

		from PySide6.QtWidgets import QSizePolicy
		canvas = FigureCanvasQTAgg(fig)
		canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		canvas.setMinimumHeight(150)
		self._canvas = canvas

		self._setup_chart_zoom_pan(ax, canvas)
		self._redraw_chart(ax, fig)
		return fig, ax, canvas

	def _setup_chart_zoom_pan(self, ax, canvas) -> None:
		"""配置滚轮缩放 + 右键拖拽平移（使用共享工具函数）。"""
		self._user_zoomed_ref, self._saved_limits = setup_chart_zoom_pan(
			ax, canvas, throttle_ms=40)

	def _redraw_chart(self, ax=None, fig=None) -> None:
		if ax is None:
			ax = self._ax
		if fig is None:
			fig = self._figure

		# 仅在用户手动缩放/平移后才恢复，首次绘制使用数据范围
		user_zoomed = (hasattr(self, '_user_zoomed_ref')
					   and self._user_zoomed_ref[0])
		if user_zoomed:
			saved_xlim = self._saved_limits["xlim"]
			saved_ylim = self._saved_limits["ylim"]

		ax.clear()
		dark = isDarkTheme()
		bg = "#2a2a2a" if dark else "#ffffff"
		fg = "#e0e0e0" if dark else "#333333"
		self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}

		times = [r[0] for r in self._rows]
		speeds = [r[2] for r in self._rows]
		cur_row = self._list.currentRow()
		cur_seg = None
		if cur_row >= 0:
			cur_seg = self._list.item(cur_row).data(Qt.ItemDataRole.UserRole)

		# 全曲线（极淡灰散点）
		bg_gray = "#666666" if not dark else "#aaaaaa"
		ax.scatter(times, speeds, c=bg_gray, s=1, alpha=0.5, zorder=0, linewidths=0, rasterized=True)

		# 当前段背景高亮
		if cur_seg:
			s, e = cur_seg['start'], cur_seg['end']
			ax.axvspan(times[s], times[min(e, len(times) - 1)],
					   facecolor="#FF9800", alpha=0.08, zorder=0)

		# 各问题段着色（散点）
		for seg in self._segments:
			s, e = seg['start'], seg['end']
			seg_t = times[s:e+1]; seg_v = speeds[s:e+1]
			is_cur = cur_seg and seg['start'] == cur_seg['start']
			# 已修正帧数超过段内一半 → 绿色标记
			seg_corrected = sum(1 for fi in range(s, e+1) if fi in self._corrections)
			if seg_corrected >= max(1, seg['count'] // 2):
				ax.scatter(seg_t, seg_v, c="#4CAF50", s=3, alpha=0.6, zorder=1, linewidths=0)
			elif is_cur:
				ax.scatter(seg_t, seg_v, c="#FF9800", s=12, zorder=4, linewidths=0)
			else:
				ax.scatter(seg_t, seg_v, c="#F44336", s=3, alpha=0.7, zorder=2, linewidths=0)

		# 已修正帧
		if self._corrections:
			cx = [times[fi] for fi in self._corrections if fi < len(times)]
			cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
			if cx:
				ax.scatter(cx, cy, c="#2196F3", s=12, zorder=5, marker='o',
						   edgecolors='white', linewidths=0.5)

		ax.set_facecolor(bg)
		fig.set_facecolor(bg)
		ax.set_xlabel("时间 (s)", color=fg)
		ax.set_ylabel("速度 (km/h)", color=fg)
		ax.tick_params(colors=fg, labelsize=8)
		ax.spines["bottom"].set_color(fg if dark else "#888")
		ax.spines["left"].set_color(fg if dark else "#888")
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.grid(True, alpha=0.15 if dark else 0.25)
		ax.set_aspect("auto")
		ax.autoscale_view()

		# 恢复用户缩放/平移状态
		if user_zoomed:
			ax.set_xlim(saved_xlim)
			ax.set_ylim(saved_ylim)

		self._canvas.draw_idle()

	def _show_frame_image(self, frame_index: int) -> None:
		"""显示指定帧的原始 ROI 图像。"""
		if 0 <= frame_index < len(self._raw_frames):
			_, crop = self._raw_frames[frame_index]
			if crop is not None and crop.size > 0:
				rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
				h, w, ch = rgb.shape
				qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
				# Scale to fit
				pm = QPixmap.fromImage(qimg)
				lw = max(50, self._img_label.width() - 8); lh = max(50, self._img_label.height() - 8)
				scaled = pm.scaled(lw, lh,
								   Qt.AspectRatioMode.KeepAspectRatio,
								   Qt.TransformationMode.SmoothTransformation)
				self._img_label.setPixmap(scaled)
				return
		self._img_label.setText("(无图像)")

	def keyPressEvent(self, event) -> None:
		"""← → 键在当前段内逐帧导航。"""
		row = self._list.currentRow()
		if row < 0:
			return super().keyPressEvent(event)
		seg = self._list.item(row).data(Qt.ItemDataRole.UserRole)
		cur = self._current_frame
		if event.key() == Qt.Key.Key_Left:
			if cur > seg['start']:
				self._navigate_to(cur - 1)
		elif event.key() == Qt.Key.Key_Right:
			if cur < seg['end']:
				self._navigate_to(cur + 1)
		else:
			super().keyPressEvent(event)

	def _navigate_to(self, fi: int) -> None:
		"""导航到指定帧并更新控件。"""
		self._current_frame = fi
		self._frame_label.setText(f"#{fi}")
		self._show_frame_image(fi)
		# 更新速度输入框
		if fi in self._partial_corrections:
			self._speed_edit.setText(self._partial_corrections[fi])
		elif fi in self._corrections:
			self._speed_edit.setText(str(int(self._corrections[fi])))
		else:
			self._speed_edit.setText(str(int(self._rows[fi][2])))
		self._btn_delete.setEnabled(
			fi in self._corrections or fi in self._partial_corrections)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		if hasattr(self, '_current_frame'):
			self._show_frame_image(self._current_frame)

	def _add_segment_item(self, seg: dict) -> None:
		text = (f"帧 {seg['start']}-{seg['end']} ({seg['count']}帧)  "
				f"置信度 {seg['avg_score']:.0f}")
		item = QListWidgetItem(text)
		item.setData(Qt.ItemDataRole.UserRole, seg)
		self._list.addItem(item)

	def _on_select(self, row: int) -> None:
		if row < 0:
			return
		seg = self._list.item(row).data(Qt.ItemDataRole.UserRole)

		for b in self._suggested_btns:
			self._suggested_widget.layout().removeWidget(b)
			b.deleteLater()
		self._suggested_btns.clear()

		for fi in seg['suggested']:
			v = self._corrections.get(fi, self._rows[fi][2])
			btn = PushButton(f"#{fi} ({v:.0f}km/h)")
			btn.setFixedWidth(110)
			btn.setProperty("frame_idx", fi)
			btn.clicked.connect(lambda checked, f=fi, val=v: self._quick_correct(f, val))
			self._suggested_widget.layout().insertWidget(
				self._suggested_widget.layout().count() - 1, btn)
			self._suggested_btns.append(btn)
		self._suggested_widget.show()

		# 恢复到该段上次查看的帧，否则默认段首
		same_seg = (self._last_selected_seg
					and self._last_selected_seg['start'] == seg['start'])
		target_frame = self._current_frame if same_seg else seg['start']
		# 确保 target_frame 在段范围内
		target_frame = max(seg['start'], min(seg['end'], target_frame))
		self._last_selected_seg = seg

		self._current_frame = target_frame
		self._frame_label.setText(f"#{target_frame}")
		self._show_frame_image(target_frame)
		if target_frame in self._partial_corrections:
			self._speed_edit.setText(self._partial_corrections[target_frame])
		elif target_frame in self._corrections:
			self._speed_edit.setText(str(int(self._corrections[target_frame])))
		else:
			self._speed_edit.setText(str(int(self._rows[target_frame][2])))
		self._btn_delete.setEnabled(
			target_frame in self._corrections or target_frame in self._partial_corrections)
		self._redraw_chart()

	def _quick_correct(self, fi: int, val: float) -> None:
		"""点击建议帧按钮：导航到该帧并自动添加修正。"""
		self._current_frame = fi; self._frame_label.setText(f"#{fi}")
		# 自动存储为修正值
		self._corrections[fi] = val
		self._partial_corrections.pop(fi, None)
		self._speed_edit.setText(str(int(val)))
		self._btn_delete.setEnabled(True)
		self._show_frame_image(fi)
		self._redraw_chart()
		# 更新建议按钮文本
		for btn in self._suggested_btns:
			try:
				f = btn.property("frame_idx")
			except Exception:
				continue
			if f == fi:
				btn.setText(f"#{fi} ({val:.0f}km/h)")
				break

	def _add_correction(self) -> None:
		fi = getattr(self, "_current_frame", 0)
		text = self._speed_edit.text().strip()
		if not text:
			return
		# Parse: pure digits = exact, contains 'x' = partial
		if 'x' in text.lower():
			self._partial_corrections[fi] = text
			self._corrections.pop(fi, None)
			label = text
		else:
			try:
				v = float(text)
			except ValueError:
				return
			self._corrections[fi] = v
			self._partial_corrections.pop(fi, None)
			label = f"{v:.0f}km/h"
		self._redraw_chart()
		for btn in self._suggested_btns:
			try:
				f = btn.property("frame_idx")
			except Exception:
				continue
			if f == fi:
				btn.setText(f"#{fi} ({label})")
				break
		self._show_frame_image(fi)
		self._btn_delete.setEnabled(True)

	def _delete_correction(self) -> None:
		fi = getattr(self, "_current_frame", 0)
		if fi not in self._corrections and fi not in self._partial_corrections:
			return
		self._corrections.pop(fi, None)
		self._partial_corrections.pop(fi, None)
		orig = self._rows[fi][2]
		self._speed_edit.setText(str(int(orig)))
		for btn in self._suggested_btns:
			try:
				f = btn.property("frame_idx")
			except Exception:
				continue
			if f == fi:
				btn.setText(f"#{fi} ({orig:.0f}km/h)")
				break
		self._redraw_chart()
		self._btn_delete.setEnabled(False)
		self._show_frame_image(fi)

	def _finish(self) -> None:
		"""完成审核：无修正时提示，确认后接受。"""
		if not self._corrections and not self._partial_corrections:
			reply = QMessageBox.question(self, "提示",
				"未添加任何修正。\n将直接使用轻量纠错结果。\n确定要完成审核吗？")
			if reply == QMessageBox.StandardButton.No:
				return
		self.accept()

	def get_corrections(self) -> dict[int, float]:
		return dict(self._corrections)

	def get_partial_corrections(self) -> dict[int, str]:
		return dict(self._partial_corrections)

	def get_confirmed(self) -> set[int]:
		return set()  # 不再使用 confirmed 机制
