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
from config import (COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE,
                     COLOR_LIGHT_GRAY, COLOR_LIGHTER_GRAY, chart_colors)

import cv2
import numpy as np

class ReviewDialog(QDialog):
	"""人工审核对话框 — 左侧问题段列表，右侧速度曲线 + 图像 + 修正控件。"""

	def __init__(self, parent: QWidget, rows: list, observations: list,
				 raw_frames: list, confidences: list[dict],
				 segments: list[dict], max_speed: float,
				 max_accel: float = 50.0,
				 final_check: bool = False) -> None:
		super().__init__(parent)
		self._final_check = final_check
		if final_check:
			self.setWindowTitle("最终检查 — 点击图中任意点修正单帧")
		else:
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
		self._max_accel = max_accel
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
		if self._final_check:
			header.addWidget(StrongBodyLabel("最终检查"))
			header.addWidget(CaptionLabel("  — 点击散点图中任意帧查看图像并修正，完成后点击「确认保存」"))
		else:
			header.addWidget(StrongBodyLabel("聚焦人工审核"))
			header.addStretch()
			total = sum(s['count'] for s in self._segments)
			header.addWidget(CaptionLabel(f"发现 {len(self._segments)} 个问题段，共 {total} 帧待审核"))
		root.addLayout(header)

		# ── 主内容：左右分栏 ──
		splitter = QSplitter(Qt.Orientation.Horizontal)
		self._splitter = splitter

		# 左侧：问题段列表（最终检查模式下隐藏）
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
		if self._final_check:
			left.setVisible(False)

		# 右侧：图表 + 原始图像 + 控件
		right = QWidget()
		rl = QVBoxLayout(right); rl.setContentsMargins(8, 0, 0, 0); rl.setSpacing(6)

		# 速度曲线
		chart_card = make_static_card()
		cl = QVBoxLayout(chart_card); cl.setContentsMargins(8, 8, 8, 4)
		if self._final_check:
			cl.addWidget(CaptionLabel("速度曲线（点击数据点选帧，蓝点=已修正，橙圈=当前帧）"))
		else:
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
		cf_row.addSpacing(12)
		self._speed_value_label = BodyLabel("")
		cf_row.addWidget(self._speed_value_label)
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
		from qfluentwidgets import CompactSpinBox
		self._speed_edit = CompactSpinBox()
		self._speed_edit.setFixedWidth(110)
		self._speed_edit.setRange(0, int(self._max_speed))
		self._speed_edit.setSuffix(" km/h")
		self._speed_edit.setSpecialValueText("(无效)")
		# 禁用 flyout 面板
		try:
			self._speed_edit.compactSpinButton.clicked.disconnect()
		except Exception:
			pass
		self._speed_edit._showFlyout = lambda: None
		# 值变化时实时预览
		self._speed_edit.valueChanged.connect(self._on_spinbox_changed)
		# 拦截 spinbox 的左右箭头键 → 传递给 dialog 用于帧导航
		self._speed_edit.installEventFilter(self)
		cr.addWidget(self._speed_edit)
		cr.addStretch()
		ctrl.addLayout(cr)

		# 部分修正行（最终检查模式下隐藏）
		part_row = QHBoxLayout()
		part_row.addWidget(CaptionLabel("部分模式"))
		self._partial_edit = QLineEdit()
		self._partial_edit.setFixedWidth(90)
		self._partial_edit.setPlaceholderText("ex: 12x")
		part_row.addWidget(self._partial_edit)
		part_row.addStretch()
		ctrl.addLayout(part_row)
		if self._final_check:
			# 隐藏部分修正行的所有子控件
			for i in range(part_row.count()):
				w = part_row.itemAt(i).widget()
				if w: w.setVisible(False)

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
		if self._final_check:
			btn_finish = PrimaryPushButton("确认保存")
			btn_finish.setFixedWidth(150)
		else:
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
		bg, fg = chart_colors(dark)
		
		fig.set_facecolor(bg)
		ax.set_facecolor(bg)
		self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}

		from PySide6.QtWidgets import QSizePolicy
		canvas = FigureCanvasQTAgg(fig)
		canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		canvas.setMinimumHeight(150)
		self._canvas = canvas

		if self._final_check:
			canvas.mpl_connect('pick_event', self._on_pick)
			canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # 让箭头键直达 dialog

		self._setup_chart_zoom_pan(ax, canvas)
		self._redraw_chart(ax, fig)
		return fig, ax, canvas

	def _setup_chart_zoom_pan(self, ax, canvas) -> None:
		"""配置滚轮缩放 + 右键拖拽平移（使用共享工具函数）。"""
		self._user_zoomed_ref, self._saved_limits = setup_chart_zoom_pan(
			ax, canvas, throttle_ms=40)

	def _redraw_chart(self, ax=None, fig=None) -> None:
		"""高性能图表渲染：预创建艺术家实例，后续调用仅更新数据。

		首次调用：创建所有 scatter/span 艺术家并缓存。
		后续调用：通过 set_offsets() 就地更新数据，避免 ax.clear() + 重建。
		仅当前选中段变化或强制刷新时才完全重建。
		"""
		if ax is None:
			ax = self._ax
		if fig is None:
			fig = self._figure

		# 仅在用户手动缩放/平移后才恢复
		user_zoomed = (hasattr(self, '_user_zoomed_ref')
					   and self._user_zoomed_ref[0])
		if user_zoomed:
			saved_xlim = self._saved_limits["xlim"]
			saved_ylim = self._saved_limits["ylim"]

		dark = isDarkTheme()
		bg, fg = chart_colors(dark)

		# ── 检查是否需要完全重建（选中段变化 / 首次 / 主题变化）──
		cur_row = self._list.currentRow()
		cur_seg = None
		if cur_row >= 0:
			cur_seg = self._list.item(cur_row).data(Qt.ItemDataRole.UserRole)
		cur_seg_start = cur_seg['start'] if cur_seg else -1
		prev_dark = getattr(self, '_chart_params', {}).get('dark')
		prev_seg = getattr(self, '_chart_seg_start', -2)
		needs_rebuild = (prev_dark != dark or prev_seg != cur_seg_start
		                 or not hasattr(self, '_chart_cache')
		                 or self._final_check)  # 最终检查模式每次全量重建

		# ── times/speeds（最终检查模式每次从 rows 实时读取）──
		if self._final_check or not hasattr(self, '_chart_cache'):
			times = [r[0] for r in self._rows]
			speeds = [r[2] for r in self._rows]
			self._chart_cache = {'times': times, 'speeds': speeds}
		else:
			times = self._chart_cache['times']
			speeds = self._chart_cache['speeds']

		if needs_rebuild:
			ax.clear()
			self._chart_params = {'dark': dark, 'bg': bg, 'fg': fg}
			self._chart_seg_start = cur_seg_start
			self._chart_artists = {}

			# ── 全曲线背景散点（最终检查模式：已修正帧用 NaN 隐藏）──
			bg_gray = COLOR_LIGHT_GRAY if not dark else COLOR_LIGHTER_GRAY
			kwargs = {"color": bg_gray, "markersize": 1, "alpha": 0.5,
					  "zorder": 0, "rasterized": True}
			if self._final_check:
				kwargs["picker"] = True
				kwargs["pickradius"] = 5
				kwargs["markersize"] = 3
				# 已修正帧用 NaN 隐藏灰色背景点
				bg_speeds = [float('nan') if i in self._corrections else s
							 for i, s in enumerate(speeds)]
			else:
				bg_speeds = speeds
			self._chart_artists['bg'] = ax.plot(
				times, bg_speeds, ".", **kwargs)

			# ── 当前选中帧红色高亮 ──
			cur_fi = self._current_frame
			if 0 <= cur_fi < len(times):
				cur_v = self._rows[cur_fi][2]
				if cur_v >= 0:
					self._chart_artists['cur_highlight'] = ax.scatter(
						[times[cur_fi]], [cur_v], c=COLOR_RED, s=12,
						zorder=5, edgecolors='white', linewidths=0.5, marker='o')
				else:
					self._chart_artists['cur_highlight'] = None
			else:
				self._chart_artists['cur_highlight'] = None

			# ── 当前段背景高亮 ──
			self._chart_artists['vspan'] = None
			if cur_seg:
				s, e = cur_seg['start'], cur_seg['end']
				self._chart_artists['vspan'] = ax.axvspan(
					times[s], times[min(e, len(times) - 1)],
					facecolor=COLOR_ORANGE, alpha=0.08, zorder=0)

			# ── 各问题段散点（按段索引存储）──
			self._chart_artists['segments'] = {}
			for seg in self._segments:
				s, e = seg['start'], seg['end']
				seg_t = times[s:e+1]; seg_v = speeds[s:e+1]
				is_cur = cur_seg and seg['start'] == cur_seg['start']
				seg_corrected = self._seg_corrected_count(seg)
				if seg_corrected >= max(1, seg['count'] // 2):
					color, sz, alpha, zo = COLOR_GREEN, 3, 0.6, 1
				elif is_cur:
					color, sz, alpha, zo = COLOR_ORANGE, 12, 1.0, 4
				else:
					color, sz, alpha, zo = COLOR_RED, 3, 0.7, 2
				artist = ax.scatter(seg_t, seg_v, c=color, s=sz,
				                    alpha=alpha, zorder=zo, linewidths=0)
				self._chart_artists['segments'][seg['start']] = artist

			# ── 已修正帧散点（可能为空）──
			cx, cy = self._get_correction_xy(times)
			self._chart_artists['corrections'] = ax.scatter(
				cx, cy, c=COLOR_BLUE, s=12, zorder=5, marker='o',
				edgecolors='white', linewidths=0.5) if cx else None

			# ── 轴样式（重建时设置）──
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

		else:
			# ── 增量更新：仅更新段颜色 + 修正帧位置 ──
			artists_seg = self._chart_artists.get('segments', {})
			for seg in self._segments:
				artist = artists_seg.get(seg['start'])
				if artist is None:
					continue
				s, e = seg['start'], seg['end']
				is_cur = cur_seg and seg['start'] == cur_seg['start']
				seg_corrected = self._seg_corrected_count(seg)
				if seg_corrected >= max(1, seg['count'] // 2):
					color, sz, alpha, zo = COLOR_GREEN, 3, 0.6, 1
				elif is_cur:
					color, sz, alpha, zo = COLOR_ORANGE, 12, 1.0, 4
				else:
					color, sz, alpha, zo = COLOR_RED, 3, 0.7, 2
				artist.set_facecolor(color)
				artist.set_sizes([sz] * (e - s + 1))
				artist.set_alpha(alpha)
				artist.set_zorder(zo)

			# 更新修正帧散点
			cx, cy = self._get_correction_xy(times)
			corr_artist = self._chart_artists.get('corrections')
			if cx:
				if corr_artist is None:
					self._chart_artists['corrections'] = ax.scatter(
						cx, cy, c=COLOR_BLUE, s=12, zorder=5, marker='o',
						edgecolors='white', linewidths=0.5)
				else:
					corr_artist.set_offsets(np.column_stack([cx, cy]) if cx
					                        else np.empty((0, 2)))
			elif corr_artist is not None:
				corr_artist.set_offsets(np.empty((0, 2)))

		# ── 恢复用户缩放 ──
		if user_zoomed:
			ax.set_xlim(saved_xlim)
			ax.set_ylim(saved_ylim)

		self._canvas.draw_idle()

	def _seg_corrected_count(self, seg: dict) -> int:
		"""统计段内已修正帧数。"""
		return sum(1 for fi in range(seg['start'], seg['end'] + 1)
		           if fi in self._corrections)

	def _get_correction_xy(self, times: list[float]
	                        ) -> tuple[list[float], list[float]]:
		"""从 _corrections 提取修正帧的 (x, y) 坐标。"""
		cx = [times[fi] for fi in self._corrections if fi < len(times)]
		cy = [self._corrections[fi] for fi in self._corrections if fi < len(times)]
		return cx, cy

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

	def eventFilter(self, obj, event) -> bool:
		"""拦截 spinbox 的左右箭头键，重定向到 dialog 用于帧导航。"""
		from PySide6.QtCore import QEvent
		if event.type() == QEvent.Type.KeyPress:
			if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
				self.keyPressEvent(event)
				return True  # 事件已处理，不再传递给 spinbox
		return super().eventFilter(obj, event)

	def keyPressEvent(self, event) -> None:
		"""← → 键在当前段内逐帧导航。最终检查模式：全范围导航。"""
		if self._final_check:
			n = len(self._rows)
			cur = self._current_frame
			if event.key() == Qt.Key.Key_Left and cur > 0:
				self._navigate_to(cur - 1)
			elif event.key() == Qt.Key.Key_Right and cur < n - 1:
				self._navigate_to(cur + 1)
			else:
				super().keyPressEvent(event)
			return

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

	def _on_pick(self, event) -> None:
		"""matplotlib pick_event：点击数据点 → 导航到该帧。"""
		ind = event.ind
		if ind is None or len(ind) == 0:
			return
		self._navigate_to(ind[0])

	@staticmethod
	def _speed_display(val: float) -> str:
		"""格式化速度显示：-1 → 空字符串，正常值 → 整数。"""
		return "" if val < 0 else str(int(val))

	@staticmethod
	def _speed_label(val: float) -> str:
		"""格式化速度标签：-1 → '失败'，正常值 → '{val:.0f}km/h'。"""
		return "失败" if val < 0 else f"{val:.0f}km/h"

	def _speed_input_value(self, fi: int) -> int:
		"""获取指定帧应在 spinbox 中显示的整数值。"""
		if fi in self._corrections:
			return int(self._corrections[fi])
		v = self._rows[fi][2]
		return int(v) if v >= 0 else 0

	def _navigate_to(self, fi: int) -> None:
		"""导航到指定帧并更新控件。"""
		# 离开当前帧时恢复未确认的预览值
		prev = self._current_frame
		backup = getattr(self, '_preview_backup', {})
		if prev != fi and prev in backup and prev not in self._corrections:
			self._rows[prev][2] = backup.pop(prev)
		self._current_frame = fi
		self._frame_label.setText(f"#{fi}")
		self._show_frame_image(fi)
		# blockSignals 避免 setValue 触发 _on_spinbox_changed
		self._speed_edit.blockSignals(True)
		self._speed_edit.setValue(self._speed_input_value(fi))
		self._speed_edit.blockSignals(False)
		# 更新速度标签
		v = self._rows[fi][2]
		self._speed_value_label.setText(f"速度: (无效)" if v < 0 else f"速度: {v:.0f} km/h")
		# 部分修正文本
		if fi in self._partial_corrections:
			self._partial_edit.setText(self._partial_corrections[fi])
		else:
			self._partial_edit.clear()
		self._btn_delete.setEnabled(
			fi in self._corrections or fi in self._partial_corrections)
		self._redraw_chart()

	def _on_spinbox_changed(self, value: int) -> None:
		"""SpinBox 值变化时实时预览：临时更新行数据并重绘图表。"""
		fi = self._current_frame
		if fi < 0 or fi >= len(self._rows):
			return
		# 首次修改时备份原始值
		if not hasattr(self, '_preview_backup'):
			self._preview_backup: dict[int, float] = {}
		if fi not in self._preview_backup and fi not in self._corrections:
			self._preview_backup[fi] = self._rows[fi][2]
		# 临时覆盖当前帧速度用于预览
		self._rows[fi][2] = float(value)
		self._redraw_chart()
		self._speed_value_label.setText(f"速度: {value:.0f} km/h (预览)")

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
			btn = PushButton(f"#{fi} ({self._speed_label(v)})")
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
		self._speed_edit.setValue(self._speed_input_value(target_frame))
		if target_frame in self._partial_corrections:
			self._partial_edit.setText(self._partial_corrections[target_frame])
		else:
			self._partial_edit.clear()
		self._btn_delete.setEnabled(
			target_frame in self._corrections or target_frame in self._partial_corrections)
		self._redraw_chart()

	def _quick_correct(self, fi: int, val: float) -> None:
		"""点击建议帧按钮：仅导航到该帧，不自动提交修正。"""
		self._navigate_to(fi)
		self._redraw_chart()

	def _check_accel(self, fi: int, v: float) -> tuple[bool, str]:
		"""使用 LCS 评分检查输入值 v 与邻居帧的物理一致性。

		Returns: (is_ok, warning_message)
		"""
		from ocr_engine import _lcs_score_for_value
		times = [r[0] for r in self._rows]
		score = _lcs_score_for_value(fi, v, self._rows, times,
		                              self._max_speed, self._max_accel)
		if score < 0.5:
			msg = (f"帧 #{fi} 输入值 {v:.0f} km/h 与周围邻居物理不一致\n\n"
			       f"LCS 一致性分数: {score:.2f} / 1.00\n"
			       f"该值在 {self._max_accel:.0f} m/s² 约束下与邻近帧矛盾。\n\n"
			       f"确定要使用此值吗？")
			return False, msg
		return True, ""

	def _add_correction(self) -> None:
		fi = getattr(self, "_current_frame", 0)
		# 优先检查部分修正输入
		part_text = self._partial_edit.text().strip()
		if part_text and 'x' in part_text.lower():
			self._partial_corrections[fi] = part_text
			self._corrections.pop(fi, None)
			self._redraw_chart()
			for btn in self._suggested_btns:
				try:
					f = btn.property("frame_idx")
				except Exception:
					continue
				if f == fi:
					btn.setText(f"#{fi} ({part_text})")
					break
			self._show_frame_image(fi)
			self._btn_delete.setEnabled(True)
			return

		v = float(self._speed_edit.value())
		# 物理一致性检查
		ok, warning = self._check_accel(fi, v)
		if not ok:
			reply = QMessageBox.warning(self, "加速度异常", warning,
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
				QMessageBox.StandardButton.No)
			if reply == QMessageBox.StandardButton.No:
				return
		self._corrections[fi] = v
		self._partial_corrections.pop(fi, None)
		self._partial_edit.clear()
		# 确认修正后清除预览备份
		getattr(self, '_preview_backup', {}).pop(fi, None)
		label = self._speed_label(v)
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
		self._partial_edit.clear()
		self._speed_edit.setValue(self._speed_input_value(fi))
		for btn in self._suggested_btns:
			try:
				f = btn.property("frame_idx")
			except Exception:
				continue
			if f == fi:
				btn.setText(f"#{fi} ({self._speed_label(orig)})")
				break
		self._redraw_chart()
		self._btn_delete.setEnabled(False)
		self._show_frame_image(fi)

	def _finish(self) -> None:
		"""完成审核：无修正时提示（最终检查模式跳过），确认后接受。"""
		if not self._final_check and not self._corrections and not self._partial_corrections:
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
