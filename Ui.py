# -*- coding: utf-8 -*-
"""
Centroid Finder のメイン UI ウィンドウ実装。

主な機能:
- 画像の読み込みと表示
- 重心検出パラメータの調整
- 参照点の設定とフィッティング
- テーブル表示と編集

依存関係:
- tables.py: テーブル操作
- interactions.py: マウス/キーボード操作
- rendering.py: 画像描画
- CalcCentroid.py: 重心計算
- Util.py: ユーティリティ
- Strings.py: UI 文字列定数
"""

import qt_compat
from qt_compat.QtWidgets import (
    QSlider, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget,
    QFileDialog, QStyle, QSizePolicy, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QScrollArea, QApplication, QMenu, QComboBox
)
from qt_compat.QtWidgets import QButtonGroup
from qt_compat.QtCore import Qt, QTimer, QObject, QEvent, QRect, QPoint, pyqtSignal, QThread
from qt_compat.QtGui import QPixmap, QFont, QCursor, QPainter, QPen, QColor, QPalette

from Util import cvimg_to_qpixmap, kmeans_posterize
from CalcCentroid import CentroidProcessor
from Config import PROC_TARGET_WIDTH, save_last_image_path, load_last_image_path, DEBUG

import numpy as np
import cv2
from datetime import datetime
from collections import deque
from time import monotonic
from widgets import ClickableSlider, RefTableDelegate
from rendering import build_zoomed_canvas, draw_crosshair
from tables import populate_tables, fix_tables_height
from interactions import ImageViewController
import unicodedata
import Strings as STR
import os
import math
import ctypes
from ctypes import wintypes


class SegmentControl(QWidget):
    """Simple segmented control: horizontal checkable buttons in an exclusive group.

    Usage: sc = SegmentControl(["A","B"], checked_index=0, btn_w=64, btn_h=24)
    Connect change via `sc.set_on_changed(callback)` where callback(index:int).
    """
    def __init__(self, labels, parent=None, checked_index=0, btn_w=64, btn_h=24, blue="#757575"):
        super().__init__(parent)
        try:
            from qt_compat.QtWidgets import QPushButton, QHBoxLayout, QButtonGroup
        except Exception:
            from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QButtonGroup

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._buttons = []
        self._group = QButtonGroup(self)
        try:
            self._group.setExclusive(True)
        except Exception:
            pass

        qss_base = (
            # Force square corners by default; we'll round only the outer corners explicitly.
            "QPushButton { border: 1px solid lightgray; padding: 2px 10px; border-radius: 0px; }"
            "QPushButton:checked { background-color: " + blue + "; color: white; }"
            "QPushButton:!checked { background-color: white; color: black; }"
        )

        for i, lbl in enumerate(labels):
            b = QPushButton(str(lbl))
            try:
                b.setCheckable(True)
            except Exception:
                pass
            try:
                b.setFixedSize(btn_w, btn_h)
            except Exception:
                pass
            # Set normal (non-bold) font
            try:
                f = b.font()
                f.setBold(False)
                b.setFont(f)
            except Exception:
                pass
            # apply corner styling depending on position
            if i == 0:
                # Left-most: round only the outer-left corners; keep inner-right corners square.
                b.setStyleSheet(
                    qss_base
                    + "QPushButton { border-top-left-radius: 10px; border-bottom-left-radius: 10px; border-top-right-radius: 0px; border-bottom-right-radius: 0px; border-right: none; font-weight: normal; }"
                )
            elif i == len(labels) - 1:
                # Right-most: round only the outer-right corners; keep inner-left corners square.
                b.setStyleSheet(
                    qss_base
                    + "QPushButton { border-top-right-radius: 10px; border-bottom-right-radius: 10px; border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: none; font-weight: normal; }"
                )
            else:
                # Middle segments: all corners square.
                b.setStyleSheet(qss_base + "QPushButton { border-radius: 0px; border-left: none; border-right: none; font-weight: normal; }")
            layout.addWidget(b)
            self._group.addButton(b, i)
            self._buttons.append(b)

        self._callback = None
        try:
            self._group.buttonClicked[int].connect(self._on_clicked)
        except Exception:
            try:
                self._group.buttonClicked.connect(self._on_clicked)
            except Exception:
                pass

        if 0 <= checked_index < len(self._buttons):
            try:
                self._buttons[checked_index].setChecked(True)
            except Exception:
                pass

    def _on_clicked(self, idx):
        try:
                if callable(self._callback):
                    # idx may be an int (QButtonGroup.buttonClicked[int]) or a QAbstractButton (PySide)
                    try:
                        # integer id
                        i = int(idx)
                    except Exception:
                        try:
                            # assume idx is the button object; find its index
                            i = self._buttons.index(idx)
                        except Exception:
                            try:
                                # try QButtonGroup id lookup
                                i = self._group.id(idx)
                            except Exception:
                                i = -1
                    try:
                        if i >= 0:
                            self._callback(int(i))
                    except Exception:
                        pass
        except Exception:
            pass

    def set_on_changed(self, cb):
        self._callback = cb

    def setCheckedIndex(self, idx: int):
        try:
            if 0 <= idx < len(self._buttons):
                self._buttons[idx].setChecked(True)
        except Exception:
            pass

    def checkedIndex(self):
        try:
            for i, b in enumerate(self._buttons):
                try:
                    if b.isChecked():
                        return i
                except Exception:
                    continue
        except Exception:
            pass
        return -1


class AreaHistogramWidget(QWidget):
    """軽量な面積ヒストグラム描画ウィジェット（Qtペイント、曲線接続、ログ軸）。"""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bins = None
        self._vals = None
        self._counts = None  # 粒子数
        self._sel_min = None
        self._sel_max = None
        self._dragging = None  # 'min'|'max'|None
        self._user_set_selection = False
        self._autoset_done = False
        try:
            self.setMinimumHeight(180)
        except Exception:
            pass

    def set_data(self, bins, vals, counts=None):
        self._bins = bins
        self._vals = vals
        self._counts = counts
        # Initialize selection to full range if not set yet.
        try:
            if self._bins and len(self._bins) >= 2:
                b0 = float(self._bins[0]); b1 = float(self._bins[-1])
                if self._sel_min is None:
                    self._sel_min = b0
                if self._sel_max is None:
                    self._sel_max = b1
                # clamp
                self._sel_min = max(b0, min(b1, float(self._sel_min)))
                self._sel_max = max(b0, min(b1, float(self._sel_max)))
                if self._sel_min > self._sel_max:
                    self._sel_min, self._sel_max = self._sel_max, self._sel_min
        except Exception:
            pass
        try:
            self.update()
        except Exception:
            pass

    def clear(self):
        self._bins = None
        self._vals = None
        self._counts = None
        self._sel_min = None
        self._sel_max = None
        self._dragging = None
        try:
            self.update()
        except Exception:
            pass

    def set_selection(self, sel_min, sel_max):
        """Set selection in area units (pixel^2). sel_max can be None to mean current max."""
        self._user_set_selection = True
        self._sel_min = None if sel_min is None else float(sel_min)
        self._sel_max = None if sel_max is None else float(sel_max)
        try:
            if self._bins and len(self._bins) >= 2:
                b0 = float(self._bins[0]); b1 = float(self._bins[-1])
                if self._sel_min is None:
                    self._sel_min = b0
                if self._sel_max is None:
                    self._sel_max = b1
                self._sel_min = max(b0, min(b1, float(self._sel_min)))
                self._sel_max = max(b0, min(b1, float(self._sel_max)))
                if self._sel_min > self._sel_max:
                    self._sel_min, self._sel_max = self._sel_max, self._sel_min
        except Exception:
            pass
        try:
            self.update()
        except Exception:
            pass

    def selection(self):
        return self._sel_min, self._sel_max

    def maybe_autoset_selection(self, sel_min, sel_max):
        """Auto-set initial selection once (startup) unless user already adjusted it."""
        try:
            if bool(getattr(self, '_user_set_selection', False)):
                return
        except Exception:
            return
        try:
            if bool(getattr(self, '_autoset_done', False)):
                return
        except Exception:
            return
        try:
            self._sel_min = None if sel_min is None else float(sel_min)
            self._sel_max = None if sel_max is None else float(sel_max)
            if self._bins and len(self._bins) >= 2:
                b0 = float(self._bins[0]); b1 = float(self._bins[-1])
                if self._sel_min is None:
                    self._sel_min = b0
                if self._sel_max is None:
                    self._sel_max = b1
                self._sel_min = max(b0, min(b1, float(self._sel_min)))
                self._sel_max = max(b0, min(b1, float(self._sel_max)))
                if self._sel_min > self._sel_max:
                    self._sel_min, self._sel_max = self._sel_max, self._sel_min
        except Exception:
            pass
        try:
            self._autoset_done = True
        except Exception:
            pass
        try:
            self.update()
        except Exception:
            pass

    def _plot_geom(self):
        w = self.width(); h = self.height()
        # Keep extra bottom space for x-axis label without overlapping tick labels.
        # margin_t includes room for the title.
        margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 50
        rect_w = max(10, w - margin_l - margin_r)
        rect_h = max(10, h - margin_t - margin_b)
        x0 = margin_l; y0 = margin_t + rect_h
        return x0, y0, rect_w, rect_h, margin_t

    def _x_to_area(self, x):
        if not self._bins or len(self._bins) < 2:
            return None
        import math
        x0, _y0, rect_w, _rect_h, _mt = self._plot_geom()
        t = (float(x) - float(x0)) / float(max(1.0, rect_w))
        t = max(0.0, min(1.0, t))
        b0 = float(self._bins[0]); b1 = float(self._bins[-1])
        if b0 <= 0 or b1 <= 0:
            return None
        lv = math.log(b0) + t * (math.log(b1) - math.log(b0))
        return float(math.exp(lv))

    def _area_to_x(self, area):
        if not self._bins or len(self._bins) < 2:
            return None
        import math
        x0, _y0, rect_w, _rect_h, _mt = self._plot_geom()
        b0 = float(self._bins[0]); b1 = float(self._bins[-1])
        if area is None or area <= 0 or b0 <= 0 or b1 <= 0:
            return x0
        t = (math.log(float(area)) - math.log(b0)) / (math.log(b1) - math.log(b0) + 1e-9)
        t = max(0.0, min(1.0, t))
        return x0 + rect_w * t

    def mousePressEvent(self, event):
        try:
            if event.button() != Qt.LeftButton:
                return
        except Exception:
            return
        if not self._bins or len(self._bins) < 2:
            return
        try:
            self._user_set_selection = True
        except Exception:
            pass
        x = event.position().x() if hasattr(event, 'position') else event.x()
        try:
            xmin = self._area_to_x(self._sel_min)
            xmax = self._area_to_x(self._sel_max)
            if abs(float(x) - float(xmin)) <= abs(float(x) - float(xmax)):
                self._dragging = 'min'
            else:
                self._dragging = 'max'
        except Exception:
            self._dragging = 'min'

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        if not self._bins or len(self._bins) < 2:
            return
        x = event.position().x() if hasattr(event, 'position') else event.x()
        v = self._x_to_area(x)
        if v is None:
            return
        b0 = float(self._bins[0]); b1 = float(self._bins[-1])
        v = max(b0, min(b1, float(v)))
        if self._dragging == 'min':
            self._sel_min = v
            if self._sel_max is not None and self._sel_min > self._sel_max:
                self._sel_max = self._sel_min
        else:
            self._sel_max = v
            if self._sel_min is not None and self._sel_max < self._sel_min:
                self._sel_min = self._sel_max
        try:
            self.rangeChanged.emit(float(self._sel_min or b0), float(self._sel_max or b1))
        except Exception:
            pass
        try:
            self.update()
        except Exception:
            pass

    def mouseReleaseEvent(self, event):
        self._dragging = None

    def paintEvent(self, event):
        from qt_compat.QtGui import QPainterPath
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = self.width(); h = self.height()
        margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 50
        rect_w = max(10, w - margin_l - margin_r)
        rect_h = max(10, h - margin_t - margin_b)
        # Background: follow surrounding palette (avoid hard-coded white)
        try:
            bg = None
            try:
                p = self.parentWidget()
                if p is not None:
                    bg = p.palette().color(QPalette.Window)
            except Exception:
                bg = None
            if bg is None:
                bg = QApplication.palette().color(QPalette.Window)
            painter.fillRect(self.rect(), bg)
        except Exception:
            pass

        tick_color = QColor("#444")
        axis_pen = QPen(QColor("#000")); axis_pen.setWidth(2)
        painter.setPen(axis_pen)
        x0 = margin_l; y0 = margin_t + rect_h
        painter.drawLine(x0, margin_t, x0, y0)
        painter.drawLine(x0, y0, x0 + rect_w, y0)

        # Right-side axis (for area curve)
        try:
            painter.setPen(axis_pen)
            painter.drawLine(x0 + rect_w, margin_t, x0 + rect_w, y0)
        except Exception:
            pass

        # Title
        try:
            painter.setPen(QPen(QColor("#000")))
            painter.setFont(self.font())
            # Align title with other left-column labels (e.g., 'Number of Groups').
            painter.drawText(QRect(0, 0, max(10, w), max(10, margin_t - 6)), Qt.AlignLeft | Qt.AlignVCenter, "Grain Size Threshold (pix)")
        except Exception:
            pass

        if not self._bins or not self._vals:
            return
        bins = self._bins; vals = self._vals
        counts = self._counts
        if len(bins) < 2 or len(vals) == 0:
            return
        
        import math
        
        # Get range for area (red curve)
        try:
            vmax = max([v for v in vals if v > 0] + [1.0])
            vmin = min([v for v in vals if v > 0] + [vmax])
        except Exception:
            vmax = 1.0
            vmin = 1.0
        if vmax <= 0 or vmin <= 0:
            return
            
        # Get range for counts (gray curve) if available
        cmax = cmin = None
        if counts:
            try:
                cmax = max([c for c in counts if c > 0] + [1.0])
                cmin = min([c for c in counts if c > 0] + [cmax])
            except Exception:
                cmax = cmin = None
        
        def _xpos(v):
            try:
                return x0 + rect_w * ((math.log(v) - math.log(bins[0])) / (math.log(bins[-1]) - math.log(bins[0]) + 1e-9))
            except Exception:
                return x0
        
        def _ypos_area(v):
            try:
                if v <= 0:
                    return y0
                return y0 - rect_h * ((math.log(v) - math.log(vmin)) / (math.log(vmax) - math.log(vmin) + 1e-9))
            except Exception:
                return y0
        
        def _ypos_count(v):
            try:
                if v <= 0 or cmin is None or cmax is None:
                    return y0
                return y0 - rect_h * ((math.log(v) - math.log(cmin)) / (math.log(cmax) - math.log(cmin) + 1e-9))
            except Exception:
                return y0

        # Fill selection range (Min..Max) with light gray
        try:
            sel_min, sel_max = self._sel_min, self._sel_max
            if sel_min is not None and sel_max is not None:
                xmn = float(self._area_to_x(sel_min))
                xmx = float(self._area_to_x(sel_max))
                if xmn > xmx:
                    xmn, xmx = xmx, xmn
                xmn = max(float(x0), min(float(x0 + rect_w), xmn))
                xmx = max(float(x0), min(float(x0 + rect_w), xmx))
                fill_col = QColor("lightgray")
                try:
                    fill_col.setAlpha(60)
                except Exception:
                    pass
                painter.fillRect(QRect(int(xmn), int(margin_t), int(max(1.0, xmx - xmn)), int(rect_h)), fill_col)
        except Exception:
            pass

        # Draw count curve (gray) on log-log scale
        if counts and cmax and cmin:
            count_pen = QPen(QColor("#888")); count_pen.setWidth(2)
            painter.setPen(count_pen)
            path_count = QPainterPath()
            first_point = True
            for i in range(len(counts)):
                if counts[i] <= 0:
                    continue
                if i + 1 >= len(bins):
                    break
                bc = math.sqrt(bins[i] * bins[i+1])
                px = _xpos(bc)
                py = _ypos_count(counts[i])
                if first_point:
                    path_count.moveTo(px, py)
                    first_point = False
                else:
                    path_count.lineTo(px, py)
            painter.drawPath(path_count)

        # Draw area curve (red) on log-log scale
        curve_pen = QPen(QColor("#bb2a2a")); curve_pen.setWidth(2)
        painter.setPen(curve_pen)
        path = QPainterPath()
        first_point = True
        for i in range(len(vals)):
            if vals[i] <= 0:
                continue
            if i + 1 >= len(bins):
                break
            bc = math.sqrt(bins[i] * bins[i+1])
            px = _xpos(bc)
            py = _ypos_area(vals[i])
            if first_point:
                path.moveTo(px, py)
                first_point = False
            else:
                path.lineTo(px, py)
        painter.drawPath(path)

        # Draw selection (min/max) as vertical dashed lines
        try:
            sel_min, sel_max = self._sel_min, self._sel_max
            if sel_min is not None and sel_max is not None:
                sel_pen = QPen(QColor("#888")); sel_pen.setWidth(1)
                try:
                    sel_pen.setStyle(Qt.DashLine)
                except Exception:
                    pass
                painter.setPen(sel_pen)
                xmn = self._area_to_x(sel_min)
                xmx = self._area_to_x(sel_max)
                painter.drawLine(int(xmn), margin_t, int(xmn), y0)
                painter.drawLine(int(xmx), margin_t, int(xmx), y0)
                # small labels
                painter.setPen(QPen(QColor("#666")))
                painter.drawText(int(xmn) - 12, margin_t + 12, "Min")
                painter.drawText(int(xmx) - 12, margin_t + 12, "Max")
        except Exception:
            pass

        # Axis labels with nice round numbers
        try:
            painter.setPen(QPen(tick_color))
            
            # X-axis: nice round numbers
            def _nice_log_labels(vmin, vmax, num_labels=3):
                """Generate nice round numbers for log scale"""
                labels = []
                log_min = math.log10(vmin)
                log_max = math.log10(vmax)
                magnitude_min = math.floor(log_min)
                magnitude_max = math.ceil(log_max)
                for mag in range(int(magnitude_min), int(magnitude_max) + 1):
                    val = 10 ** mag
                    if vmin <= val <= vmax:
                        labels.append(val)
                return labels
            
            x_labels = _nice_log_labels(bins[0], bins[-1])
            for xl in x_labels:
                px = _xpos(xl)
                if xl >= 1000:
                    label_text = f"{int(xl/1000)}k" if xl % 1000 == 0 else f"{int(xl)}"
                elif xl >= 1:
                    label_text = f"{int(xl)}"
                else:
                    label_text = f"{xl:.1f}"
                painter.drawText(int(px - 10), y0 + 18, label_text)
            
            # Y-axis numeric labels are intentionally omitted.

            # Axis labels: left=count (gray), right=area (red)
            try:
                painter.setPen(QPen(QColor("#888")))
                f = QFont(self.font());
                try:
                    f.setBold(True)
                except Exception:
                    pass
                painter.save()
                painter.setFont(f)
                # Move slightly right to avoid hugging the edge
                painter.translate(22, margin_t + rect_h / 2.0)
                painter.rotate(-90)
                painter.drawText(QRect(-rect_h // 2, -10, rect_h, 20), Qt.AlignHCenter | Qt.AlignVCenter, "Grain No.")
                painter.restore()
            except Exception:
                pass
            try:
                painter.setPen(QPen(QColor("#bb2a2a")))
                f = QFont(self.font());
                try:
                    f.setBold(True)
                except Exception:
                    pass
                painter.save()
                painter.setFont(f)
                painter.translate(x0 + rect_w + 14, margin_t + rect_h / 2.0)
                # Flip reading direction (180° from previous): use +90 instead of -90
                painter.rotate(90)
                painter.drawText(QRect(-rect_h // 2, -10, rect_h, 20), Qt.AlignHCenter | Qt.AlignVCenter, "Area")
                painter.restore()
            except Exception:
                pass

            # X-axis label: bold, same color as tick labels.
            painter.setPen(QPen(tick_color))
            try:
                fx = QFont(self.font()); fx.setBold(True)
                painter.setFont(fx)
            except Exception:
                pass
            painter.drawText(QRect(x0, y0 + 26, rect_w, 20), Qt.AlignHCenter | Qt.AlignVCenter, "Area (pix)")
        except Exception:
            pass

class TitleBar(QWidget):
    """Custom title bar: dark red background, app title and basic window buttons."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(36)
        self.setObjectName('titleBar')
        # ensure solid background using palette (avoid stylesheet inheritance issues)
        try:
            self.setAutoFillBackground(True)
            pal = self.palette()
            pal.setColor(QPalette.Window, QColor(160, 15, 15))
            self.setPalette(pal)
        except Exception:
            self.setStyleSheet('#titleBar { background-color: rgb(160,15,15); }')

        hl = QHBoxLayout(self)
        hl.setContentsMargins(8, 0, 0, 0)
        hl.setSpacing(0)
        self.label = QLabel(STR.APP_TITLE)
        self.label.setStyleSheet('color: white; font-weight: bold; font-family: "Segoe UI", sans-serif; font-size: 13px;')
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.label.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.label)
        hl.addStretch(1)

        # Minimize / Maximize / Close (small square buttons like Windows)
        self.btn_min = QPushButton('\u2212')
        self.btn_max = QPushButton('▢')
        self.btn_close = QPushButton('✕')
        # object names for targeted styling
        self.btn_min.setObjectName('titleBtnMin')
        self.btn_max.setObjectName('titleBtnMax')
        self.btn_close.setObjectName('titleBtnClose')

        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setFixedSize(34, 28)
            b.setStyleSheet('''
                QPushButton { color: white; background: transparent; border: none; }
                QPushButton:hover { background-color: rgba(255,255,255,0.08); }
            ''')

        # close button: red hover like native
        self.btn_close.setStyleSheet('''
            QPushButton { color: white; background: transparent; border: none; }
            QPushButton:hover { background-color: #E81123; }
        ''')

        hl.addWidget(self.btn_min)
        hl.addWidget(self.btn_max)
        hl.addWidget(self.btn_close)

        self.btn_close.clicked.connect(lambda: self.window().close())
        self.btn_min.clicked.connect(lambda: self.window().showMinimized())
        self.btn_max.clicked.connect(self._on_max_restore)

        self._drag_pos = None

    def mouseDoubleClickEvent(self, ev):
        # Double-click on the title bar toggles maximize/restore (like native)
        try:
            if ev.button() == Qt.LeftButton:
                self._on_max_restore()
        except Exception:
            pass

    def mousePressEvent(self, ev):
        # Re-implement left button drag and right-click system menu
        if ev.button() == Qt.RightButton:
            try:
                menu = self._build_system_menu()
                if menu is not None:
                    menu.exec_(ev.globalPos())
                    return
            except Exception:
                pass
        # fall back to original behavior for left-button drag
        super().mousePressEvent(ev)

    def _build_system_menu(self):
        try:
            from qt_compat.QtWidgets import QMenu
            m = QMenu(self)
            act_restore = m.addAction('Restore')
            act_min = m.addAction('Minimize')
            act_max = m.addAction('Maximize')
            m.addSeparator()
            act_close = m.addAction('Close')

            def on_trigger(a):
                w = self.window()
                if a == act_restore:
                    w.showNormal()
                elif a == act_min:
                    w.showMinimized()
                elif a == act_max:
                    w.showMaximized()
                elif a == act_close:
                    w.close()
            m.triggered.connect(on_trigger)
            return m
        except Exception:
            return None

    def update_max_icon(self):
        try:
            w = self.window()
            if w is not None and w.isMaximized():
                self.btn_max.setText('❐')
            else:
                self.btn_max.setText('▢')
        except Exception:
            pass

    def _on_max_restore(self):
        w = self.window()
        if w.isMaximized():
            w.showNormal()
            self.btn_max.setText('▢')
        else:
            w.showMaximized()
            self.btn_max.setText('❐')

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.globalPos() - self.window().frameGeometry().topLeft()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_pos and ev.buttons() & Qt.LeftButton:
            self.window().move(ev.globalPos() - self._drag_pos)
            ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag_pos = None

    def paintEvent(self, event):
        # Ensure title bar background is painted solid (avoid stylesheet inheritance issues)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(160, 15, 15))

    def paintEvent(self, event):
        """Force paint background to ensure color is applied."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(160, 15, 15))


class Footer(QWidget):
    """Custom footer: solid black background with status text."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(0, 0, 0))
        self.setPalette(pal)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        self.label = QLabel("")
        self.label.setStyleSheet("color: white; font-size: 11px;")
        layout.addWidget(self.label)
        layout.addStretch(1)

    def paintEvent(self, event):
        """Force paint background to ensure color is applied."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
    def showMessage(self, msg):
        self.label.setText(msg)


class RoundedWindow(QWidget):
    """Container widget that paints a rounded background and holds the app content."""
    def __init__(self, content_widget: QWidget, parent=None):
        super().__init__(parent)
        # Keep frameless (hide OS title bar) but do not use translucent background
        try:
            self.setAttribute(Qt.WA_TranslucentBackground, False)
        except Exception:
            pass
        self._content = content_widget
        self._init_ui()

    def _init_ui(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(0)
        # Title bar removed — use native window decorations instead
        # content area (already a QWidget)
        self._content.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self._content, 1)
        # Footer (black bar)
        footer = QWidget(self)
        footer.setFixedHeight(20)
        footer.setStyleSheet('background-color: black;')
        vl.addWidget(footer)

    def paintEvent(self, ev):
        # draw plain rectangular background (square corners)
        r = self.rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path_rect = QRect(r.x(), r.y(), r.width(), r.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(240, 240, 240))
        painter.drawRect(path_rect)

# 表示テーブル関連の実装は tables.py に移動

class CentroidFinderWindow(QMainWindow):
    """
    Centroid Finder のメインウィンドウクラス。

    画像処理と重心検出の GUI を提供します。
    参照点設定、フィッティング、テーブル表示を統合。
    """

    def __init__(self):
        super().__init__()
        # ウィンドウタイトル設定
        self.setWindowTitle(STR.APP_TITLE)

        # デバッグ出力ヘルパ
        def _dbg(msg):
            if DEBUG:
                try:
                    print(f"[DEBUG] {msg}", flush=True)
                    # Also write to file for persistent logging
                    with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                        f.write(f"[DEBUG] {msg}\n")
                except Exception:
                    pass
        self._dbg = _dbg

        # 画像関連変数
        self.img_full = None          # フル解像度画像 (numpy array)
        self.proc_img = None          # 処理用縮小画像
        self.scale_proc_to_full = 1.0 # 処理画像からフル画像へのスケール
        self.proc_target_width = PROC_TARGET_WIDTH  # 処理画像の目標幅

        # 重心処理関連
        self.centroid_processor = None  # CentroidProcessor インスタンス
        self.centroids = []            # 検出された重心リスト [(group_no, x, y), ...]
        self.selected_index = None     # 選択中の重心インデックス
        self.select_radius_display = 10.0  # 画像上の選択半径 (pix)

        # 参照点関連
        self.ref_points = [None] * 10  # 参照点リスト [(x_proc, y_proc) or None]
        self.ref_selected_index = 0     # 選択中の参照点インデックス
        self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]  # 参照点の観測値

        # UI 状態
        self.visible_ref_cols = 3      # 表示する参照点列数
        self.flip_mode = 'auto'        # 左右反転モード ('auto', 'normal', 'flip')
        self.overlay_mode = 'Original'  # Overlay display mode: Original / Posterized
        # Display labels: editable display strings separate from internal keys
        # Internal keys should be code-safe identifiers; change display text here.
        self.display_labels = {
            'overlay_ratio': 'Display Mode',
            'poster_level': 'Number of Groups',
            'min_area': 'Minimum Grain Area (pix)',
            'trim': 'Boundary Offset (pix)'
        }
        self.levels_value = 4          # PosterLevel の内部値
        self.show_boundaries = True    # 境界線表示フラグ
        self.view_orientation = 'Image'  # View Orientation (Image/Stage)

        # 画像表示関連
        # 仮想キャンバス関連: 実際の表示はビューポート分のみだが、スクロール範囲は仮想的に拡張する
        self._virtual_canvas_size = (0, 0)  # 仮想キャンバス幅,高さ (pix)
        # パッチ生成の安全弁 (パッチのピクセル数上限)
        self.MAX_PATCH_PIXELS = 4096 * 4096  # 大きなパッチ作成を防ぐ

        self._img_base_size = None     # ベース画像サイズ (w, h)
        self.proc_zoom = 1.0           # 処理画像のズーム倍率
        self.view_padding = 200        # 表示パディング
        self._display_offset = (0, 0)  # 表示オフセット
        self._display_img_size = (0, 0) # 表示画像サイズ
        self._display_pm_base = None   # ベース Pixmap
        self._initial_center_done = False  # 初期センタリング完了フラグ
        self._last_stage_info = None   # 最新のステージ座標変換情報（グリッド再利用用）
        self._last_pm_image_grid = None  # Imageモード用に最後に描いたグリッド付きPixmapを保持

        # パン/フリック関連
        self._mouse_pressed = False
        self._dragging = False
        self._drag_start_vp = None
        self._drag_start_scroll = (0, 0)
        self._drag_recent = deque(maxlen=8)  # 最近のドラッグ位置
        self._kinetic_timer = QTimer(self)
        self._kinetic_timer.setInterval(16)
        self._kinetic_timer.timeout.connect(self._on_kinetic_tick)
        self._kinetic_vx = 0.0  # 慣性速度 X
        self._kinetic_vy = 0.0  # 慣性速度 Y
        self._kinetic_last_t = 0.0

        # キャッシュ: パラメータ変更時の再計算を避ける
        self._cache = {
            "img_id": None,      # 画像 ID (id(proc_img))
            "levels": None,      # PosterLevel
            "min_area": None,    # Min Area
            "trim_px": None,     # Trim (pix)
            "poster": None,      # ポスタライズ画像
            "centroids": None,   # 重心リスト
        }

        # 更新タイマー (UI 更新を遅延)
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(35)  # 35ms 遅延
        self.update_timer.timeout.connect(self._update_image_actual)
        self._painting = False  # 描画中フラグ

        # 自動デバッグ: 初回更新後に自動終了するかどうか
        self._auto_exit_after_update = False

        # 画像表示ラベル (中央揃え)
        self.img_label_proc = QLabel(alignment=Qt.AlignCenter)
        self.img_label_proc.setMouseTracking(True)  # マウス追跡有効

        # 画像用スクロールエリア (ズーム/パン対応)
        self.proc_scroll = QScrollArea()
        self.proc_scroll.setWidgetResizable(False)
        # Use top-left alignment so label coordinates map directly to scroll values.
        # Centering the widget inside the viewport caused mapping offsets when zooming.
        self.proc_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.proc_scroll.setWidget(self.img_label_proc)
        self.proc_scroll.viewport().setMouseTracking(True)


        # マウス/キーボード操作コントローラ
        self.interactions = ImageViewController(self)

        # Ensure patch worker threads are cleaned up on app exit
        try:
            app = QApplication.instance()
            if app is not None:
                try:
                    app.aboutToQuit.connect(self._cleanup_threads)
                except Exception:
                    pass
        except Exception:
            pass

        # 参照点テーブル (左側: 最大10列、表示列は可変)
        self.table_ref = QTableWidget(0, 10)  # 行0、列10 (内部容量)
        # 重心テーブル (右側: 列数は動的)
        self.table = QTableWidget(0, 0)
        # 下部テーブルはウィンドウのリサイズで高さを変えたくないため固定高さにする
        try:
            # Do not hardcode a default here; compute exact height for 5 rows
            # after the tables have been populated so it matches the actual
            # font/DPI and header sizes. Initialize attribute to None.
            self.FIXED_TABLE_HEIGHT = None
        except Exception:
            pass

        # 表示する参照点列数 (起動時は3列)
        self.visible_ref_cols = 3

        # 左テーブル垂直ヘッダ設定 (行ラベル表示、太字、右揃え)
        self.table_ref.verticalHeader().setVisible(True)
        vf = self.table_ref.verticalHeader().font()
        vf.setBold(True)
        self.table_ref.verticalHeader().setFont(vf)
        try:
            self.table_ref.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        except Exception:
            pass  # 互換性確保

        # 右テーブル垂直ヘッダ設定 (行ラベル表示、太字、右揃え)
        self.table.verticalHeader().setVisible(True)
        vf2 = self.table.verticalHeader().font()
        vf2.setBold(True)
        self.table.verticalHeader().setFont(vf2)
        try:
            self.table.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        except Exception:
            pass

        # 水平ヘッダ設定 (両テーブルとも太字、中央揃え)
        hf_ref = self.table_ref.horizontalHeader().font()
        hf_ref.setBold(True)
        self.table_ref.horizontalHeader().setFont(hf_ref)
        hf = self.table.horizontalHeader().font()
        hf.setBold(True)
        self.table.horizontalHeader().setFont(hf)
        try:
            self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        except Exception:
            pass

        # Setup pseudo-headers in table_ref and table for 2-row header appearance
        try:
            # Ensure minimum rows for headers
            if self.table_ref.rowCount() < 2:
                self.table_ref.setRowCount(2)
            if self.table.rowCount() < 2:
                self.table.setRowCount(2)
            
            # Apply pseudo-headers
            self._setup_pseudo_headers_ref(self.table_ref)
            self._setup_pseudo_headers_between(self.table)
            
            # Enforce row heights for headers and data rows
            try:
                vh_ref = self.table_ref.verticalHeader()
                vh_ref.setSectionResizeMode(QHeaderView.Fixed)
                vh_ref.setDefaultSectionSize(24)
                self.table_ref.setRowHeight(0, 24)
                self.table_ref.setRowHeight(1, 20)
            except Exception:
                pass

            try:
                vh_table = self.table.verticalHeader()
                vh_table.setSectionResizeMode(QHeaderView.Fixed)
                vh_table.setDefaultSectionSize(24)
                self.table.setRowHeight(0, 24)
                self.table.setRowHeight(1, 20)
            except Exception:
                pass
        except Exception:
            pass
            
        # Startup tasks with QTimer to allow layout to settle
        try:
            # Store initial button sizes
            try:
                self._action_btn_base_w = 100  # Default base width
                self._action_btn_base_h = 56   # Default base height (increased for better visibility)
                # Try to measure from Add button if available
                add_btn = getattr(self, 'btn_add_ref', None)
                if add_btn is not None:
                    try:
                        w = int(add_btn.width() or 0)
                        h = int(add_btn.height() or 0)
                        if w > 0:
                            self._action_btn_base_w = max(100, w)
                        if h > 0:
                            self._action_btn_base_h = max(56, h)
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Apply styles and enforce heights
            QTimer.singleShot(0, self._apply_button_styles)
            QTimer.singleShot(50, self._enforce_button_heights)
            
            # Create frozen headers on startup
            QTimer.singleShot(100, self._create_frozen_header_tables)
            # Adjust column widths for Z column and others
            QTimer.singleShot(150, self._adjust_table_column_widths)
        except Exception:
            pass
        # 編集トリガー設定
        # 右テーブル: 編集不可
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        # 左テーブル: ユーザー操作時のみ編集 (Obs.* 行のみ有効)
        try:
            triggers = (
                QAbstractItemView.EditKeyPressed
                | QAbstractItemView.SelectedClicked
                | QAbstractItemView.DoubleClicked
            )
            self.table_ref.setEditTriggers(triggers)
        except Exception:
            self.table_ref.setEditTriggers(QTableWidget.AllEditTriggers)

        # 選択モード設定 (両テーブル: 列選択)
        self.table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_ref.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.table_ref.setSelectionMode(QAbstractItemView.SingleSelection)

        # 左テーブルデリゲート (Enterキーでのセル移動)
        try:
            self.table_ref.setItemDelegate(RefTableDelegate(self.table_ref))
        except Exception:
            pass

        # テーブル変更イベント接続
        self.table.currentCellChanged.connect(self._on_table_current_changed)
        self.table_ref.currentCellChanged.connect(self._on_ref_table_current_changed)
        # 左テーブルクリックイベント (Obs行即編集)
        try:
            self.table_ref.cellClicked.connect(self._on_ref_cell_clicked)
        except Exception:
            pass

        # スクロール/サイズ設定
        # Allow vertical scrollbar if content exceeds available height and let
        # the left table expand vertically so all rows can be shown when space allows.
        self.table_ref.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_ref.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 左は固定横幅
        self.table_ref.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_ref.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_ref.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        # 左テーブルの列幅は固定運用（小さめ）
        try:
            self.table_ref.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
            self.table_ref.horizontalHeader().setMinimumSectionSize(20)
        except Exception:
            pass

        # Diagnostic: connect to commitData to detect editors that don't belong to a view
        try:
            from qt_compat.QtWidgets import QApplication

            def _commit_diag(ed, view_name='table_ref', view=self.table_ref):
                try:
                    import sys, traceback
                    fw = QApplication.focusWidget()
                    print(f"[COMMITDATA_SIGNAL] view={view_name} editor={ed} focus={fw}", file=sys.stderr)
                    try:
                        is_desc = bool(view.isAncestorOf(ed))
                    except Exception:
                        is_desc = False
                    print(f"[COMMITDATA_SIGNAL] is_descendant_of_view={is_desc} editor_parent={getattr(ed, 'parent', None)}", file=sys.stderr)
                    traceback.print_stack(limit=8)
                except Exception:
                    pass

            try:
                self.table_ref.commitData.connect(lambda ed: _commit_diag(ed, 'table_ref', self.table_ref))
            except Exception:
                pass
            try:
                self.table_ref_view.commitData.connect(lambda ed: _commit_diag(ed, 'table_ref_view', self.table_ref_view))
            except Exception:
                pass
            try:
                self.table_between.commitData.connect(lambda ed: _commit_diag(ed, 'table_between', self.table_between))
            except Exception:
                pass
        except Exception:
            pass

        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # 横スクロール状態が変わったら高さも再調整（右テーブル）
        try:
            self.table.horizontalScrollBar().rangeChanged.connect(lambda _min, _max: fix_tables_height(self.table_ref, self.table))
        except Exception:
            pass

        # 統一幅（数字+ボタン領域の幅） -- 左カラムを狭めるために少し小さめに設定
        self.control_area_width = 100
        self.max_levels = 255

        # 残すのは PosterLevel と Min Area に加え、Trim(pix)
        # Use code-safe internal keys for widgets; display text comes from self.display_labels
        self.edit_levels, self.slider_levels = self._make_spin_slider('poster_level', 4, 2, 20, 1)
        self.edit_min_area, self.slider_min_area = self._make_spin_slider('min_area', 50, 10, 5000, 1)
        self.edit_trim, self.slider_trim = self._make_spin_slider('trim', 0, 0, 10, 1)
        self.edit_neck_sep, self.slider_neck_sep = self._make_spin_slider('neck_separation', 0, 0, 10, 1)
        self.edit_shape_complex, self.slider_shape_complex = self._make_spin_slider('shape_complexity', 10, 0, 10, 1)

        # PosterLevelの内部値（スライダー上限20を超えても保持）
        self.levels_value = self.slider_levels.value()

        # ボタン（画像開く / エクスポート）を作る（配置は後で画像ヘッダ等へ移動する）
        self.btn_open = QPushButton(STR.BUTTON_OPEN_IMAGE)
        self.btn_open.setFixedHeight(40)
        self.btn_open.clicked.connect(self.open_image)
        # Export ボタンは短くして隣に Clipboard を追加
        self.btn_export = QPushButton("Export")
        self.btn_export.setFixedHeight(40)
        self.btn_export.clicked.connect(self.export_centroids)
        self.btn_clipboard = QPushButton("Clipboard")
        self.btn_clipboard.setFixedHeight(40)
        self.btn_clipboard.clicked.connect(self._copy_centroids_to_clipboard)

        # 自動更新/手動再計算の UI 部品を先に作成
        # Auto Update の ON/OFF 表示・選択は不要なので、常に auto_update_mode=True とする
        self.interp_mode = 'auto'  # 常に auto モード（ズーム倍率で自動選択）
        self.auto_update_mode = True
        self.chk_auto_update = None
        # Recalc ボタン表示は不要
        self.btn_recalc = None

        # ピックモード（ルーペ制御）
        self.pick_mode = None  # None / 'add' / 'update'
        self.pick_ref_index = None
        # 全体ズーム係数（1.0=等倍）
        self.proc_zoom = 1.0
        # 最後に描いた右側オーバーレイ画像（フル解像度、numpy画像）
        self._last_overlay_full = None
        # パン/フリック用の状態
        self._mouse_pressed = False
        self._dragging = False
        self._drag_start_vp = None  # ビューポート座標での押下位置
        self._drag_start_scroll = (0, 0)
        self._drag_recent = deque(maxlen=8)  # (t, QPoint)
        self._kinetic_timer = QTimer(self)
        self._kinetic_timer.setInterval(16)
        self._kinetic_timer.timeout.connect(self._on_kinetic_tick)
        self._kinetic_vx = 0.0  # スクロール速度(px/秒)
        self._kinetic_vy = 0.0
        self._kinetic_last_t = 0.0
        # 表示用余白（スクロールの遊び）と描画状態
        self.view_padding = 200
        self._display_offset = (0, 0)   # 画像がキャンバス内で開始するラベル座標
        self._display_img_size = (0, 0) # キャンバス内の画像サイズ（ズーム後）
        self._display_pm_base = None    # クロスヘア等を描く前のベースPixmap
        # 初回表示は画像中心から開始するためのフラグ
        self._initial_center_done = False
        # 通常時は手のカーソル
        self.img_label_proc.setCursor(QCursor(Qt.OpenHandCursor))

        # 画像右上用の「境界線」トグルボタン（先に生成しておく）
        # 画像右上用の「境界線」トグル（Show/Hide の2択）
        self.show_boundaries = True
        try:
            self.boundary_toggle = SegmentControl(["Show", "Hide"], checked_index=0, btn_w=64, btn_h=24)
            # connect change: index 0 => show True, index 1 => show False
            try:
                self.boundary_toggle.set_on_changed(lambda idx: self._on_toggle_boundaries(bool(idx == 0)))
            except Exception:
                pass
            # expose button refs for backward compatibility
            try:
                # map to names used elsewhere if needed
                self.btn_boundary_show = self.boundary_toggle._buttons[0]
                self.btn_boundary_hide = self.boundary_toggle._buttons[1]
            except Exception:
                self.btn_boundary_show = None
                self.btn_boundary_hide = None
        except Exception:
            self.boundary_toggle = None
            self.btn_boundary_show = None
            self.btn_boundary_hide = None

        # Boundary ラベル + Show/Hide トグルをひとまとめに（右上に配置）
        try:
            if getattr(self, 'boundary_toggle', None) is not None:
                self.boundary_controls = QWidget()
                bcl = QHBoxLayout(self.boundary_controls)
                bcl.setContentsMargins(0, 0, 0, 0)
                bcl.setSpacing(6)
                self.lbl_boundary = QLabel("Boundary")
                try:
                    from qt_compat.QtGui import QFont as _QFont
                    fb = _QFont('Segoe UI', 12)
                    try:
                        fb.setBold(True)
                    except Exception:
                        pass
                    self.lbl_boundary.setFont(fb)
                    try:
                        self.lbl_boundary.setStyleSheet('font-weight: bold;')
                    except Exception:
                        pass
                except Exception:
                    try:
                        f = self.lbl_boundary.font()
                        f.setBold(True)
                        self.lbl_boundary.setFont(f)
                    except Exception:
                        pass
                bcl.addWidget(self.lbl_boundary)
                bcl.addWidget(self.boundary_toggle)
                # View Orientation トグル（Image / Stage）を右隣に追加
                try:
                    self.view_orientation_toggle = SegmentControl(["Image", "Stage"], checked_index=0, btn_w=69, btn_h=24)
                    try:
                        self.view_orientation_toggle.set_on_changed(lambda idx: self._on_toggle_view_orientation(int(idx)))
                    except Exception:
                        pass
                    try:
                        self.btn_view_image = self.view_orientation_toggle._buttons[0]
                        self.btn_view_stage = self.view_orientation_toggle._buttons[1]
                    except Exception:
                        self.btn_view_image = None
                        self.btn_view_stage = None
                    # small label for the control
                    self.lbl_view_orientation = QLabel("View Orientation")
                    try:
                        from qt_compat.QtGui import QFont as _QFont
                        fv = _QFont('Segoe UI', 12)
                        try:
                            fv.setBold(True)
                        except Exception:
                            pass
                        self.lbl_view_orientation.setFont(fv)
                        try:
                            self.lbl_view_orientation.setStyleSheet('font-weight: bold;')
                        except Exception:
                            pass
                    except Exception:
                        try:
                            f2 = self.lbl_view_orientation.font()
                            f2.setBold(True)
                            self.lbl_view_orientation.setFont(f2)
                        except Exception:
                            pass
                    # pack into a small widget so it can be positioned independently
                    self.view_orientation_controls = QWidget()
                    vocl = QHBoxLayout(self.view_orientation_controls)
                    vocl.setContentsMargins(0, 0, 0, 0)
                    vocl.setSpacing(6)
                    vocl.addWidget(self.lbl_view_orientation)
                    vocl.addWidget(self.view_orientation_toggle)
                except Exception:
                    self.view_orientation_toggle = None
                    self.btn_view_image = None
                    self.btn_view_stage = None
                    self.lbl_view_orientation = None
                    self.view_orientation_controls = None
            else:
                self.boundary_controls = None
                self.lbl_boundary = None
        except Exception:
            self.boundary_controls = None
            self.lbl_boundary = None

        # 左右反転モードは UI で操作しない（常に Auto 固定）
        self.flip_mode = 'auto'  # 'auto' | 'normal' | 'flip'
        # flip_mode remains fixed to 'auto' and no UI control is created
        self.combo_flip_mode = None
        self.lbl_display_mode = None

        # 画像領域レイアウト：上にボタン群（左に Open/Export/Clipboard、中央に補間/自動系、右に Flip/境界）
        img_layout = QVBoxLayout()
        img_header = QHBoxLayout()
        # 左上に Open/Export/Clipboard ボタンを横並びに配置
        try:
            button_row = QHBoxLayout()
            button_row.setContentsMargins(0, 9, 0, 0)
            button_row.setSpacing(6)
            button_row.addWidget(self.btn_open, 0)
            img_header.addLayout(button_row)
        except Exception:
            pass
        # 中央上には自動更新/手動再計算をまとめる
        try:
            center_controls = QHBoxLayout()
            # Auto Update の UI 表示は不要
            # Recalc は不要
            img_header.addLayout(center_controls)
        except Exception:
            pass
        img_header.addStretch(1)
        # 右上コントロール（左→右）: View Orientation, Boundary, Posterization Overlay
        overlay_ctrl = None
        try:
            # small overlay control placed at right-top next to Flip
            overlay_ctrl = QWidget()
            try:
                ol_layout = QHBoxLayout(overlay_ctrl)
                ol_layout.setContentsMargins(0, 0, 0, 0)
                ol_layout.setSpacing(4)
                lbl_ol = QLabel(self.display_labels.get('overlay_ratio', STR.NAME_OVERLAY_RATIO))
                try:
                    # IMPORTANT: copy the exact font from the other header labels so it matches visually.
                    # (Some environments render 'bold' subtly; copying avoids any mismatch.)
                    base_font = None
                    try:
                        base_font = getattr(self, 'lbl_view_orientation', None)
                        base_font = base_font.font() if base_font is not None else None
                    except Exception:
                        base_font = None
                    if base_font is None:
                        try:
                            base_font = getattr(self, 'lbl_boundary', None)
                            base_font = base_font.font() if base_font is not None else None
                        except Exception:
                            base_font = None
                    if base_font is None:
                        base_font = lbl_ol.font()

                    try:
                        base_font.setBold(True)
                    except Exception:
                        pass
                    lbl_ol.setFont(base_font)
                    try:
                        # Extra fallback: enforce bold with stylesheet
                        lbl_ol.setStyleSheet('font-weight: bold;')
                    except Exception:
                        pass
                except Exception:
                    pass
                ol_layout.addWidget(lbl_ol)
                try:
                    # Two-state: Original / Posterized. Widen buttons so text doesn't clip.
                    self.overlay_mode_toggle = SegmentControl(["Original", "Posterized"], checked_index=0, btn_w=108, btn_h=24)
                    try:
                        self.overlay_mode_toggle.set_on_changed(lambda idx: self._on_overlay_mode_changed(int(idx)))
                    except Exception:
                        pass
                    ol_layout.addWidget(self.overlay_mode_toggle)
                except Exception:
                    self.overlay_mode_toggle = None
            except Exception:
                pass
        except Exception:
            pass

        # 右上: View Orientation と Boundary を先に配置
        try:
            if getattr(self, 'view_orientation_controls', None) is not None:
                try:
                    img_header.addWidget(self.view_orientation_controls, 0, Qt.AlignRight)
                except Exception:
                    pass
            if getattr(self, 'boundary_controls', None) is not None:
                try:
                    # spacing between view orientation and boundary
                    if getattr(self, 'view_orientation_controls', None) is not None:
                        img_header.addSpacing(8)
                except Exception:
                    pass
                img_header.addWidget(self.boundary_controls, 0, Qt.AlignRight)
            elif getattr(self, 'boundary_toggle', None) is not None:
                img_header.addWidget(self.boundary_toggle, 0, Qt.AlignRight)
        except Exception:
            pass

        # 右上: 最後に Posterization Overlay
        try:
            if overlay_ctrl is not None:
                try:
                    # spacing between boundary and overlay
                    if getattr(self, 'boundary_controls', None) is not None or getattr(self, 'boundary_toggle', None) is not None or getattr(self, 'view_orientation_controls', None) is not None:
                        img_header.addSpacing(12)
                except Exception:
                    pass
                try:
                    img_header.addWidget(overlay_ctrl, 0, Qt.AlignRight)
                except Exception:
                    img_header.addWidget(overlay_ctrl)
        except Exception:
            pass

        img_layout.addLayout(img_header, 0)
        img_layout.addWidget(self.proc_scroll, 1)

        # スライダー/コントロールレイアウト（各項目を横一行にまとめ、アプリ共通フォントを使う）
        sliders_layout = QVBoxLayout()
        from qt_compat.QtGui import QFont
        # Use Segoe UI 12 as the control font (match app-wide font)
        try:
            ctrl_font = QFont('Segoe UI', 12)
            ctrl_font.setBold(False)
        except Exception:
            ctrl_font = QFont()
        # make rows a little taller / more airy so controls don't feel cramped
        try:
            # Reduce vertical gaps so labels feel tighter
            sliders_layout.setSpacing(10)
            sliders_layout.setContentsMargins(6, 2, 6, 2)
        except Exception:
            pass

        # NOTE: overlay slider moved to image header (right-top). See img_header insertion below.

        # Helper to build a single-row control with label, slider, and numeric box (+/-)
        def _build_control_row(key, name, edit_widget, slider_widget, nudger_minus, nudger_plus):
            try:
                row = QHBoxLayout()
                try:
                    row.setContentsMargins(0, 0, 0, 0)
                    row.setSpacing(6)
                except Exception:
                    pass
                lbl = QLabel(name)
                try:
                    # Bold only the left-column labels requested by user
                    f = QFont(ctrl_font)
                    if str(key) in ('poster_level', 'min_area'):
                        f.setBold(True)
                    lbl.setFont(f)
                except Exception:
                    pass
                try:
                    # 固定幅にして、すぐ隣に数値ボックスが来るようにする（ラベルと数値の間に可変スペースを入れない）
                    # Give labels more room so text doesn't clip; this also narrows the slider area.
                    lbl.setFixedWidth(180)
                    lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                except Exception:
                    pass
                row.addWidget(lbl)
                # numeric + +/- on the left of the slider (number left, bar right)
                box = QWidget()
                try:
                    box.setFixedWidth(self.control_area_width)
                except Exception:
                    pass

                bhl = QHBoxLayout(box)
                bhl.setContentsMargins(0, 0, 0, 0)
                bhl.setSpacing(0)

                try:
                    minus_btn = QPushButton("-")
                    minus_btn.setFixedSize(28, 28)
                    minus_btn.clicked.connect(lambda _, f=nudger_minus: f(-1))
                except Exception:
                    minus_btn = QPushButton("-")

                try:
                    plus_btn = QPushButton("+")
                    plus_btn.setFixedSize(28, 28)
                    plus_btn.clicked.connect(lambda _, f=nudger_plus: f(1))
                except Exception:
                    plus_btn = QPushButton("+")

                try:
                    # numeric edit: fixed width and height to match +/- buttons
                    edit_widget.setFixedWidth(48)
                    edit_widget.setFixedHeight(28)
                    edit_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                    edit_widget.setAlignment(Qt.AlignCenter)
                    edit_widget.setFont(ctrl_font)
                except Exception:
                    pass

                try:
                    minus_btn.setFont(ctrl_font)
                    plus_btn.setFont(ctrl_font)
                except Exception:
                    pass

                # remove internal button padding and ensure consistent spacing
                try:
                    minus_btn.setStyleSheet("padding:0px; margin:0px;")
                except Exception:
                    pass
                try:
                    plus_btn.setStyleSheet("padding:0px; margin:0px;")
                except Exception:
                    pass

                # add widgets with explicit equal spacers between them
                bhl.addWidget(minus_btn)
                bhl.addSpacing(5)            # gap between minus and number
                bhl.addWidget(edit_widget)
                bhl.addSpacing(45)            # gap between number and plus
                bhl.addWidget(plus_btn)

                row.addWidget(box)

                # slider placed to the right; give it a modest fixed height to align with buttons
                try:
                    slider_widget.setFixedHeight(28)
                except Exception:
                    pass
                row.addWidget(slider_widget, 3)
                return row
            except Exception:
                return None

        # Helper to build Number of Groups row for Basic mode
        def _build_num_groups_row_widget():
            try:
                self.edit_num_groups, self.slider_num_groups = self._make_spin_slider('num_groups', 2, 2, 20, 1)
                r = _build_control_row('num_groups', 'Number of Groups', self.edit_num_groups, self.slider_num_groups, self._nudge_num_groups, self._nudge_num_groups)
                if r is not None:
                    roww = QWidget()
                    roww.setLayout(r)
                    return roww
            except Exception as e:
                print(f"Error building num_groups row: {e}")
            return None

        # Basic mode: Number of Groups row
        try:
            self.row_num_groups = _build_num_groups_row_widget()
            if self.row_num_groups is not None:
                sliders_layout.addWidget(self.row_num_groups)
        except Exception:
            self.row_num_groups = None

        # PosterLevel row (Advanced)
        try:
            r = _build_control_row('poster_level', self.display_labels.get('poster_level', STR.NAME_POSTERLEVEL), self.edit_levels, self.slider_levels, self._nudge_levels, self._nudge_levels)
            if r is not None:
                try:
                    self.row_poster_level = QWidget()
                    self.row_poster_level.setLayout(r)
                    sliders_layout.addWidget(self.row_poster_level)
                except Exception:
                    sliders_layout.addLayout(r)
                    self.row_poster_level = None
        except Exception:
            pass

        # Min Area row (Common)
        try:
            r = _build_control_row('min_area', self.display_labels.get('min_area', STR.NAME_MIN_AREA), self.edit_min_area, self.slider_min_area, self._nudge_min_area, self._nudge_min_area)
            if r is not None:
                try:
                    self.row_min_area = QWidget()
                    self.row_min_area.setLayout(r)
                    sliders_layout.addWidget(self.row_min_area)
                except Exception:
                    sliders_layout.addLayout(r)
                    self.row_min_area = None
        except Exception:
            pass

        # Area histogram (Advanced only; shown below Min Area)
        try:
            self.area_hist = AreaHistogramWidget()
            try:
                self.area_hist.setFont(ctrl_font)
            except Exception:
                pass
            try:
                self.area_hist.rangeChanged.connect(self._on_area_hist_range_changed)
            except Exception:
                pass
            sliders_layout.addWidget(self.area_hist)
        except Exception:
            self.area_hist = None

        # Trim row (Advanced - Boundary Offset)
        try:
            r = _build_control_row('trim', self.display_labels.get('trim', STR.NAME_TRIM), self.edit_trim, self.slider_trim, self._nudge_trim, self._nudge_trim)
            if r is not None:
                try:
                    self.row_trim = QWidget()
                    self.row_trim.setLayout(r)
                    sliders_layout.addWidget(self.row_trim)
                except Exception:
                    sliders_layout.addLayout(r)
                    self.row_trim = None
        except Exception:
            pass

        # Neck Separation row (Advanced)
        try:
            r = _build_control_row('neck_separation', 'Neck Separation', self.edit_neck_sep, self.slider_neck_sep, self._nudge_neck_sep, self._nudge_neck_sep)
            if r is not None:
                try:
                    self.row_neck_sep = QWidget()
                    self.row_neck_sep.setLayout(r)
                    sliders_layout.addWidget(self.row_neck_sep)
                except Exception:
                    sliders_layout.addLayout(r)
                    self.row_neck_sep = None
        except Exception:
            self.row_neck_sep = None

        # Shape Complexity row (Advanced)
        try:
            r = _build_control_row('shape_complexity', 'Shape Complexity', self.edit_shape_complex, self.slider_shape_complex, self._nudge_shape_complex, self._nudge_shape_complex)
            if r is not None:
                try:
                    self.row_shape_complex = QWidget()
                    self.row_shape_complex.setLayout(r)
                    sliders_layout.addWidget(self.row_shape_complex)
                except Exception:
                    sliders_layout.addLayout(r)
                    self.row_shape_complex = None
        except Exception:
            self.row_shape_complex = None

        # MinAreaとテーブルの間にボタン行（左詰め）を追加
        actions_row = QHBoxLayout()
        # ここから「境界線」ボタンは削除（画像右上に移動済み）
        self.btn_add_ref = QPushButton(STR.BUTTON_ADD_REF)
        self.btn_update_xy = QPushButton(STR.BUTTON_UPDATE_XY)
        self.btn_clear_ref = QPushButton(STR.BUTTON_CLEAR)
        self.btn_add_ref.clicked.connect(self._on_add_ref_point)
        self.btn_update_xy.clicked.connect(self._on_update_xy)
        self.btn_clear_ref.clicked.connect(self._on_clear_ref)
        actions_row.addWidget(self.btn_add_ref)
        actions_row.addWidget(self.btn_update_xy)
        actions_row.addWidget(self.btn_clear_ref)
        # Flip は右上（画像ヘッダー）に配置
        actions_row.addStretch(1)  # 左詰め
        

        # メインルートレイアウト
        root = QVBoxLayout()

        # Compose main content: left column contains ReferencePoints and sliders,
        # center is image, right is centroid table
        main_row = QHBoxLayout()
        # Left column (vertical): image on top, sliders, then a transposed view
        # of the reference table (we keep the original table_ref as the data
        # backend and present `table_ref_view` to the user transposed).
        try:
            self.left_top_image = QLabel()
            self.left_top_image.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.left_top_image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            base_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(base_dir, "PiXY.png"),
                os.path.join(base_dir, "px2XY2.png"),
                os.path.join(base_dir, "px2XY.png"),
                os.path.join(base_dir, "app_icon.png"),
            ]
            pix = None
            for cand in candidates:
                try:
                    pm = QPixmap(cand)
                    if pm is not None and not pm.isNull():
                        pix = pm
                        break
                except Exception:
                    continue
            if pix is not None:
                try:
                    # Scale the logo up to a maximum width of 400px and fix label width
                    target_w = 450
                    target_h = 200
                    self._left_top_pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.left_top_image.setPixmap(self._left_top_pix)
                    try:
                        self.left_top_image.setFixedSize(target_w, target_h)
                    except Exception:
                        pass
                except Exception:
                    self._left_top_pix = pix
                    self.left_top_image.setPixmap(pix)
            else:
                self._left_top_pix = None
                self.left_top_image.setText("PiXY")
                try:
                    self.left_top_image.setFixedSize(450, 200)
                except Exception:
                    pass
        except Exception:
            self.left_top_image = QLabel("PiXY")
            self._left_top_pix = None
            try:
                self.left_top_image.setFixedSize(450, 200)
            except Exception:
                pass

        # The transposed view of the left reference table (visible to user)
        self.table_ref_view = QTableWidget()
        try:
            # ユーザー側の左カラム表示は行方向で選択する（列方向ではなく）
            self.table_ref_view.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table_ref_view.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table_ref_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.table_ref_view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.table_ref_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            # Keep scrollbar presence stable so widths don't jitter after Add/update
            try:
                self.table_ref_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                self.table_ref_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            except Exception:
                pass
            try:
                # 固定幅にして左カラム内の表の横幅を左コンテナに合わせる
                self.table_ref_view.setFixedWidth(500)
            except Exception:
                pass
            try:
                self.table_ref_view.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            except Exception:
                pass
        except Exception:
            pass

        # Apply initial visibility for Basic/Advanced groups
        try:
            self._apply_grain_ident_visibility()
        except Exception:
            pass
        # Connect edits in the view back to the data table
        try:
            self.table_ref_view.itemChanged.connect(self._on_ref_view_item_changed)
        except Exception:
            pass
        # Track selection in the transposed view so Clear/Add operate on the selected ref index
        try:
            self.table_ref_view.currentCellChanged.connect(self._on_ref_view_current_changed)
        except Exception:
            pass
        try:
            # Ensure the transposed delegate is installed so Enter commits and advances
            try:
                self._ensure_ref_view_delegate()
            except Exception:
                pass
        except Exception:
            pass

        # A transposed copy of the bottom centroid table placed between left and image
        self.table_between = QTableWidget()
        try:
            self.table_between.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            # Keep scrollbar presence stable so the center column doesn't jitter
            self.table_between.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.table_between.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            try:
                self.table_between.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            except Exception:
                pass
            try:
                # make the transposed middle table selectable by rows so image<->table sync is easier
                self.table_between.setSelectionBehavior(QAbstractItemView.SelectRows)
                self.table_between.setSelectionMode(QAbstractItemView.SingleSelection)
                self.table_between.currentCellChanged.connect(self._on_table_between_current_changed)
            except Exception:
                pass
        except Exception:
            pass

        left_col = QVBoxLayout()
        left_col.addWidget(self.left_top_image, 0, Qt.AlignTop)

        # Build Grain Identification block (to be placed below ref table)
        try:
            self.grain_ident_mode = 'basic'
        except Exception:
            pass
        try:
            self.grain_ident_controls = QWidget()
            gil = QHBoxLayout(self.grain_ident_controls)
            try:
                gil.setContentsMargins(0, 0, 0, 0)
                gil.setSpacing(6)
            except Exception:
                pass
            self.lbl_grain_ident = QLabel("Grain Identification")
            try:
                from qt_compat.QtGui import QFont as _QFont
                fgi = _QFont('Segoe UI', 12)
                try:
                    fgi.setBold(True)
                except Exception:
                    pass
                self.lbl_grain_ident.setFont(fgi)
                try:
                    self.lbl_grain_ident.setStyleSheet('font-weight: bold;')
                except Exception:
                    pass
            except Exception:
                try:
                    fgi2 = self.lbl_grain_ident.font()
                    fgi2.setBold(True)
                    self.lbl_grain_ident.setFont(fgi2)
                except Exception:
                    pass
            gil.addWidget(self.lbl_grain_ident)
            try:
                # Match Display Mode toggle size (btn_w=108, btn_h=24)
                self.toggle_grain_ident = SegmentControl(["Basic", "Advanced"], checked_index=0, btn_w=108, btn_h=24)
                try:
                    self.toggle_grain_ident.set_on_changed(lambda idx: self._on_toggle_grain_ident(int(idx)))
                except Exception:
                    pass
                gil.addWidget(self.toggle_grain_ident)
            except Exception:
                self.toggle_grain_ident = None
        except Exception:
            self.grain_ident_controls = None

        try:
            self.grain_section = QWidget()
            gl = QVBoxLayout(self.grain_section)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(6)
            if getattr(self, 'grain_ident_controls', None) is not None:
                gl.addWidget(self.grain_ident_controls, 0)
            gl.addLayout(sliders_layout)
        except Exception:
            self.grain_section = None
        # 左カラムの表の上に Add/Update/Clear ボタンを配置
        try:
            left_controls = QHBoxLayout()
            try:
                left_controls.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            try:
                left_controls.addWidget(self.btn_add_ref)
            except Exception:
                pass
            try:
                left_controls.addWidget(self.btn_update_xy)
            except Exception:
                pass
            try:
                left_controls.addWidget(self.btn_clear_ref)
            except Exception:
                pass
            try:
                left_controls.addStretch(1)
            except Exception:
                pass
            left_col.addLayout(left_controls, 0)
        except Exception:
            pass

        # Fixed 2-row header (does not scroll vertically) for the left transposed table.
        try:
            self.table_ref_view_header = QTableWidget()
            hdr = self.table_ref_view_header
            hdr.setRowCount(2)
            # Pre-allocate columns to ensure labels can be written on init;
            # prefer to match the current view column count when available.
            try:
                pref = max(9, int(getattr(self, 'table_ref_view', None).columnCount() or 9))
            except Exception:
                pref = 9
            hdr.setColumnCount(pref)
            try:
                # Show vertical header so header table reserves the same left gutter
                # as the main transposed table (prevents 1-column visual shift).
                hdr.verticalHeader().setVisible(True)
            except Exception:
                pass
            try:
                hdr.horizontalHeader().setVisible(False)
            except Exception:
                pass
            # Ensure both header rows are visible (explicit row heights + enough frame slack)
            try:
                hdr.setRowHeight(0, 24)
                hdr.setRowHeight(1, 20)
            except Exception:
                pass
            hdr.setFixedHeight(60)
            try:
                hdr.setEditTriggers(QTableWidget.NoEditTriggers)
            except Exception:
                pass
            try:
                hdr.setSelectionMode(QAbstractItemView.NoSelection)
            except Exception:
                pass
            try:
                # Keep scrollbar hidden for the fixed header (no vertical scrolling)
                hdr.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                hdr.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                try:
                    hdr.verticalHeader().setStyleSheet('QHeaderView::section { background-color: lightgray; color: lightgray; }')
                except Exception:
                    pass
            except Exception:
                pass
            try:
                hdr.setFixedWidth(500)
            except Exception:
                pass
            try:
                self._setup_pseudo_headers_ref(hdr)
            except Exception:
                pass
            try:
                # Sync horizontal scrolling between main view and fixed header
                try:
                    self.table_ref_view.horizontalScrollBar().valueChanged.connect(
                        lambda val: hdr.horizontalScrollBar().setValue(val)
                    )
                    hdr.horizontalScrollBar().valueChanged.connect(
                        lambda val: self.table_ref_view.horizontalScrollBar().setValue(val)
                    )
                except Exception:
                    pass
            except Exception:
                pass
            left_col.addWidget(hdr, 0)
        except Exception:
            self.table_ref_view_header = None

        left_col.addWidget(self.table_ref_view, 1)
        # Place Grain Identification block below the RefPoint table
        try:
            if getattr(self, 'grain_section', None) is not None:
                left_col.addWidget(self.grain_section, 0)
        except Exception:
            pass
        # Wrap left column layout in a QWidget and cap its maximum width so it doesn't grow too wide
        left_container = QWidget()
        left_container.setLayout(left_col)
        try:
            # 固定幅にして左カラムを確実に400pxにする
            left_container.setFixedWidth(477)
        except Exception:
            try:
                left_container.setMaximumWidth(477)
            except Exception:
                pass
        main_row.addWidget(left_container, 0)
        # Center area: place the transposed bottom table between left and image
        # Create a center column layout for the table_between
        try:
            center_col = QVBoxLayout()
            # Fixed 2-row header (does not scroll vertically) for the middle transposed table.
            try:
                # Add Export/Clipboard buttons above center table (aligned vertically with Open Image)
                try:
                    center_btn_row = QHBoxLayout()
                    center_btn_row.setContentsMargins(0, 0, 0, 0)
                    center_btn_row.setSpacing(6)
                    center_btn_row.addWidget(self.btn_export, 0)
                    center_btn_row.addWidget(self.btn_clipboard, 0)
                    center_btn_row.addStretch(1)
                    center_col.addLayout(center_btn_row, 0)
                except Exception:
                    pass

                self.table_between_header = QTableWidget()
                hdr_mid = self.table_between_header
                hdr_mid.setRowCount(2)
                # Pre-allocate 5 columns to ensure labels can be written on init
                hdr_mid.setColumnCount(5)
                try:
                    hdr_mid.verticalHeader().setVisible(True)
                except Exception:
                    pass
                try:
                    hdr_mid.horizontalHeader().setVisible(False)
                except Exception:
                    pass
                # Ensure both header rows are visible (explicit row heights + enough frame slack)
                try:
                    hdr_mid.setRowHeight(0, 24)
                    hdr_mid.setRowHeight(1, 20)
                    try:
                        vhw = self.table.verticalHeader().width()
                        if vhw > 0:
                            try:
                                hdr_mid.verticalHeader().setFixedWidth(vhw)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
                # Ensure initial column count covers the main table_between columns
                try:
                    pref_mid = max(5, int(getattr(self, 'table_between', None).columnCount() or 5))
                except Exception:
                    pref_mid = 5
                hdr_mid.setColumnCount(pref_mid)
                hdr_mid.setFixedHeight(60)
                try:
                    hdr_mid.setEditTriggers(QTableWidget.NoEditTriggers)
                except Exception:
                    pass
                try:
                    hdr_mid.setSelectionMode(QAbstractItemView.NoSelection)
                except Exception:
                    pass
                try:
                    hdr_mid.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    hdr_mid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    try:
                        hdr_mid.verticalHeader().setStyleSheet('QHeaderView::section { background-color: lightgray; color: lightgray; }')
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    self._setup_pseudo_headers_between(hdr_mid)
                except Exception:
                    pass
                try:
                    # Sync horizontal scrolling between center transposed and its fixed header
                    try:
                        self.table_between.horizontalScrollBar().valueChanged.connect(
                            lambda val: hdr_mid.horizontalScrollBar().setValue(val)
                        )
                        hdr_mid.horizontalScrollBar().valueChanged.connect(
                            lambda val: self.table_between.horizontalScrollBar().setValue(val)
                        )
                    except Exception:
                        pass
                    # Keep header columns in sync with the main middle table (counts, widths, content)
                    try:
                        def _sync_mid_header():
                            try:
                                hdr_mid.setColumnCount(self.table_between.columnCount())
                                for col in range(min(hdr_mid.columnCount(), self.table_between.columnCount())):
                                    w = self.table_between.columnWidth(col)
                                    if w > 0:
                                        hdr_mid.setColumnWidth(col, w)
                                # copy header rows (row 0-1) from table_between
                                for row in range(min(2, self.table_between.rowCount())):
                                    for col in range(self.table_between.columnCount()):
                                        src_item = self.table_between.item(row, col)
                                        if src_item is not None:
                                            new_item = QTableWidgetItem(src_item.text())
                                            new_item.setBackground(QColor("lightgray"))
                                            new_item.setForeground(QColor("black"))
                                            try:
                                                # Group header row (Image/Stage) should be left-aligned
                                                if int(row) == 0:
                                                    new_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                                                else:
                                                    new_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                                            except Exception:
                                                pass
                                            try:
                                                f = new_item.font()
                                                f.setBold(True)
                                                new_item.setFont(f)
                                            except Exception:
                                                pass
                                            hdr_mid.setItem(row, col, new_item)

                                # Ensure header/container are wide enough so the last column (e.g., Z) isn't clipped
                                try:
                                    total_w = 0
                                    for c in range(self.table_between.columnCount()):
                                        cw = self.table_between.columnWidth(c)
                                        if cw <= 0:
                                            cw = 50
                                        total_w += cw
                                    try:
                                        vgw = self.table_between.verticalHeader().width() or 0
                                    except Exception:
                                        vgw = 0
                                    needed_w = int(total_w + vgw + 10)
                                    try:
                                        hdr_mid.setMinimumWidth(needed_w)
                                    except Exception:
                                        pass
                                    try:
                                        cc = getattr(self, 'center_container', None)
                                        if cc is not None:
                                            cc.setFixedWidth(needed_w)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        try:
                            # Keep header width and center container width in sync when columns are resized
                            self.table_between.horizontalHeader().sectionResized.connect(
                                lambda idx, old, new: (
                                    hdr_mid.setColumnWidth(idx, new),
                                    QTimer.singleShot(0, _sync_mid_header)
                                )
                            )
                        except Exception:
                            pass
                        try:
                            mdl = self.table_between.model()
                            try:
                                mdl.modelReset.connect(_sync_mid_header)
                            except Exception:
                                pass
                            try:
                                mdl.columnsInserted.connect(lambda parent, start, end: _sync_mid_header())
                            except Exception:
                                pass
                            try:
                                mdl.columnsRemoved.connect(lambda parent, start, end: _sync_mid_header())
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # initial sync
                        try:
                            _sync_mid_header()
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
                center_col.addWidget(hdr_mid, 0)
                try:
                    # Ensure header widget is wide enough to show all columns (prevent Z cutoff)
                    try:
                        total_w = 0
                        for col in range(self.table_between.columnCount()):
                            w = self.table_between.columnWidth(col)
                            if w <= 0:
                                w = 50
                            total_w += w
                        # include vertical gutter width if visible
                        try:
                            vgw = hdr_mid.verticalHeader().width() or 0
                        except Exception:
                            vgw = 0
                        try:
                            hdr_mid.setMinimumWidth(total_w + vgw + 8)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # If _sync_mid_header exists, call it now to copy texts/widths
                    try:
                        _sync_mid_header()
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                self.table_between_header = None
            center_col.addWidget(self.table_between, 1)
            # Wrap the center column in a QWidget so we can control the column width
            self.center_container = QWidget()
            self.center_container.setLayout(center_col)
            try:
                self.center_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            except Exception:
                pass
            # Ensure center container starts wide enough to show all middle-table columns
            try:
                total_w = 0
                for c in range(self.table_between.columnCount()):
                    cw = self.table_between.columnWidth(c)
                    if cw <= 0:
                        cw = 50
                    total_w += cw
                try:
                    vgw = self.table_between.verticalHeader().width() or 0
                except Exception:
                    vgw = 0
                self.center_container.setFixedWidth(int(total_w + vgw + 24 + 30))
            except Exception:
                pass
            main_row.addWidget(self.center_container, 0)
        except Exception:
            # fallback to previous placement
            main_row.addWidget(self.table_between, 0)
        # Center: image area
        main_row.addLayout(img_layout, 1)
        # Right table will be placed below, spanning full width; do not add it to main_row
        root.addLayout(main_row, 4)

        # Place centroid table below main content, spanning the full window width
        # Keep the table's horizontal scroll policy as-is; allow expanding horizontally.
        try:
            self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        # The original bottom `self.table` is intentionally not added to the
        # layout any more (user requested it removed). It remains as the
        # canonical data table for internal calculations but is not shown.

        # 中央ウィジェット設定
        # Build top-level container with custom title bar and footer
        content_widget = QWidget()
        content_widget.setLayout(root)

        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Custom title bar removed; rely on native window decorations

        # Main content area
        main_layout.addWidget(content_widget)

        # Footer (solid black)
        self.ui_footer = Footer(main_container)
        main_layout.addWidget(self.ui_footer)

        # Remove native status bar to prevent white strip at bottom
        self.setStatusBar(None)

        # Use native window decorations and system menu buttons
        self.setWindowFlags(
            Qt.Window | 
            Qt.WindowMinMaxButtonsHint | 
            Qt.WindowCloseButtonHint |
            Qt.WindowSystemMenuHint
        )
        self.setCentralWidget(main_container)
        # Native decorations used; schedule applying DWM titlebar style after show
        try:
            QTimer.singleShot(0, self._apply_windows_titlebar_style)
        except Exception:
            pass
        # Ensure column shrinking runs after layout/show so startup view matches adjusted widths
        try:
            QTimer.singleShot(150, self._shrink_visible_columns)
        except Exception:
            pass

    def changeEvent(self, event):
        try:
            from qt_compat.QtCore import QEvent as _QE
            if event.type() == _QE.WindowStateChange:
                try:
                    if getattr(self, 'title', None) is not None:
                        self.title.update_max_icon()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            return super().changeEvent(event)
        except Exception:
            return None

        # After the layout stabilizes, shrink visible transposed-table columns
        try:
            QTimer.singleShot(100, self._shrink_visible_columns)
        except Exception:
            pass
        # 中央カラム（table_between）を1列分狭める処理も実行
        try:
            QTimer.singleShot(150, self._narrow_center_column)
        except Exception:
            pass
        # 左上画像を左カラム幅に合わせる同期処理
        try:
            QTimer.singleShot(160, self._sync_left_top_image_width)
        except Exception:
            pass

        # 配線
        self._wire_levels()
        self._wire(self.edit_min_area, self.slider_min_area)
        self._wire(self.edit_trim, self.slider_trim)
        self._wire(self.edit_neck_sep, self.slider_neck_sep)
        self._wire(self.edit_shape_complex, self.slider_shape_complex)
        # Ref の Obs.* 入力保持用（内部容量は10）
        self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]
        # 入力変更を監視（半角正規化）
        self.table_ref.itemChanged.connect(self._on_ref_item_changed)

        # 起動直後に一度テーブルを構築（左表を3列で表示しておく）
        try:
            self._safe_populate_tables(
                self.table_ref,
                self.table,
                self.ref_points,
                self.ref_obs,
                [],
                self.selected_index,
                self.ref_selected_index,
                flip_mode=self.flip_mode,
                visible_ref_cols=self.visible_ref_cols,
            )
            # 初期の列幅/高さを反映
            try:
                fix_tables_height(self.table_ref, self.table)
            except Exception:
                pass
            # 下部テーブルは5行固定なので、その表示に合わせて高さを固定する
            try:
                rows_fixed = 6
                row_h = self.table.verticalHeader().defaultSectionSize()
                hdr_h = self.table.horizontalHeader().height()
                # フレームや余白分の余裕を少し加える
                extra = 4
                try:
                    # frameWidth is a method on some styles; try to call if present
                    fw = self.table.frameWidth() if hasattr(self.table, 'frameWidth') else 2
                    extra = max(4, fw * 2)
                except Exception:
                    extra = 4
                total_h = hdr_h + int(row_h) * rows_fixed + extra
                # Reapply fixed height after layout settles to avoid later overwrites.
                def _apply_fixed_height():
                    try:
                        # Ensure vertical size policy is Fixed so layout doesn't stretch it.
                        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                        # Always compute and store the exact height for 5 rows
                        h = total_h
                        try:
                            self.FIXED_TABLE_HEIGHT = int(h)
                        except Exception:
                            pass
                        # Also set per-row height to eliminate extra blank space inside the table.
                        try:
                            # choose a reasonable per-row height; prefer existing default if available
                            row_h = int(self.table.verticalHeader().defaultSectionSize() or 20)
                            # enforce a maximum to avoid overly large rows
                            if row_h > 48:
                                row_h = 24
                            # set vertical header to fixed mode and apply default
                            try:
                                self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
                            except Exception:
                                pass
                            self.table.verticalHeader().setDefaultSectionSize(row_h)
                            # apply to all current rows
                            for rr in range(self.table.rowCount()):
                                try:
                                    self.table.setRowHeight(rr, row_h)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        self.table.setFixedHeight(h)
                    except Exception:
                        try:
                            self.table.setFixedHeight(total_h)
                        except Exception:
                            pass
                try:
                    QTimer.singleShot(0, _apply_fixed_height)
                except Exception:
                    # fallback: apply immediately
                    _apply_fixed_height()
            except Exception:
                pass
        except Exception:
            pass

        # Enforce button heights after layout settles
        try:
            QTimer.singleShot(300, self._enforce_button_heights)
        except Exception:
            pass

        # Apply button styling (colors, widths, bold, rounded corners)
        try:
            QTimer.singleShot(0, self._apply_button_styles)
        except Exception:
            pass

        self.open_image()

    # オーバーレイ表示モード（Original/Posterized）変更ハンドラ
    def _on_overlay_mode_changed(self, idx):
        try:
            modes = ['Original', 'Posterized']
            mode = modes[int(idx)] if 0 <= int(idx) < len(modes) else 'Original'
        except Exception:
            mode = 'Original'
        self.overlay_mode = mode
        # keep a numeric mix for any legacy callers (0/100)
        try:
            self.overlay_mix = {'Original': 0, 'Posterized': 100}.get(mode, 0)
        except Exception:
            pass
        self.schedule_update(force=True)

    # 境界線表示トグルハンドラ
    def _on_toggle_boundaries(self, checked):
        self.show_boundaries = bool(checked)
        try:
            if getattr(self, 'btn_boundary_show', None) is not None and getattr(self, 'btn_boundary_hide', None) is not None:
                # keep segmented toggle in sync even when called programmatically
                try:
                    self.btn_boundary_show.setChecked(bool(self.show_boundaries))
                    self.btn_boundary_hide.setChecked(not bool(self.show_boundaries))
                except Exception:
                    pass
        except Exception:
            pass
        self.schedule_update(force=True)

    # View Orientation トグルハンドラ
    def _on_toggle_view_orientation(self, idx):
        try:
            if int(idx) == 0:
                self.view_orientation = 'Image'
            else:
                self.view_orientation = 'Stage'
        except Exception:
            self.view_orientation = 'Image'
        try:
            if getattr(self, 'btn_view_image', None) is not None and getattr(self, 'btn_view_stage', None) is not None:
                try:
                    self.btn_view_image.setChecked(self.view_orientation == 'Image')
                    self.btn_view_stage.setChecked(self.view_orientation == 'Stage')
                except Exception:
                    pass
        except Exception:
            pass
        # 更新をスケジュール（必要なら表示を更新するため）
        try:
            self.schedule_update(force=True)
        except Exception:
            pass

    # Grain Identification トグルハンドラ（Basic/Advanced）
    def _on_toggle_grain_ident(self, idx):
        try:
            if int(idx) == 0:
                self.grain_ident_mode = 'basic'
            else:
                self.grain_ident_mode = 'advanced'
        except Exception:
            self.grain_ident_mode = 'basic'
        # 詳細度に応じて将来の処理分岐が可能（現状は表示更新のみ）
        try:
            self._apply_grain_ident_visibility()
            self.schedule_update(force=True)
        except Exception:
            pass

    def _apply_grain_ident_visibility(self):
        mode = str(getattr(self, 'grain_ident_mode', 'basic'))
        show_basic = bool(mode == 'basic')
        try:
            if getattr(self, 'row_num_groups', None) is not None:
                # Number of Groups is shared between Basic/Advanced
                self.row_num_groups.setVisible(True)
        except Exception:
            pass
        # Min Area slider is hidden; selection is done on the histogram in both modes.
        try:
            if getattr(self, 'row_min_area', None) is not None:
                self.row_min_area.setVisible(False)
        except Exception:
            pass
        # Advanced-only
        for name in ('row_poster_level', 'row_trim', 'row_neck_sep', 'row_shape_complex'):
            try:
                w = getattr(self, name, None)
                if w is not None:
                    # Posterization Steps row is deprecated; keep hidden.
                    if name == 'row_poster_level':
                        w.setVisible(False)
                    else:
                        w.setVisible(not show_basic)
            except Exception:
                pass
        try:
            if getattr(self, 'area_hist', None) is not None:
                self.area_hist.setVisible(True)
        except Exception:
            pass

    # スピンボックスとスライダーのペアを作成するヘルパーメソッド
    def _make_spin_slider(self, name, init, mn, mx, tick):
        edit = QLineEdit(str(init))
        edit.setAlignment(Qt.AlignRight)
        slider = ClickableSlider(Qt.Horizontal)
        slider.setMinimum(mn)
        slider.setMaximum(mx)
        slider.setSingleStep(1)
        slider.setValue(init)
        slider.setTickInterval(tick)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda v, e=edit: self._sync_from_slider(e, v))
        # name is expected to be a code-safe key (e.g. 'poster_level', 'min_area')
        try:
            if name == 'poster_level':
                slider._wheel_scale = 1.0 / 3.0
            if name == 'min_area':
                approx_div = 8
                tick_int = max(1, int(round((mx - mn) / approx_div)))
                slider.setTickInterval(tick_int)
                try:
                    edit.setFixedWidth(self.control_area_width)
                except Exception:
                    pass
        except Exception:
            pass
        return edit, slider

    # 編集ボックスとスライダーの同期配線 (Enter確定のみ)
    def _wire(self, edit, slider):
        # Enter（Return）で確定したときのみ適用する
        # Only attempt to call signal.disconnect() without a slot on PyQt5.
        # On PySide6, calling disconnect() with no arguments emits a RuntimeWarning.
        try:
            if getattr(qt_compat, 'using', '') == 'PyQt5':
                try:
                    edit.editingFinished.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        edit.returnPressed.connect(lambda e=edit, s=slider: self._sync_from_edit(e, s))

    # PosterLevel専用の配線（上限20超の内部値を保持）
    def _wire_levels(self):
        # フォーカスアウトでは適用しない。Enter確定のみ。
        try:
            if getattr(qt_compat, 'using', '') == 'PyQt5':
                try:
                    self.edit_levels.editingFinished.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        self.edit_levels.returnPressed.connect(self._on_levels_edit_finished)
        try:
            if getattr(qt_compat, 'using', '') == 'PyQt5':
                try:
                    self.slider_levels.valueChanged.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        self.slider_levels.valueChanged.connect(self._on_levels_slider_changed)

    # PosterLevelスライダー変更ハンドラ
    def _on_levels_slider_changed(self, v):
        # スライダー操作は上限20まで。内部値も更新
        self.levels_value = int(v)
        self.edit_levels.setText(str(self.levels_value))
        self.schedule_update()

    # PosterLevel編集確定ハンドラ
    def _on_levels_edit_finished(self):
        text = self.edit_levels.text().strip()
        try:
            v = int(text)
        except ValueError:
            v = self.levels_value
        if v < 1:
            v = 1
        if v > self.max_levels:
            v = self.max_levels
        self.levels_value = v
        clamped = max(self.slider_levels.minimum(), min(self.slider_levels.maximum(), v))
        try:
            self.slider_levels.blockSignals(True)
        except Exception:
            pass
        try:
            self.slider_levels.setValue(clamped)
        finally:
            try:
                self.slider_levels.blockSignals(False)
            except Exception:
                pass
        self.edit_levels.setText(str(self.levels_value))
        self.schedule_update(force=True)

    # PosterLevelの+/-ボタンで値を調整
    def _nudge_levels(self, delta):
        try:
            cur = int(self.edit_levels.text().strip())
        except Exception:
            try:
                cur = int(getattr(self, 'levels_value', 4))
            except Exception:
                cur = 4
        try:
            d = int(delta)
        except Exception:
            d = 0

        v = cur + d
        if v < 1:
            v = 1
        if v > self.max_levels:
            v = self.max_levels

        self.levels_value = v
        try:
            self.edit_levels.setText(str(v))
        except Exception:
            pass
        try:
            # クリップ範囲内ならスライダーも同期
            if v <= self.slider_levels.maximum():
                self.slider_levels.setValue(v)
        except Exception:
            pass
        self.schedule_update()

    # Number of Groups の+/-ボタンで値を調整
    def _nudge_num_groups(self, delta):
        try:
            cur = int(self.edit_num_groups.text().strip())
        except Exception:
            try:
                cur = int(getattr(self, 'slider_num_groups', None).value() if hasattr(self, 'slider_num_groups') else 4)
            except Exception:
                cur = 4
        try:
            d = int(delta)
        except Exception:
            d = 0

        v = cur + d
        if v < 2:
            v = 2
        if v > 20:
            v = 20

        try:
            self.edit_num_groups.setText(str(v))
        except Exception:
            pass
        try:
            self.slider_num_groups.setValue(v)
        except Exception:
            pass
        self.schedule_update()

        # Keep internal value even if it exceeds slider maximum
        self.levels_value = int(v)
        try:
            self.edit_levels.setText(str(self.levels_value))
        except Exception:
            pass

        clamped = max(self.slider_levels.minimum(), min(self.slider_levels.maximum(), self.levels_value))
        try:
            self.slider_levels.blockSignals(True)
        except Exception:
            pass
        try:
            self.slider_levels.setValue(int(clamped))
        finally:
            try:
                self.slider_levels.blockSignals(False)
            except Exception:
                pass
        self.schedule_update(force=True)

    def _ensure_ref_view_delegate(self):
        """Install the transposed-table delegate once.

        Avoid emitting commitData manually; doing so can trigger
        QAbstractItemView::commitData warnings when the editor association changes.
        """
        if getattr(self, '_ref_view_delegate_installed', False):
            return
        self._ref_view_delegate_installed = True

        try:
            from qt_compat.QtWidgets import QStyledItemDelegate, QLineEdit, QTableWidgetItem
            from qt_compat.QtWidgets import QAbstractItemDelegate
            from qt_compat.QtCore import Qt as _Qt, QTimer

            owner = self

            class TransposedRefDelegate(QStyledItemDelegate):
                def __init__(self, view, src_table, owner_window=None):
                    super().__init__(view)
                    self.view = view
                    self.src_table = src_table
                    self.owner_window = owner_window

                def createEditor(self, parent, option, index):
                    editor = super().createEditor(parent, option, index)
                    try:
                        if isinstance(editor, QLineEdit):
                            vr, vc = index.row(), index.column()  # view coords

                            def on_return():
                                # Capture text before the editor is potentially destroyed.
                                try:
                                    txt = editor.text()
                                except Exception:
                                    txt = None

                                # Ensure the edited value becomes visible in the cell.
                                # (Some QTableWidget setups do not immediately repaint/update on Return.)
                                try:
                                    if txt is not None:
                                        def _apply_txt():
                                            try:
                                                it = self.view.item(vr, vc)
                                                if it is None:
                                                    try:
                                                        it = QTableWidgetItem("")
                                                        self.view.setItem(vr, vc, it)
                                                    except Exception:
                                                        it = None
                                                if it is not None:
                                                    it.setText(str(txt))
                                            except Exception:
                                                pass

                                        try:
                                            QTimer.singleShot(0, _apply_txt)
                                        except Exception:
                                            _apply_txt()
                                except Exception:
                                    pass

                                # 3) Move after the event loop processes the commit
                                def _move_next():
                                    try:
                                        # Map view coords back to source table: src_row = vc, src_col = vr
                                        src_r = vc
                                        src_c = vr
                                        if src_r == 2:
                                            tgt_src_r = 3; tgt_src_c = src_c
                                        elif src_r == 3:
                                            tgt_src_r = 4; tgt_src_c = src_c
                                        elif src_r == 4:
                                            tgt_src_r = 2; tgt_src_c = min(src_c + 1, self.src_table.columnCount() - 1)
                                        else:
                                            return
                                        # Map back to view coords
                                        view_r = tgt_src_c
                                        view_c = tgt_src_r
                                        self.view.setCurrentCell(view_r, view_c)
                                        item = self.view.item(view_r, view_c)
                                        if item is not None and (item.flags() & getattr(_Qt, 'ItemIsEditable', 0)):
                                            try:
                                                self.view.setFocus()
                                            except Exception:
                                                pass
                                            # Start editing after selection/focus has settled.
                                            def _start_edit():
                                                try:
                                                    self.view.editItem(item)
                                                except Exception:
                                                    pass

                                            try:
                                                QTimer.singleShot(0, _start_edit)
                                            except Exception:
                                                _start_edit()
                                    except Exception:
                                        pass

                                try:
                                    QTimer.singleShot(0, _move_next)
                                except Exception:
                                    _move_next()

                            editor.returnPressed.connect(on_return)
                    except Exception:
                        pass
                    return editor

            try:
                self.table_ref_view.setItemDelegate(TransposedRefDelegate(self.table_ref_view, self.table_ref, owner_window=owner))
            except Exception:
                pass
        except Exception:
            pass

    def _defer_recompute_after_ref_edit(self):
        """Coalesce recompute requests triggered by transposed ref edits."""
        try:
            if getattr(self, '_recompute_ref_pending', False):
                return
            self._recompute_ref_pending = True
        except Exception:
            pass

        try:
            from qt_compat.QtCore import QTimer

            def _run():
                try:
                    self._recompute_ref_pending = False
                except Exception:
                    pass
                try:
                    self._safe_populate_tables(
                        self.table_ref,
                        self.table,
                        self.ref_points,
                        self.ref_obs,
                        self.centroids,
                        self.selected_index,
                        self.ref_selected_index,
                        flip_mode=self.flip_mode,
                        visible_ref_cols=self.visible_ref_cols,
                    )
                except Exception:
                    pass
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass

            # Delay helps avoid racing the editor close + next-cell edit sequence.
            QTimer.singleShot(150, _run)
        except Exception:
            # Fallback: run immediately
            try:
                self._recompute_ref_pending = False
            except Exception:
                pass
            try:
                self._safe_populate_tables(
                    self.table_ref,
                    self.table,
                    self.ref_points,
                    self.ref_obs,
                    self.centroids,
                    self.selected_index,
                    self.ref_selected_index,
                    flip_mode=self.flip_mode,
                    visible_ref_cols=self.visible_ref_cols,
                )
            except Exception:
                pass
            try:
                self._refresh_transposed_views()
            except Exception:
                pass

    # スライダーから編集ボックスへ同期
    def _sync_from_slider(self, edit, val):
        edit.setText(str(val))
        self.schedule_update()

    # 編集ボックスからスライダーへ同期 (Enter確定)
    def _sync_from_edit(self, edit, slider):
        try:
            v = int(edit.text())
        except ValueError:
            v = slider.value()
        v = max(slider.minimum(), min(slider.maximum(), v))
        slider.setValue(v)
        edit.setText(str(v))
        self.schedule_update()

    def _nudge_min_area(self, delta):
        try:
            cur = int(self.edit_min_area.text())
        except Exception:
            cur = self.slider_min_area.value()
        cur = max(self.slider_min_area.minimum(), min(self.slider_min_area.maximum(), cur + int(delta)))
        self.slider_min_area.setValue(cur)
        self.edit_min_area.setText(str(cur))
        self.schedule_update(force=True)

    def _nudge_trim(self, delta):
        try:
            cur = int(self.edit_trim.text())
        except Exception:
            cur = self.slider_trim.value()
        cur = max(self.slider_trim.minimum(), min(self.slider_trim.maximum(), cur + int(delta)))
        self.slider_trim.setValue(cur)
        self.edit_trim.setText(str(cur))
        self.schedule_update(force=True)

    def _nudge_neck_sep(self, delta):
        try:
            cur = int(self.edit_neck_sep.text())
        except Exception:
            cur = self.slider_neck_sep.value()
        cur = max(self.slider_neck_sep.minimum(), min(self.slider_neck_sep.maximum(), cur + int(delta)))
        self.slider_neck_sep.setValue(cur)
        self.edit_neck_sep.setText(str(cur))
        self.schedule_update(force=True)

    def _nudge_shape_complex(self, delta):
        try:
            cur = int(self.edit_shape_complex.text())
        except Exception:
            cur = self.slider_shape_complex.value()
        cur = max(self.slider_shape_complex.minimum(), min(self.slider_shape_complex.maximum(), cur + int(delta)))
        self.slider_shape_complex.setValue(cur)
        self.edit_shape_complex.setText(str(cur))
        self.schedule_update(force=True)

    # 画像ファイルを開くダイアログを表示
    def open_image(self):
        last_path = load_last_image_path()
        fname, _ = QFileDialog.getOpenFileName(self, STR.OPEN_DIALOG_TITLE, last_path, STR.FILE_FILTER)
        if not fname:
            return
        self._open_image_from_path(fname)

    # 指定パスから画像を読み込み、処理画像を構築
    def _open_image_from_path(self, fname: str):
        # 大きなファイルかどうかチェックして、必要なら軽負荷モードを有効化
        try:
            fsize = os.path.getsize(fname)
        except Exception:
            fsize = 0
        LARGE_THRESHOLD = 100 * 1024 * 1024  # 100MB
        if fsize >= LARGE_THRESHOLD:
            # 大きい画像なので処理用幅を縮小して操作性を確保
            try:
                self._dbg(f"Large image detected: {fsize} bytes. Enabling lightweight processing.")
            except Exception:
                pass
            # 保存してから縮小
            self._prev_proc_target_width = getattr(self, 'proc_target_width', PROC_TARGET_WIDTH)
            self.proc_target_width = max(200, PROC_TARGET_WIDTH // 2)
            self._large_file_hint = True
        else:
            self._large_file_hint = False

        # update status
        try:
            if getattr(self, '_large_file_hint', False):
                self.ui_footer.showMessage("軽負荷モード: 大きな画像を簡易処理中")
            else:
                self.ui_footer.showMessage("")
        except Exception:
            pass

        try:
            self.img_full = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.img_full is None:
                raise ValueError("画像の読み込みに失敗しました")
            save_last_image_path(fname)
        except Exception as e:
            print("画像読み込みエラー:", e)
            self.img_full = None
            return
        self._build_processing_image()
        try:
            if getattr(self, '_large_file_hint', False):
                self._dbg(f"Processing image with proc_target_width={self.proc_target_width}")
        except Exception:
            pass
        # 画像が変わったのでキャッシュ破棄
        self._cache = {"img_id": id(self.proc_img), "levels": None, "min_area": None, "trim_px": None, "poster": None, "centroids": None}
        # 次回更新時に画像中心へスクロール
        self._initial_center_done = False
        self.schedule_update(force=True)

    # 自動デバッグ実行: 前回画像を読み込み、更新後に終了
    def run_auto_and_exit(self):
        """前回の画像を自動で読み込み、初回更新が完了したらアプリを終了する。"""
        last_path = load_last_image_path()
        if not last_path or not os.path.isfile(last_path):
            # 対象が無ければ即終了
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.quit)
            return
        self._auto_exit_after_update = True
        self._open_image_from_path(last_path)

    # 処理用画像を構築 (リサイズしてPROC_TARGET_WIDTHに合わせる)
    def _build_processing_image(self):
        if self.img_full is None:
            self.proc_img = None
            self.scale_proc_to_full = 1.0
            return
        h, w = self.img_full.shape[:2]
        try:
            self._dbg(f"_build_processing_image: full size={w}x{h}, target_width={self.proc_target_width}")
        except Exception:
            pass
        if w <= self.proc_target_width:
            self.proc_img = self.img_full.copy()
            self.scale_proc_to_full = 1.0
        else:
            scale = self.proc_target_width / float(w)
            new_w = self.proc_target_width
            new_h = max(1, int(round(h * scale)))
            self.proc_img = cv2.resize(self.img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.scale_proc_to_full = 1.0 / scale
        self.centroid_processor = CentroidProcessor(self.proc_img, self.scale_proc_to_full, self.img_full)
        try:
            self._dbg(f"_build_processing_image: proc_img size={self.proc_img.shape[1]}x{self.proc_img.shape[0]}")
        except Exception:
            pass

    def _disable_win_shadow(self):
        """Disable Windows DWM non-client rendering to remove the OS drop-shadow/frame.

        This is a best-effort Windows-only call. It will be a no-op on other platforms.
        """
        if os.name != 'nt':
            return
        try:
            hwnd = int(self.winId())
            # DWMWA_NCRENDERING_POLICY = 2, DWMNCRP_DISABLED = 0
            DWMWA_NCRENDERING_POLICY = 2
            DWMNCRP_DISABLED = 0
            val = ctypes.c_int(DWMNCRP_DISABLED)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_NCRENDERING_POLICY), ctypes.byref(val), ctypes.sizeof(val))
        except Exception:
            pass

    def _force_win_frameless(self):
        """Force-remove Windows non-client styles (caption/border) using SetWindowLongW.

        Note: This disables native resize grips; current UI already provides custom controls.
        """
        if os.name != 'nt':
            return
        GWL_STYLE = -16
        WS_OVERLAPPED = 0x00000000
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            # Remove caption + thickframe + sysmenu boxes; keep sysmenu disabled for true frameless
            style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_OVERLAPPED)
            ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                                              0x0002 | 0x0001 | 0x0020 | 0x0040)  # SWP_NOSIZE|SWP_NOMOVE|SWP_NOZORDER|SWP_FRAMECHANGED
        except Exception:
            pass

    def _apply_windows_titlebar_style(self):
        """Attempt to set native Windows titlebar colors/text/border and corner preference.

        This is best-effort and will silently no-op on unsupported Windows versions.
        """
        if os.name != 'nt':
            return
        try:
            hwnd = int(self.winId())
            # Common DWM attribute IDs (may vary by OS build)
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWA_BORDER_COLOR = 34
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            DWMWCP_DONOTROUND = 1

            # COLORREF values are 0x00BBGGRR (BGR). RGB(160,15,15) => 0x000F0FA0
            caption_color = ctypes.c_uint(0x000F0FA0)
            text_color = ctypes.c_uint(0x00FFFFFF)
            border_color = ctypes.c_uint(0x000F0FA0)
            dark_mode = ctypes.c_int(1)
            corner_pref = ctypes.c_int(DWMWCP_DONOTROUND)

            dwm = ctypes.windll.dwmapi
            # Apply attributes individually and ignore failures
            try:
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE), ctypes.byref(dark_mode), ctypes.sizeof(dark_mode))
            except Exception:
                pass
            try:
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_CAPTION_COLOR), ctypes.byref(caption_color), ctypes.sizeof(caption_color))
            except Exception:
                pass
            try:
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_TEXT_COLOR), ctypes.byref(text_color), ctypes.sizeof(text_color))
            except Exception:
                pass
            try:
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_BORDER_COLOR), ctypes.byref(border_color), ctypes.sizeof(border_color))
            except Exception:
                pass
            try:
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_WINDOW_CORNER_PREFERENCE), ctypes.byref(corner_pref), ctypes.sizeof(corner_pref))
            except Exception:
                pass
        except Exception:
            pass

    # 更新をスケジュール (タイマーで遅延実行、forceで即時)
    def schedule_update(self, force=False):
        if force:
            self.update_timer.stop()
            self._update_image_actual()
        else:
            self.update_timer.start()

        # 現在の処理パラメータを取得
    def _get_params(self):
        # Number of Groups is the single source of truth for k-means levels.
        try:
            levels = int(getattr(self, 'slider_num_groups', None).value() if hasattr(self, 'slider_num_groups') else 2)
        except Exception:
            levels = 2

        # Grain size thresholds: use histogram selection in both Basic/Advanced.
        if getattr(self, 'area_hist', None) is not None:
            try:
                sel_min, sel_max = self.area_hist.selection()
            except Exception:
                sel_min, sel_max = (None, None)
        else:
            sel_min, sel_max = (None, None)

        try:
            min_area = int(round(float(sel_min))) if sel_min is not None else int(self.slider_min_area.value())
        except Exception:
            min_area = int(self.slider_min_area.value())
        try:
            max_area = int(round(float(sel_max))) if sel_max is not None else None
        except Exception:
            max_area = None

        params = dict(
            levels=levels,
            min_area=min_area,
            max_area=max_area,
            trim_px=self.slider_trim.value(),
            neck_separation=int(getattr(self, 'slider_neck_sep', None).value() if hasattr(self, 'slider_neck_sep') else 0),
            shape_complexity=int(getattr(self, 'slider_shape_complex', None).value() if hasattr(self, 'slider_shape_complex') else 10),
        )
        return params

    def _on_area_hist_range_changed(self, sel_min, sel_max):
        # Histogram selection drives Advanced min/max area filters.
        try:
            # Keep the hidden slider in sync for consistency (best-effort; clamped).
            v = int(round(float(sel_min)))
            v = max(self.slider_min_area.minimum(), min(self.slider_min_area.maximum(), v))
            try:
                self.slider_min_area.setValue(v)
            except Exception:
                pass
            try:
                self.edit_min_area.setText(str(v))
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.schedule_update()
        except Exception:
            pass

    def _update_area_histogram(self, areas):
        if getattr(self, 'area_hist', None) is None:
            return
        try:
            import numpy as _np
            import math
            arr = _np.array([a for a in areas if a is not None and a > 0], dtype=float)
            if arr.size == 0:
                self.area_hist.clear(); return
            mn = float(arr.min()); mx = float(arr.max())
            if mx <= 0:
                self.area_hist.clear(); return
            if mn <= 0:
                mn = min([v for v in arr if v > 0] + [1.0])
            if mx <= mn:
                mx = mn * 1.1
            bins = _np.logspace(math.log10(mn), math.log10(mx), num=21)
            # 面積の総和（赤線）
            vals, edges = _np.histogram(arr, bins=bins, weights=arr)
            # 粒子数（灰色線）
            counts, _ = _np.histogram(arr, bins=bins)
            self.area_hist.set_data(edges.tolist(), vals.tolist(), counts.tolist())

            # Auto-initialize Min/Max based on curve inflection points.
            # Min: left inflection of particle count peak
            # Max: right inflection of area peak
            try:
                if counts.size > 0 and vals.size > 0:
                    # Find particle count (Grain No.) peak
                    if counts.max() > 0:
                        count_peak_idx = int(_np.argmax(counts))
                        
                        # Find left inflection point of counts curve (max slope on left side)
                        if count_peak_idx > 1:
                            left_counts = counts[:count_peak_idx + 1]
                            # Compute derivative
                            diff_counts = _np.diff(left_counts)
                            if diff_counts.size > 0:
                                # Find where slope is steepest (max derivative)
                                max_slope_idx = int(_np.argmax(_np.abs(diff_counts)))
                                sel_min = float(edges[max_slope_idx])
                            else:
                                sel_min = float(edges[0])
                        else:
                            sel_min = float(edges[0])
                    else:
                        sel_min = float(edges[0])
                    
                    # Find area (total area) peak
                    if vals.max() > 0:
                        area_peak_idx = int(_np.argmax(vals))
                        
                        # Find right inflection point of area curve (max slope on right side)
                        if area_peak_idx < len(vals) - 1:
                            right_vals = vals[area_peak_idx:]
                            # Compute derivative
                            diff_vals = _np.diff(right_vals)
                            if diff_vals.size > 0:
                                # Find where slope is steepest (max derivative on right)
                                max_slope_idx = int(_np.argmax(_np.abs(diff_vals)))
                                sel_max = float(edges[area_peak_idx + max_slope_idx + 1])
                            else:
                                sel_max = float(edges[-1])
                        else:
                            sel_max = float(edges[-1])
                    else:
                        sel_max = float(edges[-1])
                    
                    # Clamp to valid range
                    sel_min = max(float(mn), min(float(mx), sel_min))
                    sel_max = max(float(mn), min(float(mx), sel_max))
                    if sel_min > sel_max:
                        sel_min, sel_max = sel_max, sel_min
                    
                    self.area_hist.maybe_autoset_selection(sel_min, sel_max)
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] Auto-init histogram selection failed: {e}")
                pass
        except Exception:
            try:
                self.area_hist.clear()
            except Exception:
                pass

    def _update_image_actual(self):
        if self._painting:
            return
        self._painting = True
        try:
            if self.img_full is None:
                self.img_label_proc.clear()
                self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, [], self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
                self.centroids = []
                self._img_base_size = None
                try:
                    if getattr(self, 'area_hist', None) is not None:
                        self.area_hist.clear()
                except Exception:
                    pass
                return
            # 単一ビュー表示のため、左側原画像の描画は廃止
            params = self._get_params()
            if self.proc_img is None:
                self._build_processing_image()
            overlay_full = self.img_full.copy()
            centroids = []
            if self.centroid_processor:
                # 判定: 自動更新モードか手動モードかで重い処理の実行を切り替える
                cache_img_id = self._cache.get("img_id")
                cache_levels = self._cache.get("levels")
                cache_min_area = self._cache.get("min_area")
                cache_max_area = self._cache.get("max_area")
                cache_trim = self._cache.get("trim_px")
                cache_neck = self._cache.get("neck_separation")
                cache_shape = self._cache.get("shape_complexity")
                cache_poster = self._cache.get("poster")
                cache_centroids = self._cache.get("centroids")
                cache_areas = self._cache.get("areas")
                areas_now = cache_areas
                boundary_mask_now = self._cache.get("boundary_mask")

                need_poster_recalc = (
                    cache_poster is None
                    or cache_levels != params["levels"]
                    or cache_img_id != id(self.proc_img)
                    or cache_neck != params.get("neck_separation")
                    or cache_shape != params.get("shape_complexity")
                )

                # 自動モードでは通常通り重い処理を行う
                if self.auto_update_mode:
                    if need_poster_recalc:
                        poster = kmeans_posterize(self.proc_img, params["levels"])
                        centroids = self.centroid_processor.get_centroids(params, poster=poster)
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                        self._cache.update({
                            "img_id": id(self.proc_img),
                            "levels": params["levels"],
                            "min_area": params["min_area"],
                            "max_area": params.get("max_area"),
                            "trim_px": params["trim_px"],
                            "neck_separation": params.get("neck_separation"),
                            "shape_complexity": params.get("shape_complexity"),
                            "poster": poster,
                            "centroids": centroids,
                            "areas": areas_now,
                            "boundary_mask": boundary_mask_now,
                        })
                    else:
                        # reuse cached poster
                        poster = cache_poster
                        # If only min_area/trim changed, recompute centroids from cached poster
                        if (cache_poster is not None) and (
                            cache_min_area != params.get("min_area")
                            or cache_max_area != params.get("max_area")
                            or cache_trim != params.get("trim_px")
                            or cache_neck != params.get("neck_separation")
                            or cache_shape != params.get("shape_complexity")
                        ):
                            try:
                                centroids = self.centroid_processor.get_centroids(params, poster=poster)
                                areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                                boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                                # update cached params and centroids (keep poster and img_id/levels)
                                self._cache.update({
                                    "trim_px": params["trim_px"],
                                    "max_area": params.get("max_area"),
                                    "neck_separation": params.get("neck_separation"),
                                    "shape_complexity": params.get("shape_complexity"),
                                    "centroids": centroids,
                                    "areas": areas_now,
                                    "boundary_mask": boundary_mask_now,
                                })
                            except Exception:
                                centroids = cache_centroids
                                areas_now = cache_areas
                        else:
                            centroids = cache_centroids
                            areas_now = cache_areas
                            boundary_mask_now = self._cache.get("boundary_mask")
                else:
                    # 手動モード: 可能ならキャッシュを使い、重い poster 再生成は行わない
                    if cache_poster is not None and cache_img_id == id(self.proc_img):
                        poster = cache_poster
                        # Use centroid_processor to recompute centroids from cached poster with current params
                        try:
                            centroids = self.centroid_processor.get_centroids(params, poster=poster)
                            areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                            boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                            # Keep cache in sync for histogram/boundary reuse
                            self._cache.update({
                                "trim_px": params.get("trim_px"),
                                "max_area": params.get("max_area"),
                                "neck_separation": params.get("neck_separation"),
                                "shape_complexity": params.get("shape_complexity"),
                                "centroids": centroids,
                                "areas": areas_now,
                                "boundary_mask": boundary_mask_now,
                            })
                        except Exception:
                            # fallback to cached centroids if recompute fails
                            if cache_centroids is not None:
                                centroids = cache_centroids
                                areas_now = cache_areas
                                boundary_mask_now = self._cache.get("boundary_mask")
                    else:
                        # キャッシュが無ければフォールバックで軽めに計算（呼び出し元でエラーは吸収）
                        poster = kmeans_posterize(self.proc_img, params["levels"])
                        centroids = self.centroid_processor.get_centroids(params, poster=poster)
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                        self._cache.update({
                            "img_id": id(self.proc_img),
                            "levels": params["levels"],
                            "min_area": params["min_area"],
                            "max_area": params.get("max_area"),
                            "trim_px": params["trim_px"],
                            "neck_separation": params.get("neck_separation"),
                            "shape_complexity": params.get("shape_complexity"),
                            "poster": poster,
                            "centroids": centroids,
                            "areas": areas_now,
                            "boundary_mask": boundary_mask_now,
                        })
                # 表示用にポスター画像をフル解像度へ拡大
                scale = 1.0 / self.scale_proc_to_full
                if scale != 1.0:
                    new_w = self.img_full.shape[1]
                    new_h = self.img_full.shape[0]
                    poster_full = cv2.resize(poster, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    # Boundary のエッジ検出は最近傍で拡大したポスターを使う（線が太くなる原因を避ける）
                    poster_edges_full = cv2.resize(poster, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    poster_full = poster.copy()
                    poster_edges_full = poster_full
                # Overlay selection by mode: Original / Posterized (Mixed removed)
                try:
                    overlay_mode = str(getattr(self, 'overlay_mode', 'Mixed')).lower()
                except Exception:
                    overlay_mode = 'original'
                if overlay_mode == 'original':
                    overlay_full = self.img_full.copy()
                else:
                    overlay_full = poster_full.copy()

                try:
                    self._update_area_histogram(areas_now or [])
                except Exception:
                    pass
                # ポスタリゼーション境界に白線を描画（オプション）
                try:
                    if self.show_boundaries:
                        # エッジ検出は最近傍補間（ギザ）版を使って細い境界を得る
                        # Build poster_for_edges at full resolution and apply trim in full-pixel units
                        try:
                            trim_px_full = int(params.get('trim_px', 0) or 0)
                        except Exception:
                            trim_px_full = 0
                        try:
                            # edge detection uses nearest-upscaled poster to avoid thick edges
                            poster_fe = poster_edges_full.copy()
                            if trim_px_full > 0:
                                kf = int(trim_px_full)
                                ker = np.ones((3, 3), np.uint8)
                                out_full = np.zeros_like(poster_fe)
                                unique_colors_full = np.unique(poster_fe.reshape(-1, 3), axis=0)
                                for color in unique_colors_full:
                                    mask = cv2.inRange(poster_fe, color, color)
                                    mask_e = cv2.erode(mask, ker, iterations=kf)
                                    out_full[mask_e == 255] = color
                                edge_src = out_full
                            else:
                                edge_src = poster_fe
                        except Exception:
                            # fallback to poster_full if anything goes wrong
                            try:
                                edge_src = poster_full
                            except Exception:
                                edge_src = poster
                        h, w = edge_src.shape[:2]
                        edge_mask = None
                        # Prefer using the post-filter boundary mask from centroid_processor if available.
                        try:
                            if boundary_mask_now is not None:
                                bm = boundary_mask_now
                                if bm.shape[:2] != (h, w):
                                    bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
                                edge_mask = bm.astype(np.uint8)
                        except Exception:
                            edge_mask = None

                        if edge_mask is None:
                            # Use Canny edge detector on nearest-upscaled poster to get crisp 1px edges.
                            try:
                                gray = cv2.cvtColor(edge_src, cv2.COLOR_BGR2GRAY)
                                # thresholds chosen to be permissive; poster edges are high-contrast
                                edges = cv2.Canny(gray, 30, 100)
                                # If Canny finds nothing (possible for some posters), fallback to diff-based
                                if edges is None or not edges.any():
                                    diff_h = np.any(edge_src[:, 1:, :] != edge_src[:, :-1, :], axis=2)
                                    diff_v = np.any(edge_src[1:, :, :] != edge_src[:-1, :, :], axis=2)
                                    edge_mask = np.zeros((h, w), dtype=np.uint8)
                                    edge_mask[:, 1:][diff_h] = 255
                                    edge_mask[1:, :][diff_v] = 255
                                else:
                                    edge_mask = edges.copy()
                            except Exception:
                                # Fallback to difference-based detection if Canny fails
                                diff_h = np.any(edge_src[:, 1:, :] != edge_src[:, :-1, :], axis=2)
                                diff_v = np.any(edge_src[1:, :, :] != edge_src[:-1, :, :], axis=2)
                                edge_mask = np.zeros((h, w), dtype=np.uint8)
                                edge_mask[:, 1:][diff_h] = 255
                                edge_mask[1:, :][diff_v] = 255
                        # 黒枠は不要 → スムージング（ガウシアン）で柔らかい白線へ
                        # trim_px_full==0 のときは、重なって太く見えるのを抑えるため
                        # - 事前に軽く erode して線を細くする
                        # - ブラー強度を小さくして細い線を作る
                        # - 最終的な alpha を少し抑えて視覚的な太さを揃える
                        try:
                            is_zero = int(trim_px_full) == 0
                        except Exception:
                            is_zero = False
                        # Keep boundaries thin: avoid blur (which makes them look thicker)
                        # Note: avoid aggressive erosion which can remove 1px edges.
                        try:
                            alpha = (edge_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
                            # Make edges clearly visible but not too heavy; slightly lower weight for trim=0 case
                            alpha *= 0.60 if is_zero else 0.80
                        except Exception:
                            alpha = (edge_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
                        # 白を alpha でブレンド
                        overlay_full = overlay_full.astype(np.float32)
                        overlay_full = overlay_full * (1.0 - alpha) + 255.0 * alpha
                        overlay_full = np.clip(overlay_full, 0, 255).astype(np.uint8)
                except Exception:
                    # 万一の失敗時は何もしない（オーバーレイはそのまま）
                    pass
                # マーカーは等倍時の画像に焼き込まず、ズーム後にQPainterで上描きする
            # 右画像のベースサイズを保存（フル画像サイズ)
            self._img_base_size = (overlay_full.shape[1], overlay_full.shape[0])

            # データ反映を先に行い、描画前に最新の点群を反映させる（灰色丸を即表示）
            self.centroids = centroids

            # 真ん中の転置表へ raw X/Y を即反映（populate/refresh が遅延しても見えるように）
            try:
                rv = getattr(self, 'table_between', None)
                if rv is not None:
                    from qt_compat.QtWidgets import QTableWidgetItem
                    # columns correspond to right-table row labels: X,Y,CalcX,CalcY,CalcZ
                    need_cols = 5
                    if rv.columnCount() != need_cols:
                        rv.setColumnCount(need_cols)
                        try:
                            rv.setHorizontalHeaderLabels(STR.TABLE_RIGHT_ROW_LABELS)
                        except Exception:
                            pass
                    need_rows = int(len(self.centroids)) if self.centroids is not None else 0
                    if rv.rowCount() != need_rows:
                        rv.setRowCount(need_rows)
                        try:
                            rv.setVerticalHeaderLabels([str(i + 1) for i in range(need_rows)])
                        except Exception:
                            pass
                    for i, c in enumerate(self.centroids or []):
                        try:
                            _, x, y = c
                            sx = str(int(round(x)))
                            sy = str(int(round(y)))
                            itx = rv.item(i, 0)
                            if itx is None:
                                itx = QTableWidgetItem("")
                                rv.setItem(i, 0, itx)
                            ity = rv.item(i, 1)
                            if ity is None:
                                ity = QTableWidgetItem("")
                                rv.setItem(i, 1, ity)
                            itx.setText(sx)
                            ity.setText(sy)
                        except Exception:
                            pass
            except Exception:
                pass
            # 選択インデックスが範囲外なら解除
            if self.selected_index is not None and not (0 <= self.selected_index < len(self.centroids)):
                self.selected_index = None

            # 右側オーバーレイ画像を保持（フル解像度）
            self._last_overlay_full = overlay_full
            self._apply_proc_zoom()

            # 初回描画後に画像中心へスクロール（スクロール範囲反映後に行うため 0ms ディレイ）
            if not self._initial_center_done and self._img_base_size is not None:
                cx = self._img_base_size[0] / 2.0
                cy = self._img_base_size[1] / 2.0
                try:
                    QTimer.singleShot(0, lambda: self._ensure_full_pos_visible(cx, cy))
                except Exception:
                    # 何かあっても一度だけ試みる
                    pass

                self._initial_center_done = True

            # テーブル更新
            self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, self.centroids, self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
            try:
                self._refresh_transposed_views()
            except Exception:
                pass
            try:
                # ensure selection sync after refresh
                QTimer.singleShot(0, self._sync_table_selection)
            except Exception:
                pass
            # 画像表示更新
            self._apply_proc_zoom()
        finally:
            self._painting = False
            # 自動デバッグモードなら、一度処理が走ったら終了
            if self._auto_exit_after_update:
                self._auto_exit_after_update = False
                app = QApplication.instance()
                if app is not None:
                    try:
                        QTimer.singleShot(0, app.quit)
                    except Exception:
                        app.quit()

    def _apply_proc_zoom(self):
        # Simplified rendering: do not use virtual canvas or PatchWorker.
        # Build a pixmap for the current overlay (or proc_img fallback) and then draw grid/rotation if needed.
        source_img = self._last_overlay_full if self._last_overlay_full is not None else self.proc_img
        if source_img is None:
            self.img_label_proc.clear()
            return

        try:
            pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
                source_img,
                self.proc_zoom,
                self.view_padding,
                self.centroids,
                self.selected_index,
                self.ref_points,
                self.scale_proc_to_full,
                colors=None,
                interp_mode=self.interp_mode,
            )
        except Exception:
            pm = None
            off_x = off_y = 0
            new_w = new_h = 0

        if pm is None:
            self.img_label_proc.clear()
            return

        # Helper: compute similarity transform (scale, rotation, translation)
        def _compute_similarity_transform(img_pts, stage_pts):
            try:
                import numpy as _np
                if len(img_pts) < 2 or len(stage_pts) < 2:
                    return None
                src = _np.asarray(img_pts, dtype=_np.float64)
                dst = _np.asarray(stage_pts, dtype=_np.float64)
                # use Umeyama-like similarity estimation
                src_mean = src.mean(axis=0)
                dst_mean = dst.mean(axis=0)
                src_c = src - src_mean
                dst_c = dst - dst_mean
                # covariance
                cov = (dst_c.T @ src_c) / src.shape[0]
                U, S, Vt = _np.linalg.svd(cov)
                R = U @ Vt
                # ensure proper rotation (no accidental reflection)
                det = _np.linalg.det(R)
                reflect = False
                if det < 0:
                    # force proper rotation by flipping sign on last column of U
                    U[:, -1] *= -1
                    R = U @ Vt
                    reflect = True
                # scale
                var_src = (src_c ** 2).sum() / src.shape[0]
                s = _np.trace(_np.diag(S)) / var_src if var_src > 1e-12 else 1.0
                t = dst_mean - s * (R @ src_mean)
                # rotation angle (radians)
                angle = _np.arctan2(R[1, 0], R[0, 0])
                return {'s': float(s), 'R': R, 't': t, 'angle_rad': float(angle), 'angle_deg': float(_np.degrees(angle)), 'reflect': bool(reflect)}
            except Exception:
                return None

        # Helper: build stage transform info from available reference points
        def _get_stage_alignment_info():
            try:
                pts_img = []
                pts_stage = []
                # Gather pairs where we have both image (proc->full) and numeric stage obs
                for i, rp in enumerate(getattr(self, 'ref_points', []) or []):
                    if rp is None:
                        continue
                    # image full coords
                    try:
                        x_full = float(rp[0]) * float(getattr(self, 'scale_proc_to_full', 1.0))
                        y_full = float(rp[1]) * float(getattr(self, 'scale_proc_to_full', 1.0))
                    except Exception:
                        continue
                    # stage obs
                    try:
                        ro = (getattr(self, 'ref_obs', []) or [])
                        ro_i = ro[i] if i < len(ro) else None
                        if not ro_i:
                            continue
                        sx = ro_i.get('x', '')
                        sy = ro_i.get('y', '')
                        if sx is None or sy is None:
                            continue
                        sx_f = float(str(sx).replace(',', '').strip())
                        sy_f = float(str(sy).replace(',', '').strip())
                    except Exception:
                        continue
                    pts_img.append((x_full, y_full))
                    pts_stage.append((sx_f, sy_f))
                self._dbg(f"Stage alignment: found {len(pts_img)} valid point pairs")
                if len(pts_img) < 2:
                    self._dbg(f"Insufficient ref points for stage transform (need ≥2, have {len(pts_img)})")
                    return None
                result = _compute_similarity_transform(pts_img, pts_stage)
                if result:
                    self._dbg(f"Transform computed: angle={result.get('angle_deg', 0):.2f}deg, scale={result.get('s', 1):.3f}")
                return result
            except Exception as e:
                self._dbg(f"Stage alignment error: {e}")
                return None

        # Helper: map a stage point to display pixmap coords given build_zoomed_canvas outputs
        def _stage_to_display(p_stage, info, off_x, off_y, display_scale, pad, img_full_w, img_full_h, angle_deg, reflect):
            try:
                import math
                import numpy as _np
                # inverse similarity: img = (1/s) R^T (stage - t)
                s = float(info['s'])
                R = _np.asarray(info['R'], dtype=_np.float64)
                t = _np.asarray(info['t'], dtype=_np.float64)
                stage = _np.asarray(p_stage, dtype=_np.float64)
                img_full = (1.0 / s) * (R.T @ (stage - t))
                # display coords before rotation/flip
                xd = float(img_full[0]) * display_scale + off_x
                yd = float(img_full[1]) * display_scale + off_y
                # rotate around image center
                cx = pad + (img_full_w * display_scale) / 2.0
                cy = pad + (img_full_h * display_scale) / 2.0
                theta = -math.radians(angle_deg)
                # apply flip if needed (reflect implies mirror across X in stage->image, map to horizontal flip of image)
                x_rel = xd - cx
                y_rel = yd - cy
                if reflect:
                    x_rel = -x_rel
                xr = x_rel * math.cos(theta) - y_rel * math.sin(theta)
                yr = x_rel * math.sin(theta) + y_rel * math.cos(theta)
                return (int(round(cx + xr)), int(round(cy + yr)))
            except Exception:
                return None

        # Build pixmap from the full overlay and display it directly
        pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
            self._last_overlay_full,
            self.proc_zoom,
            self.view_padding,
            self.centroids,
            self.selected_index,
            self.ref_points,
            self.scale_proc_to_full,
            colors=None,
            interp_mode=self.interp_mode,
        )
        try:
            # Compute display_scale from actual drawn pixels so full<->display mapping stays consistent
            pad = int(self.view_padding)
            full_w = int(self._img_base_size[0]) if self._img_base_size is not None else max(1, new_w)
            drawn_w = max(1, pm.width() - 2 * pad)
            self._display_scale = float(drawn_w) / float(full_w)
            # physical offset in label coordinates
            self._display_offset = (pad, pad)
        except Exception:
            try:
                self._display_scale = float(self.proc_zoom)
            except Exception:
                self._display_scale = 1.0
            self._display_offset = (off_x, off_y)
        if pm is None:
            self.img_label_proc.clear()
            return
        self._display_img_size = (new_w, new_h)
        self._display_pm_base = pm
        # update statusbar
        try:
            msg = ""
            if getattr(self, '_large_file_hint', False):
                msg = "軽負荷モード有効"
            if msg:
                self.ui_footer.showMessage(msg)
        except Exception:
            pass

        # Directly set pixmap and resize label to match pixmap size (no virtual canvas)
        try:
            # If we have at least 2-3 reference points with observed stage coords, compute alignment
            try:
                info = _get_stage_alignment_info()
            except Exception as e:
                import traceback
                self._dbg(f"_get_stage_alignment_info failed: {e}")
                self._dbg(traceback.format_exc())
                info = None

            # Reuse last valid stage transform in Stage mode so grid/images don't vanish when alignment is temporarily unavailable
            try:
                current_orient = getattr(self, 'view_orientation', 'Image')
                if info is not None:
                    self._last_stage_info = info
                elif current_orient == 'Stage' and getattr(self, '_last_stage_info', None) is not None:
                    info = self._last_stage_info
            except Exception:
                pass

            pm_to_show = pm
            try:
                # Apply rotation when Stage is selected, grid for both modes
                view_orient = getattr(self, 'view_orientation', 'Image')
                if view_orient == 'Stage':
                    if info is None:
                        self._dbg(f"Stage mode selected but transform info not available (need ≥2 ref points with obs)")
                    else:
                        self._dbg(f"Applying Stage rotation: angle={info.get('angle_deg', 0.0):.2f}deg")
                
                # Draw rotated image for Stage mode
                if info is not None and view_orient == 'Stage':
                    # draw rotated/possibly reflected image and overlay grid
                    from qt_compat.QtGui import QPixmap, QPainter, QTransform, QPen, QColor
                    pad = int(self.view_padding)
                    draw_w, draw_h = (new_w, new_h)
                    # copy base pixmap and extract the drawn image region
                    try:
                        img_region = pm.copy(pad, pad, draw_w, draw_h)
                    except Exception:
                        img_region = pm
                    # create transform: rotate by -angle so stage X -> right, Y -> up
                    angle = float(info.get('angle_deg', 0.0))
                    transform = QTransform()
                    # rotate around center
                    cx = img_region.width() / 2.0
                    cy = img_region.height() / 2.0
                    # If reflection was detected, apply horizontal flip
                    if info.get('reflect', False):
                        transform.translate(cx, cy)
                        transform.scale(-1, 1)
                        transform.translate(-cx, -cy)
                    # rotate so that stage X points to the right (negative because displayed Y axis is downward)
                    transform.translate(cx, cy)
                    transform.rotate(-angle)
                    transform.translate(-cx, -cy)
                    rotated = img_region.transformed(transform, Qt.SmoothTransformation)
                    # compose new canvas sized to fit the rotated pixmap (avoid clipping)
                    rot_w = rotated.width()
                    rot_h = rotated.height()
                    pm2 = QPixmap(rot_w + 2 * pad, rot_h + 2 * pad)
                    pm2.fill(QColor(30, 30, 30))
                    p = QPainter(pm2)
                    p.drawPixmap(pad, pad, rotated)

                    # Draw stage grid lines on rotated image (simple orthogonal grid)
                    try:
                        import numpy as _np
                        import math
                        
                        # After rotation, draw straight grid based on stage coordinates
                        display_scale = getattr(self, '_display_scale', None)
                        if display_scale is None:
                            display_scale = float(self.proc_zoom)
                        
                        # Determine stage pixel scale: display pixels per stage unit = display_scale / s
                        s = float(info.get('s', 1.0))
                        px_per_stage = display_scale / max(1e-12, s)
                        
                        # Choose nice spacing in stage units so spacing in px is in [50,200]
                        target_px = 100.0
                        candidates = []
                        base = [1, 2, 5]
                        for e in range(-3, 6):
                            for b in base:
                                candidates.append(b * (10 ** e))
                        spacing = candidates[0]
                        for c in candidates:
                            if 50 <= (c * px_per_stage) <= 220:
                                spacing = c
                                break
                        
                        # Get stage bounds
                        w_full = int(self._img_base_size[0]) if getattr(self, '_img_base_size', None) else draw_w
                        h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else draw_h
                        corners = [(0.0, 0.0), (w_full, 0.0), (w_full, h_full), (0.0, h_full)]
                        R = _np.asarray(info.get('R'))
                        t = _np.asarray(info.get('t'))
                        s_val = float(info.get('s', 1.0))
                        stage_pts = [_np.asarray([c[0], c[1]]) for c in corners]
                        stage_corners = [s_val * (R @ p) + t for p in stage_pts]
                        xs = [p[0] for p in stage_corners]
                        ys = [p[1] for p in stage_corners]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        
                        start_x = math.floor(xmin / spacing) * spacing
                        end_x = math.ceil(xmax / spacing) * spacing
                        start_y = math.floor(ymin / spacing) * spacing
                        end_y = math.ceil(ymax / spacing) * spacing
                        
                        pen = QPen(QColor(200, 200, 200, 140))
                        pen.setWidth(1)
                        p.setPen(pen)
                        font = p.font()
                        font.setPointSize(9)
                        p.setFont(font)
                        
                        # Image is already rotated, so draw straight orthogonal grid
                        # Map stage coords to rotated canvas coords
                        # Center of rotated image
                        cx = pad + rotated.width() / 2.0
                        cy = pad + rotated.height() / 2.0
                        
                        # Draw vertical lines (constant stage X)
                        x = start_x
                        while x <= end_x + 1e-9:
                            # In stage coords: vertical line at x, from ymin to ymax
                            # Convert to display: stage units to pixels, centered on rotated canvas
                            # Stage origin to display
                            x_offset = (x - (xmin + xmax) / 2.0) * px_per_stage
                            x_disp = cx + x_offset
                            y_top = pad
                            y_bottom = pad + rotated.height()
                            p.drawLine(int(x_disp), int(y_top), int(x_disp), int(y_bottom))
                            try:
                                lbl = f"{x:.3g}"
                                p.drawText(int(x_disp) + 4, int(y_top) + 14, lbl)
                            except Exception:
                                pass
                            x += spacing
                        
                        # Draw horizontal lines (constant stage Y)
                        y = start_y
                        while y <= end_y + 1e-9:
                            # In stage coords: horizontal line at y, from xmin to xmax
                            y_offset = (y - (ymin + ymax) / 2.0) * px_per_stage
                            y_disp = cy + y_offset
                            x_left = pad
                            x_right = pad + rotated.width()
                            p.drawLine(int(x_left), int(y_disp), int(x_right), int(y_disp))
                            try:
                                lbl = f"{y:.3g}"
                                p.drawText(int(x_left) + 4, int(y_disp) - 4, lbl)
                            except Exception:
                                pass
                            y += spacing
                    except Exception:
                        pass
                    try:
                        p.end()
                    except Exception:
                        pass
                    pm_to_show = pm2
                
                # Draw pixel grid overlay for Image mode (simple pixel coordinates)
                elif view_orient == 'Image':
                    try:
                        from qt_compat.QtGui import QPixmap, QPainter, QPen, QColor
                        import math
                        
                        # Create a copy to draw grid on
                        pm2 = QPixmap(pm)
                        p = QPainter(pm2)
                        
                        pad = int(self.view_padding)
                        
                        # For Image mode, use pixel coordinates
                        display_scale = getattr(self, '_display_scale', None)
                        if display_scale is None or display_scale <= 0:
                            try:
                                display_scale = float(self.proc_zoom)
                            except Exception:
                                display_scale = 1.0
                        display_scale = max(1e-4, float(display_scale))

                        # Choose spacing so that about ~8 lines appear across the visible area.
                        # Compute spacing in DISPLAY pixels, then convert to image pixels.
                        visible_w = max(20.0, pm.width() - 2 * pad)
                        target_lines = 8.0
                        spacing_display = max(30.0, min(180.0, visible_w / target_lines))
                        pixel_spacing = spacing_display / display_scale
                        # snap to nearest 10px in image coords for cleaner labels
                        pixel_spacing = max(10.0, round(pixel_spacing / 10.0) * 10.0)
                        
                        w_full = int(self._img_base_size[0]) if getattr(self, '_img_base_size', None) else new_w
                        h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else new_h
                        
                        pen = QPen(QColor(200, 200, 200, 140))
                        pen.setWidth(1)
                        p.setPen(pen)
                        font = p.font()
                        font.setPointSize(9)
                        p.setFont(font)
                        
                        # Vertical lines (constant X in image pixels)
                        x_px = 0.0
                        while x_px <= w_full + 1e-6:
                            x_disp = pad + x_px * display_scale
                            y_top = pad
                            y_bottom = pad + h_full * display_scale
                            p.drawLine(int(x_disp), int(y_top), int(x_disp), int(y_bottom))
                            try:
                                lbl = f"{int(round(x_px))}"
                                p.drawText(int(x_disp) + 4, int(y_top) + 14, lbl)
                            except Exception:
                                pass
                            x_px += pixel_spacing

                        # Horizontal lines (constant Y in image pixels)
                        y_px = 0.0
                        while y_px <= h_full + 1e-6:
                            x_left = pad
                            x_right = pad + w_full * display_scale
                            y_disp = pad + y_px * display_scale
                            p.drawLine(int(x_left), int(y_disp), int(x_right), int(y_disp))
                            try:
                                lbl = f"{int(round(y_px))}"
                                p.drawText(int(x_left) + 4, int(y_disp) - 4, lbl)
                            except Exception:
                                pass
                            y_px += pixel_spacing
                        
                        p.end()
                        pm_to_show = pm2
                        try:
                            self._last_pm_image_grid = pm2
                        except Exception:
                            pass
                    except Exception as e:
                        self._dbg(f"Image grid drawing failed: {e}")
                        # fallback: reuse last successful grid pixmap if available
                        try:
                            if getattr(self, '_last_pm_image_grid', None) is not None:
                                pm_to_show = self._last_pm_image_grid
                            else:
                                pm_to_show = pm
                        except Exception:
                            pm_to_show = pm
                        
            except Exception as e:
                import traceback
                self._dbg(f"Stage rotation failed with exception: {e}")
                self._dbg(traceback.format_exc())
                pm_to_show = pm

            # IMPORTANT: keep the latest rendered pixmap (including grid/rotation) as the base.
            # Some interaction handlers restore `img_label_proc` from `_display_pm_base` after a click
            # to clear transient overlays (e.g., crosshair). If `_display_pm_base` is kept as the
            # pre-grid pixmap, the grid appears to "disappear" right after Add/Update.
            try:
                self._display_pm_base = pm_to_show
            except Exception:
                pass

            self.img_label_proc.setPixmap(pm_to_show)
            try:
                self.img_label_proc.resize(pm_to_show.width(), pm_to_show.height())
            except Exception:
                pass
        except Exception:
            self.img_label_proc.clear()
            return

        # If pick mode active, redraw crosshair
        try:
            if self.pick_mode in ('add', 'update'):
                global_pt = QCursor.pos()
                vp = self.proc_scroll.viewport()
                pos_vp = vp.mapFromGlobal(global_pt)
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                self._draw_crosshair(pos_label)
        except Exception:
            pass

    def _viewport_pos_to_label_pos(self, pos):
        # スクロールビュー座標をラベル座標へ変換
        # ラベルの実位置（センタリング/スクロール反映）を考慮
        if self.proc_scroll is None:
            return pos
        label_pos_in_vp = self.img_label_proc.pos()  # QPoint (相対: viewport)
        return QPoint(pos.x() - label_pos_in_vp.x(), pos.y() - label_pos_in_vp.y())

    def _cleanup_threads(self):
        # Cancel and wait for any running patch worker to avoid QThread destroy errors
        try:
            prev = getattr(self, '_patch_worker', None)
            if prev is not None:
                try:
                    prev.cancel()
                except Exception:
                    pass
                try:
                    # wait briefly for thread to exit
                    prev.wait(1000)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_patch_ready(self, patch_array, left_label, top_label, request_id):
        # Only apply if request_id matches latest
        try:
            if request_id != getattr(self, '_patch_request_id', None):
                return
            if patch_array is None:
                return
            # convert to QPixmap and composite over base pixmap
            try:
                pm = QPixmap(self._display_pm_base)
                # If patch_array size != viewport size, resize it to viewport to fill region
                vp = self.proc_scroll.viewport()
                tgt_w = vp.width()
                tgt_h = vp.height()
                if patch_array.shape[1] != tgt_w or patch_array.shape[0] != tgt_h:
                    try:
                        patch_resized = cv2.resize(patch_array, (max(1, tgt_w), max(1, tgt_h)), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        patch_resized = patch_array
                else:
                    patch_resized = patch_array
                painter = QPainter(pm)
                painter.drawPixmap(left_label, top_label, cvimg_to_qpixmap(patch_resized))
                painter.end()
                self._display_pm_base = pm
                self.img_label_proc.setPixmap(pm)
            except Exception:
                pass
        except Exception:
            pass

    def _label_pos_to_viewport_pos(self, pos):
        # ラベル座標をビューポート座標へ変換
        if self.proc_scroll is None:
            return pos
        label_pos_in_vp = self.img_label_proc.pos()  # QPoint (相対: viewport)
        return QPoint(pos.x() + label_pos_in_vp.x(), pos.y() + label_pos_in_vp.y())

    def _ensure_full_pos_visible(self, x_full, y_full):
        # 指定フル座標がビューポート中心付近に来るようスクロールを調整
        dxy = self._full_to_display(x_full, y_full)
        if dxy is None:
            return
        lx, ly = dxy
        vp = self.proc_scroll.viewport()
        cx = lx - vp.width() / 2.0
        cy = ly - vp.height() / 2.0
        self._set_scroll(cx, cy)

    def _set_scroll(self, sx, sy):
        # スクロールバー値を範囲内に設定
        hsb = self.proc_scroll.horizontalScrollBar()
        vsb = self.proc_scroll.verticalScrollBar()
        hsb.setValue(max(hsb.minimum(), min(hsb.maximum(), int(round(sx)))))
        vsb.setValue(max(vsb.minimum(), min(vsb.maximum(), int(round(sy)))))

    def _start_kinetic(self, vx, vy):
        # 慣性スクロール開始（vx,vy は px/秒、スクロール方向の速度）
        self._kinetic_vx = float(vx)
        self._kinetic_vy = float(vy)
        self._kinetic_last_t = monotonic()
        if not self._kinetic_timer.isActive():
            self._kinetic_timer.start()

    def _stop_kinetic(self):
        self._kinetic_timer.stop()
        self._kinetic_vx = 0.0
        self._kinetic_vy = 0.0

    def _on_kinetic_tick(self):
        # 慣性スクロールの更新
        t = monotonic()
        dt = max(0.0, t - self._kinetic_last_t)
        self._kinetic_last_t = t
        if dt <= 0.0:
            return
        hsb = self.proc_scroll.horizontalScrollBar()
        vsb = self.proc_scroll.verticalScrollBar()
        # 現在位置更新
        sx = hsb.value() + self._kinetic_vx * dt
        sy = vsb.value() + self._kinetic_vy * dt
        # 端でのバウンド抑制：はみ出す方向の速度は殺す
        hit_edge_x = False
        hit_edge_y = False
        if sx <= hsb.minimum():
            sx = hsb.minimum(); hit_edge_x = True
        elif sx >= hsb.maximum():
            sx = hsb.maximum(); hit_edge_x = True
        if sy <= vsb.minimum():
            sy = vsb.minimum(); hit_edge_y = True
        elif sy >= vsb.maximum():
            sy = vsb.maximum(); hit_edge_y = True
        self._set_scroll(sx, sy)
        # 減衰（指数的）
        decay = 0.92  # 1ティックごと
        self._kinetic_vx *= decay
        self._kinetic_vy *= decay
        # エッジ命中で該当軸の速度を強制減衰
        if hit_edge_x:
            self._kinetic_vx *= 0.3
        if hit_edge_y:
            self._kinetic_vy *= 0.3
        # 終了条件
        if abs(self._kinetic_vx) < 5 and abs(self._kinetic_vy) < 5:
            self._stop_kinetic()

    # テーブル構築関連は tables.py に移動

    def _on_ref_table_current_changed(self, curRow, curCol, prevRow, prevCol):
        if curCol is None or curCol < 0:
            return
        self.ref_selected_index = curCol
        # 参照選択の変更では描画更新は不要

    def _on_ref_view_current_changed(self, curRow, curCol, prevRow, prevCol):
        """Selection change in transposed ref view.

        In the transposed view, each *row* corresponds to a reference-point index (source column).
        """
        try:
            if curRow is None or curRow < 0:
                return
            header_rows = 2
            if int(curRow) < header_rows:
                return
            self.ref_selected_index = int(curRow) - header_rows
        except Exception:
            pass

    def _on_add_ref_point(self):
        # キャンセルモード中なら、ピックモードを終了
        if self.pick_mode == 'add':
            self._end_pick_mode()
            return
        
        # 空きのRef列があればそこを選択し、RefクリックモードをON。
        # Ensure any pending edits in the transposed left view are committed
        try:
            self._flush_ref_view()
        except Exception:
            pass
        target = None
        for i, pt in enumerate(self.ref_points):
            if pt is None:
                target = i
                break
        if target is None:
            # 既存選択が有効ならそれを使う
            target = self.ref_selected_index if 0 <= self.ref_selected_index < len(self.ref_points) else 0
        self.ref_selected_index = target
        # 新しい列が表示範囲外なら表示列数を拡張し、テーブルを更新
        if (target + 1) > self.visible_ref_cols:
            self.visible_ref_cols = min(len(self.ref_points), target + 1)
            try:
                self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, self.centroids, self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass
            except Exception:
                pass
        # 左テーブル側の選択を更新
        try:
            self.table_ref.blockSignals(True)
            # canonical table_ref has 2 pseudo-header rows
            self.table_ref.setCurrentCell(2, target)
            self.table_ref.selectColumn(target)
        finally:
            self.table_ref.blockSignals(False)
        # ピックモード開始（Add）
        self._start_pick_mode('add', ref_index=target)
        # カーソルを画像中心にジャンプ
        self._move_cursor_to_image_center()

    def _on_update_xy(self):
        # Toggle pick-mode（Update）: 押し直すとキャンセル
        if self.pick_mode == 'update':
            self._end_pick_mode()
            return
        # ピックモード開始（Update）
        if not (0 <= self.ref_selected_index < len(self.ref_points)):
            return
        self._start_pick_mode('update', ref_index=self.ref_selected_index)
        # 既存のRef座標があればそこへカーソル移動。無ければ選択点/画像中央。
        x_full = y_full = None
        if self._img_base_size is not None:
            pt = self.ref_points[self.ref_selected_index]
            if pt is not None:
                x_full = pt[0] * self.scale_proc_to_full
                y_full = pt[1] * self.scale_proc_to_full
            elif self.selected_index is not None and 0 <= self.selected_index < len(self.centroids):
                _, xp, yp = self.centroids[self.selected_index]
                x_full = xp * self.scale_proc_to_full
                y_full = yp * self.scale_proc_to_full
            else:
                x_full = self._img_base_size[0] / 2.0
                y_full = self._img_base_size[1] / 2.0
        if x_full is not None and y_full is not None:
            # まず対象座標が見えるようスクロール
            self._ensure_full_pos_visible(x_full, y_full)
            dxy = self._full_to_display(x_full, y_full)
            if dxy is not None:
                local_pt = QPoint(int(round(dxy[0])), int(round(dxy[1])))
                global_pt = self.img_label_proc.mapToGlobal(local_pt)
                QCursor.setPos(global_pt)
                # ルーペは廃止
                # 十字線を即時表示
                self._draw_crosshair(local_pt)

    def _move_cursor_to_image_center(self):
        """カーソルを右画像の中心にジャンプさせる"""
        x_full = y_full = None
        if self._img_base_size is not None:
            x_full = self._img_base_size[0] / 2.0
            y_full = self._img_base_size[1] / 2.0
        if x_full is not None and y_full is not None:
            # まず対象座標が見えるようスクロール
            self._ensure_full_pos_visible(x_full, y_full)
            dxy = self._full_to_display(x_full, y_full)
            if dxy is not None:
                local_pt = QPoint(int(round(dxy[0])), int(round(dxy[1])))
                global_pt = self.img_label_proc.mapToGlobal(local_pt)
                QCursor.setPos(global_pt)
                # 十字線を即時表示
                self._draw_crosshair(local_pt)

    def _on_clear_ref(self):
        # 選択中のRef列をクリア
        if not (0 <= self.ref_selected_index < len(self.ref_points)):
            return
        # Commit any pending edits before clearing
        try:
            self._flush_ref_view()
        except Exception:
            pass

        idx = int(self.ref_selected_index)
        self.ref_points[idx] = None
        try:
            if 0 <= idx < len(self.ref_obs):
                self.ref_obs[idx] = {"x": "", "y": "", "z": ""}
        except Exception:
            pass
        # テーブル更新と再描画
        try:
            self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, self.centroids, self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
            try:
                self._refresh_transposed_views()
            except Exception:
                pass
            self._apply_proc_zoom()
        except Exception:
            pass

    def _on_cycle_flip_mode(self):
        # Auto -> Normal -> Flip -> Auto と循環
        cur = str(getattr(self, 'flip_mode', 'auto')).lower()
        nxt = 'normal' if cur == 'auto' else ('flip' if cur == 'normal' else 'auto')
        try:
            self._set_flip_mode(nxt)
        except Exception:
            # fallback (shouldn't happen)
            self.flip_mode = nxt
    def _set_flip_mode(self, mode: str, refresh: bool = True):
        try:
            m = str(mode or '').lower().strip()
        except Exception:
            m = 'auto'
        if m not in ('auto', 'normal', 'flip'):
            m = 'auto'
        self.flip_mode = m

        # combobox の選択を更新
        try:
            combo = getattr(self, 'combo_flip_mode', None)
            idx_map = {'auto': 0, 'normal': 1, 'flip': 2}
            if combo is not None:
                old = combo.blockSignals(True)
                combo.setCurrentIndex(idx_map.get(m, 0))
                combo.blockSignals(old)
        except Exception:
            pass

        if not refresh:
            return

        # 再描画・テーブル更新
        try:
            self._safe_populate_tables(
                self.table_ref,
                self.table,
                self.ref_points,
                self.ref_obs,
                self.centroids,
                self.selected_index,
                self.ref_selected_index,
                flip_mode=self.flip_mode,
                visible_ref_cols=self.visible_ref_cols,
            )
            try:
                self._refresh_transposed_views()
            except Exception:
                pass
            self._apply_proc_zoom()
        except Exception:
            pass

    def _on_combo_flip_changed(self, index: int):
        try:
            idx_map = {0: 'auto', 1: 'normal', 2: 'flip'}
            mode = idx_map.get(int(index), 'auto')
        except Exception:
            mode = 'auto'
        try:
            self._set_flip_mode(mode, refresh=True)
        except Exception:
            try:
                self.flip_mode = mode
            except Exception:
                pass


    def export_centroids(self):
        if self.img_full is None or self.centroid_processor is None:
            return
        # 表示と一致させるため、キャッシュを優先
        if self._cache.get("centroids") is not None and self._cache.get("img_id") == id(self.proc_img):
            centroids = self._cache["centroids"]
        else:
            params = self._get_params()
            poster = None
            if (
                self._cache.get("poster") is not None
                and self._cache.get("img_id") == id(self.proc_img)
                and self._cache.get("levels") == params["levels"]
                and self._cache.get("min_area") == params["min_area"]
                and self._cache.get("trim_px") == params.get("trim_px")
            ):
                poster = self._cache.get("poster")
            centroids = self.centroid_processor.get_centroids(params, poster=poster)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{STR.EXPORT_FILENAME_PREFIX}{dt_str}.txt"

        # Always ask where to save.
        try:
            last_path = load_last_image_path()
            start_dir = os.path.dirname(last_path) if last_path else os.getcwd()
        except Exception:
            start_dir = os.getcwd()
        try:
            start_path = os.path.join(start_dir, default_name)
        except Exception:
            start_path = default_name

        try:
            outpath, _ = QFileDialog.getSaveFileName(
                self,
                "Export Centroids",
                start_path,
                "Text Files (*.txt);;All Files (*)",
            )
        except Exception:
            outpath = ""

        if not outpath:
            return

        try:
            with open(outpath, "w", encoding="utf-8") as f:
                # Header: No,Group,Stage X,Stage Y,Stage Z
                try:
                    hdr = getattr(STR, 'EXPORT_HEADER', None)
                except Exception:
                    hdr = None
                if hdr is None or not hdr.strip():
                    f.write("No,Group,Stage X,Stage Y,Stage Z\n")
                else:
                    # If existing header is different, replace with desired header
                    f.write("No,Group,Stage X,Stage Y,Stage Z\n")

                # Use table items (Calc.* rows) for Stage values when available
                tbl = getattr(self, 'table', None)
                for i, cent in enumerate(centroids):
                    try:
                        g = ""
                        try:
                            g = str(int(round(float(cent[0]))))
                        except Exception:
                            g = ""
                        sx = sy = sz = ""
                        if tbl is not None and tbl.columnCount() > i:
                            try:
                                itx = tbl.item(4, i)
                                ity = tbl.item(5, i)
                                itz = tbl.item(6, i)
                                sx = itx.text() if itx is not None else ""
                                sy = ity.text() if ity is not None else ""
                                sz = itz.text() if itz is not None else ""
                            except Exception:
                                sx = sy = sz = ""
                        f.write(f"{i+1},{g},{sx},{sy},{sz}\n")
                    except Exception:
                        try:
                            f.write(f"{i+1},,, ,\n")
                        except Exception:
                            pass
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", f"Saved centroids to:\n{outpath}")
        except Exception as e:
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_table_current_changed(self, curRow, curCol, prevRow, prevCol):
        if curCol is None or curCol < 0:
            return
        # 右テーブルはデータ列のみ（オフセット無し）
        idx = curCol
        if self.selected_index != idx:
            self.selected_index = idx
            self.schedule_update(force=True)

    def _on_table_between_current_changed(self, curRow, curCol, prevRow, prevCol):
        # transposed view row maps to original table column (selected centroid index)
        try:
            if curRow is None or curRow < 0:
                return
            header_rows = 2
            if int(curRow) < header_rows:
                return
            idx = int(curRow) - header_rows
            if self.selected_index != idx:
                self.selected_index = idx
                self.schedule_update(force=True)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # UI 側ではイベント処理を行わず、標準の処理へ委譲
        return super().eventFilter(obj, event)

    def _on_ref_cell_clicked(self, row, col):
        # 左テーブルクリック時に、Obs.X/Y/Z(行2,3,4)なら即編集を開始する
        try:
            # canonical table_ref has 2 pseudo-header rows
            if row in (4, 5, 6):
                item = self.table_ref.item(row, col)
                if item is not None and (item.flags() & Qt.ItemIsEditable):
                    # 列選択は維持しつつ、そのセルを編集開始
                    self.table_ref.setCurrentCell(row, col)
                    self.table_ref.selectColumn(col)
                    self.table_ref.editItem(item)
        except Exception:
            pass

    def _display_to_full(self, pos):
        # ラベル座標 pos からフル画像座標へ（ズームとスクロールを考慮）
        if self._img_base_size is None:
            return None
        img_w, img_h = self._img_base_size
        # use actual display_scale (display pixels per full-image pixel)
        z = max(0.0001, float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom)))))
        off_x, off_y = self._display_offset
        x_full = (pos.x() - off_x) / z
        y_full = (pos.y() - off_y) / z
        if not (0 <= x_full <= img_w and 0 <= y_full <= img_h):
            return None
        return x_full, y_full

    def _full_to_display(self, x_full, y_full):
        # フル画像座標からラベル座標へ（ズームのみ）
        if self._img_base_size is None:
            return None
        z = max(0.0001, float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom)))))
        off_x, off_y = self._display_offset
        return x_full * z + off_x, y_full * z + off_y

    def _draw_crosshair(self, pos_label):
        # ピックモード中に、画像端まで届く白い＋線（黒縁）を描画
        if self._display_pm_base is None:
            return
        pm2 = draw_crosshair(self._display_pm_base, self._display_offset, self._display_img_size, pos_label)
        if pm2 is not None:
            self.img_label_proc.setPixmap(pm2)

    # ルーペ更新は不要

    def _start_pick_mode(self, mode, ref_index=None):
        self.pick_mode = mode
        self.pick_ref_index = ref_index
        self.img_label_proc.setCursor(QCursor(Qt.CrossCursor))
        # ルーペ表示は廃止
        # While waiting for image click after "Add Ref. Point", gray-invert the button and change text to "Cancel".
        try:
            if str(mode) == 'add' or str(mode) == 'update':
                # pick モード開始時は対応ボタンをキャンセル表示にする
                target_btn = self.btn_add_ref if str(mode) == 'add' else getattr(self, 'btn_update_xy', None)
                if target_btn is not None:
                    # ボタンのテキストを「Cancel」に変更
                    try:
                        target_btn.setText(STR.BUTTON_ADD_REF_CANCEL)
                    except Exception:
                        pass

                    # Lock current size so switching QSS won't change layout/height.
                    try:
                        h0 = int(target_btn.height() or 0)
                    except Exception:
                        h0 = 0
                    if h0 <= 0:
                        try:
                            h0 = int(target_btn.sizeHint().height())
                        except Exception:
                            h0 = 0
                    if h0 > 0:
                        try:
                            target_btn.setFixedHeight(h0)
                        except Exception:
                            pass

                    # Also lock width to avoid becoming thinner.
                    try:
                        w0 = int(target_btn.width() or 0)
                    except Exception:
                        w0 = 0
                    if w0 <= 0:
                        try:
                            w0 = int(target_btn.sizeHint().width())
                        except Exception:
                            w0 = 0
                    if w0 > 0:
                        try:
                            target_btn.setFixedWidth(w0)
                        except Exception:
                            pass

                    radius = 8
                    style_wait = (
                        f"QPushButton {{ background-color: rgb(230,230,230); color: black; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,220,220); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,210,210); }}"
                    )
                    # Also lock height in QSS to avoid style-driven recalculation.
                    try:
                        if h0 > 0:
                            style_wait += f"\nQPushButton {{ min-height: {h0}px; max-height: {h0}px; }}"
                    except Exception:
                        pass
                    # Also lock width in QSS to avoid style-driven recalculation.
                    try:
                        if w0 > 0:
                            style_wait += f"\nQPushButton {{ min-width: {w0}px; max-width: {w0}px; }}"
                    except Exception:
                        pass
                    try:
                        target_btn.setStyleSheet(style_wait)
                    except Exception:
                        pass
        except Exception:
            pass

    def _end_pick_mode(self):
        self.pick_mode = None
        self.pick_ref_index = None
        # 通常は手のカーソル
        self.img_label_proc.setCursor(QCursor(Qt.OpenHandCursor))
        # ルーペは存在しない
        # Restore button styles and text (undo gray-invert and "Cancel" text)
        try:
            btn = getattr(self, 'btn_add_ref', None)
            if btn is not None:
                btn.setText(STR.BUTTON_ADD_REF)
        except Exception:
            pass
        try:
            btn_up = getattr(self, 'btn_update_xy', None)
            if btn_up is not None:
                btn_up.setText(STR.BUTTON_UPDATE_XY)
        except Exception:
            pass
        try:
            self._apply_button_styles()
        except Exception:
            pass
        # Re-enforce heights after style change
        try:
            QTimer.singleShot(0, self._enforce_button_heights)
        except Exception:
            pass

        # Clear any crosshair overlay by re-rendering the base pixmap.
        try:
            self._apply_proc_zoom()
        except Exception:
            pass

    def _handle_image_click(self, pos):
        # クリック座標を右画像の元サイズ（overlay_full）座標に変換（ズームのみ考慮）
        xy = self._display_to_full(pos)
        if xy is None:
            return
        x_full, y_full = xy

        # ピックモード中は、マウス位置の座標をRefに保存して終了
        if self.pick_mode in ('add', 'update'):
            if self.scale_proc_to_full != 0:
                x_proc = x_full / self.scale_proc_to_full
                y_proc = y_full / self.scale_proc_to_full
                idx = self.pick_ref_index if self.pick_ref_index is not None else self.ref_selected_index
                if idx is not None and 0 <= idx < len(self.ref_points):
                    self.ref_points[idx] = (x_proc, y_proc)

                    # すぐに左表へ反映（populate_tables が遅延しても X/Y を見せる）
                    try:
                        from qt_compat.QtWidgets import QTableWidgetItem
                        from qt_compat.QtCore import Qt as _Qt

                        xi = str(int(round(x_proc)))
                        yi = str(int(round(y_proc)))

                        # canonical: table_ref (data starts at row 2 because rows 0-1 are pseudo-headers)
                        t = getattr(self, 'table_ref', None)
                        if t is not None:
                            try:
                                src_row_offset = 2
                                if t.columnCount() <= int(idx):
                                    t.setColumnCount(int(idx) + 1)
                                    try:
                                        t.setHorizontalHeaderLabels([str(i + 1) for i in range(t.columnCount())])
                                    except Exception:
                                        pass
                                if t.rowCount() >= (src_row_offset + 2):
                                    itx = t.item(src_row_offset + 0, int(idx))
                                    if itx is None:
                                        itx = QTableWidgetItem("")
                                        t.setItem(src_row_offset + 0, int(idx), itx)
                                    ity = t.item(src_row_offset + 1, int(idx))
                                    if ity is None:
                                        ity = QTableWidgetItem("")
                                        t.setItem(src_row_offset + 1, int(idx), ity)
                                    itx.setText(xi)
                                    ity.setText(yi)
                                    try:
                                        itx.setFlags(itx.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                        ity.setFlags(ity.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                        # transposed: table_ref_view (row idx + 2 header rows, col 0/1)
                        rv = getattr(self, 'table_ref_view', None)
                        if rv is not None:
                            try:
                                header_rows = 2
                                view_r = int(idx) + header_rows
                                if rv.rowCount() <= view_r:
                                    rv.setRowCount(view_r + 1)
                                # Ensure at least X/Y columns exist
                                if rv.columnCount() < 2:
                                    try:
                                        rv.setColumnCount(max(2, len(STR.TABLE_LEFT_ROW_LABELS)))
                                    except Exception:
                                        rv.setColumnCount(2)
                                if rv.columnCount() >= 2:
                                    vix = rv.item(view_r, 0)
                                    if vix is None:
                                        vix = QTableWidgetItem("")
                                        rv.setItem(view_r, 0, vix)
                                    viy = rv.item(view_r, 1)
                                    if viy is None:
                                        viy = QTableWidgetItem("")
                                        rv.setItem(view_r, 1, viy)
                                    vix.setText(xi)
                                    viy.setText(yi)
                                    try:
                                        vix.setFlags(vix.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                        viy.setFlags(viy.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # 新しく追加された列が表示範囲外なら可視列を拡張
                    if (idx + 1) > self.visible_ref_cols:
                        self.visible_ref_cols = min(len(self.ref_points), idx + 1)
                    self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, self.centroids, self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
                    try:
                        self._refresh_transposed_views()
                    except Exception:
                        pass
                    # 要望: Add で点を指定したら即赤点を描画し、Addモードを抜ける
                    try:
                        self._apply_proc_zoom()  # ref_points を反映して再描画（赤点が即時出る）
                    except Exception:
                        pass
                    # End pick mode for both add and update after handling click
                    if self.pick_mode in ('add', 'update'):
                        self._end_pick_mode()

    def _on_ref_item_changed(self, item):
        # 左テーブル（Ref）の Obs.* 行（2,3,4行目）入力を半角へ正規化し、内部配列に反映
        row = item.row()
        col = item.column()
        # canonical table_ref has 2 pseudo-header rows
        if row not in (4, 5, 6):
            return
        text = item.text() or ""
        # 全角を半角へ（英数記号）
        normalized = unicodedata.normalize('NFKC', text)
        if normalized != text:
            # ループ防止のため一旦シグナル停止
            self.table_ref.blockSignals(True)
            try:
                item.setText(normalized)
            finally:
                self.table_ref.blockSignals(False)
        key = 'x' if row == 4 else ('y' if row == 5 else 'z')
        if 0 <= col < len(self.ref_obs):
            self.ref_obs[col][key] = normalized

        # 右表の Calc.* 更新はイベントループに回して commit/closeEditor と競合させない
        try:
            self._defer_recompute_after_ref_edit()
        except Exception:
            pass

    def _on_ref_view_item_changed(self, item):
        # Map edits in the transposed view back to the underlying `self.table_ref`.
        try:
            if item is None:
                return
            r = item.row()
            c = item.column()
            header_rows = 2
            src_row_offset = 2  # canonical table_ref has 2 pseudo-header rows
            # Ignore edits in our in-cell header rows
            if r < header_rows:
                return
            # transposed: view[r,c] corresponds to source[src_row_offset + c, r - header_rows]
            src_r = src_row_offset + c
            src_c = r - header_rows
            if 0 <= src_r < self.table_ref.rowCount() and 0 <= src_c < self.table_ref.columnCount():
                try:
                    # prevent recursion on the view; allow the source table to emit its itemChanged
                    try:
                        self.table_ref_view.blockSignals(True)
                    except Exception:
                        pass
                    txt = item.text() if item.text() is not None else ""
                    # normalize full-width -> half-width (keep consistent with _on_ref_item_changed)
                    try:
                        normalized = unicodedata.normalize('NFKC', txt)
                    except Exception:
                        normalized = txt
                    txt = normalized
                    # While we mirror the edit into the canonical table, block its signals too.
                    # Otherwise _on_ref_item_changed may fire synchronously (re-entrant) while
                    # the editor is still being committed/closed.
                    try:
                        self.table_ref.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        src_item = self.table_ref.item(src_r, src_c)
                        if src_item is None:
                            src_item = QTableWidgetItem(txt)
                            self.table_ref.setItem(src_r, src_c, src_item)
                        else:
                            src_item.setText(txt)
                    finally:
                        try:
                            self.table_ref.blockSignals(False)
                        except Exception:
                            pass

                    # Update internal ref_obs immediately when editing Obs rows (2,3,4)
                    try:
                        # canonical rows: X,Y,ObsX,ObsY,ObsZ,... start at src_row_offset
                        obs_rows = (src_row_offset + 2, src_row_offset + 3, src_row_offset + 4)
                        if src_r in obs_rows and 0 <= src_c < len(self.ref_obs):
                            key = 'x' if src_r == obs_rows[0] else ('y' if src_r == obs_rows[1] else 'z')
                            self.ref_obs[src_c][key] = txt
                    except Exception:
                        pass
                finally:
                    try:
                        self.table_ref_view.blockSignals(False)
                    except Exception:
                        pass
        except Exception:
            pass
        # Recompute after any transposed-view edit (coalesced)
        try:
            self._defer_recompute_after_ref_edit()
        except Exception:
            pass

        # If we skipped refreshing while editing, try again now.
        try:
            if getattr(self, '_pending_ref_view_refresh', False):
                from qt_compat.QtCore import QTimer
                QTimer.singleShot(0, self._refresh_transposed_views)
        except Exception:
            pass

        # Fallback: if editor movement didn't occur (delegate didn't handle Return),
        # move to next editable cell: Obs X (col 2) -> Obs Y (col3), Obs Y -> Obs Z (col4),
        # Obs Z -> next row Obs X.
        try:
            header_rows = 2
            if item is not None:
                r = item.row()
                c = item.column()
                # only consider movement for data rows (not header rows)
                if r is not None and c is not None and r >= header_rows:
                    def _move_next_fallback():
                        try:
                            try:
                                vr = int(r)
                                vc = int(c)
                            except Exception:
                                return
                            # editable Obs columns in transposed view are 2,3,4
                            if vc == 2:
                                tgt_r, tgt_c = vr, 3
                            elif vc == 3:
                                tgt_r, tgt_c = vr, 4
                            elif vc == 4:
                                tgt_r, tgt_c = vr + 1, 2
                            else:
                                return
                            # bounds check
                            tv = getattr(self, 'table_ref_view', None)
                            if tv is None:
                                return
                            if tgt_r < 0 or tgt_c < 0:
                                return
                            if tgt_r >= tv.rowCount():
                                return
                            if tgt_c >= tv.columnCount():
                                return
                            try:
                                tv.setCurrentCell(tgt_r, tgt_c)
                            except Exception:
                                pass
                            try:
                                itm = tv.item(tgt_r, tgt_c)
                                from qt_compat.QtCore import Qt as _Qt
                                if itm is not None and (itm.flags() & getattr(_Qt, 'ItemIsEditable', 0)):
                                    try:
                                        tv.setFocus()
                                        tv.editItem(itm)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        except Exception:
                            pass
                    try:
                        from qt_compat.QtCore import QTimer
                        QTimer.singleShot(0, _move_next_fallback)
                    except Exception:
                        _move_next_fallback()
        except Exception:
            pass

    def _flush_ref_view(self):
        """Commit transposed left-view edits into the canonical `table_ref` and `self.ref_obs`.

        This avoids losing values if there are uncommitted edits when other actions
        (like AddRefPoint) rebuild the tables.
        """
        try:
            # If an editor widget is active in the view, avoid emitting commitData
            # for an arbitrary editor (can trigger "does not belong to this view").
            # Instead move focus away so delegate commits naturally, then process events.
            try:
                from qt_compat.QtWidgets import QApplication
                editor = QApplication.focusWidget()
                if editor is not None:
                    # verify editor is a child of our transposed view
                    p = editor
                    is_child = False
                    while p is not None:
                        if p is self.table_ref_view:
                            is_child = True
                            break
                        try:
                            p = p.parent()
                        except Exception:
                            p = None
                    if is_child:
                        try:
                            # move focus to a safe widget to commit editor
                            try:
                                if getattr(self, 'btn_add_ref', None) is not None:
                                    self.btn_add_ref.setFocus()
                                else:
                                    self.table_ref_view.clearFocus()
                            except Exception:
                                try:
                                    self.table_ref_view.clearFocus()
                                except Exception:
                                    pass
                            try:
                                QApplication.processEvents()
                            except Exception:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass

            # Block signals while copying to avoid repeated repopulates
            try:
                self.table_ref.blockSignals(True)
            except Exception:
                pass
            # Copy only editable cells from the transposed view into the source-of-truth.
            # IMPORTANT: The transposed view contains 2 in-cell header rows (0-1). Do not
            # copy those, or header strings like "Stage"/"X" will pollute the data model.
            rv = self.table_ref_view
            if rv is None:
                return
            header_rows = 2
            src_row_offset = 2  # canonical table_ref has 2 pseudo-header rows
            # Only Stage (Obs) columns are editable in the view
            editable_cols = (2, 3, 4)  # ObsX, ObsY, ObsZ in transposed view
            header_tokens = {"image", "stage", "residual", "x", "y", "z", "u", "v", "|r|", "|r|\n", "|r|\r\n"}
            rows = int(rv.rowCount())
            cols = int(rv.columnCount())
            max_ref = max(0, rows - header_rows)
            for r in range(header_rows, rows):
                ref_idx = int(r) - header_rows
                if ref_idx < 0:
                    continue
                # ensure arrays exist
                try:
                    if ref_idx >= len(self.ref_obs):
                        # extend with empty dicts if needed
                        self.ref_obs.extend({"x": "", "y": "", "z": ""} for _ in range(ref_idx - len(self.ref_obs) + 1))
                except Exception:
                    pass
                for c in editable_cols:
                    if c < 0 or c >= cols:
                        continue
                    try:
                        it = rv.item(r, c)
                        txt = it.text() if it is not None else ""
                        try:
                            txt = unicodedata.normalize('NFKC', txt)
                        except Exception:
                            pass
                        # sanitize accidental header tokens
                        try:
                            if (txt or "").strip().lower() in header_tokens:
                                txt = ""
                        except Exception:
                            pass

                        # Update internal model
                        try:
                            if 0 <= ref_idx < len(self.ref_obs):
                                key = 'x' if c == 2 else ('y' if c == 3 else 'z')
                                self.ref_obs[ref_idx][key] = txt
                        except Exception:
                            pass

                        # Mirror into canonical table_ref so downstream code stays consistent
                        try:
                            src_r = int(src_row_offset + c)  # canonical row
                            src_c = int(ref_idx)             # canonical column
                            if src_r >= self.table_ref.rowCount():
                                self.table_ref.setRowCount(src_r + 1)
                            if src_c >= self.table_ref.columnCount():
                                self.table_ref.setColumnCount(src_c + 1)
                                try:
                                    self.table_ref.setHorizontalHeaderLabels([str(i + 1) for i in range(self.table_ref.columnCount())])
                                except Exception:
                                    pass
                            src_item = self.table_ref.item(src_r, src_c)
                            if src_item is None:
                                src_item = QTableWidgetItem(txt)
                                self.table_ref.setItem(src_r, src_c, src_item)
                            else:
                                src_item.setText(txt)
                        except Exception:
                            pass
                    except Exception:
                        pass
        finally:
            try:
                self.table_ref.blockSignals(False)
            except Exception:
                pass
        # After flushing edits into `self.ref_obs`, recompute derived values
        try:
            from qt_compat.QtCore import QTimer

            def _do_populate():
                try:
                    self._safe_populate_tables(
                        self.table_ref,
                        self.table,
                        self.ref_points,
                        self.ref_obs,
                        self.centroids,
                        self.selected_index,
                        self.ref_selected_index,
                        flip_mode=self.flip_mode,
                        visible_ref_cols=self.visible_ref_cols,
                    )
                except Exception:
                    pass
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass

            try:
                # Allow any pending editor commits to settle before rebuilding tables.
                QTimer.singleShot(10, _do_populate)
            except Exception:
                # Fallback: run immediately
                _do_populate()
        except Exception:
            pass

    def _copy_centroids_to_clipboard(self):
        """Copy sequential index and CalcX/CalcY/CalcZ to the clipboard as TSV."""
        try:
            # Visual feedback: flash the Clipboard button gray briefly.
            try:
                btn = getattr(self, 'btn_clipboard', None)
                if btn is not None:
                    prev_style = btn.styleSheet() or ""
                    try:
                        btn.setStyleSheet(
                            "QPushButton { background-color: rgb(140,140,140); color: white; border: none; border-radius: 8px; }"
                        )
                    except Exception:
                        prev_style = None

                    def _restore_btn_style(_btn=btn, _prev=prev_style):
                        try:
                            if _prev is not None and _prev != "":
                                _btn.setStyleSheet(_prev)
                            else:
                                # Fallback to standard styling
                                try:
                                    self._apply_button_styles()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    QTimer.singleShot(150, _restore_btn_style)
            except Exception:
                pass

            app = QApplication.instance()
            if app is None:
                return
            tbl = self.table
            if tbl is None or tbl.columnCount() == 0:
                return
            lines = []
            # header: No, Group, Stage X, Stage Y, Stage Z
            lines.append("No\tGroup\tStage X\tStage Y\tStage Z")
            for c in range(tbl.columnCount()):
                try:
                    # Group from self.centroids (first element). Stage values are Calc.* at rows 2,3,4 per tables.populate_tables
                    grp = ""
                    try:
                        if getattr(self, 'centroids', None) is not None and 0 <= c < len(self.centroids):
                            grp = str(int(round(float(self.centroids[c][0]))))
                    except Exception:
                        grp = ""
                    # Calc.* rows live at offsets 4,5,6 (DATA_ROW_OFFSET + 2..4 in tables.populate_tables)
                    itx = tbl.item(4, c)
                    ity = tbl.item(5, c)
                    itz = tbl.item(6, c)
                    sx = itx.text() if itx is not None else ""
                    sy = ity.text() if ity is not None else ""
                    sz = itz.text() if itz is not None else ""
                    lines.append(f"{c+1}\t{grp}\t{sx}\t{sy}\t{sz}")
                except Exception:
                    lines.append(f"{c+1}\t\t\t\t")
            txt = "\n".join(lines)
            try:
                QApplication.clipboard().setText(txt)
            except Exception:
                # fallback: print to stdout
                print(txt)
        except Exception:
            pass

    def _safe_populate_tables(self, *args, **kwargs):
        """Call populate_tables but defer if an editor in the left views is active.

        This prevents calling into populate_tables while a cell editor widget
        is still active, which can trigger Qt's "commitData called with an
        editor that does not belong to this view" warning.

        To avoid spamming logs and scheduling many deferred calls while the
        user is typing, we schedule at most one deferred call at a time.
        """
        try:
            from qt_compat.QtCore import QTimer
            from qt_compat.QtWidgets import QAbstractItemView

            editing = False
            try:
                w = getattr(self, 'table_ref', None)
                if w is not None and w.state() == QAbstractItemView.EditingState:
                    editing = True
            except Exception:
                pass
            try:
                # NOTE:
                # Edits in the *transposed* view (`table_ref_view`) should not block
                # recalculation of derived values (middle/right tables). We avoid
                # rebuilding that view while it is editing in `_refresh_transposed_views`.
                # Here we only defer when a table that `populate_tables` mutates is editing.
                w = getattr(self, 'table', None)
                if w is not None and w.state() == QAbstractItemView.EditingState:
                    editing = True
            except Exception:
                pass
            try:
                # table_between is a derived/transposed view; it should not block populate.
                # Keep this check disabled unless the view is made editable in the future.
                pass
            except Exception:
                pass

            if editing:
                # Only schedule one deferred update at a time
                if getattr(self, '_safe_populate_scheduled', False):
                    return
                try:
                    self._safe_populate_scheduled = True
                except Exception:
                    pass

                def _run_deferred():
                    try:
                        # Clear the scheduled flag early so re-scheduling can occur if needed
                        try:
                            self._safe_populate_scheduled = False
                        except Exception:
                            pass
                        # Ensure any active editors are flushed/committed safely before modifying tables
                        # try:
                        #     self._flush_ref_view()
                        # except Exception:
                        #     pass
                    except Exception:
                        pass
                    try:
                        populate_tables(*args, **kwargs)
                    except Exception:
                        pass
                    # Reinstall pseudo-headers after populate (data might overwrite them)
                    try:
                        self._setup_pseudo_headers_ref(self.table_ref)
                        self._setup_pseudo_headers_between(self.table)
                    except Exception:
                        pass
                    # Sync frozen headers after populate completes
                    try:
                        self._sync_frozen_headers()
                    except Exception:
                        pass
                    # Re-apply pseudo-headers after populate to ensure they're visible
                    try:
                        QTimer.singleShot(50, lambda: self._setup_pseudo_headers_ref(self.table_ref))
                        QTimer.singleShot(100, lambda: self._setup_pseudo_headers_between(self.table))
                    except Exception:
                        pass
                    # If populate_tables was deferred, the caller may already have refreshed
                    # transposed views using stale data. Refresh again now to ensure X/Y and
                    # Calc tables reflect the latest population.
                    try:
                        self._refresh_transposed_views()
                    except Exception:
                        pass

                QTimer.singleShot(250, _run_deferred)
                return
        except Exception:
            pass
        try:
            populate_tables(*args, **kwargs)
            # Reinstall pseudo-headers after populate (data might overwrite them)
            try:
                self._setup_pseudo_headers_ref(self.table_ref)
                self._setup_pseudo_headers_between(self.table)
            except Exception:
                pass
            # Sync frozen headers after populate completes
            try:
                QTimer.singleShot(100, self._sync_frozen_headers)
            except Exception:
                pass
            # Re-apply pseudo-headers after populate to ensure they're visible
            try:
                QTimer.singleShot(150, lambda: self._setup_pseudo_headers_ref(self.table_ref))
                QTimer.singleShot(200, lambda: self._setup_pseudo_headers_between(self.table))
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_transposed_views(self):
        # Create/update transposed copies of `self.table_ref` and `self.table`.
        try:
            header_rows = 2
            ref_src_row_offset = 2  # canonical table_ref has 2 pseudo-header rows
            mid_src_row_offset = 2  # canonical table has 2 pseudo-header rows

            # If the user is actively editing the transposed left table, do not rebuild
            # *that* view's items. But still refresh the middle table (Calc results)
            # so recomputation is visible immediately after Enter.
            editing_left = False
            try:
                from qt_compat.QtWidgets import QAbstractItemView
                rv = getattr(self, 'table_ref_view', None)
                if rv is not None:
                    try:
                        editing_left = (rv.state() == QAbstractItemView.EditingState)
                    except Exception:
                        editing_left = False
            except Exception:
                pass
            if editing_left:
                try:
                    self._pending_ref_view_refresh = True
                except Exception:
                    pass

                # Even while editing, keep key *computed/display* rows in the left transposed
                # view up-to-date so they reflect the latest recomputation / newly added refs.
                try:
                    rv = getattr(self, 'table_ref_view', None)
                    src = getattr(self, 'table_ref', None)
                    if rv is not None and src is not None:
                        from qt_compat.QtWidgets import QTableWidgetItem
                        try:
                            rv.blockSignals(True)
                        except Exception:
                            pass
                        try:
                            # Rows in the canonical ref table we want to keep visible even while editing:
                            # - RefX/RefY (data starts at ref_src_row_offset)
                            # - Residual rows
                            update_src_rows = (
                                ref_src_row_offset + 0,
                                ref_src_row_offset + 1,
                                ref_src_row_offset + 5,
                                ref_src_row_offset + 6,
                                ref_src_row_offset + 7,
                                ref_src_row_offset + 8,
                            )
                            cur = None
                            try:
                                cur = rv.currentItem()
                            except Exception:
                                cur = None
                            cur_r = cur.row() if cur is not None else -1
                            cur_c = cur.column() if cur is not None else -1

                            max_view_rows = rv.rowCount()
                            max_view_cols = rv.columnCount()

                            for view_r in range(header_rows, max_view_rows):
                                src_c = view_r - header_rows  # source column == view row (minus header rows)
                                if not (0 <= src_c < src.columnCount()):
                                    continue
                                for src_r in update_src_rows:
                                    view_c = src_r - ref_src_row_offset  # source row == view column (minus src offset)
                                    if not (0 <= view_c < max_view_cols):
                                        continue
                                    # Avoid touching the actively edited cell
                                    if view_r == cur_r and view_c == cur_c:
                                        continue
                                    # Prefer source-of-truth arrays for Image X/Y so they never get polluted by table items.
                                    try:
                                        txt = ""
                                        if src_r == (ref_src_row_offset + 0):
                                            pt = self.ref_points[src_c] if 0 <= src_c < len(self.ref_points) else None
                                            txt = "" if pt is None else str(int(round(float(pt[0]))))
                                        elif src_r == (ref_src_row_offset + 1):
                                            pt = self.ref_points[src_c] if 0 <= src_c < len(self.ref_points) else None
                                            txt = "" if pt is None else str(int(round(float(pt[1]))))
                                        else:
                                            src_item = src.item(src_r, src_c)
                                            txt = src_item.text() if src_item is not None else ""
                                    except Exception:
                                        txt = ""
                                    it = None
                                    try:
                                        it = rv.item(view_r, view_c)
                                    except Exception:
                                        it = None
                                    if it is None:
                                        try:
                                            it = QTableWidgetItem("")
                                            rv.setItem(view_r, view_c, it)
                                        except Exception:
                                            it = None
                                    if it is not None:
                                        try:
                                            it.setText(str(txt))
                                        except Exception:
                                            pass
                                        # Ensure these computed/display cells are non-editable
                                        try:
                                            from qt_compat.QtCore import Qt as _Qt
                                            it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                        except Exception:
                                            pass
                        finally:
                            try:
                                rv.blockSignals(False)
                            except Exception:
                                pass
                except Exception:
                    pass

            def make_transposed(src: QTableWidget) -> QTableWidget:
                if src is None:
                    return QTableWidget()
                rows = src.columnCount()
                cols = src.rowCount()
                tw = QTableWidget(rows, cols)
                # Horizontal header -> source vertical header labels
                try:
                    vlabels = [src.verticalHeaderItem(i).text() if src.verticalHeaderItem(i) is not None else "" for i in range(src.rowCount())]
                except Exception:
                    vlabels = [str(i) for i in range(src.rowCount())]
                try:
                    hlabels = [src.horizontalHeaderItem(i).text() if src.horizontalHeaderItem(i) is not None else "" for i in range(src.columnCount())]
                except Exception:
                    hlabels = [str(i) for i in range(src.columnCount())]
                try:
                    tw.setHorizontalHeaderLabels(vlabels)
                except Exception:
                    pass
                try:
                    tw.setVerticalHeaderLabels(hlabels)
                except Exception:
                    pass
                # copy items (text + alignment)
                for r in range(rows):
                    for c in range(cols):
                        try:
                            src_item = src.item(c, r)
                            txt = src_item.text() if src_item is not None else ""
                            it = QTableWidgetItem(txt)
                            if src_item is not None:
                                try:
                                    it.setTextAlignment(src_item.textAlignment())
                                except Exception:
                                    pass
                            tw.setItem(r, c, it)
                        except Exception:
                            pass
                return tw

            def _apply_incell_two_row_header(tbl, group_configs, sub_labels):
                try:
                    from qt_compat.QtWidgets import QTableWidgetItem
                    from qt_compat.QtGui import QColor
                    from qt_compat.QtCore import Qt as _Qt

                    # Hide the built-in header (we render a 2-row header inside the table)
                    try:
                        tbl.horizontalHeader().setVisible(False)
                    except Exception:
                        pass

                    # Row 0 group labels (left-aligned only for group headers)
                    for col_start, col_span, label in group_configs:
                        if col_start >= tbl.columnCount():
                            continue
                        span = max(1, min(int(col_span), int(tbl.columnCount() - col_start)))
                        it = QTableWidgetItem(str(label))
                        try:
                            it.setTextAlignment(_Qt.AlignLeft | _Qt.AlignVCenter)
                            f = it.font(); f.setBold(True); it.setFont(f)
                            it.setBackground(QColor("lightgray"))
                            it.setForeground(QColor("black"))
                            it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                        except Exception:
                            pass
                        tbl.setItem(0, col_start, it)
                        try:
                            tbl.setSpan(0, col_start, 1, span)
                        except Exception:
                            pass

                    # Row 1 sub labels
                    for c, label in enumerate(sub_labels):
                        if c >= tbl.columnCount():
                            break
                        it = QTableWidgetItem(str(label))
                        try:
                            it.setTextAlignment(_Qt.AlignHCenter | _Qt.AlignVCenter)
                            f = it.font(); f.setBold(True); it.setFont(f)
                            it.setBackground(QColor("lightgray"))
                            it.setForeground(QColor("black"))
                            it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                        except Exception:
                            pass
                        tbl.setItem(1, c, it)

                    # Fixed heights for the 2 header rows
                    try:
                        tbl.setRowHeight(0, 24)
                        tbl.setRowHeight(1, 20)
                    except Exception:
                        pass
                except Exception:
                    pass

            def _build_ref_transposed_view():
                src = getattr(self, 'table_ref', None)
                dst = getattr(self, 'table_ref_view', None)
                if src is None or dst is None:
                    return
                try:
                    from qt_compat.QtWidgets import QTableWidgetItem
                    from qt_compat.QtCore import Qt as _Qt
                    import Strings as STR

                    data_rows = int(src.columnCount())
                    # Always map columns based on the canonical row-label definition.
                    # This prevents pseudo-header rows (0-1) from leaking into data.
                    try:
                        src_row_count = len(getattr(STR, 'TABLE_LEFT_ROW_LABELS', []) or [])
                    except Exception:
                        src_row_count = 0
                    data_cols = max(0, int(src_row_count))
                    src_row_map = [ref_src_row_offset + i for i in range(data_cols)]
                    dst.blockSignals(True)
                    try:
                        try:
                            dst.clearSpans()
                        except Exception:
                            pass
                        try:
                            dst.clearContents()
                        except Exception:
                            pass
                        dst.setRowCount(data_rows + header_rows)
                        dst.setColumnCount(data_cols)

                        # Keep scrollbar presence stable to avoid width/layout shifts
                        try:
                            dst.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
                            dst.setHorizontalScrollBarPolicy(_Qt.ScrollBarAlwaysOff)
                        except Exception:
                            pass

                        # Vertical header: blank for header rows, then 1..N (source horizontal headers)
                        try:
                            hlabels = []
                            for i in range(src.columnCount()):
                                hi = src.horizontalHeaderItem(i)
                                hlabels.append(hi.text() if hi is not None else str(i + 1))
                            dst.setVerticalHeaderLabels(["", ""] + hlabels)
                        except Exception:
                            pass

                        # Fill data (shifted down by header_rows)
                        for r in range(data_rows):
                            for c in range(data_cols):
                                # Render from source-of-truth arrays for Image/Stage values.
                                # Use canonical table_ref only for computed residual columns.
                                try:
                                    txt = ""
                                    # Columns: 0..8 == X,Y,ObsX,ObsY,ObsZ,ResX,ResY,ResZ,|R|
                                    if c == 0:
                                        pt = self.ref_points[r] if 0 <= r < len(self.ref_points) else None
                                        txt = "" if pt is None else str(int(round(float(pt[0]))))
                                    elif c == 1:
                                        pt = self.ref_points[r] if 0 <= r < len(self.ref_points) else None
                                        txt = "" if pt is None else str(int(round(float(pt[1]))))
                                    elif c in (2, 3, 4):
                                        obs = self.ref_obs[r] if 0 <= r < len(self.ref_obs) else None
                                        if isinstance(obs, dict):
                                            key = 'x' if c == 2 else ('y' if c == 3 else 'z')
                                            txt = str(obs.get(key, "") or "")
                                            # If the model was previously polluted by header labels, hide them.
                                            try:
                                                if txt.strip().lower() in {"image", "stage", "residual", "x", "y", "z", "u", "v", "|r|"}:
                                                    txt = ""
                                            except Exception:
                                                pass
                                        else:
                                            txt = ""
                                    else:
                                        src_item = src.item(src_row_map[c], r)
                                        txt = src_item.text() if src_item is not None else ""
                                except Exception:
                                    txt = ""
                                it = QTableWidgetItem(str(txt))
                                try:
                                    it.setTextAlignment(_Qt.AlignHCenter | _Qt.AlignVCenter)
                                except Exception:
                                    pass
                                # Make Stage columns (X/Y/Z) visually bold in the transposed/ref view
                                try:
                                    if c in (2, 3, 4):
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                # Editability: only Obs columns (X/Y/Z) are editable
                                try:
                                    if c in (2, 3, 4):
                                        pass
                                    else:
                                        it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                except Exception:
                                    pass
                                dst.setItem(header_rows + r, c, it)

                        # Style the row-number gutter (vertical header): bold + readable gray
                        try:
                            dst.verticalHeader().setStyleSheet(
                                'QHeaderView::section { background-color: lightgray; color: gray; font-weight: bold; border: none; }'
                            )
                        except Exception:
                            pass

                        # Apply in-cell 2-row header (Image/Stage/Residual)
                        group_configs = [(0, 2, "Image"), (2, 3, "Stage (input)"), (5, max(1, data_cols - 5), "Residual")]
                        # Row 1 labels: Image(u,v), Stage(X,Y,Z), Residual(X,Y,Z,|R|)
                        sub_labels = ["u", "v", "X", "Y", "Z", "X", "Y", "Z", "|R|"]
                        if len(sub_labels) != data_cols:
                            if len(sub_labels) > data_cols:
                                sub_labels = sub_labels[:data_cols]
                            else:
                                sub_labels = sub_labels + [""] * (data_cols - len(sub_labels))
                        _apply_incell_two_row_header(dst, group_configs, sub_labels)

                        # If a fixed header widget exists, hide in-table header rows
                        try:
                            if getattr(self, 'table_ref_view_header', None) is not None and not self.table_ref_view_header.isHidden():
                                dst.setRowHidden(0, True)
                                dst.setRowHidden(1, True)
                        except Exception:
                            pass

                        # Keep row heights stable to reduce layout shifts
                        try:
                            vh = dst.verticalHeader()
                            vh.setSectionResizeMode(QHeaderView.Fixed)
                            vh.setDefaultSectionSize(24)
                        except Exception:
                            pass
                    finally:
                        dst.blockSignals(False)
                except Exception:
                    try:
                        dst.blockSignals(False)
                    except Exception:
                        pass

            def _build_mid_transposed_view():
                src = getattr(self, 'table', None)
                dst = getattr(self, 'table_between', None)
                if src is None or dst is None:
                    return
                try:
                    from qt_compat.QtWidgets import QTableWidgetItem
                    from qt_compat.QtCore import Qt as _Qt
                    import Strings as STR

                    data_rows = int(src.columnCount())
                    # Add one extra column at the left for Posterization Level (group_no)
                    base_cols = 0
                    try:
                        base_cols = len(getattr(STR, 'TABLE_RIGHT_ROW_LABELS', []) or [])
                    except Exception:
                        base_cols = 0
                    base_cols = max(0, int(base_cols))
                    data_cols = base_cols + 1
                    src_row_map = [mid_src_row_offset + i for i in range(base_cols)]
                    dst.blockSignals(True)
                    try:
                        try:
                            dst.clearSpans()
                        except Exception:
                            pass
                        try:
                            dst.clearContents()
                        except Exception:
                            pass
                        dst.setRowCount(data_rows + header_rows)
                        dst.setColumnCount(data_cols)

                        try:
                            dst.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
                            dst.setHorizontalScrollBarPolicy(_Qt.ScrollBarAlwaysOff)
                        except Exception:
                            pass

                        # Vertical header: blank for header rows, then 1..N
                        try:
                            hlabels = []
                            for i in range(src.columnCount()):
                                hi = src.horizontalHeaderItem(i)
                                hlabels.append(hi.text() if hi is not None else str(i + 1))
                            dst.setVerticalHeaderLabels(["", ""] + hlabels)
                        except Exception:
                            pass

                        for r in range(data_rows):
                            for c in range(data_cols):
                                try:
                                    if c == 0:
                                        # Posterization level/group number for this centroid
                                        try:
                                            g = None
                                            if self.centroids is not None and 0 <= r < len(self.centroids):
                                                g = self.centroids[r][0]
                                            txt = "" if g is None else str(int(g))
                                        except Exception:
                                            txt = ""
                                    else:
                                        src_item = src.item(src_row_map[c - 1], r)
                                        txt = src_item.text() if src_item is not None else ""
                                except Exception:
                                    txt = ""
                                it = QTableWidgetItem(str(txt))
                                try:
                                    it.setTextAlignment(_Qt.AlignHCenter | _Qt.AlignVCenter)
                                except Exception:
                                    pass
                                # All cells in this transposed view should be non-editable
                                try:
                                    it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                except Exception:
                                    pass
                                # Bold the leftmost Grp column values
                                try:
                                    if c == 0:
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                # Bold Stage X/Y/Z columns for readability
                                try:
                                    tmp_sub_labels = ["Lvl", "u", "v", "X", "Y", "Z"]
                                    sub_lbl = tmp_sub_labels[c] if 0 <= c < len(tmp_sub_labels) else None
                                    if sub_lbl in ("X", "Y", "Z"):
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                dst.setItem(header_rows + r, c, it)

                        # Style the row-number gutter (vertical header): bold + readable gray
                        try:
                            dst.verticalHeader().setStyleSheet(
                                'QHeaderView::section { background-color: lightgray; color: gray; font-weight: bold; border: none; }'
                            )
                        except Exception:
                            pass

                        # In-cell header (Posterization/Image/Stage)
                        group_configs = [(0, 1, ""), (1, 2, "Image"), (3, 3, "Stage")]
                        sub_labels = ["Grp", "u", "v", "X", "Y", "Z"]
                        _apply_incell_two_row_header(dst, group_configs, sub_labels)

                        # If a fixed header widget exists, hide in-table header rows
                        try:
                            if getattr(self, 'table_between_header', None) is not None and not self.table_between_header.isHidden():
                                dst.setRowHidden(0, True)
                                dst.setRowHidden(1, True)
                        except Exception:
                            pass

                        try:
                            vh = dst.verticalHeader()
                            vh.setSectionResizeMode(QHeaderView.Fixed)
                            vh.setDefaultSectionSize(24)
                        except Exception:
                            pass
                    finally:
                        dst.blockSignals(False)
                except Exception:
                    try:
                        dst.blockSignals(False)
                    except Exception:
                        pass

            # update left ref view (skip if currently editing)
            if not editing_left:
                try:
                    _build_ref_transposed_view()
                except Exception:
                    pass
                try:
                    self._pending_ref_view_refresh = False
                except Exception:
                    pass

            # update bottom/transposed table_between
            try:
                _build_mid_transposed_view()
            except Exception:
                pass

            # Rebuild fixed header widgets now that column counts exist
            try:
                self._rebuild_fixed_headers()
            except Exception:
                pass
            # After updating transposed views, ensure fixed pixel widths are applied
            try:
                # schedule immediately so layout has applied sizes
                QTimer.singleShot(0, self._shrink_visible_columns)
            except Exception:
                pass
            # After widths are applied, sync fixed headers to final widths
            try:
                QTimer.singleShot(0, self._rebuild_fixed_headers)
            except Exception:
                pass
            # Keep center column width stable to avoid subtle layout shifts
            try:
                QTimer.singleShot(0, lambda: self._adjust_center_column_widths(fixed_px=350))
            except Exception:
                pass
        except Exception:
            pass

    def _shrink_visible_columns(self):
        """Apply fixed pixel widths to transposed tables so startup/更新後の幅が決まるようにする。
        幅は必要に応じて変更してください（単位 px）。"""
        try:
            # --- Left transposed reference view ---
            tbl = getattr(self, 'table_ref_view', None)
            if tbl is not None:
                try:
                    hdr = tbl.horizontalHeader()
                    try:
                        hdr.setSectionResizeMode(QHeaderView.Fixed)
                    except Exception:
                        pass
                    cnt = tbl.columnCount()
                    if cnt > 0:
                        # すべて同じ幅に
                        widths_ref = [50] * cnt
                        for i in range(cnt):
                            w = int(widths_ref[i]) if i < len(widths_ref) else int(widths_ref[-1])
                            try:
                                tbl.setColumnWidth(i, max(8, w))
                            except Exception:
                                pass
                        try:
                            # center-align existing items
                            for r in range(tbl.rowCount()):
                                for c in range(tbl.columnCount()):
                                    try:
                                        it = tbl.item(r, c)
                                        if it is not None:
                                            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            hdr.setDefaultSectionSize(max(8, int(widths_ref[0])))
                        except Exception:
                            pass
                except Exception:
                    pass

            # --- Middle transposed table_between (optional) ---
            tbl2 = getattr(self, 'table_between', None)
            if tbl2 is not None:
                try:
                    hdr2 = tbl2.horizontalHeader()
                    try:
                        hdr2.setSectionResizeMode(QHeaderView.Fixed)
                    except Exception:
                        pass
                    cnt2 = tbl2.columnCount()
                    if cnt2 > 0:
                        # Match widths to the left transposed reference view when possible
                        ref_tbl = getattr(self, 'table_ref_view', None)
                        for i in range(cnt2):
                            try:
                                if ref_tbl is not None and i < ref_tbl.columnCount():
                                    w = int(ref_tbl.columnWidth(i))
                                else:
                                    w = 40
                                tbl2.setColumnWidth(i, max(8, w))
                            except Exception:
                                pass
                        try:
                            # center-align existing items in middle transposed
                            for r in range(tbl2.rowCount()):
                                for c in range(tbl2.columnCount()):
                                    try:
                                        it = tbl2.item(r, c)
                                        if it is not None:
                                            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            if ref_tbl is not None and ref_tbl.columnCount() > 0:
                                hdr2.setDefaultSectionSize(max(8, 40))
                        except Exception:
                            pass
                except Exception:
                    pass

            # Sync fixed header tables (if present) to the resized column widths
            try:
                self._sync_fixed_header_table(getattr(self, 'table_ref_view_header', None), getattr(self, 'table_ref_view', None))
            except Exception:
                pass
            try:
                self._sync_fixed_header_table(getattr(self, 'table_between_header', None), getattr(self, 'table_between', None))
            except Exception:
                pass
        except Exception:
            pass

    def _adjust_center_column_widths(self, fixed_px: int = 300):
        """Fix the middle column and `table_between` width to a constant (px)."""
        tbl = getattr(self, 'table_between', None)
        if tbl is None:
            return
        try:
            new_w = max(64, int(fixed_px))
        except Exception:
            new_w = 150

        try:
            tbl.setFixedWidth(int(new_w))
        except Exception:
            try:
                tbl.setMinimumWidth(int(new_w))
            except Exception:
                pass

        col = getattr(self, 'center_container', None)
        if col is not None:
            try:
                col.setFixedWidth(int(new_w))
            except Exception:
                try:
                    col.setMinimumWidth(int(new_w))
                except Exception:
                    pass

    def _sync_table_selection(self):
        """Sync selected_index to visible transposed table selection and canonical table selection."""
        try:
            idx = getattr(self, 'selected_index', None)
            if idx is None:
                return
            # Select column in canonical table (if exists)
            try:
                if 0 <= idx < self.table.columnCount():
                    try:
                        self.table.blockSignals(True)
                        self.table.setCurrentCell(0, idx)
                        self.table.selectColumn(idx)
                    finally:
                        try:
                            self.table.blockSignals(False)
                        except Exception:
                            pass
            except Exception:
                pass
            # Select corresponding row in transposed middle table (row == original column index)
            try:
                header_rows = 2
                view_r = int(idx) + header_rows
                if hasattr(self, 'table_between') and 0 <= view_r < self.table_between.rowCount():
                    try:
                        self.table_between.blockSignals(True)
                        # choose column 0 for current cell; selection behavior is rows
                        self.table_between.setCurrentCell(view_r, 0)
                        self.table_between.selectRow(view_r)
                    finally:
                        try:
                            self.table_between.blockSignals(False)
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass

    def _narrow_center_column(self):
        """Compatibility shim: adjust center widths after layout settle."""
        try:
            self._adjust_center_column_widths(fixed_px=300)
        except Exception:
            pass

    def _sync_left_top_image_width(self):
        """Set `left_top_image` width to match the left column's table width.

        We try to use the visible `table_ref_view` width; if that is not available
        yet, estimate from header/column sizes. This runs after layout settle.
        """
        try:
            img = getattr(self, 'left_top_image', None)
            tbl = getattr(self, 'table_ref_view', None)
            if img is None or tbl is None:
                return
            # Prefer to size image to the sum of the visible table column widths
            try:
                cnt = tbl.columnCount()
                content_w = 0
                for i in range(cnt):
                    try:
                        content_w += int(tbl.columnWidth(i))
                    except Exception:
                        # fallback to default section size
                        try:
                            content_w += int(tbl.horizontalHeader().defaultSectionSize() or 16)
                        except Exception:
                            content_w += 16
                # include vertical header width and a small padding/frame
                try:
                    vh = int(tbl.verticalHeader().width() or 0)
                except Exception:
                    vh = 0
                pad = 8
                # ensure a visually large minimum so logo is prominent
                min_display = 500
                w = max(min_display, content_w + vh + pad)
                # Left column is fixed-width; do not grow beyond it.
                try:
                    w = min(int(w), 500)
                except Exception:
                    pass
            except Exception:
                # ultimate fallback: widget width
                w = tbl.width()
            try:
                img.setFixedWidth(int(w))
            except Exception:
                try:
                    img.setMaximumWidth(int(w))
                except Exception:
                    pass
            # If we saved original pixmap, rescale it to exactly the width so it doesn't get clipped
            try:
                if getattr(self, '_left_top_pix', None) is not None:
                    pm = self._left_top_pix.scaledToWidth(int(w), Qt.SmoothTransformation)
                    img.setPixmap(pm)
            except Exception:
                pass
        except Exception:
            pass

    def _on_toggle_auto_update(self, enabled: bool):
        """Toggle automatic poster/centroid recalculation.

        When disabled (manual mode), heavy poster regeneration is skipped until the user clicks "重心再計算".
        """
        try:
            self.auto_update_mode = bool(enabled)
            # すぐに UI に反映させる（自動に切り替えたら即時再計算）
            if self.auto_update_mode:
                self.schedule_update(force=True)
        except Exception:
            pass

    def _on_manual_recalc(self):
        """Perform heavy poster generation and centroid calculation immediately.

        This is the button handler for manual recalculation. Disables the button while running.
        """
        if self.proc_img is None or self.centroid_processor is None:
            return
        try:
            self.btn_recalc.setEnabled(False)
            params = self._get_params()
            # poster は重いので明示的に生成
            poster = kmeans_posterize(self.proc_img, params["levels"])
            centroids = self.centroid_processor.get_centroids(params, poster=poster)
            self._cache.update({
                "img_id": id(self.proc_img),
                "levels": params["levels"],
                "min_area": params["min_area"],
                "trim_px": params["trim_px"],
                "poster": poster,
                "centroids": centroids,
            })
            # Rebuild overlay_full (boundaries/mask) from the newly generated poster
            try:
                # poster is at proc_img resolution; upscale to full
                scale = 1.0 / self.scale_proc_to_full if getattr(self, 'scale_proc_to_full', 1.0) != 0 else 1.0
                if scale != 1.0 and self.img_full is not None:
                    new_w = self.img_full.shape[1]
                    new_h = self.img_full.shape[0]
                    poster_full = cv2.resize(poster, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    # Boundary edge detection should use nearest to prevent thick/blurred edges
                    poster_edges_full = cv2.resize(poster, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    poster_full = poster.copy()
                    poster_edges_full = poster_full
                # Overlay selection by mode: Original / Mixed(50:50) / Posterized
                try:
                    overlay_mode = str(getattr(self, 'overlay_mode', 'Mixed')).lower()
                except Exception:
                    overlay_mode = 'mixed'
                if overlay_mode == 'original':
                    overlay_full = self.img_full.copy()
                elif overlay_mode == 'posterized':
                    overlay_full = poster_full.copy()
                else:
                    overlay_full = cv2.addWeighted(self.img_full, 0.5, poster_full, 0.5, 0)
                # draw boundaries if enabled
                try:
                    if self.show_boundaries:
                        # Build poster_for_edges at full resolution and apply trim in full-pixel units
                        try:
                            trim_px_full = int(params.get('trim_px', 0) or 0)
                        except Exception:
                            trim_px_full = 0
                        try:
                            poster_fe = poster_edges_full.copy()
                            if trim_px_full > 0:
                                kf = int(trim_px_full)
                                ker = np.ones((3, 3), np.uint8)
                                out_full = np.zeros_like(poster_fe)
                                unique_colors_full = np.unique(poster_fe.reshape(-1, 3), axis=0)
                                for color in unique_colors_full:
                                    mask = cv2.inRange(poster_fe, color, color)
                                    mask_e = cv2.erode(mask, ker, iterations=kf)
                                    out_full[mask_e == 255] = color
                                edge_src = out_full
                            else:
                                edge_src = poster_fe
                        except Exception:
                            edge_src = poster_full
                        h, w = edge_src.shape[:2]
                        diff_h = np.any(edge_src[:, 1:, :] != edge_src[:, :-1, :], axis=2)
                        diff_v = np.any(edge_src[1:, :, :] != edge_src[:-1, :, :], axis=2)
                        edge_mask = np.zeros((h, w), dtype=np.uint8)
                        edge_mask[:, 1:][diff_h] = 255
                        edge_mask[1:, :][diff_v] = 255
                        # trim_px_full==0 のときは見た目が太くなるため軽い erode と alpha 調整を行う
                        try:
                            is_zero = int(trim_px_full) == 0
                        except Exception:
                            is_zero = False
                        # Keep boundaries thin: avoid blur (which makes them look thicker) and
                        # blend a 1px mask with a modest alpha.
                        try:
                            if is_zero:
                                ker = np.ones((2, 2), np.uint8)
                                try:
                                    edge_mask = cv2.erode(edge_mask, ker, iterations=1)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            alpha = (edge_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
                            alpha *= 0.30 if is_zero else 0.45
                        except Exception:
                            alpha = (edge_mask.astype(np.float32) / 255.0).reshape(h, w, 1)
                        overlay_full = overlay_full.astype(np.float32)
                        overlay_full = overlay_full * (1.0 - alpha) + 255.0 * alpha
                        overlay_full = np.clip(overlay_full, 0, 255).astype(np.uint8)
                except Exception:
                    pass
                # store and display
                self._last_overlay_full = overlay_full
                try:
                    self._apply_proc_zoom()
                except Exception:
                    pass
            except Exception:
                pass
            # 強制再描画
            self.schedule_update(force=True)
        finally:
            try:
                self.btn_recalc.setEnabled(True)
            except Exception:
                pass

    def keyPressEvent(self, event):
        # ピックモード中の操作
        if self.pick_mode in ('add', 'update'):
            key = event.key()
            # Escでキャンセル
            if key == Qt.Key_Escape:
                self._end_pick_mode()
                return
            # 矢印キーでルーペ中心とカーソルを移動（倍率に反比例したステップ）
            if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                if self._img_base_size is None:
                    return
                zoom = max(1.0, float(getattr(self.magnifier, "_zoom", 4.0)))
                # 以前: 1/zoom に比例 → 今回: 1/(zoom^1.6) で“2倍強”に減速し、倍率が上がるほどどんどん遅く
                step = max(1, int(round(6 / (zoom ** 2))))
                dx = dy = 0
                
                if key == Qt.Key_Left:
                    dx = -step
                elif key == Qt.Key_Right:
                    dx = step
                elif key == Qt.Key_Up:
                    dy = -step
                elif key == Qt.Key_Down:
                    dy = step
                mods = event.modifiers()
                if mods & Qt.ShiftModifier:
                    dx *= 5; dy *= 5
                if mods & Qt.ControlModifier:
                    dx = int(dx * 0.5) if dx != 0 else 0
                    dy = int(dy * 0.5) if dy != 0 else 0
                # 現在のルーペ中心（フル座標）
                # 現在の視界中心を起点とする（簡易）
                vp = self.proc_scroll.viewport()
                # ビューポート左上のフル座標（display_scale を使用）
                ds = max(0.0001, float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom)))))
                x0_full = self.proc_scroll.horizontalScrollBar().value() / ds
                y0_full = self.proc_scroll.verticalScrollBar().value() / ds
                x_full = x0_full + vp.width() / (2.0 * ds)
                y_full = y0_full + vp.height() / (2.0 * ds)
                # 画面内に収める
                x_full = max(0, min(self._img_base_size[0] - 1, x_full + dx))
                y_full = max(0, min(self._img_base_size[1] - 1, y_full + dy))
                dxy = self._full_to_display(x_full, y_full)
                if dxy is not None:
                    local_pt = QPoint(int(round(dxy[0])), int(round(dxy[1])))
                    global_pt = self.img_label_proc.mapToGlobal(local_pt)
                    # カーソルも移動
                    QCursor.setPos(global_pt)
                return
        super().keyPressEvent(event)

    def _apply_button_styles(self):
        """Apply button styling: colors, widths, bold font, rounded corners."""
        try:
            from qt_compat.QtWidgets import QPushButton
        except Exception:
            return

        red = "rgb(160,15,15)"    # Add button color
        blue = "#757575"          # Update/Clear/Export/Clipboard color (gray)
        radius = 8
        
        # Helper to make button bold
        def _set_bold(btn):
            try:
                f = btn.font()
                f.setBold(True)
                btn.setFont(f)
            except Exception:
                pass

        # Get base width from stored or compute
        try:
            prev_base_w = int(getattr(self, '_action_btn_base_w', 0) or 0)
        except Exception:
            prev_base_w = 100
        
        base_w = max(90, prev_base_w)

        # Make all buttons bold + rounded corners
        try:
            for b in self.findChildren(QPushButton):
                try:
                    if (b.text() or "") in ("−", "▢", "✕"):
                        continue
                except Exception:
                    pass
                _set_bold(b)
                try:
                    s = b.styleSheet() or ""
                    if "border-radius" not in s:
                        b.setStyleSheet(s + f"\nQPushButton {{ border-radius: {radius}px; }}")
                except Exception:
                    pass
        except Exception:
            pass

        # Apply color-specific styles and widths
        try:
            # Add Ref. Point: dark red, wide width (1.5x)
            add_btn = getattr(self, 'btn_add_ref', None)
            if add_btn is not None:
                try:
                    style_add = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    add_btn.setStyleSheet(style_add)
                except Exception:
                    pass
                try:
                    add_btn.setFixedWidth(int(round(base_w * 1.5)))
                except Exception:
                    pass

            # Update XY + Clear: blue, same width as Export/Clipboard
            upd_btn = getattr(self, 'btn_update_xy', None)
            clr_btn = getattr(self, 'btn_clear_ref', None)
            for btn in (upd_btn, clr_btn):
                if btn is not None:
                    try:
                        style_blue = f"QPushButton {{ background-color: {blue}; color: white; border: none; border-radius: {radius}px; }}"
                        btn.setStyleSheet(style_blue)
                    except Exception:
                        pass
                    try:
                        btn.setFixedWidth(int(base_w) + 10)
                    except Exception:
                        pass

            # Export + Clipboard + Open Image (+ Flip): base style (blue) and widths
            exp_btn = getattr(self, 'btn_export', None)
            clip_btn = getattr(self, 'btn_clipboard', None)
            open_btn = getattr(self, 'btn_open', None)
            flip_btn = getattr(self, 'btn_flip_mode', None)
            combo_flip = getattr(self, 'combo_flip_mode', None)
            for btn in (exp_btn, clip_btn, open_btn, flip_btn):
                if btn is not None:
                    try:
                        style_blue = f"QPushButton {{ background-color: {blue}; color: white; border: none; border-radius: {radius}px; }}"
                        btn.setStyleSheet(style_blue)
                    except Exception:
                        pass
                    try:
                        btn.setFixedWidth(int(base_w) + 10)
                    except Exception:
                        pass

            # Open Image / Export: make them red like "Add Ref. Point"
            if open_btn is not None:
                try:
                    style_red = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    open_btn.setStyleSheet(style_red)
                except Exception:
                    pass

            if exp_btn is not None:
                try:
                    style_red = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    exp_btn.setStyleSheet(style_red)
                except Exception:
                    pass

            # Store the base width for future calls.
            # NOTE: Do not add padding here; _apply_button_styles may run many times
            # (e.g. after Update/Add pick-mode), and adding here would grow widths
            # cumulatively on every call.
            try:
                self._action_btn_base_w = int(base_w)
            except Exception:
                pass
            # Style combobox similarly (rounded right corner + down-arrow area)
            try:
                if combo_flip is not None:
                    try:
                        style_combo = (
                            f"QComboBox {{ background-color: {blue}; color: white; border: none; border-radius: {radius}px; padding: 6px 8px; }}"
                            f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border-left: none; }}"
                            f"QComboBox::down-arrow {{ width: 10px; height: 10px; }}"
                        )
                        combo_flip.setStyleSheet(style_combo)
                    except Exception:
                        pass
                    try:
                        # reduce width by 15px as requested
                        try:
                            new_w = int(base_w) + 10 - 15
                        except Exception:
                            new_w = int(base_w) if base_w is not None else 80
                        new_w = max(48, int(new_w))
                        combo_flip.setFixedWidth(new_w)
                    except Exception:
                        pass
                    try:
                        combo_flip.setFixedHeight(40)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _enforce_button_heights(self):
        """Enforce all buttons to have fixed height of 40px for better visibility."""
        try:
            from qt_compat.QtWidgets import QPushButton
            for b in self.findChildren(QPushButton):
                try:
                    if (b.text() or "") in ("−", "▢", "✕"):
                        continue
                    b.setFixedHeight(40)
                except Exception:
                    pass
        except Exception:
            pass

    def _adjust_table_column_widths(self):
        """Adjust Z column width and ensure no columns are cut off."""
        try:
            # Adjust middle table (table) columns if Z column exists
            if hasattr(self, 'table') and self.table is not None:
                # Set minimum width for Z column (column index 4 typically)
                try:
                    if self.table.columnCount() >= 5:
                        # Z column is usually the 5th column (index 4)
                        # Set it to 50px to prevent cutoff
                        self.table.setColumnWidth(4, 50)
                except Exception:
                    pass
            
            # Do the same for table_ref if needed
            if hasattr(self, 'table_ref') and self.table_ref is not None:
                try:
                    # Check if there are Z columns to adjust
                    # table_ref might have multiple Z columns (one per coordinate set)
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_frozen_headers(self):
        """Sync frozen header tables with main tables after data update."""
        try:
            # Sync table_ref_header with table_ref
            hdr_ref = getattr(self, 'table_ref_header', None)
            if hdr_ref is not None:
                try:
                    # Update column count
                    hdr_ref.setColumnCount(self.table_ref.columnCount())
                    # Sync column widths
                    for col in range(min(hdr_ref.columnCount(), self.table_ref.columnCount())):
                        w = self.table_ref.columnWidth(col)
                        if w > 0:
                            hdr_ref.setColumnWidth(col, w)
                    # Refresh display
                    hdr_ref.update()
                except Exception:
                    pass
            
            # Sync table_header with table
            hdr_mid = getattr(self, 'table_header', None)
            if hdr_mid is not None:
                try:
                    # Update column count
                    hdr_mid.setColumnCount(self.table.columnCount())
                    # Sync column widths
                    for col in range(min(hdr_mid.columnCount(), self.table.columnCount())):
                        w = self.table.columnWidth(col)
                        if w > 0:
                            hdr_mid.setColumnWidth(col, w)
                    # Refresh display
                    hdr_mid.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _create_frozen_header_tables(self):
        """Create and layout separate header tables above main tables for frozen header effect."""
        try:
            # ===== Frozen header for table_ref =====
            try:
                # Create header table with same column count as main table
                hdr_ref = getattr(self, 'table_ref_header', None)
                if hdr_ref is None or hdr_ref.isHidden():
                    hdr_ref = QTableWidget()
                    self.table_ref_header = hdr_ref
                    
                    # Setup header table
                    hdr_ref.setRowCount(2)
                    hdr_ref.setColumnCount(self.table_ref.columnCount())
                    hdr_ref.verticalHeader().setVisible(False)
                    # Ensure both header rows are visible (explicit row heights + enough frame slack)
                    try:
                        hdr_ref.setRowHeight(0, 24)
                        hdr_ref.setRowHeight(1, 20)
                        try:
                            # Ensure vertical gutter width matches main table_ref view
                            vhw = self.table_ref.verticalHeader().width()
                            if vhw > 0:
                                try:
                                    hdr_ref.verticalHeader().setFixedWidth(vhw)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    hdr_ref.setFixedHeight(60)
                    hdr_ref.setSelectionBehavior(QAbstractItemView.SelectColumns)
                    hdr_ref.setSelectionMode(QAbstractItemView.SingleSelection)
                    hdr_ref.setEditTriggers(QTableWidget.NoEditTriggers)
                    
                    # Disable scrollbars for header table and style gutter text to match background
                    try:
                        hdr_ref.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        hdr_ref.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    except Exception:
                        pass
                    try:
                        hdr_ref.verticalHeader().setStyleSheet('QHeaderView::section { background-color: lightgray; color: lightgray; }')
                    except Exception:
                        pass
                    # Sync horizontal scrollbar with main table
                    try:
                        self.table_ref.horizontalScrollBar().valueChanged.connect(
                            lambda val: hdr_ref.horizontalScrollBar().setValue(val)
                        )
                        hdr_ref.horizontalScrollBar().valueChanged.connect(
                            lambda val: self.table_ref.horizontalScrollBar().setValue(val)
                        )
                    except Exception:
                        pass
                    
                    # Sync column widths
                    try:
                        for col in range(min(hdr_ref.columnCount(), self.table_ref.columnCount())):
                            w = self.table_ref.columnWidth(col)
                            if w > 0:
                                hdr_ref.setColumnWidth(col, w)
                    except Exception:
                        pass
                    
                    # Copy header row content from main table (row 0-1 content if exists)
                    try:
                        for row in range(min(2, self.table_ref.rowCount())):
                            for col in range(self.table_ref.columnCount()):
                                src_item = self.table_ref.item(row, col)
                                if src_item is not None:
                                    new_item = QTableWidgetItem(src_item.text())
                                    new_item.setBackground(QColor("lightgray"))
                                    new_item.setForeground(QColor("black"))
                                    hdr_ref.setItem(row, col, new_item)
                    except Exception:
                        pass
                    
                    # Set row heights same as main table
                    hdr_ref.setRowHeight(0, 24)
                    hdr_ref.setRowHeight(1, 20)
            except Exception:
                pass
            
            # ===== Frozen header for table (middle table) =====
            try:
                # Create header table with same column count as main table
                hdr_mid = getattr(self, 'table_header', None)
                if hdr_mid is None or hdr_mid.isHidden():
                    hdr_mid = QTableWidget()
                    self.table_header = hdr_mid
                    
                    # Setup header table
                    hdr_mid.setRowCount(2)
                    hdr_mid.setColumnCount(self.table.columnCount())
                    hdr_mid.verticalHeader().setVisible(False)
                    # Ensure both header rows are visible (explicit row heights + enough frame slack)
                    try:
                        hdr_mid.setRowHeight(0, 24)
                        hdr_mid.setRowHeight(1, 20)
                    except Exception:
                        pass
                    hdr_mid.setFixedHeight(60)
                    hdr_mid.setSelectionBehavior(QAbstractItemView.SelectColumns)
                    hdr_mid.setSelectionMode(QAbstractItemView.SingleSelection)
                    hdr_mid.setEditTriggers(QTableWidget.NoEditTriggers)
                    
                    # Disable scrollbars and style gutter for middle header; copy header text from main table
                    try:
                        hdr_mid.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        hdr_mid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    except Exception:
                        pass
                    try:
                        hdr_mid.verticalHeader().setStyleSheet('QHeaderView::section { background-color: lightgray; color: lightgray; }')
                    except Exception:
                        pass
                    # Copy header row content from main table (row 0-1 content if exists)
                    try:
                        for row in range(min(2, self.table.rowCount())):
                            for col in range(self.table.columnCount()):
                                src_item = self.table.item(row, col)
                                if src_item is not None:
                                    new_item = QTableWidgetItem(src_item.text())
                                    new_item.setBackground(QColor("lightgray"))
                                    new_item.setForeground(QColor("black"))
                                    hdr_mid.setItem(row, col, new_item)
                    except Exception:
                        pass
                    # Sync horizontal scrollbar with main table
                    try:
                        self.table.horizontalScrollBar().valueChanged.connect(
                            lambda val: hdr_mid.horizontalScrollBar().setValue(val)
                        )
                        hdr_mid.horizontalScrollBar().valueChanged.connect(
                            lambda val: self.table.horizontalScrollBar().setValue(val)
                        )
                    except Exception:
                        pass
                    
                    # Sync column widths
                    try:
                        for col in range(min(hdr_mid.columnCount(), self.table.columnCount())):
                            w = self.table.columnWidth(col)
                            if w > 0:
                                hdr_mid.setColumnWidth(col, w)
                    except Exception:
                        pass
                    
                    # Copy header row content from main table (row 0-1 content if exists)
                    try:
                        for row in range(min(2, self.table.rowCount())):
                            for col in range(self.table.columnCount()):
                                src_item = self.table.item(row, col)
                                if src_item is not None:
                                    new_item = QTableWidgetItem(src_item.text())
                                    new_item.setBackground(QColor("lightgray"))
                                    new_item.setForeground(QColor("black"))
                                    hdr_mid.setItem(row, col, new_item)
                    except Exception:
                        pass
                    
                    # Set row heights same as main table
                    hdr_mid.setRowHeight(0, 24)
                    hdr_mid.setRowHeight(1, 20)
            except Exception:
                pass
        except Exception:
            pass

    def _setup_pseudo_headers_ref(self, tbl):
        """Setup pseudo-header rows (0-1) in left reference table using setSpan."""
        try:
            # Ensure we have at least 2 rows
            try:
                tbl.setRowCount(max(2, tbl.rowCount()))
            except Exception:
                pass

            # Row 0: Group labels (Image, Stage, Residual)
            group_configs = [
                (0, 2, "Image"),      # cols 0-1
                (2, 3, "Stage (input)"),      # cols 2-4
                (5, 4, "Residual"),   # cols 5-8
            ]
            for col_start, col_span, label in group_configs:
                item = QTableWidgetItem(label)
                try:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    font = item.font()
                    font.setBold(True)
                    font.setPointSize(font.pointSize())
                    item.setFont(font)
                    item.setBackground(QColor("lightgray"))
                    item.setForeground(QColor("black"))
                except Exception:
                    pass
                tbl.setItem(0, col_start, item)
                try:
                    tbl.setSpan(0, col_start, 1, col_span)
                except Exception:
                    pass

            # IMPORTANT: After spans are set on row 0, NOW set row 1 labels
            # This ensures row 1 doesn't get inadvertently cleared or overwritten
            # Row 1: Individual labels (u, v, X, Y, Z, X, Y, Z, |R|)
            sub_labels = ["u", "v", "X", "Y", "Z", "X", "Y", "Z", "|R|"]
            for col, label in enumerate(sub_labels):
                item = QTableWidgetItem(label)
                try:
                    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor("lightgray"))
                    item.setForeground(QColor("black"))
                except Exception:
                    pass
                tbl.setItem(1, col, item)

            # Set row heights
            try:
                tbl.setRowHeight(0, 24)
                tbl.setRowHeight(1, 20)
            except Exception:
                pass
        except Exception:
            pass

    def _setup_pseudo_headers_between(self, tbl):
        """Setup pseudo-header rows (0-1) in middle table using setSpan."""
        try:
            # Ensure we have at least 2 rows
            try:
                tbl.setRowCount(max(2, tbl.rowCount()))
            except Exception:
                pass

            # Row 0: Group labels
            # - Default (5 cols): Image(u,v) + Stage(X,Y,Z)
            # - With Level column (6 cols): Posterization(Grp) + Image(u,v) + Stage(X,Y,Z)
            try:
                ncols = int(tbl.columnCount() or 0)
            except Exception:
                ncols = 0

            if ncols >= 6:
                group_configs = [
                    (0, 1, ""),  # col 0
                    (1, 2, "Image"),          # cols 1-2
                    (3, 3, "Stage"),          # cols 3-5
                ]
                sub_labels = ["Grp", "u", "v", "X", "Y", "Z"]
            else:
                group_configs = [
                    (0, 2, "Image"),      # cols 0-1
                    (2, 3, "Stage"),      # cols 2-4
                ]
                sub_labels = ["u", "v", "X", "Y", "Z"]
            for col_start, col_span, label in group_configs:
                item = QTableWidgetItem(label)
                try:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    font = item.font()
                    font.setBold(True)
                    font.setPointSize(font.pointSize())
                    item.setFont(font)
                    item.setBackground(QColor("lightgray"))
                    item.setForeground(QColor("black"))
                except Exception:
                    pass
                tbl.setItem(0, col_start, item)
                try:
                    tbl.setSpan(0, col_start, 1, col_span)
                except Exception:
                    pass

            # IMPORTANT: After spans are set on row 0, NOW set row 1 labels
            # This ensures row 1 doesn't get inadvertently cleared or overwritten
            for col, label in enumerate(sub_labels):
                item = QTableWidgetItem(label)
                try:
                    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor("lightgray"))
                    item.setForeground(QColor("black"))
                except Exception:
                    pass
                tbl.setItem(1, col, item)

            # Set row heights
            try:
                tbl.setRowHeight(0, 24)
                tbl.setRowHeight(1, 20)
            except Exception:
                pass
        except Exception:
            pass

    def _sync_fixed_header_table(self, header_tbl, main_tbl):
        """Keep a 2-row fixed header table aligned to the scrolling main table."""
        try:
            if header_tbl is None or main_tbl is None:
                return
            try:
                if header_tbl.isHidden():
                    return
            except Exception:
                pass

            # Keep column count aligned
            try:
                if header_tbl.columnCount() != main_tbl.columnCount():
                    header_tbl.setColumnCount(main_tbl.columnCount())
            except Exception:
                pass

            # Match column widths
            try:
                for col in range(min(header_tbl.columnCount(), main_tbl.columnCount())):
                    w = main_tbl.columnWidth(col)
                    if w > 0:
                        header_tbl.setColumnWidth(col, w)
            except Exception:
                pass

            # Also match the left gutter (vertical header width) so columns line up
            try:
                vh_w = main_tbl.verticalHeader().width()
                if vh_w > 0:
                    try:
                        header_tbl.verticalHeader().setFixedWidth(vh_w)
                    except Exception:
                        pass
            except Exception:
                pass

            # Match overall width (best-effort)
            try:
                header_tbl.setFixedWidth(main_tbl.width())
            except Exception:
                pass
        except Exception:
            pass

    def _rebuild_fixed_headers(self):
        """Rebuild fixed header widgets to match current transposed tables."""
        try:
            # Left
            hdr = getattr(self, 'table_ref_view_header', None)
            main = getattr(self, 'table_ref_view', None)
            if hdr is not None and main is not None:
                try:
                    hdr.blockSignals(True)
                except Exception:
                    pass
                try:
                    nc = main.columnCount()
                    if nc > 0:
                        # Ensure column count matches main table
                        try:
                            hdr.setColumnCount(nc)
                        except Exception:
                            pass
                        # Clear all cells individually before rebuilding
                        try:
                            hdr.clearSpans()
                        except Exception:
                            pass
                        try:
                            for r in range(hdr.rowCount()):
                                for c in range(hdr.columnCount()):
                                    try:
                                        hdr.setItem(r, c, None)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        # Now rebuild the 2-row pseudo-header
                        try:
                            self._setup_pseudo_headers_ref(hdr)
                        except Exception:
                            pass
                        try:
                            # Row 0 group labels: left aligned; Row 1 sub labels: centered
                            for r in range(hdr.rowCount()):
                                for c in range(hdr.columnCount()):
                                    try:
                                        it = hdr.item(r, c)
                                        if it is None:
                                            continue
                                        if r == 0:
                                            it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                                        else:
                                            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                finally:
                    try:
                        hdr.blockSignals(False)
                    except Exception:
                        pass
                try:
                    self._sync_fixed_header_table(hdr, main)
                except Exception:
                    pass

            # Middle
            hdrm = getattr(self, 'table_between_header', None)
            mainm = getattr(self, 'table_between', None)
            if hdrm is not None and mainm is not None:
                try:
                    hdrm.blockSignals(True)
                except Exception:
                    pass
                try:
                    nc = mainm.columnCount()
                    if nc > 0:
                        # Ensure column count matches main table
                        try:
                            hdrm.setColumnCount(nc)
                        except Exception:
                            pass
                        # Clear all cells individually before rebuilding
                        try:
                            hdrm.clearSpans()
                        except Exception:
                            pass
                        try:
                            for r in range(hdrm.rowCount()):
                                for c in range(hdrm.columnCount()):
                                    try:
                                        hdrm.setItem(r, c, None)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        # Now rebuild the 2-row pseudo-header
                        try:
                            self._setup_pseudo_headers_between(hdrm)
                        except Exception:
                            pass
                        try:
                            # Row 0 group labels: left aligned; Row 1 sub labels: centered
                            for r in range(hdrm.rowCount()):
                                for c in range(hdrm.columnCount()):
                                    try:
                                        it = hdrm.item(r, c)
                                        if it is None:
                                            continue
                                        if r == 0:
                                            it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                                        else:
                                            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                finally:
                    try:
                        hdrm.blockSignals(False)
                    except Exception:
                        pass
                try:
                    self._sync_fixed_header_table(hdrm, mainm)
                except Exception:
                    pass
        except Exception:
            pass