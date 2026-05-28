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
    QHeaderView, QScrollArea, QApplication, QMenu, QComboBox, QTabWidget, QFrame
)
from qt_compat.QtWidgets import QButtonGroup
from qt_compat.QtCore import Qt, QTimer, QObject, QEvent, QRect, QPoint, pyqtSignal, QThread, QSettings
from qt_compat.QtGui import QPixmap, QFont, QCursor, QPainter, QPen, QColor, QPalette

from Util import cvimg_to_qpixmap, kmeans_posterize
from CalcCentroid import CentroidProcessor, CalculationCancelled
from Config import (
    PROC_TARGET_WIDTH,
    save_last_image_path,
    load_last_image_path,
    DEBUG,
    DEFAULT_MAX_GRAIN_AREA,
    DEFAULT_MIN_GRAIN_AREA,
)

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
import sys
import math
import ctypes
from ctypes import wintypes


# Unified table row-height constants (single source of truth)
TABLE_HEADER_ROW0_HEIGHT = 24
TABLE_HEADER_ROW1_HEIGHT = 20
TABLE_DEFAULT_ROW_HEIGHT = 24


class SegmentControl(QWidget):
    """Simple segmented control: horizontal checkable buttons in an exclusive group.

    Usage: sc = SegmentControl(["A","B"], checked_index=0, btn_w=64, btn_h=24)
    Connect change via `sc.set_on_changed(callback)` where callback(index:int).
    """
    def __init__(self, labels, parent=None, checked_index=0, btn_w=64, btn_h=35, blue="#757575"):
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
            "QPushButton { border: 1px solid lightgray; padding: 2px 10px; border-radius: 0px; font-weight: normal; }"
            # Selected: keep existing colors, but emphasize with bold.
            "QPushButton:checked { background-color: " + blue + "; color: white; font-weight: bold; }"
            # Unselected: dark gray text instead of black.
            "QPushButton:!checked { background-color: white; color: #555555; font-weight: normal; }"
            # Special-case: allow a checked segment to be rendered in normal weight (e.g. 'Calculating').
            "QPushButton[pixy_calc_in_progress=\"true\"] { font-weight: normal; }"
            "QPushButton[pixy_calc_in_progress=\"true\"]:checked { font-weight: normal; }"
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
                    + "QPushButton { border-top-left-radius: 10px; border-bottom-left-radius: 10px; border-top-right-radius: 0px; border-bottom-right-radius: 0px; border-right: none; }"
                )
            elif i == len(labels) - 1:
                # Right-most: round only the outer-right corners; keep inner-left corners square.
                b.setStyleSheet(
                    qss_base
                    + "QPushButton { border-top-right-radius: 10px; border-bottom-right-radius: 10px; border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: none; }"
                )
            else:
                # Middle segments: all corners square.
                b.setStyleSheet(qss_base + "QPushButton { border-radius: 0px; border-left: none; border-right: none; }")
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
            # Single-line title (no note line)
            painter.setFont(self.font())
            painter.drawText(QRect(0, 0, max(10, w), max(10, int(margin_t - 6))), Qt.AlignLeft | Qt.AlignVCenter, "Grain Size Threshold (pix)")
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
        layout.setSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: white; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(self.status_label, 1)

        self.version_label = QLabel("")
        self.version_label.setStyleSheet("color: white; font-size: 11px;")
        self.version_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        layout.addWidget(self.version_label, 0)

    def paintEvent(self, event):
        """Force paint background to ensure color is applied."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
    def showMessage(self, msg):
        self.status_label.setText(msg)

    def setVersion(self, version: str | None):
        if version:
            self.version_label.setText(f"Ver. {version}")
        else:
            self.version_label.setText("")


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
        # v1.1.6+: window title is stable; version is shown in the footer.
        self._app_version = None
        try:
            import sys

            def _read_version_from_pyproject(pyproject_path: str) -> str | None:
                try:
                    if not pyproject_path or not os.path.isfile(pyproject_path):
                        return None
                except Exception:
                    return None
                # Prefer tomllib if available
                try:
                    import tomllib
                    with open(pyproject_path, 'rb') as tf:
                        data = tomllib.load(tf)
                    v = data.get('project', {}).get('version')
                    return str(v) if v else None
                except Exception:
                    pass
                # Fallback: simple regex parse
                try:
                    import re
                    with open(pyproject_path, 'r', encoding='utf-8') as tf:
                        txt = tf.read()
                    m = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]", txt, re.M)
                    return m.group(1) if m else None
                except Exception:
                    return None

            version = None
            here = os.path.dirname(__file__)
            candidates = [
                os.path.join(here, 'pyproject.toml'),
            ]
            try:
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    candidates.insert(0, os.path.join(sys._MEIPASS, 'pyproject.toml'))
            except Exception:
                pass
            try:
                # If running from an EXE without _MEIPASS (edge cases), also try the EXE directory.
                candidates.append(os.path.join(os.path.dirname(sys.executable), 'pyproject.toml'))
            except Exception:
                pass
            try:
                # Repo root when running from a worktree (../..)
                candidates.append(os.path.abspath(os.path.join(here, os.pardir, os.pardir, 'pyproject.toml')))
            except Exception:
                pass

            for pth in candidates:
                version = _read_version_from_pyproject(pth)
                if version:
                    break

            if not version:
                # If installed as a package, try package metadata
                try:
                    import importlib.metadata as _im
                    version = _im.version('pixy')
                except Exception:
                    version = None

            self._app_version = version
        except Exception:
            self._app_version = None

        try:
            self.setWindowTitle(STR.APP_TITLE)
        except Exception:
            pass

        # デバッグ出力ヘルパ
        def _dbg(msg):
            if DEBUG:
                try:
                    import os as _os
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    pid = _os.getpid()
                    line = f"[DEBUG {ts} pid={pid}] {msg}"
                    print(line, flush=True)
                    # Also write to file for persistent logging
                    with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
        self._dbg = _dbg

        # Always start a fresh log per run so tailing doesn't show stale traces.
        # Also provide an error logger that works even when DEBUG is off.
        def _log_error(msg):
            try:
                import os as _os
                from datetime import datetime as _dt
                ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                pid = _os.getpid()
                line = f"[ERROR {ts} pid={pid}] {msg}"
                print(line, flush=True)
                with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        self._log_error = _log_error

        def _log_info(msg):
            """Write lightweight run-time info to the log even when DEBUG is off."""
            try:
                import os as _os
                from datetime import datetime as _dt
                ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                pid = _os.getpid()
                line = f"[INFO {ts} pid={pid}] {msg}"
                # Do not spam stdout unless DEBUG is enabled
                if DEBUG:
                    print(line, flush=True)
                with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        self._log_info = _log_info

        try:
            import os as _os
            from datetime import datetime as _dt
            ui_path = _os.path.abspath(__file__)
            with open("debug_px2xy.log", "w", encoding="utf-8") as f:
                f.write(
                    f"[RUN {_dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')} pid={_os.getpid()}] "
                    f"DEBUG={'1' if bool(DEBUG) else '0'} Ui={ui_path}\n"
                )
        except Exception:
            pass

        # 画像関連変数
        self.img_full = None          # フル解像度画像 (numpy array)
        self.proc_img = None          # 処理用縮小画像
        self.scale_proc_to_full = 1.0 # 処理画像からフル画像へのスケール
        self.proc_target_width = PROC_TARGET_WIDTH  # 処理画像の目標幅

        # 重心処理関連
        self.centroid_processor = None  # CentroidProcessor インスタンス
        self.centroids = []            # 検出された重心リスト [(group_no, x, y), ...]
        self._auto_centroids = []      # 自動検出のみの重心リスト（手動加算前）
        self.manual_target_mode = False  # True: 手動ターゲット(Group0)を優先
        self.manual_targets = []         # [(0, x_proc, y_proc), ...]
        self.excluded_centroid_indices = set()  # 出力除外する重心インデックス
        self._explicit_excluded_centroid_indices = set()  # ユーザー操作/置換により明示的に除外された重心インデックス
        self._force_visible_centroid_indices = set()  # フィルタ非表示中でも一時的に表示する重心インデックス
        self.visible_groups = None       # None=all visible, otherwise set[int]
        self.center_list_indices = []    # 中カラムに追加された重心インデックス
        self.center_numeric_rows = []    # 中カラム表示値の数値スナップショット(dict list)
        self._table_between_row_indices = []  # table_between行 -> centroidsインデックス
        self.overlay_point_source = 'left'  # 'left': 左リスト(全検出), 'center': 中リスト(Add済み)
        self._replace_target_source_index = None
        self._replace_target_source_group = None
        self.selected_index = None     # 選択中の重心インデックス
        self.select_radius_display = 10.0  # 画像上の選択半径 (pix)

        # 参照点関連
        self.ref_points = [None] * 10  # 参照点リスト [(x_proc, y_proc) or None]
        self.ref_selected_index = 0     # 選択中の参照点インデックス
        self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]  # 参照点の観測値
        self.excluded_ref_indices = set()  # 座標変換から除外する参照点インデックス

        # UI 状態
        self.visible_ref_cols = 3      # 表示する参照点列数
        self.flip_mode = 'auto'        # 左右反転モード ('auto', 'normal', 'flip')
        self.overlay_mode = 'Original'  # Overlay display mode: Original / Posterized
        self._centroid_extraction_overlay_mode = 'Posterized'  # CentroidExtraction時の既定/記憶値
        self._centroid_extraction_show_boundaries = True
        try:
            self._load_centroid_extraction_preferences()
        except Exception:
            pass
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
        self.view_orientation = 'Image'  # Coordinate (Image/Stage)

        # Stage座標表示の符号（Stageモード時のみ表示に反映）
        self.stage_axis_x_sign = 1  # +1: right is +X, -1: right is -X
        self.stage_axis_y_sign = 1  # +1: up is +Y,   -1: up is -Y

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

        # Stage情報オーバーレイ（左上固定）
        self.stage_info_overlay = QLabel(self.proc_scroll.viewport())
        self.stage_info_overlay.setWordWrap(False)
        self.stage_info_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.stage_info_overlay.setStyleSheet(
            "QLabel {"
            "color: rgb(235, 235, 235);"
            "background-color: rgba(20, 20, 20, 140);"
            "padding: 3px 6px;"
            "border-radius: 4px;"
            "}"
        )
        try:
            f_ov = self.stage_info_overlay.font()
            f_ov.setPointSize(10)
            f_ov.setBold(True)
            self.stage_info_overlay.setFont(f_ov)
        except Exception:
            pass
        self.stage_info_overlay.hide()

        # カーソル座標オーバーレイ（右下固定）
        self.cursor_info_overlay = QLabel(self.proc_scroll.viewport())
        self.cursor_info_overlay.setWordWrap(False)
        self.cursor_info_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.cursor_info_overlay.setStyleSheet(
            "QLabel {"
            "color: rgb(235, 235, 235);"
            "background-color: rgba(20, 20, 20, 140);"
            "padding: 3px 6px;"
            "border-radius: 4px;"
            "}"
        )
        try:
            f_cv = self.cursor_info_overlay.font()
            f_cv.setPointSize(10)
            f_cv.setBold(True)
            self.cursor_info_overlay.setFont(f_cv)
        except Exception:
            pass
        self.cursor_info_overlay.hide()
        try:
            self._reposition_viewport_overlays()
        except Exception:
            pass


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

        # Setup pseudo-headers in table_ref for 2-row header appearance.
        # Keep canonical self.table free from pseudo-header cell writes.
        try:
            # Ensure minimum rows for headers
            if self.table_ref.rowCount() < 2:
                self.table_ref.setRowCount(2)
            if self.table.rowCount() < 2:
                self.table.setRowCount(2)
            
            # Apply pseudo-headers
            self._setup_pseudo_headers_ref(self.table_ref)
            
            # Enforce row heights for headers and data rows
            try:
                vh_ref = self.table_ref.verticalHeader()
                vh_ref.setSectionResizeMode(QHeaderView.Fixed)
                vh_ref.setDefaultSectionSize(TABLE_DEFAULT_ROW_HEIGHT)
                self.table_ref.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                self.table_ref.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
            except Exception:
                pass

            try:
                vh_table = self.table.verticalHeader()
                vh_table.setSectionResizeMode(QHeaderView.Fixed)
                vh_table.setDefaultSectionSize(TABLE_DEFAULT_ROW_HEIGHT)
                self.table.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                self.table.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
        # 左テーブル: ユーザー操作時のみ編集 (Stage.* 行のみ有効)
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
        # 左テーブルクリックイベント (Stage行即編集)
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
        self.btn_open = QPushButton("Export Image")
        self.btn_open.setFixedHeight(40)
        self.btn_open.clicked.connect(self._on_export_image_clicked)
        self.btn_replace_image = QPushButton("Replace Image")
        self.btn_replace_image.setFixedHeight(40)
        self.btn_replace_image.clicked.connect(self._on_replace_image_clicked)
        # Export ボタンは短くして隣に Clipboard を追加
        self.btn_export = QPushButton("Export XYZ")
        self.btn_export.setFixedHeight(40)
        self.btn_export.clicked.connect(self.export_centroids)
        self.btn_clipboard = QPushButton("Clipboard")
        self.btn_clipboard.setFixedHeight(40)
        self.btn_clipboard.clicked.connect(self._copy_centroids_to_clipboard)
        self.btn_filter = QPushButton("Filter")
        self.btn_filter.setFixedHeight(40)
        self.btn_filter.clicked.connect(self._show_group_filter_popup)
        self.btn_add_target = QPushButton("Add Target")
        self.btn_add_target.setFixedHeight(40)
        self.btn_add_target.clicked.connect(self._on_add_target_point)
        self.btn_update_target_uv = QPushButton("Update u, v")
        self.btn_update_target_uv.setFixedHeight(40)
        self.btn_update_target_uv.clicked.connect(self._on_update_target_uv)
        self.btn_clear_target = QPushButton("Clear")
        self.btn_clear_target.setFixedHeight(40)
        self.btn_clear_target.clicked.connect(self._on_clear_target)
        self.btn_clear_target_all = QPushButton("Clear ALL")
        self.btn_clear_target_all.setFixedHeight(40)
        self.btn_clear_target_all.clicked.connect(self._on_clear_target_all)

        # 自動更新/手動再計算の UI 部品を先に作成
        # Auto Update の ON/OFF 表示・選択は不要なので、常に auto_update_mode=True とする
        self.interp_mode = 'auto'  # 常に auto モード（ズーム倍率で自動選択）
        self.auto_update_mode = True
        self.chk_auto_update = None
        # Recalc ボタン表示は不要
        self.btn_recalc = None

        # v1.1.7+: heavy recomputation can be manual-triggered
        self.calc_mode = 'auto'  # 'auto' | 'manual'
        self._manual_recompute_request = False
        # Auto/Manualで計算パラメータを分離保持（切替時に保存/復元）
        self._calc_params_by_mode = {'auto': None, 'manual': None}
        self.calc_mode_controls = None
        self.lbl_calc_mode = None
        self.toggle_calc_mode = None
        self.btn_stop_calc = None
        self._calc_in_progress = False
        self._calc_stop_requested = False
        self._calc_trace_seq = 0
        self._calc_trace_last_reason = ""
        # New Project 初期化用: Area閾値を下位/上位1/3で一度だけ自動設定
        self._area_init_tercile_pending = False
        self._pending_recompute_after_area_init = False
        self._ref_add_has_added = False
        self.centroid_extraction_mode = False

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
        self._max_render_pixels_override = None  # ホイール中の軽量描画用（Noneで通常）
        self.max_zoom_target_visible_px = 220    # 最大拡大時に長辺方向で見える元画像pxの目標
        self._normal_max_render_pixels = 8192 * 8192
        self._hard_max_render_pixels = 12288 * 12288
        # 初回表示は画像中心から開始するためのフラグ
        self._initial_center_done = False
        # 通常時は手のカーソル
        self.img_label_proc.setCursor(QCursor(Qt.OpenHandCursor))

        # 画像右上用の「境界線」トグルボタン（先に生成しておく）
        # 画像右上用の「境界線」トグル（Show/Hide の2択）
        self.show_boundaries = True
        try:
            self.boundary_toggle = SegmentControl(["Show", "Hide"], checked_index=0, btn_w=64, btn_h=27)
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
                # Coordinate  トグル（Image / Stage）を右隣に追加
                try:
                    self.view_orientation_toggle = SegmentControl(["Image", "Stage"], checked_index=0, btn_w=69, btn_h=27)
                    try:
                        # Handler name is `_on_toggle_coordinate` (label is "Coordinate"), keep wiring consistent.
                        self.view_orientation_toggle.set_on_changed(lambda idx: self._on_toggle_coordinate(int(idx)))
                    except Exception:
                        pass
                    try:
                        self.btn_view_image = self.view_orientation_toggle._buttons[0]
                        self.btn_view_stage = self.view_orientation_toggle._buttons[1]
                    except Exception:
                        self.btn_view_image = None
                        self.btn_view_stage = None
                    # small label for the control
                    self.lbl_view_orientation = QLabel("Coordinate")
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

        # 手動画像回転（Imageビュー専用）と左右反転モード（ビュー別）
        self.manual_image_rotation_deg = 0
        self.flip_mode_image = 'normal'  # 'normal' | 'flip'
        self.flip_mode_stage = 'auto'    # 'auto' | 'normal' | 'flip'
        # Flipトグル（Image/Stage用を用意し、Coordinateに応じて出し分け）
        self.flip_toggle_image = None
        self.flip_toggle_stage = None
        self.lbl_scale_val = None
        self.lbl_tx_val = None
        self.lbl_ty_val = None
        self.lbl_angle_val = None

        # 画像領域レイアウト：上にボタン群（左に Open/Export/Clipboard、中央に補間/自動系、右に Flip/境界）
        img_layout = QVBoxLayout()
        img_header = QHBoxLayout()
        img_header.setContentsMargins(0, 0, 0, 0)
        img_header.setSpacing(8)
        # 1段目: Open / Boundary / Display Mode
        try:
            # Simply add the btn_open directly to avoid nested layout alignment issues
            img_header.addWidget(self.btn_open, 0, Qt.AlignLeft | Qt.AlignVCenter)
            img_header.addWidget(self.btn_replace_image, 0, Qt.AlignLeft | Qt.AlignVCenter)
        except Exception:
            pass
        # 中央上には自動更新/手動再計算をまとめる
        try:
            center_controls = QHBoxLayout()
            center_controls.setContentsMargins(0, 0, 0, 0)
            center_controls.setSpacing(6)
            img_header.addLayout(center_controls)
        except Exception:
            pass
        # 右上コントロール（左→右）: Coordinate, Boundary, Posterization Overlay
        overlay_ctrl = None
        self.overlay_mode_controls = None
        try:
            # small overlay control placed at right-top next to Flip
            overlay_ctrl = QWidget()
            self.overlay_mode_controls = overlay_ctrl
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
                    self.overlay_mode_toggle = SegmentControl(["Original", "Posterized"], checked_index=0, btn_w=108, btn_h=27)
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
            self.overlay_mode_controls = None
            pass

        # Boundary / Display Mode は左詰めで配置（Openの右側）
        try:
            if getattr(self, 'boundary_controls', None) is not None:
                img_header.addWidget(self.boundary_controls, 0, Qt.AlignLeft | Qt.AlignVCenter)
            elif getattr(self, 'boundary_toggle', None) is not None:
                img_header.addWidget(self.boundary_toggle, 0, Qt.AlignLeft | Qt.AlignVCenter)
        except Exception:
            pass

        # Display Mode（Original/Posterized）も左詰め
        try:
            if overlay_ctrl is not None:
                try:
                    # spacing between boundary and overlay
                    if getattr(self, 'boundary_controls', None) is not None or getattr(self, 'boundary_toggle', None) is not None:
                        img_header.addSpacing(12)
                except Exception:
                    pass
                try:
                    img_header.addWidget(overlay_ctrl, 0, Qt.AlignLeft | Qt.AlignVCenter)
                except Exception:
                    img_header.addWidget(overlay_ctrl)
        except Exception:
            pass

        # Push any remaining space to the right so the above controls stay left-aligned
        img_header.addStretch(1)

        img_layout.addLayout(img_header, 0)

        # 2段目: Coordinate / Image Rotate / Normal/Flip（Stage時はMagnification/X/Y/Z）
        try:
            midbar = QWidget()
            midbar.setFixedHeight(36)
            mb = QHBoxLayout(midbar)
            mb.setContentsMargins(6, 0, 6, 0)
            mb.setSpacing(10)

            # Coordinate toggle on the left of the 2nd row
            try:
                if getattr(self, 'view_orientation_controls', None) is not None:
                    mb.addWidget(self.view_orientation_controls, 0, Qt.AlignVCenter)
                    mb.addSpacing(10)
            except Exception:
                pass

            # --- Axis sign controls (Stage only)
            self._mid_axis_controls = QWidget()
            try:
                ahl = QHBoxLayout(self._mid_axis_controls)
                ahl.setContentsMargins(0, 0, 0, 0)
                ahl.setSpacing(8)

                lbl_xa = QLabel("Right")
                try:
                    f = lbl_xa.font(); f.setBold(True); lbl_xa.setFont(f)
                except Exception:
                    pass
                try:
                    self.axis_toggle_x = SegmentControl(["+X", "-X"], checked_index=0, btn_w=44, btn_h=27)
                    self.axis_toggle_x.set_on_changed(lambda idx: self._on_stage_axis_changed('x', int(idx)))
                except Exception:
                    self.axis_toggle_x = None

                lbl_ya = QLabel("Top")
                try:
                    f = lbl_ya.font(); f.setBold(True); lbl_ya.setFont(f)
                except Exception:
                    pass
                try:
                    self.axis_toggle_y = SegmentControl(["+Y", "-Y"], checked_index=0, btn_w=44, btn_h=27)
                    self.axis_toggle_y.set_on_changed(lambda idx: self._on_stage_axis_changed('y', int(idx)))
                except Exception:
                    self.axis_toggle_y = None

                ahl.addWidget(lbl_xa)
                if self.axis_toggle_x is not None:
                    ahl.addWidget(self.axis_toggle_x)
                ahl.addSpacing(10)
                ahl.addWidget(lbl_ya)
                if self.axis_toggle_y is not None:
                    ahl.addWidget(self.axis_toggle_y)
            except Exception:
                pass

            # Build 3 groups so we can toggle visibility cleanly by Coordinate.
            self._mid_rotate_controls = QWidget()
            self._mid_flip_controls = QWidget()
            self._mid_stats_controls = QWidget()

            # --- Rotate group (Image only)
            try:
                rhl = QHBoxLayout(self._mid_rotate_controls)
                rhl.setContentsMargins(0, 0, 0, 0)
                rhl.setSpacing(10)

                lbl_rot = QLabel("Image Rotate")
                try:
                    f = lbl_rot.font()
                    f.setBold(True)
                    lbl_rot.setFont(f)
                except Exception:
                    pass

                self.slider_img_rotate = ClickableSlider(Qt.Horizontal)
                try:
                    self.slider_img_rotate.setMinimum(-180)
                    self.slider_img_rotate.setMaximum(180)
                    self.slider_img_rotate.setSingleStep(10)
                    # Clicking on the groove uses pageStep in many styles; keep it 10 deg.
                    self.slider_img_rotate.setPageStep(10)
                    # Use only Qt standard ticks at 30-degree intervals.
                    self.slider_img_rotate.setTickInterval(30)
                    self.slider_img_rotate.setTickPosition(QSlider.TicksBelow)
                    # Allow continuous wheel rotation by wrapping at ±180.
                    self.slider_img_rotate._wheel_wrap = True
                    # Disable custom tick overlay for this slider (Qt standard ticks only).
                    self.slider_img_rotate._use_custom_ticks = False
                    # Make slider wider and taller so tick marks can be drawn below it
                    self.slider_img_rotate.setFixedWidth(260)
                    try:
                        self.slider_img_rotate.setFixedHeight(28)
                    except Exception:
                        pass
                    self.slider_img_rotate.setValue(int(self.manual_image_rotation_deg))
                except Exception:
                    pass

                self.lbl_rot_val = QLabel("0°")
                self.lbl_rot_val.setFixedWidth(38)
                self.lbl_rot_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                try:
                    self.slider_img_rotate.valueChanged.connect(self._on_manual_image_rotation_changed)
                except Exception:
                    pass

                rhl.addWidget(lbl_rot)
                # Wrap slider in a small vertical container so we can nudge it downward
                slider_container = QWidget()
                try:
                    svl = QVBoxLayout(slider_container)
                    svl.setContentsMargins(0, 0, 0, 0)
                    svl.setSpacing(0)
                    # vertically center the slider so its groove aligns with the label
                    svl.setAlignment(self.slider_img_rotate, Qt.AlignVCenter)
                    svl.addWidget(self.slider_img_rotate)
                except Exception:
                    # fallback: add slider directly if layout creation fails
                    try:
                        rhl.addWidget(self.slider_img_rotate)
                    except Exception:
                        pass
                try:
                    rhl.addWidget(slider_container, 0, Qt.AlignVCenter)
                except Exception:
                    try:
                        rhl.addWidget(self.slider_img_rotate)
                    except Exception:
                        pass
                rhl.addWidget(self.lbl_rot_val)
            except Exception:
                pass

            # --- Flip group (Image only per latest request)
            try:
                fhl = QHBoxLayout(self._mid_flip_controls)
                fhl.setContentsMargins(0, 0, 0, 0)
                fhl.setSpacing(6)
                try:
                    self.flip_toggle_image = SegmentControl(["Normal", "Flip"], checked_index=0, btn_w=77, btn_h=27)
                    self.flip_toggle_image.set_on_changed(lambda idx: self._on_flip_changed('image', int(idx)))
                except Exception:
                    self.flip_toggle_image = None
                # Keep stage flip toggle object for backward compatibility, but do not show it here.
                try:
                    self.flip_toggle_stage = SegmentControl(["Auto", "Normal", "Flip"], checked_index=0, btn_w=77, btn_h=27)
                    self.flip_toggle_stage.set_on_changed(lambda idx: self._on_flip_changed('stage', int(idx)))
                    self.flip_toggle_stage.setVisible(False)
                except Exception:
                    self.flip_toggle_stage = None
                if self.flip_toggle_image is not None:
                    fhl.addWidget(self.flip_toggle_image)
            except Exception:
                pass

            # --- Stats group (Stage only)
            try:
                shl = QHBoxLayout(self._mid_stats_controls)
                shl.setContentsMargins(0, 0, 0, 0)
                shl.setSpacing(10)

                def _mk_stat(label_text, min_width=0):
                    box = QWidget()
                    hb = QHBoxLayout(box)
                    hb.setContentsMargins(0, 0, 0, 0)
                    hb.setSpacing(4)
                    lbl = QLabel(label_text)
                    val = QLabel("-")
                    try:
                        # Keep values readable & tight (avoid large gap like "X:      158")
                        if int(min_width) > 0:
                            val.setMinimumWidth(int(min_width))
                        val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    except Exception:
                        pass
                    hb.addWidget(lbl)
                    hb.addWidget(val)
                    return box, val

                w_s, self.lbl_scale_val = _mk_stat("Magnification:", min_width=56)
                w_rot, self.lbl_angle_val = _mk_stat("Rotation:", min_width=56)
                w_flip, self.lbl_flip_val = _mk_stat("Flip:", min_width=44)
                w_tx, self.lbl_tx_val = _mk_stat("Shift X:", min_width=46)
                w_ty, self.lbl_ty_val = _mk_stat("Shift Y:", min_width=46)
                w_pitch, self.lbl_pitch_val = _mk_stat("Pitch:", min_width=46)
                w_roll, self.lbl_roll_val = _mk_stat("Roll:", min_width=46)

                shl.addWidget(w_s)
                shl.addWidget(w_rot)
                shl.addWidget(w_flip)
                shl.addWidget(w_tx)
                shl.addWidget(w_ty)
                shl.addWidget(w_pitch)
                shl.addWidget(w_roll)
            except Exception:
                pass

            # Add groups to the bar (left aligned, vertically centered)
            mb.addWidget(self._mid_axis_controls, 0, Qt.AlignVCenter)
            mb.addWidget(self._mid_rotate_controls, 0, Qt.AlignVCenter)
            mb.addWidget(self._mid_flip_controls, 0, Qt.AlignVCenter)
            mb.addWidget(self._mid_stats_controls, 0, Qt.AlignVCenter)
            mb.addStretch(1)

            # Initial visibility: Image mode
            try:
                if getattr(self, '_mid_axis_controls', None) is not None:
                    self._mid_axis_controls.setVisible(False)
                self._mid_rotate_controls.setVisible(True)
                self._mid_flip_controls.setVisible(True)
                self._mid_stats_controls.setVisible(False)
            except Exception:
                pass

            img_layout.addWidget(midbar, 0)
        except Exception:
            pass
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
            sliders_layout.setSpacing(0)
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
                    minus_btn.setFixedSize(28, 23)
                    minus_btn.clicked.connect(lambda _, f=nudger_minus: f(-1))
                except Exception:
                    minus_btn = QPushButton("-")

                try:
                    plus_btn = QPushButton("+")
                    plus_btn.setFixedSize(28, 23)
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

        # Area histogram (Advanced only; place at the bottom under Shape Complexity)
        try:
            self.area_hist = AreaHistogramWidget()
            try:
                self.area_hist.setFont(ctrl_font)
            except Exception:
                pass
            try:
                # Keep chart area stable so axis labels do not visually collide with next row.
                self.area_hist.setMinimumHeight(190)
                self.area_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            except Exception:
                pass
            try:
                self.area_hist.rangeChanged.connect(self._on_area_hist_range_changed)
            except Exception:
                pass
            sliders_layout.addWidget(self.area_hist)
        except Exception:
            self.area_hist = None

        # MinAreaとテーブルの間にボタン行（左詰め）を追加
        actions_row = QHBoxLayout()
        # ここから「境界線」ボタンは削除（画像右上に移動済み）
        self.btn_start_centroid_extraction = QPushButton("START Centroid Extraction")
        self.btn_add_ref = QPushButton(STR.BUTTON_ADD_REF)
        self.btn_update_xy = QPushButton(STR.BUTTON_UPDATE_XY)
        self.btn_clear_ref = QPushButton(STR.BUTTON_CLEAR)
        self.btn_start_centroid_extraction.clicked.connect(self._on_toggle_centroid_extraction_mode)
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
            base_dirs = []
            try:
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    base_dirs.append(sys._MEIPASS)
            except Exception:
                pass
            base_dir = os.path.dirname(__file__)
            if base_dir not in base_dirs:
                base_dirs.append(base_dir)

            file_candidates = [
                "PiXY_Pix.png",  # Image mode default
                "PiXY.png",
                "px2XY2.png",
                "px2XY.png",
                "app_icon.png",
            ]
            candidates = [os.path.join(root, name) for root in base_dirs for name in file_candidates]
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

        # Use explicit font size for numeric contents (requested: specify directly)
        try:
            f = self.table_ref_view.font()
            if f is not None:
                f.setPointSize(10)
                self.table_ref_view.setFont(f)
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
        # Single-click edit in transposed view for Stage X/Y/Z columns
        try:
            self.table_ref_view.cellClicked.connect(self._on_ref_view_cell_clicked)
        except Exception:
            pass
        # Enable user-initiated edits (Stage columns only are editable at item-level)
        try:
            triggers = (
                QAbstractItemView.EditKeyPressed
                | QAbstractItemView.SelectedClicked
                | QAbstractItemView.DoubleClicked
            )
            self.table_ref_view.setEditTriggers(triggers)
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
                self.table_between.cellClicked.connect(self._on_table_between_cell_clicked)
            except Exception:
                pass
        except Exception:
            pass

        # Use explicit font size for numeric contents (requested: specify directly)
        try:
            f2 = self.table_between.font()
            if f2 is not None:
                f2.setPointSize(10)
                self.table_between.setFont(f2)
        except Exception:
            pass

        left_col = QVBoxLayout()
        left_col.addWidget(self.left_top_image, 0, Qt.AlignTop)

        # Save / Load Project ボタン（ロゴの下、Add Fiducial の上）
        try:
            self.left_project_row = QWidget()
            try:
                self.left_project_row.setObjectName('leftProjectRow')
                self.left_project_row.setStyleSheet('#leftProjectRow { border-bottom: 1px solid #c6c6c6; }')
            except Exception:
                pass
            _proj_row = QHBoxLayout(self.left_project_row)
            _proj_row.setContentsMargins(0, 0, 0, 0)
            _proj_row.setSpacing(6)
            self.btn_new_project = QPushButton("New Project")
            self.btn_save_project = QPushButton("Save Project")
            self.btn_load_project = QPushButton("Load Project")
            for _btn in (self.btn_new_project, self.btn_save_project, self.btn_load_project):
                _btn.setFixedHeight(28)
            self.btn_new_project.clicked.connect(self.open_image)
            self.btn_save_project.clicked.connect(self.save_project)
            self.btn_load_project.clicked.connect(self.load_project)
            _proj_row.addWidget(self.btn_new_project)
            _proj_row.addWidget(self.btn_save_project)
            _proj_row.addWidget(self.btn_load_project)
            _proj_row.addStretch(1)
            left_col.addWidget(self.left_project_row, 0)
        except Exception:
            self.left_project_row = None
            self.btn_new_project = None
            self.btn_save_project = None
            self.btn_load_project = None

        # Build Grain Identification block (to be placed below ref table)
        try:
            self.grain_ident_mode = 'advanced'
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
            self.lbl_grain_ident = QLabel(STR.SECTION_AUTO_DETECT)
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
                self.lbl_grain_ident.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            except Exception:
                pass
            try:
                gil.addStretch(1)
            except Exception:
                pass
            try:
                # Match Display Mode toggle size (btn_w=108, btn_h=24)
                self.toggle_grain_ident = SegmentControl(["Basic", "Advanced"], checked_index=0, btn_w=108, btn_h=27)
                try:
                    self.toggle_grain_ident.set_on_changed(lambda idx: self._on_toggle_grain_ident(int(idx)))
                except Exception:
                    pass
                try:
                    self.toggle_grain_ident.setCheckedIndex(1)
                    self.toggle_grain_ident.setVisible(False)
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
                try:
                    # User requested to remove the Auto-detect title row.
                    self.grain_ident_controls.setVisible(False)
                except Exception:
                    pass

            # Body: wraps calc_mode_controls + sliders_layout
            self.grain_body = QWidget()
            gb_layout = QVBoxLayout(self.grain_body)
            gb_layout.setContentsMargins(0, 0, 0, 0)
            gb_layout.setSpacing(6)

            # Recalculation Trigger controls (v1.1.9-style): Auto/Manual with Manual -> ReCalculate
            try:
                self.calc_mode_controls = QWidget()
                cml = QHBoxLayout(self.calc_mode_controls)
                cml.setContentsMargins(0, 0, 0, 0)
                cml.setSpacing(6)
                self.lbl_calc_mode = QLabel("Recalculation Trigger")
                try:
                    fcm = self.lbl_calc_mode.font()
                    fcm.setBold(True)
                    self.lbl_calc_mode.setFont(fcm)
                except Exception:
                    pass
                cml.addWidget(self.lbl_calc_mode)
                try:
                    self.lbl_calc_mode.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
                except Exception:
                    pass
                try:
                    cml.addStretch(1)
                except Exception:
                    pass
                try:
                    self.toggle_calc_mode = SegmentControl(["Auto", "Manual"], checked_index=0, btn_w=108, btn_h=27)
                    try:
                        self.toggle_calc_mode.set_on_changed(lambda idx: self._on_toggle_calc_mode(int(idx)))
                    except Exception:
                        pass
                    try:
                        # Manual button click triggers recalculation when already in manual mode
                        self.toggle_calc_mode._buttons[1].clicked.connect(self._on_manual_recalculate_clicked)
                    except Exception:
                        pass
                    cml.addWidget(self.toggle_calc_mode)
                except Exception:
                    self.toggle_calc_mode = None
                try:
                    self.btn_stop_calc = QPushButton("Stop Calc.")
                    self.btn_stop_calc.setFixedSize(108, 27)
                    self.btn_stop_calc.setEnabled(False)
                    self.btn_stop_calc.setVisible(False)
                    self.btn_stop_calc.clicked.connect(lambda: self._request_calc_stop("button"))
                    cml.addWidget(self.btn_stop_calc)
                except Exception:
                    self.btn_stop_calc = None
                gb_layout.addWidget(self.calc_mode_controls, 0)
            except Exception:
                self.calc_mode_controls = None
                self.lbl_calc_mode = None
                self.toggle_calc_mode = None
                self.btn_stop_calc = None

            gb_layout.addLayout(sliders_layout)
            self.grain_body.setVisible(True)  # always expanded
            gl.addWidget(self.grain_body, 0)
        except Exception:
            self.grain_section = None
            self.grain_body = None

        # Left tabs: Off-line Targeting / On-line Alignment
        try:
            self.left_tabs = QTabWidget()
            self.left_tabs.setObjectName('leftWorkflowTabs')
            self.left_tabs.setTabPosition(QTabWidget.North)
            try:
                # Improve readability: slightly wider tabs and stronger selected/inactive contrast.
                self.left_tabs.tabBar().setExpanding(False)
                self.left_tabs.tabBar().setElideMode(Qt.ElideNone)
            except Exception:
                pass
            self.left_tabs.setStyleSheet(
                """
                QTabWidget#leftWorkflowTabs::pane {
                    border: 1px solid #b8b8b8;
                    top: -1px;
                    background: #f4f4f4;
                }
                QTabWidget#leftWorkflowTabs QTabBar::tab {
                    background: #dfdfdf;
                    color: #3a3a3a;
                    border: 1px solid #b8b8b8;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    min-width: 166px;
                    padding: 5px 10px;
                    margin-right: 2px;
                    font-weight: 600;
                }
                QTabWidget#leftWorkflowTabs QTabBar::tab:selected {
                    background: #ffffff;
                    color: #121212;
                    border-color: #8e8e8e;
                }
                QTabWidget#leftWorkflowTabs QTabBar::tab:!selected {
                    margin-top: 2px;
                }
                """
            )

            # Off-line Targeting tab: Auto-detect (Auxiliary)
            self.tab_offline = QWidget()
            offline_col = QVBoxLayout(self.tab_offline)
            offline_col.setContentsMargins(0, 5, 0, 0)
            offline_col.setSpacing(6)
            try:
                offline_overlay_row = QHBoxLayout()
                offline_overlay_row.setContentsMargins(0, 0, 0, 0)
                offline_overlay_row.setSpacing(6)
                self.lbl_overlay_source = QLabel("Overlay Points")
                try:
                    fov = self.lbl_overlay_source.font()
                    fov.setBold(True)
                    self.lbl_overlay_source.setFont(fov)
                except Exception:
                    pass
                offline_overlay_row.addWidget(self.lbl_overlay_source, 0)
                offline_overlay_row.addStretch(1)
                self.toggle_overlay_source = SegmentControl(["Left List", "Center List"], checked_index=0, btn_w=108, btn_h=27)
                try:
                    if str(getattr(self, 'overlay_point_source', 'left')) == 'center':
                        self.toggle_overlay_source.setCheckedIndex(1)
                except Exception:
                    pass
                try:
                    self.toggle_overlay_source.set_on_changed(lambda idx: self._on_toggle_overlay_source(int(idx)))
                except Exception:
                    pass
                offline_overlay_row.addWidget(self.toggle_overlay_source, 0)
                offline_col.addLayout(offline_overlay_row, 0)
            except Exception:
                self.lbl_overlay_source = None
                self.toggle_overlay_source = None
            if getattr(self, 'grain_section', None) is not None:
                offline_col.addWidget(self.grain_section, 0)
            try:
                offline_global_row = QHBoxLayout()
                offline_global_row.setContentsMargins(0, 0, 0, 0)
                offline_global_row.setSpacing(6)
                self.btn_add_all_grp_list = QPushButton("Add ALL Group to List")
                self.btn_add_all_grp_list.setFixedSize(316, 27)
                self.btn_add_all_grp_list.clicked.connect(self._add_all_groups_to_center_list)
                offline_global_row.addWidget(self.btn_add_all_grp_list, 0)
                self.toggle_show_all_groups = SegmentControl(["Show ALL", "Hide ALL"], checked_index=0, btn_w=94, btn_h=27)
                try:
                    self.toggle_show_all_groups.set_on_changed(lambda idx: self._on_toggle_show_all_groups(int(idx)))
                except Exception:
                    pass
                offline_global_row.addWidget(self.toggle_show_all_groups, 0)
                offline_global_row.addStretch(1)
                offline_col.addLayout(offline_global_row, 0)
            except Exception:
                self.btn_add_all_grp_list = None
                self.toggle_show_all_groups = None
            try:
                self.offline_group_scroll = QScrollArea()
                self.offline_group_scroll.setWidgetResizable(True)
                self.offline_group_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                self.offline_group_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.offline_group_scroll.setFrameShape(QScrollArea.NoFrame)
                self.offline_group_scroll.setMinimumHeight(220)
                try:
                    self.offline_group_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                except Exception:
                    pass

                self.offline_group_inner = QWidget()
                self.offline_group_layout = QHBoxLayout(self.offline_group_inner)
                self.offline_group_layout.setContentsMargins(4, 2, 4, 2)
                self.offline_group_layout.setSpacing(8)
                self.offline_group_scroll.setWidget(self.offline_group_inner)
                offline_col.addWidget(self.offline_group_scroll, 1)
            except Exception:
                self.offline_group_scroll = None
                self.offline_group_inner = None
                self.offline_group_layout = None
            self.left_tabs.addTab(self.tab_offline, 'Off-line Targeting')

            # On-line Alignment tab: fiducial controls + table
            self.tab_online = QWidget()
            online_col = QVBoxLayout(self.tab_online)
            online_col.setContentsMargins(0, 0, 0, 0)
            online_col.setSpacing(6)
            self.left_tabs.addTab(self.tab_online, 'On-line Alignment')
            try:
                self.left_tabs.tabBar().hide()
            except Exception:
                pass
            self.left_tabs.setCurrentIndex(1)
        except Exception:
            self.left_tabs = None
            self.tab_offline = None
            self.tab_online = None
            self.offline_group_scroll = None
            self.offline_group_inner = None
            self.offline_group_layout = None
            offline_col = None
            online_col = None

        # Centroid Extraction切替ボタンは左カラム直下に置き、Off-line表示中も常に見えるようにする
        try:
            start_extract_row = QHBoxLayout()
            start_extract_row.setContentsMargins(0, 5, 0, 0)
            start_extract_row.setSpacing(6)
            try:
                start_extract_row.addWidget(self.btn_start_centroid_extraction)
            except Exception:
                pass
            try:
                start_extract_row.addStretch(1)
            except Exception:
                pass
            left_col.addLayout(start_extract_row, 0)
        except Exception:
            pass

        # 左カラムの表の上に Add/Update/Clear ボタンを配置
        try:
            left_controls = QHBoxLayout()
            try:
                # Add a small top gap on On-line Alignment tab for better separation from tabs.
                if online_col is not None:
                    left_controls.setContentsMargins(0, 5, 0, 0)
                else:
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
            if online_col is not None:
                online_col.addLayout(left_controls, 0)
            else:
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
                hdr.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                hdr.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
            if online_col is not None:
                online_col.addWidget(hdr, 0)
            else:
                left_col.addWidget(hdr, 0)
        except Exception:
            self.table_ref_view_header = None

        if online_col is not None:
            online_col.addWidget(self.table_ref_view, 1)
        else:
            left_col.addWidget(self.table_ref_view, 1)

        if self.left_tabs is not None:
            left_col.addWidget(self.left_tabs, 1)
            try:
                self._set_centroid_extraction_mode(False)
            except Exception:
                pass
        # Wrap left column layout in a QWidget and cap its maximum width so it doesn't grow too wide
        self.left_container = QWidget()
        self.left_container.setLayout(left_col)
        try:
            # 初期幅 (動的に再計算される)
            self.left_container.setFixedWidth(500)
        except Exception:
            try:
                self.left_container.setMaximumWidth(500)
            except Exception:
                pass
        main_row.addWidget(self.left_container, 0)
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
                    try:
                        self.btn_filter.setVisible(False)
                    except Exception:
                        pass
                    center_btn_row.addStretch(1)
                    center_col.addLayout(center_btn_row, 0)

                    center_target_row = QHBoxLayout()
                    center_target_row.setContentsMargins(0, 0, 0, 0)
                    center_target_row.setSpacing(6)
                    center_target_row.addWidget(self.btn_update_target_uv, 0)
                    center_target_row.addWidget(self.btn_clear_target, 0)
                    center_target_row.addWidget(self.btn_clear_target_all, 0)
                    center_target_row.addStretch(1)
                    center_col.addLayout(center_target_row, 0)

                    center_add_row = QHBoxLayout()
                    center_add_row.setContentsMargins(0, 0, 0, 0)
                    center_add_row.setSpacing(6)
                    center_add_row.addWidget(self.btn_add_target, 0)
                    self.lbl_add_target_as = QLabel("as")
                    try:
                        self.lbl_add_target_as.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                    except Exception:
                        pass
                    center_add_row.addWidget(self.lbl_add_target_as, 0)
                    self.combo_add_target_group = QComboBox()
                    try:
                        self.combo_add_target_group.setFixedHeight(40)
                        self.combo_add_target_group.setMinimumWidth(120)
                    except Exception:
                        pass
                    center_add_row.addWidget(self.combo_add_target_group, 0)
                    center_add_row.addStretch(1)
                    center_col.addLayout(center_add_row, 0)
                    try:
                        self._refresh_target_group_combo()
                    except Exception:
                        pass

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
                    hdr_mid.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                    hdr_mid.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
        try:
            self.ui_footer.setVersion(getattr(self, '_app_version', None))
        except Exception:
            pass
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
        # Ref の Stage.* 入力保持用（内部容量は10）
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

        self._open_startup_image()

    # オーバーレイ表示モード（Original/Posterized）変更ハンドラ
    def _on_overlay_mode_changed(self, idx):
        if not bool(getattr(self, 'centroid_extraction_mode', False)):
            try:
                self.overlay_mode = 'Original'
                self.overlay_mix = 0
            except Exception:
                pass
            try:
                tog = getattr(self, 'overlay_mode_toggle', None)
                if tog is not None:
                    tog.setCheckedIndex(0)
            except Exception:
                pass
            self.schedule_update(force=True, recompute_centroids=False)
            return
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

    # Recalculation Trigger toggle handler (Auto/Manual)
    def _on_toggle_calc_mode(self, idx):
        # 計算中はモード切り替えではなく緊急停止として扱う
        try:
            if bool(getattr(self, '_calc_in_progress', False)):
                self._request_calc_stop("mode-toggle")
                return
        except Exception:
            pass

        try:
            if int(idx) == 1:
                self.calc_mode = 'manual'
            else:
                self.calc_mode = 'auto'
        except Exception:
            self.calc_mode = 'auto'

        try:
            self.auto_update_mode = (str(getattr(self, 'calc_mode', 'auto')) == 'auto')
        except Exception:
            self.auto_update_mode = True

        # Important behavior: mode toggle must not alter any calc parameter values.
        # Only trigger policy (Auto/Manual) is switched here.

        # Optional trace: confirm mode switch actually applied
        try:
            trace = bool(str(os.environ.get('PIXY_UPDATE_TRACE', '')).strip())
        except Exception:
            trace = False
        if trace:
            try:
                import sys
                print(
                    f"[TRACE][calc_mode] idx={int(idx)} calc_mode={str(getattr(self, 'calc_mode', 'auto'))} auto_update_mode={bool(getattr(self, 'auto_update_mode', False))}",
                    file=sys.stderr,
                )
            except Exception:
                pass

        # Swap label on the Manual segment
        try:
            if getattr(self, 'toggle_calc_mode', None) is not None:
                buttons = getattr(self.toggle_calc_mode, '_buttons', [None, None])
                btn0 = buttons[0] if len(buttons) > 0 else None
                btn1 = buttons[1] if len(buttons) > 1 else None
                if btn0 is not None:
                    btn0.setText('Auto')
                if btn1 is not None:
                    try:
                        btn1.setProperty('pixy_calc_in_progress', False)
                    except Exception:
                        pass
                    if str(getattr(self, 'calc_mode', 'auto')) == 'manual':
                        btn1.setText('ReCalculate')
                    else:
                        btn1.setText('Manual')
        except Exception:
            pass

        # Switching to manual should update visuals but avoid heavy recompute
        try:
            if str(getattr(self, 'calc_mode', 'auto')) == 'manual':
                self.schedule_update(force=True, recompute_centroids=False)
            else:
                self.schedule_update(force=True)
        except Exception:
            pass

    def _snapshot_calc_params(self):
        """Capture current grain-identification calc parameters from UI.

        This snapshot is used to keep separate parameter sets for Auto/Manual.
        """
        try:
            if hasattr(self, 'slider_num_groups') and getattr(self, 'slider_num_groups', None) is not None:
                levels = int(self.slider_num_groups.value())
            else:
                levels = int(getattr(self, 'slider_levels', None).value() if hasattr(self, 'slider_levels') else 2)
        except Exception:
            levels = 2

        try:
            trim_px = int(self.slider_trim.value()) if getattr(self, 'slider_trim', None) is not None else 0
        except Exception:
            trim_px = 0
        try:
            neck = int(getattr(self, 'slider_neck_sep', None).value() if hasattr(self, 'slider_neck_sep') else 0)
        except Exception:
            neck = 0
        try:
            shape = int(getattr(self, 'slider_shape_complex', None).value() if hasattr(self, 'slider_shape_complex') else 10)
        except Exception:
            shape = 10

        try:
            if getattr(self, 'area_hist', None) is not None:
                sel_min, sel_max = self.area_hist.selection()
            else:
                sel_min, sel_max = (None, None)
        except Exception:
            sel_min, sel_max = (None, None)

        return {
            'levels': int(levels),
            'trim_px': int(trim_px),
            'neck_separation': int(neck),
            'shape_complexity': int(shape),
            'area_sel_min': sel_min,
            'area_sel_max': sel_max,
        }

    def _apply_calc_params_snapshot(self, snap: dict):
        """Apply a previously captured calc-param snapshot to the UI widgets."""
        # Block signals to avoid triggering recomputation while restoring
        widgets = []
        try:
            for nm in ('slider_num_groups', 'slider_trim', 'slider_neck_sep', 'slider_shape_complex', 'slider_min_area'):
                w = getattr(self, nm, None)
                if w is not None:
                    widgets.append(w)
        except Exception:
            pass
        try:
            ah = getattr(self, 'area_hist', None)
            if ah is not None:
                widgets.append(ah)
        except Exception:
            ah = None

        try:
            for w in widgets:
                try:
                    w.blockSignals(True)
                except Exception:
                    pass

            # levels / groups
            try:
                lv = int(snap.get('levels', 2))
                if getattr(self, 'slider_num_groups', None) is not None:
                    self.slider_num_groups.setValue(lv)
                    try:
                        self.edit_num_groups.setText(str(int(lv)))
                    except Exception:
                        pass
            except Exception:
                pass

            # trim
            try:
                tv = int(snap.get('trim_px', 0))
                if getattr(self, 'slider_trim', None) is not None:
                    self.slider_trim.setValue(tv)
                    try:
                        self.edit_trim.setText(str(int(tv)))
                    except Exception:
                        pass
            except Exception:
                pass

            # neck
            try:
                nv = int(snap.get('neck_separation', 0))
                if getattr(self, 'slider_neck_sep', None) is not None:
                    self.slider_neck_sep.setValue(nv)
                    try:
                        self.edit_neck_sep.setText(str(int(nv)))
                    except Exception:
                        pass
            except Exception:
                pass

            # shape complexity
            try:
                sv = int(snap.get('shape_complexity', 10))
                if getattr(self, 'slider_shape_complex', None) is not None:
                    self.slider_shape_complex.setValue(sv)
                    try:
                        self.edit_shape_complex.setText(str(int(sv)))
                    except Exception:
                        pass
            except Exception:
                pass

            # histogram selection (min/max area)
            try:
                sel_min = snap.get('area_sel_min', None)
                sel_max = snap.get('area_sel_max', None)
                if ah is not None:
                    ah.set_selection(sel_min, sel_max)
            except Exception:
                pass
            # Keep hidden min-area widget in sync (best-effort)
            try:
                if sel_min is not None and getattr(self, 'slider_min_area', None) is not None:
                    v = int(round(float(sel_min)))
                    v = max(self.slider_min_area.minimum(), min(self.slider_min_area.maximum(), v))
                    self.slider_min_area.setValue(v)
                    try:
                        self.edit_min_area.setText(str(int(v)))
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            for w in widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

    def _on_manual_recalculate_clicked(self):
        # 計算中に同ボタンが押された場合は緊急停止として扱う
        try:
            if bool(getattr(self, '_calc_in_progress', False)):
                self._request_calc_stop("manual-segment")
                return
        except Exception:
            pass

        # Only act if Manual is active; otherwise ignore
        try:
            if str(getattr(self, 'calc_mode', 'auto')) != 'manual':
                return
        except Exception:
            return
        try:
            if bool(getattr(self, '_manual_recalc_in_progress', False)):
                return
        except Exception:
            pass

        try:
            self._manual_recalc_in_progress = True
        except Exception:
            pass

        try:
            QApplication.processEvents()
        except Exception:
            pass

        def _do_recalc():
            try:
                try:
                    self._manual_recompute_request = True
                except Exception:
                    pass
                try:
                    self.schedule_update(force=True, recompute_centroids=True)
                except Exception:
                    pass
            finally:
                try:
                    self._manual_recalc_in_progress = False
                except Exception:
                    pass

        try:
            QTimer.singleShot(0, _do_recalc)
        except Exception:
            _do_recalc()

    def _request_calc_stop(self, source="ui"):
        """Request cancellation of the currently running heavy calculation."""
        try:
            self._calc_stop_requested = True
        except Exception:
            pass
        try:
            if bool(getattr(self, '_calc_in_progress', False)) and getattr(self, 'ui_footer', None) is not None:
                self.ui_footer.showMessage(f"Stopping calculation... ({source})")
        except Exception:
            pass

    def _calc_stop_requested_callback(self):
        """Process UI events and return True when a stop is requested."""
        try:
            QApplication.processEvents()
        except Exception:
            pass
        try:
            return bool(getattr(self, '_calc_stop_requested', False))
        except Exception:
            return False

    def _set_calc_in_progress(self, running: bool):
        try:
            self._calc_in_progress = bool(running)
        except Exception:
            pass
        try:
            if not bool(running):
                self._calc_stop_requested = False
        except Exception:
            pass
        try:
            if getattr(self, 'btn_stop_calc', None) is not None:
                self.btn_stop_calc.setEnabled(bool(running))
        except Exception:
            pass

        # 計算中は現在採用中モードのセグメントを「Stop Calc.」として使う
        try:
            tcm = getattr(self, 'toggle_calc_mode', None)
            if tcm is not None:
                buttons = getattr(tcm, '_buttons', [None, None])
                btn_auto = buttons[0] if len(buttons) > 0 else None
                btn_manual = buttons[1] if len(buttons) > 1 else None
                mode_now = str(getattr(self, 'calc_mode', 'auto'))
                active_idx = 1 if mode_now == 'manual' else 0

                for i, b in enumerate((btn_auto, btn_manual)):
                    if b is None:
                        continue
                    if bool(running):
                        if i == active_idx:
                            b.setText('Stop Calc.')
                            b.setEnabled(True)
                            try:
                                b.setProperty('pixy_calc_in_progress', True)
                            except Exception:
                                pass
                        else:
                            b.setEnabled(False)
                            b.setText('Auto' if i == 0 else ('ReCalculate' if mode_now == 'manual' else 'Manual'))
                            try:
                                b.setProperty('pixy_calc_in_progress', False)
                            except Exception:
                                pass
                    else:
                        b.setEnabled(True)
                        b.setText('Auto' if i == 0 else ('ReCalculate' if mode_now == 'manual' else 'Manual'))
                        try:
                            b.setProperty('pixy_calc_in_progress', False)
                        except Exception:
                            pass
                    try:
                        b.style().unpolish(b)
                        b.style().polish(b)
                        b.update()
                    except Exception:
                        pass
        except Exception:
            pass

    def _compute_centroids_cancellable(self, params, poster):
        """Run centroid calculation with periodic cancellation checks."""
        trace_id = None
        t0 = monotonic()
        status = 'ok'
        try:
            self._calc_trace_seq = int(getattr(self, '_calc_trace_seq', 0) or 0) + 1
            trace_id = int(self._calc_trace_seq)
        except Exception:
            trace_id = None

        try:
            if hasattr(self, '_log_info'):
                mode = str(getattr(self, 'calc_mode', 'auto'))
                reason = str(getattr(self, '_calc_trace_last_reason', '') or '')
                self._log_info(
                    "CalcTrace START "
                    + f"id={trace_id} mode={mode} reason={reason} "
                    + f"levels={params.get('levels')} min={params.get('min_area')} max={params.get('max_area')} "
                    + f"trim={params.get('trim_px')} neck={params.get('neck_separation')} shape={params.get('shape_complexity')}"
                )
        except Exception:
            pass

        self._calc_stop_requested = False
        self._set_calc_in_progress(True)
        try:
            return self.centroid_processor.get_centroids(
                params,
                poster=poster,
                stop_requested=self._calc_stop_requested_callback,
                stop_check_interval_sec=1.0,
            )
        except CalculationCancelled:
            status = 'cancelled'
            raise
        except Exception:
            status = 'error'
            raise
        finally:
            self._set_calc_in_progress(False)
            try:
                dt = float(monotonic() - t0)
            except Exception:
                dt = 0.0
            try:
                if hasattr(self, '_log_info'):
                    self._log_info(
                        "CalcTrace END "
                        + f"id={trace_id} status={status} stop_requested={bool(getattr(self, '_calc_stop_requested', False))} "
                        + f"elapsed={dt:.3f}s"
                    )
            except Exception:
                pass

    # 境界線表示トグルハンドラ
    def _on_toggle_boundaries(self, checked):
        if not bool(getattr(self, 'centroid_extraction_mode', False)):
            checked = False
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
        # Boundary の表示/非表示を変えるだけなので再計算は不要
        self.schedule_update(force=True, recompute_centroids=False)
    def _on_toggle_coordinate(self, idx):
        center_full = None
        try:
            if getattr(self, 'proc_scroll', None) is not None:
                vp = self.proc_scroll.viewport()
                pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                center_full = self._display_to_full(pos_label)
        except Exception:
            center_full = None

        try:
            if int(idx) == 0:
                self.coordinate = 'Image'
            else:
                self.coordinate = 'Stage'
        except Exception:
            self.coordinate = 'Image'

        # Keep the canonical flag used by rendering logic in sync.
        try:
            self.view_orientation = str(self.coordinate)
        except Exception:
            self.view_orientation = 'Image'
        try:
            if getattr(self, 'btn_view_image', None) is not None and getattr(self, 'btn_view_stage', None) is not None:
                try:
                    self.btn_view_image.setChecked(self.coordinate == 'Image')
                    self.btn_view_stage.setChecked(self.coordinate == 'Stage')
                except Exception:
                    pass
        except Exception:
            pass
        # Update left_top_image based on coordinate selection
        try:
            if getattr(self, 'left_top_image', None) is not None:
                # Look for resource in frozen bundle first (sys._MEIPASS),
                # then fall back to the source directory.
                base_dirs = []
                try:
                    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                        base_dirs.append(sys._MEIPASS)
                except Exception:
                    pass
                base_dirs.append(os.path.dirname(__file__))

                if self.coordinate == 'Image':
                    names = ['PiXY_Pix.png', 'PiXY.png']
                else:
                    names = ['PiXY_XY.png', 'PiXY.png']
                pm = None
                for bd in base_dirs:
                    for name in names:
                        img_path = os.path.join(bd, name)
                        try:
                            candidate = QPixmap(img_path)
                            if candidate is not None and not candidate.isNull():
                                pm = candidate
                                break
                        except Exception:
                            continue
                    if pm is not None:
                        break
                try:
                    if pm is not None:
                        target_w, target_h = 450, 200
                        self._left_top_pix = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.left_top_image.setPixmap(self._left_top_pix)
                except Exception:
                    pass
        except Exception:
            pass
        # Mid toolbar visibility by Coordinate:
        # - Image: show Image Rotate + Normal/Flip
        # - Stage: show Scale/X/Y/Z
        try:
            is_image = (self.coordinate == 'Image')
            if getattr(self, '_mid_axis_controls', None) is not None:
                self._mid_axis_controls.setVisible(not is_image)
            if getattr(self, '_mid_rotate_controls', None) is not None:
                self._mid_rotate_controls.setVisible(is_image)
            if getattr(self, '_mid_flip_controls', None) is not None:
                self._mid_flip_controls.setVisible(is_image)
            if getattr(self, '_mid_stats_controls', None) is not None:
                # stats are now overlaid inside the image area; keep hidden here
                self._mid_stats_controls.setVisible(False)
        except Exception:
            pass
        # Keep objects around for backward compatibility but do not show Stage flip per latest request.
        try:
            if getattr(self, 'flip_toggle_stage', None) is not None:
                self.flip_toggle_stage.setVisible(False)
        except Exception:
            pass
        # Rotate slider is only meaningful in Image view
        try:
            if getattr(self, 'slider_img_rotate', None) is not None:
                self.slider_img_rotate.setEnabled(is_image)
        except Exception:
            pass
        # 更新をスケジュール（必要なら表示を更新するため）
        try:
            self._apply_proc_zoom()
            if center_full is not None:
                try:
                    self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
                except Exception:
                    pass
        except Exception:
            try:
                self.schedule_update(force=True)
            except Exception:
                pass

    def _on_stage_axis_changed(self, axis, idx):
        """Stage座標表示の符号（X/Y）を切り替える。0:+, 1:-"""
        center_full = None
        try:
            if str(getattr(self, 'view_orientation', 'Image')) == 'Stage' and getattr(self, 'proc_scroll', None) is not None:
                vp = self.proc_scroll.viewport()
                pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                center_full = self._display_to_full(pos_label)
        except Exception:
            center_full = None

        try:
            sign = 1 if int(idx) == 0 else -1
        except Exception:
            sign = 1
        try:
            if str(axis).lower().startswith('x'):
                self.stage_axis_x_sign = int(sign)
            else:
                self.stage_axis_y_sign = int(sign)
        except Exception:
            pass
        try:
            # display-only; repaint is enough
            self._apply_proc_zoom()
            if center_full is not None and str(getattr(self, 'view_orientation', 'Image')) == 'Stage':
                try:
                    self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
                except Exception:
                    pass
        except Exception:
            try:
                self.schedule_update(force=True)
            except Exception:
                pass

    def _on_manual_image_rotation_changed(self, val):
        center_full = None
        try:
            if str(getattr(self, 'view_orientation', 'Image')) == 'Image' and getattr(self, 'proc_scroll', None) is not None:
                vp = self.proc_scroll.viewport()
                pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                center_full = self._display_to_full(pos_label)
        except Exception:
            center_full = None

        try:
            # Snap to 10-degree increments even when clicking the slider groove.
            try:
                ival = int(val)
            except Exception:
                ival = 0
            snapped = int(round(ival / 10.0)) * 10
            snapped = max(-180, min(180, snapped))
            if snapped != ival:
                if not bool(getattr(self, '_in_img_rot_snap', False)):
                    try:
                        self._in_img_rot_snap = True
                        if getattr(self, 'slider_img_rotate', None) is not None:
                            self.slider_img_rotate.blockSignals(True)
                            self.slider_img_rotate.setValue(int(snapped))
                            self.slider_img_rotate.blockSignals(False)
                    except Exception:
                        try:
                            if getattr(self, 'slider_img_rotate', None) is not None:
                                self.slider_img_rotate.blockSignals(False)
                        except Exception:
                            pass
                    finally:
                        try:
                            self._in_img_rot_snap = False
                        except Exception:
                            pass
                ival = snapped
            self.manual_image_rotation_deg = int(ival)
            if getattr(self, 'lbl_rot_val', None) is not None:
                try:
                    self.lbl_rot_val.setText(f"{int(ival)}°")
                except Exception:
                    pass
        except Exception:
            self.manual_image_rotation_deg = 0
        # 表示のみ更新（Imageモード回転では表示中心を維持）
        try:
            self._apply_proc_zoom()
            if center_full is not None and str(getattr(self, 'view_orientation', 'Image')) == 'Image':
                try:
                    self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
                except Exception:
                    pass
        except Exception:
            try:
                self.schedule_update(force=True, recompute_centroids=False)
            except Exception:
                pass

    def _on_flip_changed(self, mode, idx):
        center_full = None
        try:
            view_orient = str(getattr(self, 'view_orientation', 'Image'))
            mode_s = str(mode)
            should_lock_center = (
                (mode_s == 'image' and view_orient == 'Image')
                or (mode_s == 'stage' and view_orient == 'Stage')
            )
            if should_lock_center and getattr(self, 'proc_scroll', None) is not None:
                vp = self.proc_scroll.viewport()
                pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                center_full = self._display_to_full(pos_label)
        except Exception:
            center_full = None

        try:
            if str(mode) == 'image':
                self.flip_mode_image = 'normal' if int(idx) == 0 else 'flip'
            else:
                self.flip_mode_stage = {0: 'auto', 1: 'normal', 2: 'flip'}.get(int(idx), 'auto')
        except Exception:
            pass
        # 表示のみ更新
        try:
            self._apply_proc_zoom()
            try:
                view_orient = str(getattr(self, 'view_orientation', 'Image'))
                mode_s = str(mode)
                should_lock_center = (
                    (mode_s == 'image' and view_orient == 'Image')
                    or (mode_s == 'stage' and view_orient == 'Stage')
                )
            except Exception:
                should_lock_center = False
            if center_full is not None and should_lock_center:
                try:
                    self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
                except Exception:
                    pass
        except Exception:
            pass

    # Grain Identification トグルハンドラ（Basic/Advanced）
    def _on_toggle_grain_ident(self, idx):
        self.grain_ident_mode = 'advanced'
        try:
            tog = getattr(self, 'toggle_grain_ident', None)
            if tog is not None:
                tog.setCheckedIndex(1)
                tog.setVisible(False)
        except Exception:
            pass
        # Basic/Advanced is UI visibility only: do not trigger heavy recomputation here.
        try:
            self._apply_grain_ident_visibility()
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _apply_grain_ident_visibility(self):
        mode = 'advanced'
        self.grain_ident_mode = 'advanced'
        show_basic = False
        try:
            tog = getattr(self, 'toggle_grain_ident', None)
            if tog is not None:
                tog.setCheckedIndex(1)
                tog.setVisible(False)
        except Exception:
            pass
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
        # Store key for diagnostics and for Auto/Manual parameter snapshots
        try:
            setattr(edit, '_pixy_key', str(name))
        except Exception:
            pass
        try:
            setattr(slider, '_pixy_key', str(name))
        except Exception:
            pass
        slider.valueChanged.connect(lambda v, e=edit, k=str(name): self._sync_from_slider(e, v, key=k))
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
            # Avoid duplicate schedule_update via slider callback; trigger once below.
            self.slider_num_groups.blockSignals(True)
            self.slider_num_groups.setValue(v)
            self.slider_num_groups.blockSignals(False)
        except Exception:
            try:
                self.slider_num_groups.blockSignals(False)
            except Exception:
                pass
            pass

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
        # Single recompute trigger for num-groups adjustment.
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
                try:
                    self._apply_proc_zoom()
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
                self._apply_proc_zoom()
            except Exception:
                pass
            try:
                self._refresh_transposed_views()
            except Exception:
                pass

    # スライダーから編集ボックスへ同期
    def _sync_from_slider(self, edit, val, key=None):
        try:
            edit.setText(str(val))
        except Exception:
            pass

        # Param-level trace: helps identify which control triggered updates.
        try:
            trace = bool(str(os.environ.get('PIXY_UPDATE_TRACE', '')).strip())
        except Exception:
            trace = False
        if trace:
            try:
                if key is None:
                    key = getattr(edit, '_pixy_key', None)
                self._last_update_reason = f"param:{key}" if key is not None else "param:?"
            except Exception:
                pass
            try:
                import sys
                mode = str(getattr(self, 'calc_mode', 'auto'))
                print(f"[TRACE][param] key={key} val={val} mode={mode}", file=sys.stderr)
            except Exception:
                pass

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

        try:
            trace = bool(str(os.environ.get('PIXY_UPDATE_TRACE', '')).strip())
        except Exception:
            trace = False
        if trace:
            try:
                key = getattr(slider, '_pixy_key', None)
            except Exception:
                key = None
            try:
                if key is None:
                    key = getattr(edit, '_pixy_key', None)
            except Exception:
                pass
            try:
                self._last_update_reason = f"edit:{key}" if key is not None else "edit:?"
            except Exception:
                pass
            try:
                import sys
                mode = str(getattr(self, 'calc_mode', 'auto'))
                print(f"[TRACE][param] key={key} val={v} mode={mode} (enter)", file=sys.stderr)
            except Exception:
                pass

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
        self._open_image_from_path(fname, reset_project_state=True)

    def _on_replace_image_clicked(self):
        """Replace only the base image while keeping current project data."""
        last_path = load_last_image_path()
        fname, _ = QFileDialog.getOpenFileName(self, "Replace Image", last_path, STR.FILE_FILTER)
        if not fname:
            return
        try:
            self._open_image_from_path(fname, reset_project_state=False, auto_detect=False)
        except Exception:
            pass

    def _open_startup_image(self):
        """起動時はデモ画像を自動読み込みし、見つからない場合のみ手動選択へフォールバック。"""
        try:
            base_dirs = []
            try:
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    base_dirs.append(sys._MEIPASS)
            except Exception:
                pass
            try:
                here = os.path.dirname(__file__)
                if here not in base_dirs:
                    base_dirs.append(here)
            except Exception:
                pass

            # Prefer explicit demo files first.
            demo_names = [
                'DemoBSE.png',
                'DemoBMP.bmp',
                'DemoBMP.png',
            ]
            demo_candidates = [os.path.join(root, name) for root in base_dirs for name in demo_names]

            demo_path = None
            for p in demo_candidates:
                try:
                    if p and os.path.isfile(p):
                        demo_path = p
                        break
                except Exception:
                    continue

            if demo_path:
                ok = self._open_image_from_path(demo_path, reset_project_state=True)
                if ok:
                    return

            # Fallback: no demo available (or load failed), let user pick an image.
            self.open_image()
        except Exception:
            try:
                self.open_image()
            except Exception:
                pass

    def _reset_project_coordinates(self):
        """Reset coordinate-related project state for New Project."""
        try:
            self._end_pick_mode(redraw=False)
        except Exception:
            pass
        try:
            self.ref_points = [None] * 10
            self.ref_selected_index = 0
            self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]
            self.excluded_ref_indices = set()
        except Exception:
            pass
        try:
            self.manual_targets = []
            self.centroids = []
            self._auto_centroids = []
            self.center_list_indices = []
            self.center_numeric_rows = []
            self._table_between_row_indices = []
            self.overlay_point_source = 'left'
            self.selected_index = None
            self.excluded_centroid_indices = set()
            self._explicit_excluded_centroid_indices = set()
            self._force_visible_centroid_indices = set()
            self._replace_target_source_index = None
            self._replace_target_source_group = None
            self._target_add_has_added = False
        except Exception:
            pass
        try:
            self.visible_ref_cols = 3
        except Exception:
            pass
        try:
            tog = getattr(self, 'toggle_overlay_source', None)
            if tog is not None:
                tog.setCheckedIndex(0)
        except Exception:
            pass
        try:
            self._set_centroid_extraction_mode(False)
        except Exception:
            pass

    def _show_open_image_prompt_message(self):
        try:
            msg = "Failed to load startup image. Please select an image using Open Image."
            self.ui_footer.showMessage(msg)
        except Exception:
            pass

    # 指定パスから画像を読み込み、処理画像を構築
    def _open_image_from_path(self, fname: str, show_startup_prompt_on_fail: bool = False, auto_detect: bool = False, reset_project_state: bool = False):
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
            try:
                self.image_path = str(fname)
            except Exception:
                self.image_path = ""
        except Exception as e:
            print("画像読み込みエラー:", e)
            self.img_full = None
            try:
                if bool(show_startup_prompt_on_fail):
                    self._show_open_image_prompt_message()
            except Exception:
                pass
            return False
        self._build_processing_image()
        try:
            if getattr(self, '_large_file_hint', False):
                self._dbg(f"Processing image with proc_target_width={self.proc_target_width}")
        except Exception:
            pass
        # New Project: discard previous coordinate state before next update.
        try:
            if bool(reset_project_state):
                self._reset_project_coordinates()
        except Exception:
            pass
        # New Project 読み込み時の初期条件:
        # - Number of Groups を 2 に固定
        # - Area Min/Max の初期化を「下位1/3・上位1/3」に設定
        try:
            if bool(reset_project_state):
                # Keep current manual/auto mode untouched; only reset detection defaults.
                lv = 2
                try:
                    if getattr(self, 'slider_num_groups', None) is not None:
                        self.slider_num_groups.blockSignals(True)
                        self.slider_num_groups.setValue(int(lv))
                        self.slider_num_groups.blockSignals(False)
                except Exception:
                    try:
                        if getattr(self, 'slider_num_groups', None) is not None:
                            self.slider_num_groups.blockSignals(False)
                    except Exception:
                        pass
                try:
                    if getattr(self, 'edit_num_groups', None) is not None:
                        self.edit_num_groups.setText(str(int(lv)))
                except Exception:
                    pass
                try:
                    self.levels_value = int(lv)
                    if getattr(self, 'edit_levels', None) is not None:
                        self.edit_levels.setText(str(int(lv)))
                    if getattr(self, 'slider_levels', None) is not None:
                        self.slider_levels.blockSignals(True)
                        self.slider_levels.setValue(max(self.slider_levels.minimum(), min(self.slider_levels.maximum(), int(lv))))
                        self.slider_levels.blockSignals(False)
                except Exception:
                    try:
                        if getattr(self, 'slider_levels', None) is not None:
                            self.slider_levels.blockSignals(False)
                    except Exception:
                        pass

                try:
                    ah = getattr(self, 'area_hist', None)
                    if ah is not None:
                        ah.clear()
                        ah._user_set_selection = False
                        ah._autoset_done = False
                except Exception:
                    pass
                self._area_init_tercile_pending = True
                self._pending_recompute_after_area_init = False
        except Exception:
            pass
        # 画像が変わったのでキャッシュ破棄
        self._cache = {"img_id": id(self.proc_img), "levels": None, "min_area": None, "trim_px": None, "poster": None, "centroids": None}
        # 次回更新時に画像中心へスクロール
        self._initial_center_done = False
        # New Project 時は最初から計算を実行（groups=2 で開始）
        do_initial_detect = bool(auto_detect) or bool(reset_project_state)
        self.schedule_update(force=True, recompute_centroids=bool(do_initial_detect))
        return True

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
    def schedule_update(self, force=False, recompute_centroids=True):
        """Schedule UI update.

        Args:
            force (bool): if True, run update immediately; otherwise start timer.
            recompute_centroids (bool): if True allow heavy centroid recomputation;
                if False, reuse cached centroids when possible.
        """
        # Optional debug: trace heavy recomputation triggers.
        # Enable by setting environment variable PIXY_UPDATE_TRACE=1.
        try:
            trace = bool(str(os.environ.get('PIXY_UPDATE_TRACE', '')).strip())
        except Exception:
            trace = False

        # Keep original caller intent for diagnostics.
        try:
            _requested_recompute = bool(recompute_centroids)
        except Exception:
            _requested_recompute = True

        try:
            _reason = str(getattr(self, '_last_update_reason', '') or '')
        except Exception:
            _reason = ''
        try:
            if bool(recompute_centroids):
                if _reason:
                    self._calc_trace_last_reason = str(_reason)
                elif bool(getattr(self, '_manual_recompute_request', False)):
                    self._calc_trace_last_reason = 'manual-recompute'
                elif bool(force):
                    self._calc_trace_last_reason = 'force-update'
                else:
                    self._calc_trace_last_reason = 'update'
        except Exception:
            pass
        try:
            # In manual mode, skip heavy recompute unless forced explicitly
            if str(getattr(self, 'calc_mode', 'auto')) == 'manual' and recompute_centroids:
                if not bool(getattr(self, '_manual_recompute_request', False)):
                    recompute_centroids = False
            try:
                self._manual_recompute_request = False
            except Exception:
                pass
            # store preference for the immediate update
            self._next_recompute_centroids = bool(recompute_centroids)
        except Exception:
            pass

        # If recompute was requested but gated off (manual mode), emit a short trace line.
        if trace and bool(_requested_recompute) and not bool(recompute_centroids):
            try:
                from time import perf_counter as _perf_counter
                now = float(_perf_counter())
                last = float(getattr(self, '_update_trace_last_t_gated', 0.0) or 0.0)
                if (now - last) >= 0.5:
                    self._update_trace_last_t_gated = now
                    try:
                        import sys
                        msg = f"[TRACE][schedule_update] force={bool(force)} requested=True effective=False (gated) calc_mode={str(getattr(self, 'calc_mode', 'auto'))}"
                        if _reason:
                            msg += f" reason={_reason}"
                        print(msg, file=sys.stderr)
                    except Exception:
                        pass
            except Exception:
                pass

        # Only print stack trace when heavy recompute is effectively allowed.
        if trace and bool(_requested_recompute) and bool(recompute_centroids):
            try:
                from time import perf_counter as _perf_counter
                now = float(_perf_counter())
                last = float(getattr(self, '_update_trace_last_t', 0.0) or 0.0)
                if (now - last) >= 0.5:
                    self._update_trace_last_t = now
                    try:
                        import traceback
                        st = ''.join(traceback.format_stack(limit=10))
                    except Exception:
                        st = ''
                    print(
                        (f"[TRACE][schedule_update] force={bool(force)} requested={bool(_requested_recompute)} effective={bool(recompute_centroids)} auto_update_mode={bool(getattr(self, 'auto_update_mode', False))}"
                         + (f" reason={_reason}" if _reason else "")
                         + f"\n{st}")
                    )
            except Exception:
                pass
        if force:
            self.update_timer.stop()
            self._update_image_actual(recompute_centroids=recompute_centroids)
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
                # Avoid duplicate schedule_update via slider callback.
                self.slider_min_area.blockSignals(True)
                self.slider_min_area.setValue(v)
                self.slider_min_area.blockSignals(False)
            except Exception:
                try:
                    self.slider_min_area.blockSignals(False)
                except Exception:
                    pass
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

            # New Project 初回のみ: Min/Max を Log 変換後レンジの 1/3, 2/3 で初期化。
            try:
                if bool(getattr(self, '_area_init_tercile_pending', False)):
                    log_mn = float(_np.log(float(mn)))
                    log_mx = float(_np.log(float(mx)))
                    log_rng = float(log_mx - log_mn)
                    sel_min = float(_np.exp(log_mn + (log_rng / 3.0)))
                    sel_max = float(_np.exp(log_mn + (2.0 * log_rng / 3.0)))
                    sel_min = max(float(mn), min(float(mx), sel_min))
                    sel_max = max(float(mn), min(float(mx), sel_max))
                    if sel_min > sel_max:
                        sel_min, sel_max = sel_max, sel_min
                    self.area_hist.set_selection(sel_min, sel_max)
                    self._area_init_tercile_pending = False
                    # 初期閾値が確定したので、次フレームでその値を使って再計算する。
                    self._pending_recompute_after_area_init = True
                    return
            except Exception:
                pass

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
                    # Apply configured default maximum cap to initial selection
                    try:
                        if sel_max is not None:
                            sel_max = min(sel_max, float(DEFAULT_MAX_GRAIN_AREA))
                    except Exception:
                        pass
                    # Enforce a minimum cap so autoset doesn't choose too-small grains
                    try:
                        if sel_min is not None:
                            sel_min = max(sel_min, float(DEFAULT_MIN_GRAIN_AREA))
                    except Exception:
                        pass
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

    def _update_image_actual(self, recompute_centroids=None):
        try:
            perf_enabled = bool(str(os.environ.get('PIXY_PERF', '')).strip())
        except Exception:
            perf_enabled = False
        t_update0 = None
        if perf_enabled:
            try:
                from time import perf_counter as _perf_counter
                t_update0 = float(_perf_counter())
            except Exception:
                t_update0 = None

        if recompute_centroids is None:
            try:
                recompute_centroids = bool(getattr(self, '_next_recompute_centroids', True))
            except Exception:
                recompute_centroids = True
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
            # Avoid copying the full image unless we actually need to mutate it.
            # Downstream boundary overlay uses astype(), which creates a new array.
            overlay_full = self.img_full
            centroids = []
            # Ensure poster is always defined before use in downstream rendering.
            poster = None
            poster_dt = None
            did_centroid_recompute = False
            manual_visibility_before_recompute = None
            try:
                if bool(recompute_centroids):
                    manual_visibility_before_recompute = self._capture_manual_target_visibility()
            except Exception:
                manual_visibility_before_recompute = None
            if self.centroid_processor:
                # 判定: 自動更新モードか手動モードかで重い処理の実行を切り替える
                try:
                    is_manual_mode = (str(getattr(self, 'calc_mode', 'auto')) == 'manual')
                except Exception:
                    is_manual_mode = False
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

                # Manualモードでは、トリガーされていない更新では poster も再計算しない。
                # (Autoでは随時計算、ManualではReCalculate等で recompute_centroids=True の時だけ再計算)
                if bool(is_manual_mode) and (not bool(recompute_centroids)):
                    need_poster_recalc = False

                if need_poster_recalc:
                    try:
                        if perf_enabled:
                            try:
                                from time import perf_counter as _perf_counter
                                t_p0 = float(_perf_counter())
                            except Exception:
                                t_p0 = None
                        poster = kmeans_posterize(self.proc_img, params["levels"])
                        if perf_enabled:
                            try:
                                if t_p0 is not None:
                                    poster_dt = float(_perf_counter()) - float(t_p0)
                            except Exception:
                                poster_dt = None
                        self._cache.update({
                            "img_id": id(self.proc_img),
                            "levels": params["levels"],
                            "neck_separation": params.get("neck_separation"),
                            "shape_complexity": params.get("shape_complexity"),
                            "poster": poster,
                        })
                    except Exception:
                        poster = cache_poster
                else:
                    poster = cache_poster

                # If caller requested skipping centroid recompute, still refresh poster (above)
                # but avoid heavy centroid work; reuse cache where possible.
                if not bool(recompute_centroids):
                    if cache_centroids is not None and cache_img_id == id(self.proc_img):
                        centroids = cache_centroids
                        areas_now = cache_areas
                        boundary_mask_now = self._cache.get("boundary_mask")
                    else:
                        centroids = getattr(self, 'centroids', []) or []
                        areas_now = getattr(self, '_cache', {}).get('areas')
                        boundary_mask_now = getattr(self, '_cache', {}).get('boundary_mask')
                # 自動モードでは通常通り重い処理を行う
                elif self.auto_update_mode:
                    if poster is None:
                        poster = kmeans_posterize(self.proc_img, params["levels"])
                    did_centroid_recompute = True
                    try:
                        centroids = self._compute_centroids_cancellable(params, poster)
                    except CalculationCancelled:
                        did_centroid_recompute = False
                        centroids = cache_centroids if cache_centroids is not None else (getattr(self, 'centroids', []) or [])
                        areas_now = cache_areas
                        boundary_mask_now = self._cache.get("boundary_mask")
                        try:
                            if getattr(self, 'ui_footer', None) is not None:
                                self.ui_footer.showMessage("Calculation stopped.")
                        except Exception:
                            pass
                    if did_centroid_recompute:
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                        try:
                            self._compute_group_header_colors(poster)
                        except Exception:
                            pass
                        # Also compute and cache full-image u,v coordinates for centroids
                        try:
                            # centroids are returned in proc coords (group, x_proc, y_proc)
                            img_base = getattr(self, '_img_base_size', None)
                            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                            uvs = []
                            if img_base is not None:
                                h_full0 = int(img_base[1])
                            else:
                                h_full0 = int(self.img_full.shape[0]) if getattr(self, 'img_full', None) is not None else None
                            for g, xp, yp in centroids:
                                x_full = float(xp) * spf
                                y_full = float(yp) * spf
                                u = int(round(x_full))
                                if h_full0 is not None:
                                    v = int(round((h_full0 - 1) - y_full))
                                else:
                                    v = int(round(-y_full))
                                uvs.append((g, u, v))
                            self._cache['centroids_full_uv'] = uvs
                        except Exception:
                            pass
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
                    # 手動モードで再計算を許可したケース（force/manual recompute）
                    if poster is None:
                        poster = kmeans_posterize(self.proc_img, params["levels"])
                    did_centroid_recompute = True
                    try:
                        centroids = self._compute_centroids_cancellable(params, poster)
                    except CalculationCancelled:
                        did_centroid_recompute = False
                        centroids = cache_centroids if cache_centroids is not None else (getattr(self, 'centroids', []) or [])
                        areas_now = cache_areas
                        boundary_mask_now = self._cache.get("boundary_mask")
                        try:
                            if getattr(self, 'ui_footer', None) is not None:
                                self.ui_footer.showMessage("Calculation stopped.")
                        except Exception:
                            pass
                    if did_centroid_recompute:
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        boundary_mask_now = getattr(self.centroid_processor, 'last_boundary_mask', None)
                        try:
                            self._compute_group_header_colors(poster)
                        except Exception:
                            pass
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
                poster_full = None
                poster_edges_full = None
                try:
                    if poster is not None:
                        # Manual モードで heavy recompute が許可されていない (スライダー移動中など) 場合、
                        # フル解像度への cv2.resize をスキップして UI を軽くする。
                        is_manual_mode = (str(getattr(self, 'calc_mode', 'auto')) == 'manual')
                        try:
                            trace = bool(str(os.environ.get('PIXY_UPDATE_TRACE', '')).strip())
                        except Exception:
                            trace = False

                        if is_manual_mode and (not bool(recompute_centroids)):
                            # Skip the expensive full resize; prefer using cached full-size poster if available
                            poster_full = self._cache.get('poster_full') if isinstance(self._cache, dict) else None
                            poster_edges_full = self._cache.get('poster_edges_full') if isinstance(self._cache, dict) else None
                            if trace:
                                try:
                                    import sys
                                    print(f"[TRACE][resize] skipped poster_full resize (manual mode, recompute=False) cached_full={'yes' if poster_full is not None else 'no'}", file=sys.stderr)
                                except Exception:
                                    pass
                        else:
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
                            # Cache the full-size poster for future skipped-resize renders
                            try:
                                if isinstance(self._cache, dict):
                                    self._cache['poster_full'] = poster_full
                                    self._cache['poster_edges_full'] = poster_edges_full
                            except Exception:
                                pass
                except Exception:
                    poster_full = None
                    poster_edges_full = None

                # Overlay selection by mode: Original / Posterized
                try:
                    overlay_mode = str(getattr(self, 'overlay_mode', 'Mixed')).lower()
                except Exception:
                    overlay_mode = 'original'
                # In Manual mode without recompute, prefer keeping the last rendered overlay
                # to avoid expensive full-frame composition during slider moves.
                reuse_last_overlay = False
                try:
                    is_manual_mode = (str(getattr(self, 'calc_mode', 'auto')) == 'manual')
                except Exception:
                    is_manual_mode = False
                try:
                    last_mode = str(getattr(self, '_last_overlay_mode', ''))
                    last_b = bool(getattr(self, '_last_show_boundaries', False))
                except Exception:
                    last_mode = ''
                    last_b = False
                try:
                    cur_b = bool(getattr(self, 'show_boundaries', True))
                except Exception:
                    cur_b = True
                try:
                    last_overlay = getattr(self, '_last_overlay_full', None)
                except Exception:
                    last_overlay = None
                if is_manual_mode and (not bool(recompute_centroids)) and last_overlay is not None:
                    if (str(overlay_mode) == str(last_mode)) and (bool(cur_b) == bool(last_b)):
                        reuse_last_overlay = True

                if reuse_last_overlay:
                    overlay_full = last_overlay
                elif overlay_mode == 'posterized':
                    if poster_full is not None:
                        # Use reference (no copy) to avoid full-frame memcpy.
                        overlay_full = poster_full
                        try:
                            self._last_overlay_full_poster = poster_full
                        except Exception:
                            pass
                    else:
                        # Reuse cached full poster if available, else reuse last poster overlay
                        prev = None
                        try:
                            if isinstance(self._cache, dict):
                                prev = self._cache.get('poster_full')
                        except Exception:
                            prev = None
                        if prev is None:
                            prev = getattr(self, '_last_overlay_full_poster', None)
                        overlay_full = prev if prev is not None else self.img_full
                else:
                    overlay_full = self.img_full

                try:
                    self._update_area_histogram(areas_now or [])
                except Exception:
                    pass

                # Optional perf log (only when PIXY_PERF=1)
                if perf_enabled and t_update0 is not None:
                    try:
                        from time import perf_counter as _perf_counter
                        dt_total = float(_perf_counter()) - float(t_update0)
                        # Avoid spamming: rate-limit to ~3 logs/sec and only show slow frames.
                        last = float(getattr(self, '_perf_last_log_t', 0.0) or 0.0)
                        nowt = float(_perf_counter())
                        if dt_total >= 0.15 and (nowt - last) >= 0.33:
                            self._perf_last_log_t = nowt
                            try:
                                import sys
                                mode = str(getattr(self, 'calc_mode', 'auto'))
                                auto = bool(getattr(self, 'auto_update_mode', False))
                                pdt = (f"{poster_dt:.3f}s" if isinstance(poster_dt, (int, float)) else "-")
                                print(
                                    "[PERF][update] "
                                    + f"total={dt_total:.3f}s mode={mode} auto_update={auto} "
                                    + f"recompute_centroids={bool(recompute_centroids)} did_centroid={did_centroid_recompute} "
                                    + f"poster_dt={pdt} params={{levels:{params.get('levels')}, trim:{params.get('trim_px')}, neck:{params.get('neck_separation')}, shape:{params.get('shape_complexity')}}}",
                                    file=sys.stderr,
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                # ポスタリゼーション境界に白線を描画（オプション）
                try:
                    # If we decided to reuse the last overlay (manual slider moves),
                    # skip boundary composition entirely to keep the UI responsive.
                    if (not bool(reuse_last_overlay)) and self.show_boundaries and poster_edges_full is not None:
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

                # Remember the last-rendered visual state so manual slider moves can reuse it.
                try:
                    self._last_overlay_mode = str(overlay_mode)
                except Exception:
                    pass
                try:
                    self._last_show_boundaries = bool(getattr(self, 'show_boundaries', True))
                except Exception:
                    pass
            # 右画像のベースサイズを保存（フル画像サイズ)
            self._img_base_size = (overlay_full.shape[1], overlay_full.shape[0])

            # 自動検出結果を保持（手動ターゲット合成前）
            try:
                self._auto_centroids = list(centroids or [])
            except Exception:
                self._auto_centroids = []

            # データ反映を先に行い、描画前に最新の点群を反映させる（灰色丸を即表示）
            # 手動ターゲットは常に自動重心へ加算（+α）する。
            self.centroids = self._compose_centroids_with_manual(self._auto_centroids)
            try:
                if bool(did_centroid_recompute):
                    self._reset_visibility_after_recompute(manual_visibility_before_recompute)
            except Exception:
                pass
            self._sanitize_excluded_indices()

            # 真ん中の転置表へ raw X/Y を即反映（populate/refresh が遅延しても見えるように）
            try:
                rv = getattr(self, 'table_between', None)
                if rv is not None:
                    from qt_compat.QtWidgets import QTableWidgetItem, QWidget, QHBoxLayout
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
                            try:
                                spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                            except Exception:
                                spf = 1.0
                            try:
                                h_full = int(self._img_base_size[1]) if self._img_base_size is not None else None
                            except Exception:
                                h_full = None
                            x_full = float(x) * spf
                            y_full = float(y) * spf
                            sx = str(int(round(x_full)))
                            if h_full is not None and h_full > 0:
                                sy = str(int(round((h_full - 1) - y_full)))
                            else:
                                sy = str(int(round(-y_full)))
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
            # New Project 初期閾値（下位/上位1/3）適用後の再計算を1回だけ実行
            try:
                if bool(getattr(self, '_pending_recompute_after_area_init', False)):
                    self._pending_recompute_after_area_init = False
                    try:
                        QTimer.singleShot(0, lambda: self.schedule_update(force=True, recompute_centroids=True))
                    except Exception:
                        self.schedule_update(force=True, recompute_centroids=True)
            except Exception:
                pass

    def _apply_proc_zoom(self):
        # Simplified rendering: do not use virtual canvas or PatchWorker.
        # Build a pixmap for the current overlay (or proc_img fallback) and then draw grid/rotation if needed.
        source_img = self._last_overlay_full if self._last_overlay_full is not None else self.proc_img
        if source_img is None:
            self.img_label_proc.clear()
            try:
                self._update_stage_info_overlay(None, getattr(self, 'view_orientation', 'Image'))
            except Exception:
                pass
            try:
                self._update_cursor_info_overlay(None)
            except Exception:
                pass
            return

        try:
            ov = self._get_overlay_render_payload()
            pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
                source_img,
                self.proc_zoom,
                self.view_padding,
                ov['centroids'],
                ov['selected_index'],
                self.ref_points,
                self.scale_proc_to_full,
                ref_selected_index=getattr(self, 'ref_selected_index', None),
                manual_indices=ov['manual_indices'],
                excluded_indices=ov['excluded_indices'],
                force_visible_indices=ov['force_visible_indices'],
                visible_groups=ov['visible_groups'],
                label_texts=ov.get('label_texts'),
                colors=None,
                debug_ref_coords=True,
                interp_mode=self.interp_mode,
                max_pixels=self._get_render_max_pixels(),
            )
        except Exception:
            pm = None
            off_x = off_y = 0
            new_w = new_h = 0

        if pm is None:
            self.img_label_proc.clear()
            try:
                self._update_stage_info_overlay(None, getattr(self, 'view_orientation', 'Image'))
            except Exception:
                pass
            try:
                self._update_cursor_info_overlay(None)
            except Exception:
                pass
            return

        # Helper: compute similarity transform (scale, rotation, translation)
        def _compute_similarity_transform(img_pts, stage_pts, prefer_reflect=None):
            """Estimate 2D similarity (scale/rotation/translation) from ref points.

            To handle possible left-right mirror between coordinate systems, we try both:
              - normal:  (x, y)
              - flipped: (-x, y)
            and pick the one with the smaller RMS residual.
            """
            try:
                import numpy as _np

                if len(img_pts) < 2 or len(stage_pts) < 2:
                    return None

                src0 = _np.asarray(img_pts, dtype=_np.float64)
                dst = _np.asarray(stage_pts, dtype=_np.float64)
                if src0.ndim != 2 or dst.ndim != 2 or src0.shape[1] != 2 or dst.shape[1] != 2:
                    return None
                if src0.shape[0] != dst.shape[0] or src0.shape[0] < 2:
                    return None

                def _fit(src, reflect_flag: bool):
                    # Umeyama-like similarity estimation
                    src_mean = src.mean(axis=0)
                    dst_mean = dst.mean(axis=0)
                    src_c = src - src_mean
                    dst_c = dst - dst_mean
                    # covariance
                    cov = (dst_c.T @ src_c) / src.shape[0]
                    U, S, Vt = _np.linalg.svd(cov)
                    R = U @ Vt
                    # ensure proper rotation (no accidental reflection)
                    if _np.linalg.det(R) < 0:
                        U[:, -1] *= -1
                        R = U @ Vt
                    # scale
                    var_src = (src_c ** 2).sum() / src.shape[0]
                    s = _np.sum(S) / var_src if var_src > 1e-12 else 1.0
                    t = dst_mean - s * (R @ src_mean)
                    # residual (RMS)
                    pred = (s * (src @ R.T)) + t
                    err = dst - pred
                    rms = float(_np.sqrt(_np.mean(_np.sum(err * err, axis=1)))) if err.size else float('inf')
                    # rotation angle (radians)
                    angle = float(_np.arctan2(R[1, 0], R[0, 0]))
                    return {
                        's': float(s),
                        'R': R,
                        't': t,
                        'angle_rad': float(angle),
                        'angle_deg': float(_np.degrees(angle)),
                        'reflect': bool(reflect_flag),
                        'rms': rms,
                    }

                # Decide reflection handling based on Stage flip mode (auto/normal/flip)
                # NOTE: This is the transform used for Stage alignment, so it must NOT use image flip mode.
                try:
                    mode = str(getattr(self, 'flip_mode_stage', 'auto') or 'auto').lower().strip()
                except Exception:
                    mode = 'auto'
                if mode not in ('auto', 'normal', 'flip'):
                    mode = 'auto'

                # Precompute both candidates
                normal = _fit(src0, reflect_flag=False)
                src1 = src0.copy()
                src1[:, 0] *= -1.0
                flipped = _fit(src1, reflect_flag=True)

                # Sticky selection in auto mode: if a previous reflect choice exists,
                # do not flip reflect unless it is meaningfully better.
                sticky_ratio = 0.02  # 2% RMS improvement required to switch away from preferred

                if mode == 'normal':
                    best = normal
                elif mode == 'flip':
                    best = flipped
                else:
                    # auto
                    pref = None
                    try:
                        if prefer_reflect is not None:
                            pref = bool(prefer_reflect)
                    except Exception:
                        pref = None

                    rn = float(normal.get('rms', float('inf')))
                    rf = float(flipped.get('rms', float('inf')))

                    if pref is True:
                        # keep flipped unless normal is clearly better
                        if rn < rf * (1.0 - sticky_ratio):
                            best = normal
                        else:
                            best = flipped
                    elif pref is False:
                        # keep normal unless flipped is clearly better
                        if rf < rn * (1.0 - sticky_ratio):
                            best = flipped
                        else:
                            best = normal
                    else:
                        # no preference
                        best = flipped if (rf < rn) else normal

                return best
            except Exception:
                return None

        # Helper: build stage transform info from available reference points
        def _get_stage_alignment_info():
            try:
                pts_img = []
                pts_stage = []
                pts_stage_xyz = []
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
                    # Fit in the same image coordinate convention as the tables/grid labels:
                    # u = x_full, v = (h_full - 1) - y_full  (origin bottom-left, +v upward)
                    try:
                        h_full0 = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else None
                    except Exception:
                        h_full0 = None
                    u_full = float(x_full)
                    if h_full0 is not None and h_full0 > 0:
                        v_full = float((h_full0 - 1) - float(y_full))
                    else:
                        v_full = float(-float(y_full))
                    pts_img.append((u_full, v_full))
                    pts_stage.append((sx_f, sy_f))

                    # Optional: z observation for pitch/roll estimation
                    try:
                        sz = ro_i.get('z', '')
                        if sz is not None and str(sz).strip() != '':
                            sz_f = float(str(sz).replace(',', '').strip())
                            pts_stage_xyz.append((sx_f, sy_f, sz_f))
                    except Exception:
                        pass
                self._dbg(f"Stage alignment: found {len(pts_img)} valid point pairs")
                try:
                    # Avoid high-frequency file I/O during redraw loops.
                    _pairs = int(len(pts_img))
                    _insuf = bool(_pairs < 2)
                    _state = (_pairs, _insuf)
                    if getattr(self, '_stage_align_log_state', None) != _state:
                        self._stage_align_log_state = _state
                        if hasattr(self, '_log_info'):
                            self._log_info(f"Stage alignment: pairs={_pairs}")
                except Exception:
                    pass
                if len(pts_img) < 2:
                    self._dbg(f"Insufficient ref points for stage transform (need ≥2, have {len(pts_img)})")
                    try:
                        _pairs = int(len(pts_img))
                        _state_ins = ("insufficient", _pairs)
                        if getattr(self, '_stage_align_ins_log_state', None) != _state_ins:
                            self._stage_align_ins_log_state = _state_ins
                            if hasattr(self, '_log_info'):
                                self._log_info(f"Stage alignment: insufficient (need>=2 have={_pairs})")
                    except Exception:
                        pass
                    return None
                try:
                    prev_info = getattr(self, '_last_stage_info', None)
                    pref_reflect = None if prev_info is None else prev_info.get('reflect', None)
                except Exception:
                    pref_reflect = None
                result = _compute_similarity_transform(pts_img, pts_stage, prefer_reflect=pref_reflect)
                if result:
                    self._dbg(f"Transform computed: angle={result.get('angle_deg', 0):.2f}deg, scale={result.get('s', 1):.3f}")
                    try:
                        _angle = round(float(result.get('angle_deg', 0.0)), 4)
                        _scale = round(float(result.get('s', 1.0)), 6)
                        _reflect = bool(result.get('reflect', False))
                        _rms = round(float(result.get('rms', 0.0)), 6)
                        _tf_state = (_angle, _scale, _reflect, _rms)
                        if getattr(self, '_stage_transform_log_state', None) != _tf_state:
                            self._stage_transform_log_state = _tf_state
                            if hasattr(self, '_log_info'):
                                self._log_info(
                                    f"Stage transform: angle_deg={_angle:.4f} "
                                    f"scale={_scale:.6g} reflect={_reflect} rms={_rms:.6g}"
                                )
                    except Exception:
                        pass

                # Display magnification should be based on proc (u,v) -> stage (X,Y)
                # while internal transform remains based on full coords for rendering consistency.
                try:
                    if result is not None:
                        spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                        result['s_proc'] = float(result.get('s', 1.0)) * spf
                except Exception:
                    pass

                # Estimate pitch/roll from z observations: fit plane z = a*x + b*y + c
                try:
                    import numpy as _np
                    if len(pts_stage_xyz) >= 3:
                        A = _np.asarray([[p[0], p[1], 1.0] for p in pts_stage_xyz], dtype=_np.float64)
                        z = _np.asarray([p[2] for p in pts_stage_xyz], dtype=_np.float64)
                        coeff, *_ = _np.linalg.lstsq(A, z, rcond=None)
                        a = float(coeff[0])
                        b = float(coeff[1])
                        # Convention: slope along Y -> pitch, slope along X -> roll
                        import math
                        pitch_deg = float(_np.degrees(math.atan(b)))
                        roll_deg = float(_np.degrees(math.atan(a)))
                        if result is None:
                            result = {}
                        result['pitch_deg'] = pitch_deg
                        result['roll_deg'] = roll_deg
                        result['z_plane'] = (a, b, float(coeff[2]))
                except Exception:
                    pass
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

        def _build_display_mapping(mode: str, draw_w: float, draw_h: float, pad: float, z: float, qt_transform=None):
            """QPixmap.trueMatrix() を使い、transformed() と完全一致する変換行列を取得する。

            Qt の transformed() は内部で toAlignedRect() により整数丸めしたシフトを使う。
            mapRect().topLeft() の浮動小数とは最大1px ズレるため、trueMatrix で統一する。
            """
            try:
                from qt_compat.QtCore import QRectF
                from qt_compat.QtGui import QTransform, QPixmap as _QPixmap

                if qt_transform is None:
                    qt_transform = QTransform()

                # trueMatrix incorporates the integer-pixel shift that Qt's
                # transformed() applies internally.  Using it guarantees that
                # our forward / inverse mapping matches the actual pixmap layout.
                true_qt = _QPixmap.trueMatrix(qt_transform, int(round(draw_w)), int(round(draw_h)))

                inv, invertible = true_qt.inverted()
                if not invertible:
                    inv = QTransform()

                # Compute rotated bounding box size for informational purposes
                rect = QRectF(0.0, 0.0, float(draw_w), float(draw_h))
                mapped = true_qt.mapRect(rect)
                rot_w = float(abs(mapped.width()))
                rot_h = float(abs(mapped.height()))

                return {
                    'type': 'qt',
                    'mode': str(mode),
                    'pad': float(pad),
                    'z': float(z),
                    'draw_w': float(draw_w),
                    'draw_h': float(draw_h),
                    'qt': true_qt,
                    'qt_inv': inv,
                    'shift_x': 0.0,
                    'shift_y': 0.0,
                    'rot_w': float(rot_w),
                    'rot_h': float(rot_h),
                }
            except Exception:
                return None

        # Build pixmap from the full overlay and display it directly
        ov = self._get_overlay_render_payload()
        pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
            self._last_overlay_full,
            self.proc_zoom,
            self.view_padding,
            ov['centroids'],
            ov['selected_index'],
            self.ref_points,
            self.scale_proc_to_full,
            ref_selected_index=getattr(self, 'ref_selected_index', None),
            manual_indices=ov['manual_indices'],
            excluded_indices=ov['excluded_indices'],
            force_visible_indices=ov['force_visible_indices'],
            visible_groups=ov['visible_groups'],
            label_texts=ov.get('label_texts'),
            colors=None,
            interp_mode=self.interp_mode,
            max_pixels=self._get_render_max_pixels(),
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

            # reset display mapping each render; branches below will set it as needed
            self._display_mapping = None

            # Update status labels (Stage: Magnification/Rotation/Shift X/Shift Y/Pitch/Roll)
            try:
                if getattr(self, 'lbl_scale_val', None) is not None:
                    if info is not None:
                        try:
                            self.lbl_scale_val.setText(f"{float(info.get('s_proc', info.get('s', 1.0))):.4g}")
                        except Exception:
                            self.lbl_scale_val.setText(f"{info.get('s', 1.0):.4g}")
                    else:
                        self.lbl_scale_val.setText("-")
                if getattr(self, 'lbl_tx_val', None) is not None:
                    if info is not None and info.get('t', None) is not None:
                        try:
                            tx = float(info['t'][0])
                            self.lbl_tx_val.setText(f"{tx:.3g}")
                        except Exception:
                            self.lbl_tx_val.setText("-")
                    else:
                        self.lbl_tx_val.setText("-")
                if getattr(self, 'lbl_ty_val', None) is not None:
                    if info is not None and info.get('t', None) is not None:
                        try:
                            ty = float(info['t'][1])
                            self.lbl_ty_val.setText(f"{ty:.3g}")
                        except Exception:
                            self.lbl_ty_val.setText("-")
                    else:
                        self.lbl_ty_val.setText("-")
                if getattr(self, 'lbl_angle_val', None) is not None:
                    if info is not None:
                        try:
                            self.lbl_angle_val.setText(f"{info.get('angle_deg', 0.0):.2f}")
                        except Exception:
                            self.lbl_angle_val.setText("-")
                    else:
                        self.lbl_angle_val.setText("-")
                if getattr(self, 'lbl_flip_val', None) is not None:
                    try:
                        view_orient0 = getattr(self, 'view_orientation', 'Image')
                    except Exception:
                        view_orient0 = 'Image'
                    if info is not None and view_orient0 == 'Stage':
                        rf = bool(info.get('reflect', False))
                        try:
                            mode0 = str(getattr(self, 'flip_mode_stage', 'auto') or 'auto').lower().strip()
                        except Exception:
                            mode0 = 'auto'
                        if mode0 == 'normal':
                            rf = False
                        elif mode0 == 'flip':
                            rf = True
                        self.lbl_flip_val.setText("On" if rf else "Off")
                    else:
                        self.lbl_flip_val.setText("-")
                if getattr(self, 'lbl_pitch_val', None) is not None:
                    if info is not None and info.get('pitch_deg', None) is not None:
                        try:
                            self.lbl_pitch_val.setText(f"{float(info.get('pitch_deg', 0.0)):.3g}")
                        except Exception:
                            self.lbl_pitch_val.setText("-")
                    else:
                        self.lbl_pitch_val.setText("-")
                if getattr(self, 'lbl_roll_val', None) is not None:
                    if info is not None and info.get('roll_deg', None) is not None:
                        try:
                            self.lbl_roll_val.setText(f"{float(info.get('roll_deg', 0.0)):.3g}")
                        except Exception:
                            self.lbl_roll_val.setText("-")
                    else:
                        self.lbl_roll_val.setText("-")
            except Exception:
                pass

            pm_to_show = pm
            try:
                # Apply rotation when Stage is selected, grid for both modes
                view_orient = getattr(self, 'view_orientation', 'Image')
                reflect_flag = False
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
                    # Reflect mode: use the fitted value (表の計算パラメータ)
                    reflect_fit = bool(info.get('reflect', False))
                    # NOTE: Do NOT override with flip_mode_stage. flip_mode_stage is for override purposes,
                    # but here we use the computed reflect value. Right/Top toggles will be applied later.

                    # Compute scale/image properties
                    s_val = float(info.get('s_proc', info.get('s', 1.0)))
                    
                    # === COORDINATE FLIP (from fitted model) ===
                    # This is applied in the image coordinate system BEFORE rotation alignment
                    sx_reflect = -1 if bool(reflect_fit) else 1
                    
                    # === DISPLAY FLIP (from user Right/Top settings) ===
                    # These are purely display orientation (screen axes).
                    # `stage_axis_x_sign`: +1 => screen-right is +X,  -1 => screen-right is -X
                    # `stage_axis_y_sign`: +1 => screen-up    is +Y,  -1 => screen-up    is -Y
                    # Note: our fit uses (u,v) where +v is upward; after converting back to pixmap (y-down),
                    # stage +Y already points upward on screen when stage_axis_y_sign == +1.
                    try:
                        sx_sign = int(getattr(self, 'stage_axis_x_sign', 1) or 1)
                    except Exception:
                        sx_sign = 1
                    try:
                        sy_sign = int(getattr(self, 'stage_axis_y_sign', 1) or 1)
                    except Exception:
                        sy_sign = 1
                    sx_axis = 1 if sx_sign > 0 else -1
                    sy_axis = 1 if sy_sign > 0 else -1

                    # Image rotation angle for QTransform:
                    # Fit is performed in (u,v) where +v is upward, but QPixmap coords have +y downward.
                    # Therefore we invert the sign when rotating the pixmap.
                    angle_to_rotate = -float(angle)  # angle_deg from info (table convention)
                    
                    # === Build QTransform (single chain) ===
                    # Desired application on points:
                    #   p -> T(-c) -> S_reflect (coordinate flip) -> R(angle) -> S_axis (display flip) -> T(c)
                    # With Qt's right-multiplication, append in reverse:
                    #   T(c) -> S_axis -> R -> S_reflect -> T(-c)
                    transform.translate(cx, cy)
                    transform.scale(float(sx_axis), float(sy_axis))  # Display flip (Right/Top)
                    transform.rotate(float(angle_to_rotate))         # Rotation (from fitted model)
                    if sx_reflect != 1:
                        transform.scale(float(sx_reflect), 1.0)      # Coordinate flip (from fitted model)
                    transform.translate(-cx, -cy)
                    rotated = img_region.transformed(transform, Qt.SmoothTransformation)
                    try:
                        z = float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom))))
                        dm = _build_display_mapping('Stage', float(draw_w), float(draw_h), float(pad), z, qt_transform=transform)
                        if dm is not None:
                            self._display_mapping = dm
                            try:
                                self._display_img_size = (int(rotated.width()), int(rotated.height()))
                                self._display_offset = (int(pad), int(pad))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # compose new canvas sized to fit the rotated pixmap (avoid clipping)
                    rot_w = rotated.width()
                    rot_h = rotated.height()
                    pm2 = QPixmap(rot_w + 2 * pad, rot_h + 2 * pad)
                    pm2.fill(QColor(30, 30, 30))
                    p = QPainter(pm2)
                    p.drawPixmap(pad, pad, rotated)

                    # 参照点は build_zoomed_canvas 側で描画され、画像回転と一緒に回転されるため、ここでの再描画は不要

                    # Draw stage grid lines on rotated image (simple orthogonal grid)
                    try:
                        import numpy as _np
                        import math
                        
                        # Draw grid based on the *actual* fitted stage<->image transform.
                        # This keeps axes consistent with StageInput: X is horizontal, Y is vertical.
                        display_scale = getattr(self, '_display_scale', None)
                        if display_scale is None:
                            display_scale = float(self.proc_zoom)

                        s_val = float(info.get('s', 1.0))
                        px_per_stage = float(display_scale) / max(1e-12, s_val)

                        # Choose nice spacing in stage units (1-2-5 series).
                        # Keep display spacing around ~120 px while clamping to [50,220].
                        candidates = []
                        for e in range(-6, 9):
                            for b in (1, 2, 5):
                                candidates.append(float(b) * (10.0 ** e))
                        target_px = 120.0
                        spacing = candidates[0]
                        best_score = float('inf')
                        for c in candidates:
                            s_px = c * px_per_stage
                            if s_px <= 0:
                                continue
                            penalty = 0.0
                            if s_px < 50.0:
                                penalty += (50.0 - s_px) * 10.0
                            elif s_px > 220.0:
                                penalty += (s_px - 220.0) * 10.0
                            score = abs(s_px - target_px) + penalty
                            if score < best_score:
                                best_score = score
                                spacing = c

                        # Bounds: map image corners into stage coords using the fitted forward model.
                        w_full = int(self._img_base_size[0]) if getattr(self, '_img_base_size', None) else draw_w
                        h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else draw_h
                        # Corners in full-image pixel coords are (x,y) with +y downward.
                        # Convert them to the fitted (u,v) convention: v = (h_full-1) - y.
                        corners = [(0.0, 0.0), (float(w_full - 1), 0.0), (float(w_full - 1), float(h_full - 1)), (0.0, float(h_full - 1))]
                        R = _np.asarray(info.get('R'), dtype=_np.float64)
                        t = _np.asarray(info.get('t'), dtype=_np.float64)
                        reflect_fit = bool(info.get('reflect', False))
                        stage_corners = []
                        for cx0, cy0 in corners:
                            x0 = float(cx0)
                            y0 = float(cy0)
                            u0 = x0
                            v0 = float((h_full - 1) - y0) if (h_full is not None and h_full > 0) else -y0
                            if reflect_fit:
                                u0 = -u0
                            p_uv = _np.asarray([u0, v0], dtype=_np.float64)
                            stage_corners.append((s_val * (R @ p_uv)) + t)
                        xs = [float(p0[0]) for p0 in stage_corners]
                        ys = [float(p0[1]) for p0 in stage_corners]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)

                        start_x = math.floor(xmin / spacing) * spacing
                        end_x = math.ceil(xmax / spacing) * spacing
                        start_y = math.floor(ymin / spacing) * spacing
                        end_y = math.ceil(ymax / spacing) * spacing

                        pen = QPen(QColor(200, 200, 200, 140))
                        pen.setWidth(1)
                        p.setPen(pen)
                        font = p.font(); font.setPointSize(9); p.setFont(font)

                        # stage -> image(full) -> display mapping, consistent with current QTransform
                        def _stage_to_disp_xy(sx, sy):
                            try:
                                stage = _np.asarray([float(sx), float(sy)], dtype=_np.float64)
                                # inverse of stage ~= s*R*[x',y] + t, where (x',y) is the fitted source coords.
                                # Convert back to original image coords: (x,y) = F*(x',y) with F=diag(-1,1) when reflect.
                                uv = (1.0 / max(1e-12, s_val)) * (R.T @ (stage - t))
                                if reflect_fit:
                                    uv[0] = -uv[0]
                                # Convert back to full-image coords (x,y with +y downward)
                                x_full = float(uv[0])
                                try:
                                    h_full0 = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else None
                                except Exception:
                                    h_full0 = None
                                if h_full0 is not None and h_full0 > 0:
                                    y_full = float((h_full0 - 1) - float(uv[1]))
                                else:
                                    y_full = -float(uv[1])
                                dxy = self._full_to_display(x_full, y_full)
                                return None if dxy is None else (int(round(dxy[0])), int(round(dxy[1])))
                            except Exception:
                                return None

                        def _fmt_grid_label(v):
                            try:
                                fv = float(v)
                                if abs(fv - round(fv)) < 1e-9:
                                    return f"{int(round(fv))}"
                                afv = abs(fv)
                                if afv >= 1000.0 or (afv > 0.0 and afv < 1e-2):
                                    return f"{fv:.3g}"
                                return f"{fv:.4f}".rstrip('0').rstrip('.')
                            except Exception:
                                return f"{v}"

                        # Vertical grid lines: X = const, Y varies
                        x = start_x
                        while x <= end_x + 1e-9:
                            p1 = _stage_to_disp_xy(x, ymin)
                            p2 = _stage_to_disp_xy(x, ymax)
                            if p1 is not None and p2 is not None:
                                p.drawLine(p1[0], p1[1], p2[0], p2[1])
                                try:
                                    lbl = _fmt_grid_label(x)
                                    p.drawText(int(p1[0]) + 4, int(min(p1[1], p2[1])) + 14, lbl)
                                except Exception:
                                    pass
                            x += spacing

                        # Horizontal grid lines: Y = const, X varies
                        y = start_y
                        while y <= end_y + 1e-9:
                            p1 = _stage_to_disp_xy(xmin, y)
                            p2 = _stage_to_disp_xy(xmax, y)
                            if p1 is not None and p2 is not None:
                                p.drawLine(p1[0], p1[1], p2[0], p2[1])
                                try:
                                    lbl = _fmt_grid_label(y)
                                    p.drawText(int(min(p1[0], p2[0])) + 4, int(p1[1]) - 4, lbl)
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
                        from qt_compat.QtGui import QPixmap, QPainter, QPen, QColor, QTransform
                        import math
                        
                        # まずImageビュー用の回転／反転を適用
                        pad = int(self.view_padding)
                        draw_w, draw_h = (new_w, new_h)
                        try:
                            img_region = pm.copy(pad, pad, draw_w, draw_h)
                        except Exception:
                            img_region = pm
                        transform = QTransform()
                        cx = img_region.width() / 2.0
                        cy = img_region.height() / 2.0
                        # Image用Flip
                        try:
                            if getattr(self, 'flip_mode_image', 'normal') == 'flip':
                                transform.translate(cx, cy)
                                transform.scale(-1, 1)
                                transform.translate(-cx, -cy)
                        except Exception:
                            pass
                        # Image用回転（手動）
                        try:
                            angle_img = float(getattr(self, 'manual_image_rotation_deg', 0.0))
                        except Exception:
                            angle_img = 0.0
                        transform.translate(cx, cy)
                        transform.rotate(-angle_img)
                        transform.translate(-cx, -cy)
                        rotated_img = img_region.transformed(transform, Qt.SmoothTransformation)

                        try:
                            z = float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom))))
                            flip_flag = bool(getattr(self, 'flip_mode_image', 'normal') == 'flip')
                            dm = _build_display_mapping('Image', float(draw_w), float(draw_h), float(pad), z, qt_transform=transform)
                            if dm is not None:
                                self._display_mapping = dm
                                try:
                                    self._display_img_size = (int(rotated_img.width()), int(rotated_img.height()))
                                    self._display_offset = (int(pad), int(pad))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # 回転後キャンバス作成
                        pm2 = QPixmap(rotated_img.width() + 2 * pad, rotated_img.height() + 2 * pad)
                        pm2.fill(QColor(30, 30, 30))
                        p = QPainter(pm2)
                        p.drawPixmap(pad, pad, rotated_img)

                        pad = int(self.view_padding)
                        
                        # For Image mode, use pixel coordinates
                        display_scale = getattr(self, '_display_scale', None)
                        if display_scale is None or display_scale <= 0:
                            try:
                                display_scale = float(self.proc_zoom)
                            except Exception:
                                display_scale = 1.0
                        display_scale = max(1e-4, float(display_scale))

                        # Choose spacing from 1-2-5 series in image-pixel units.
                        # Keep about ~8 lines across visible width while preserving "nice" values.
                        visible_w = max(20.0, pm2.width() - 2 * pad)
                        target_lines = 8.0
                        spacing_display_target = max(30.0, min(180.0, visible_w / target_lines))
                        pixel_spacing_target = spacing_display_target / display_scale

                        # Build candidate spacings from powers of 10 with multipliers 1,2,5.
                        nice_spacings = []
                        for e in range(-2, 8):
                            base = 10.0 ** e
                            for m in (1.0, 2.0, 5.0):
                                s = m * base
                                if s >= 1.0:
                                    nice_spacings.append(s)
                        pixel_spacing = 10.0
                        best_diff = float('inf')
                        for s in nice_spacings:
                            diff = abs(s - pixel_spacing_target)
                            if diff < best_diff:
                                best_diff = diff
                                pixel_spacing = s
                        
                        w_full = int(self._img_base_size[0]) if getattr(self, '_img_base_size', None) else new_w
                        h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else new_h
                        
                        pen = QPen(QColor(200, 200, 200, 140))
                        pen.setWidth(1)
                        p.setPen(pen)
                        font = p.font()
                        font.setPointSize(9)
                        p.setFont(font)
                        
                        # グリッド/点の描画は、クリック判定と同じ座標変換（_full_to_display）に統一する
                        def transform_grid_coords(x_px, y_px):
                            try:
                                dxy = self._full_to_display(float(x_px), float(y_px))
                                if dxy is None:
                                    return None, None
                                return int(round(dxy[0])), int(round(dxy[1]))
                            except Exception:
                                return None, None
                        
                        # Vertical lines (constant X in image pixels)
                        # IMPORTANT: Image pixel indices are [0..w_full-1], [0..h_full-1].
                        # Using w_full/h_full as coordinates caused v=-1 at the bottom due to (h_full-1)-h_full.
                        x_px = 0.0
                        x_max = max(0.0, float(w_full - 1))
                        y_top_px = 0.0
                        y_bottom_px = max(0.0, float(h_full - 1))
                        while x_px <= x_max + 1e-6:
                            # 線の始終点を計算（回転適用）
                            x_top, y_top = transform_grid_coords(x_px, y_top_px)
                            x_bottom, y_bottom = transform_grid_coords(x_px, y_bottom_px)
                            if x_top is not None and x_bottom is not None:
                                p.drawLine(x_top, y_top, x_bottom, y_bottom)
                            # ラベル位置（下端付近）: 左下原点を明確にする
                            x_lbl, y_lbl = transform_grid_coords(x_px, y_bottom_px)
                            if x_lbl is not None and y_lbl is not None:
                                try:
                                    lbl = f"{int(round(x_px))}"
                                    p.drawText(x_lbl + 4, y_lbl - 6, lbl)
                                except Exception:
                                    pass
                            x_px += pixel_spacing

                        # Horizontal lines (constant Y in image pixels)
                        # v-axis origin is bottom-left: v=0 at y=(h_full-1), increasing upward.
                        y_max = max(0.0, float(h_full - 1))
                        x_left_px = 0.0
                        x_right_px = max(0.0, float(w_full - 1))
                        v_px = 0.0
                        v_max = y_max
                        while v_px <= v_max + 1e-6:
                            y_px = y_max - v_px
                            # 線の始終点を計算（回転適用）
                            x_left, y_left = transform_grid_coords(x_left_px, y_px)
                            x_right, y_right = transform_grid_coords(x_right_px, y_px)
                            if x_left is not None and x_right is not None:
                                p.drawLine(x_left, y_left, x_right, y_right)
                            # ラベル位置（左端付近）: 左下原点を明確にする
                            x_lbl, y_lbl = transform_grid_coords(x_left_px, y_px)
                            if x_lbl is not None and y_lbl is not None:
                                try:
                                    lbl = f"{int(round(v_px))}"
                                    p.drawText(x_lbl + 4, y_lbl - 4, lbl)
                                except Exception:
                                    pass
                            v_px += pixel_spacing
                        
                        # 参照点は build_zoomed_canvas 側で描画されており、画像回転と一緒に回転される。
                        # ここでの再描画は不要（重複するとズレに見える原因になる）。

                        p.end()
                        pm_to_show = pm2
                        try:
                            self._last_pm_image_grid = pm2
                        except Exception:
                            pass
                        # mappingは rotated_img 作成直後に設定済み
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
                        try:
                            self._display_mapping = None
                        except Exception:
                            pass
                        
            except Exception as e:
                import traceback
                self._dbg(f"Stage rotation failed with exception: {e}")
                self._dbg(traceback.format_exc())
                try:
                    if hasattr(self, '_log_error'):
                        self._log_error(f"Stage rotation failed: {e}\n{traceback.format_exc()}")
                except Exception:
                    pass
                pm_to_show = pm
                try:
                    self._display_mapping = None
                except Exception:
                    pass

            # Note: Scale/X/Y/Z overlay text inside the image is intentionally disabled.

            # Fallback mapping (identity) if none was set above
            try:
                if self._display_mapping is None:
                    z = float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom))))
                    dm = _build_display_mapping(str(view_orient), float(new_w), float(new_h), float(int(self.view_padding)), z, qt_transform=None)
                    if dm is not None:
                        self._display_mapping = dm
            except Exception:
                pass

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
            try:
                self._update_stage_info_overlay(None, getattr(self, 'view_orientation', 'Image'))
            except Exception:
                pass
            try:
                self._update_cursor_info_overlay(None)
            except Exception:
                pass
            return

        try:
            self._update_stage_info_overlay(info, getattr(self, 'view_orientation', 'Image'))
        except Exception:
            pass
        try:
            vp = self.proc_scroll.viewport() if getattr(self, 'proc_scroll', None) is not None else None
            if vp is not None:
                pos_vp = vp.mapFromGlobal(QCursor.pos())
                pos_label = self._viewport_pos_to_label_pos(pos_vp)
                self._update_cursor_info_overlay(pos_label)
        except Exception:
            pass

        # If pick mode active, redraw crosshair
        try:
            if self.pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
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

    def _reposition_viewport_overlays(self):
        try:
            vp = self.proc_scroll.viewport() if getattr(self, 'proc_scroll', None) is not None else None
            if vp is None:
                return
            margin = 8

            ov_tl = getattr(self, 'stage_info_overlay', None)
            if ov_tl is not None:
                try:
                    x = int(max(margin, vp.width() - ov_tl.width() - margin))
                    y = int(margin)
                    ov_tl.move(x, y)
                except Exception:
                    ov_tl.move(int(max(margin, vp.width() - ov_tl.width() - margin)), int(margin))

            ov_br = getattr(self, 'cursor_info_overlay', None)
            if ov_br is not None:
                try:
                    x = int(margin)
                    y = int(max(margin, vp.height() - ov_br.height() - margin))
                    ov_br.move(x, y)
                except Exception:
                    ov_br.move(int(margin), int(max(margin, vp.height() - ov_br.height() - margin)))
        except Exception:
            pass

    def _get_render_max_pixels(self):
        try:
            # Fast mode override during wheel interaction
            ov = getattr(self, '_max_render_pixels_override', None)
            if ov is not None:
                try:
                    iv = int(ov)
                    if iv > 0:
                        return iv
                except Exception:
                    pass

            base = int(getattr(self, '_normal_max_render_pixels', 6144 * 6144) or (6144 * 6144))
            hard = int(getattr(self, '_hard_max_render_pixels', 12288 * 12288) or (12288 * 12288))

            img_sz = getattr(self, '_img_base_size', None)
            if not img_sz or len(img_sz) < 2:
                return max(1, min(base, hard))
            w = int(img_sz[0])
            h = int(img_sz[1])
            if w <= 0 or h <= 0:
                return max(1, min(base, hard))

            vp = self.proc_scroll.viewport() if getattr(self, 'proc_scroll', None) is not None else None
            if vp is None:
                return max(1, min(base, hard))
            vw = max(1, int(vp.width()))
            vh = max(1, int(vp.height()))

            target_px = float(getattr(self, 'max_zoom_target_visible_px', 300) or 300)
            if target_px <= 0:
                return max(1, min(base, hard))

            # Desired display scale so that visible full-image span ~ target_px on viewport long side
            req_scale = float(max(vw, vh)) / float(target_px)
            req_scale = max(1.0, req_scale)
            needed = int(round(float(w) * float(h) * (req_scale * req_scale)))

            dyn = max(base, needed)
            dyn = min(dyn, hard)
            return max(1, int(dyn))
        except Exception:
            return int(6144 * 6144)

    def _stage_input_decimal_digits(self):
        try:
            max_digits = 0
            for ro in (getattr(self, 'ref_obs', None) or []):
                if not ro:
                    continue
                for key in ('x', 'y', 'z'):
                    try:
                        raw = ro.get(key, '')
                    except Exception:
                        raw = ''
                    if raw is None:
                        continue
                    s = str(raw).strip().replace(',', '')
                    if not s:
                        continue
                    low = s.lower()
                    if 'e' in low:
                        try:
                            from decimal import Decimal
                            d = Decimal(s)
                            digits = int(max(0, -int(d.as_tuple().exponent)))
                        except Exception:
                            digits = 0
                    elif '.' in s:
                        try:
                            frac = s.split('.', 1)[1].rstrip('0')
                            digits = max(0, len(frac))
                        except Exception:
                            digits = 0
                    else:
                        digits = 0
                    if digits > max_digits:
                        max_digits = digits
            return int(max_digits)
        except Exception:
            return 0

    def _format_stage_numeric(self, value, decimal_digits):
        try:
            fv = float(value)
        except Exception:
            return "-"
        try:
            if not np.isfinite(fv):
                return "-"
        except Exception:
            pass
        try:
            d = int(decimal_digits)
        except Exception:
            d = 0
        if d <= 0:
            try:
                return str(int(round(fv)))
            except Exception:
                return "-"
        s = f"{fv:.{d}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        if s in ('-0', '-0.0'):
            s = '0'
        return s

    def _format_plain_decimal(self, value, max_digits=6):
        try:
            fv = float(value)
        except Exception:
            return "-"
        try:
            if not np.isfinite(fv):
                return "-"
        except Exception:
            pass
        try:
            d = int(max(0, max_digits))
        except Exception:
            d = 6
        s = f"{fv:.{d}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        if s in ('-0', '-0.0'):
            s = '0'
        return s

    def _update_stage_info_overlay(self, info, view_orient):
        try:
            overlay = getattr(self, 'stage_info_overlay', None)
            if overlay is None:
                return

            if not (view_orient == 'Stage' and info is not None):
                overlay.hide()
                return

            # Magnification: 3 significant figures + 'x'
            try:
                mag_v = float(info.get('s_proc', info.get('s', 1.0)))
                mag = format(mag_v, '.3g') + 'x'
            except Exception:
                mag = '-'

            # Rotation: one decimal place with degree symbol
            try:
                ang_v = float(info.get('angle_deg', 0.0))
                ang = f"{ang_v:.1f}°"
            except Exception:
                ang = '-'

            try:
                rf = bool(info.get('reflect', False))
                flip_txt = "On" if rf else "Off"
            except Exception:
                flip_txt = "-"

            try:
                t_xy = info.get('t', None)
                dec_digits = int(self._stage_input_decimal_digits())
                tx = self._format_stage_numeric(t_xy[0] if t_xy is not None else None, dec_digits)
                ty = self._format_stage_numeric(t_xy[1] if t_xy is not None else None, dec_digits)
            except Exception:
                tx, ty = "-", "-"

            pitch = self._format_plain_decimal(info.get('pitch_deg', None), max_digits=3)
            roll = self._format_plain_decimal(info.get('roll_deg', None), max_digits=3)

            line1 = f"Magnification: {mag}   Rotation: {ang}   Flip: {flip_txt}"
            line2 = f"Shift X: {tx}   Shift Y: {ty}   Pitch: {pitch}   Roll: {roll}"
            overlay.setText(f"{line1}\n{line2}")
            try:
                overlay.adjustSize()
            except Exception:
                pass
            try:
                self._reposition_viewport_overlays()
            except Exception:
                pass
            overlay.raise_()
            overlay.show()
        except Exception:
            pass

    def _update_cursor_info_overlay(self, pos_label=None):
        try:
            overlay = getattr(self, 'cursor_info_overlay', None)
            if overlay is None:
                return

            if pos_label is None:
                overlay.hide()
                return

            xy = self._display_to_full(pos_label)
            if xy is None:
                overlay.hide()
                return
            x_full, y_full = xy

            try:
                h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else None
            except Exception:
                h_full = None

            u_img = int(round(float(x_full)))
            if h_full is not None and h_full > 0:
                v_img = int(round(float(h_full - 1) - float(y_full)))
                v_model = float(h_full - 1) - float(y_full)
            else:
                v_img = int(round(-float(y_full)))
                v_model = -float(y_full)
            u_model = float(x_full)

            stage_x = None
            stage_y = None
            stage_z = None
            info = getattr(self, '_last_stage_info', None)
            try:
                if info is not None and info.get('R', None) is not None and info.get('t', None) is not None:
                    import numpy as _np
                    s_val = float(info.get('s', 1.0))
                    R = _np.asarray(info.get('R'), dtype=_np.float64)
                    t = _np.asarray(info.get('t'), dtype=_np.float64)

                    if bool(info.get('reflect', False)):
                        u_model = -u_model

                    uv = _np.asarray([float(u_model), float(v_model)], dtype=_np.float64)
                    st_xy = (s_val * (R @ uv)) + t
                    stage_x = float(st_xy[0])
                    stage_y = float(st_xy[1])

                    zp = info.get('z_plane', None)
                    if zp is not None and len(zp) == 3:
                        a, b, c = float(zp[0]), float(zp[1]), float(zp[2])
                        stage_z = float(a * stage_x + b * stage_y + c)
            except Exception:
                pass

            dec_digits = int(self._stage_input_decimal_digits())
            def _fmt_stage(v):
                return self._format_stage_numeric(v, dec_digits)

            line1 = f"Image (u, v) = ({u_img}, {v_img})"
            line2 = f"Stage (X, Y, Z) = ({_fmt_stage(stage_x)}, {_fmt_stage(stage_y)}, {_fmt_stage(stage_z)})"
            overlay.setText(f"{line1}\n{line2}")
            try:
                overlay.adjustSize()
            except Exception:
                pass
            try:
                self._reposition_viewport_overlays()
            except Exception:
                pass
            overlay.raise_()
            overlay.show()
        except Exception:
            pass

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
        # QScrollArea is aligned top-left, so scroll values map directly.
        cx = lx - vp.width() / 2.0
        cy = ly - vp.height() / 2.0
        self._set_scroll(cx, cy)

    def _center_on_ref_index(self, idx):
        try:
            i = int(idx)
            pts = getattr(self, 'ref_points', []) or []
            if not (0 <= i < len(pts)):
                return
            pt = pts[i]
            if pt is None:
                return
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
            self._ensure_full_pos_visible(float(pt[0]) * spf, float(pt[1]) * spf)
        except Exception:
            pass

    def _center_on_centroid_index(self, idx):
        try:
            i = int(idx)
            cents = getattr(self, 'centroids', []) or []
            if not (0 <= i < len(cents)):
                return
            _, xp, yp = cents[i]
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
            self._ensure_full_pos_visible(float(xp) * spf, float(yp) * spf)
        except Exception:
            pass

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
        # Ref選択変更はRefマーカーの大小に影響するので、表/描画を同期する
        try:
            self._sync_ref_selection()
        except Exception:
            pass
        try:
            self._center_on_ref_index(self.ref_selected_index)
        except Exception:
            pass

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
        # Ref選択変更はRefマーカーの大小に影響するので、表/描画を同期する
        try:
            self._sync_ref_selection()
        except Exception:
            pass
        try:
            self._center_on_ref_index(self.ref_selected_index)
        except Exception:
            pass

    def _sync_ref_selection(self):
        """Sync ref_selected_index to both ref tables and redraw.

        - canonical: table_ref selects the column
        - transposed: table_ref_view selects the row
        """
        try:
            idx = getattr(self, 'ref_selected_index', None)
            if idx is None:
                return
            idx = int(idx)
            if idx < 0:
                return
        except Exception:
            return

        # Ensure the selected ref column is visible
        try:
            vis = int(getattr(self, 'visible_ref_cols', 1) or 1)
            if (idx + 1) > vis:
                self.visible_ref_cols = min(len(getattr(self, 'ref_points', []) or []), idx + 1)
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
                except Exception:
                    pass
        except Exception:
            pass

        # Canonical ref table selection (column)
        try:
            t = getattr(self, 'table_ref', None)
            if t is not None and 0 <= idx < t.columnCount():
                try:
                    t.blockSignals(True)
                    # canonical table_ref has 2 pseudo-header rows.
                    # Do not force currentRow back to row 2 when the user is trying to edit
                    # Stage X/Y/Z (rows 4-6). Forcing it breaks edit initiation.
                    row_for_current = 2
                    try:
                        cur_r = int(t.currentRow())
                    except Exception:
                        cur_r = -1
                    try:
                        # If currently editing, never change the current cell here.
                        if int(getattr(t, 'state', lambda: 0)()) == int(getattr(QAbstractItemView, 'EditingState', 0)):
                            row_for_current = cur_r
                    except Exception:
                        pass
                    if cur_r in (4, 5, 6):
                        row_for_current = cur_r
                    if row_for_current is not None and int(row_for_current) >= 0:
                        t.setCurrentCell(int(row_for_current), idx)
                    t.selectColumn(idx)
                finally:
                    try:
                        t.blockSignals(False)
                    except Exception:
                        pass
        except Exception:
            pass

        # Transposed ref view selection (row)
        try:
            rv = getattr(self, 'table_ref_view', None)
            if rv is not None:
                header_rows = 2
                view_r = int(idx) + header_rows
                if 0 <= view_r < rv.rowCount():
                    try:
                        rv.blockSignals(True)
                        # Do not force column 0 when user is editing Stage columns (2-4).
                        col_for_current = 0
                        try:
                            cur_c = int(rv.currentColumn())
                        except Exception:
                            cur_c = -1
                        try:
                            if int(getattr(rv, 'state', lambda: 0)()) == int(getattr(QAbstractItemView, 'EditingState', 0)):
                                col_for_current = cur_c
                        except Exception:
                            pass
                        if cur_c in (2, 3, 4):
                            col_for_current = cur_c
                        if col_for_current is None or int(col_for_current) < 0:
                            col_for_current = 0
                        rv.setCurrentCell(view_r, int(col_for_current))
                        rv.selectRow(view_r)
                    finally:
                        try:
                            rv.blockSignals(False)
                        except Exception:
                            pass
        except Exception:
            pass

        # Debounced redraw: selecting a different ref changes marker emphasis.
        # Calling _apply_proc_zoom() directly here can be expensive and can visually
        # flicker during rapid selection/table updates, so coalesce into one tick.
        try:
            if not getattr(self, '_ref_redraw_pending', False):
                self._ref_redraw_pending = True
                try:
                    from qt_compat.QtCore import QTimer
                    QTimer.singleShot(0, self._do_ref_redraw)
                except Exception:
                    self._do_ref_redraw()
        except Exception:
            pass

    def _do_ref_redraw(self):
        try:
            self._ref_redraw_pending = False
        except Exception:
            pass
        try:
            self._apply_proc_zoom()
        except Exception:
            pass

    def _on_ref_view_cell_clicked(self, row, col):
        # Transposed view: data rows start at row>=2; editable Stage columns are 2,3,4.
        # Show column toggle is handled by the cell widget (SegmentControl), not here.
        try:
            header_rows = 2
            if row is None or col is None:
                return
            if int(row) < header_rows:
                return

            # Show column handled by cell widget — skip
            tbl = getattr(self, 'table_ref_view', None)
            if tbl is None:
                return
            excl_col = int(tbl.columnCount()) - 1
            if int(col) == excl_col:
                return

            if int(col) not in (2, 3, 4):
                return
            item = self.table_ref_view.item(int(row), int(col))
            if item is None:
                return
            if not (item.flags() & Qt.ItemIsEditable):
                return
            try:
                self.table_ref_view.setCurrentCell(int(row), int(col))
                self.table_ref_view.selectRow(int(row))
            except Exception:
                pass
            try:
                self.table_ref_view.setFocus(Qt.MouseFocusReason)
            except Exception:
                try:
                    self.table_ref_view.setFocus()
                except Exception:
                    pass
            # Defer edit start slightly so selection updates don't immediately cancel it.
            try:
                from qt_compat.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.table_ref_view.editItem(item))
            except Exception:
                self.table_ref_view.editItem(item)
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
        self._ref_add_has_added = False
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
        """カーソルを現在表示されているビューポート領域の中心にジャンプさせる（スクロール・ズーム変更なし）"""
        try:
            vp = self.proc_scroll.viewport()
            # ビューポート中央のビューポート座標
            vp_center_x = vp.width() / 2.0
            vp_center_y = vp.height() / 2.0
            vp_center = QPoint(int(round(vp_center_x)), int(round(vp_center_y)))
            # グローバル座標に変換
            global_pt = vp.mapToGlobal(vp_center)
            QCursor.setPos(global_pt)
            # 十字線を即時表示
            pos_label = self._viewport_pos_to_label_pos(vp_center)
            self._draw_crosshair(pos_label)
        except Exception:
            pass

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

    def _on_add_target_point(self):
        # Toggle pick-mode（Add Target）
        if self.pick_mode == 'target_add':
            try:
                if hasattr(self, '_log_info'):
                    self._log_info("AddTarget: cancel requested by button")
            except Exception:
                pass
            self._end_pick_mode()
            return
        try:
            if getattr(self, 'manual_targets', None) is None:
                self.manual_targets = []
        except Exception:
            pass
        self._target_add_has_added = False  # reset: no point added yet this session
        try:
            if hasattr(self, '_log_info'):
                self._log_info("AddTarget: mode start (waiting first click)")
        except Exception:
            pass
        self._start_pick_mode('target_add')
        self._move_cursor_to_image_center()

    def _manual_target_base_index(self):
        try:
            auto_only = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())
            return sum(1 for c in auto_only if int(c[0]) == 0)
        except Exception:
            return 0

    def _selected_manual_target_index(self):
        try:
            idx = getattr(self, 'selected_index', None)
            if idx is None:
                return None
            idx = int(idx)
            base = self._manual_target_base_index()
            m_idx = idx - base
            mt_n = len(getattr(self, 'manual_targets', []) or [])
            if 0 <= m_idx < mt_n:
                return m_idx
        except Exception:
            pass
        return None

    def _manual_centroid_indices(self):
        try:
            mt_n = len(getattr(self, 'manual_targets', []) or [])
            if mt_n <= 0:
                return set()
            base = self._manual_target_base_index()
            return set(range(int(base), int(base) + mt_n))
        except Exception:
            return set()

    def _capture_manual_target_visibility(self):
        """Capture current show/hide state of manual Group 0 targets by manual-target order."""
        try:
            mt_n = len(getattr(self, 'manual_targets', []) or [])
            if mt_n <= 0:
                return []
            base = int(self._manual_target_base_index())
            excluded = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            vis = []
            for j in range(mt_n):
                idx = int(base + j)
                vis.append(idx not in excluded)
            return vis
        except Exception:
            return []

    def _reset_visibility_after_recompute(self, manual_visibility_by_order=None):
        """Reset per-particle visibility after recompute.

        Rule:
        - Recomputed particles are all set to visible by default.
        - Manual Group 0 particles keep their previous show/hide state by order.
        """
        try:
            n = len(getattr(self, 'centroids', []) or [])
            if n <= 0:
                self.excluded_centroid_indices = set()
                self._explicit_excluded_centroid_indices = set()
                self._force_visible_centroid_indices = set()
                return

            excluded = set()
            explicit = set()
            force_visible = set()

            states = list(manual_visibility_by_order or [])
            mt_n = len(getattr(self, 'manual_targets', []) or [])
            base = int(self._manual_target_base_index())
            for j in range(min(mt_n, len(states))):
                idx = int(base + j)
                if not (0 <= idx < n):
                    continue
                if not bool(states[j]):
                    excluded.add(idx)
                    explicit.add(idx)

            self.excluded_centroid_indices = excluded
            self._explicit_excluded_centroid_indices = explicit
            self._force_visible_centroid_indices = force_visible
            self._sanitize_excluded_indices()
        except Exception:
            pass

    def _sanitize_excluded_indices(self):
        try:
            n = len(getattr(self, 'centroids', []) or [])
            old = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            self.excluded_centroid_indices = {int(i) for i in old if 0 <= int(i) < n}
        except Exception:
            self.excluded_centroid_indices = set()
        try:
            n = len(getattr(self, 'centroids', []) or [])
            old_exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            self._explicit_excluded_centroid_indices = {int(i) for i in old_exp if 0 <= int(i) < n}
        except Exception:
            self._explicit_excluded_centroid_indices = set()
        try:
            n = len(getattr(self, 'centroids', []) or [])
            old_force = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            self._force_visible_centroid_indices = {int(i) for i in old_force if 0 <= int(i) < n}
        except Exception:
            self._force_visible_centroid_indices = set()

    def _is_centroid_excluded(self, idx):
        try:
            return int(idx) in set(getattr(self, 'excluded_centroid_indices', set()) or set())
        except Exception:
            return False

    def _is_ref_excluded(self, idx):
        try:
            return int(idx) in set(getattr(self, 'excluded_ref_indices', set()) or set())
        except Exception:
            return False

    def _available_group_numbers(self):
        try:
            groups = set()
            for c in (getattr(self, 'centroids', []) or []):
                try:
                    groups.add(int(c[0]))
                except Exception:
                    pass
            return sorted(groups)
        except Exception:
            return []

    def _safe_float_or_nan(self, v):
        try:
            s = str(v).strip()
            if s == "":
                return float('nan')
            return float(s.replace(',', ''))
        except Exception:
            return float('nan')

    def _center_uv_from_proc(self, x_proc, y_proc):
        try:
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
        except Exception:
            spf = 1.0
        try:
            h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) is not None else None
        except Exception:
            h_full = None
        if h_full is None:
            try:
                h_full = int(self.img_full.shape[0]) if getattr(self, 'img_full', None) is not None else None
            except Exception:
                h_full = None
        try:
            x_full = float(x_proc) * spf
            y_full = float(y_proc) * spf
            u = int(round(x_full))
            if h_full is not None and h_full > 0:
                v = int(round((h_full - 1) - y_full))
            else:
                v = int(round(-y_full))
            return u, v
        except Exception:
            return 0, 0

    def _proc_from_center_uv(self, u_val, v_val):
        try:
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
        except Exception:
            spf = 1.0
        if abs(spf) < 1e-12:
            spf = 1.0
        try:
            h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) is not None else None
        except Exception:
            h_full = None
        if h_full is None:
            try:
                h_full = int(self.img_full.shape[0]) if getattr(self, 'img_full', None) is not None else None
            except Exception:
                h_full = None
        try:
            u = float(u_val)
            v = float(v_val)
            x_full = u
            if h_full is not None and h_full > 0:
                y_full = float((h_full - 1) - v)
            else:
                y_full = float(-v)
            return float(x_full / spf), float(y_full / spf)
        except Exception:
            return None

    def _snapshot_center_row_from_centroid(self, cidx):
        try:
            idx = int(cidx)
            cents = list(getattr(self, 'centroids', []) or [])
            if not (0 <= idx < len(cents)):
                return None
            g, xp, yp = cents[idx]
            u, v = self._center_uv_from_proc(float(xp), float(yp))

            x_val = float('nan')
            y_val = float('nan')
            z_val = float('nan')
            try:
                src = getattr(self, 'table', None)
                if src is not None and 0 <= idx < src.columnCount():
                    x_it = src.item(4, idx)
                    y_it = src.item(5, idx)
                    z_it = src.item(6, idx)
                    x_val = self._safe_float_or_nan(x_it.text() if x_it is not None else "")
                    y_val = self._safe_float_or_nan(y_it.text() if y_it is not None else "")
                    z_val = self._safe_float_or_nan(z_it.text() if z_it is not None else "")
            except Exception:
                pass

            try:
                is_manual = 1.0 if idx in set(self._manual_centroid_indices() or set()) else 0.0
            except Exception:
                is_manual = 0.0

            return {
                'source_idx': int(idx),
                'grp': float(int(g)),
                'u': float(u),
                'v': float(v),
                'x': float(x_val),
                'y': float(y_val),
                'z': float(z_val),
                'x_proc': float(xp),
                'y_proc': float(yp),
                'show': 1.0,
                'manual': float(is_manual),
            }
        except Exception:
            return None

    def _append_center_numeric_rows_from_indices(self, indices):
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            existing = set()
            for r in rows:
                try:
                    existing.add(int(r.get('source_idx', -1)))
                except Exception:
                    pass
            for i in (indices or []):
                try:
                    ii = int(i)
                except Exception:
                    continue
                if ii in existing:
                    continue
                snap = self._snapshot_center_row_from_centroid(ii)
                if snap is None:
                    continue
                rows.append(snap)
                existing.add(ii)
            self.center_numeric_rows = rows
        except Exception:
            pass

    def _remove_center_numeric_row_by_source_idx(self, cidx):
        try:
            tgt = int(cidx)
        except Exception:
            return
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            self.center_numeric_rows = [r for r in rows if int(r.get('source_idx', -1)) != tgt]
        except Exception:
            pass

    def _get_center_list_indices(self):
        try:
            n = len(getattr(self, 'centroids', []) or [])
            cur = list(getattr(self, 'center_list_indices', []) or [])
            seen = set()
            out = []
            for i in cur:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if not (0 <= ii < n):
                    continue
                if ii in seen:
                    continue
                seen.add(ii)
                out.append(ii)
            explicit_only = list(out)

            # Manual targets are always part of the middle list.
            try:
                manual_idxs = sorted(int(i) for i in (self._manual_centroid_indices() or set()))
            except Exception:
                manual_idxs = []
            for mi in manual_idxs:
                if mi in seen:
                    continue
                if 0 <= mi < n:
                    seen.add(mi)
                    out.append(mi)

            # Persist only explicit additions; manual points are implicit.
            self.center_list_indices = explicit_only
            return out
        except Exception:
            self.center_list_indices = []
            return []

    def _get_explicit_center_list_indices(self):
        try:
            n = len(getattr(self, 'centroids', []) or [])
            cur = list(getattr(self, 'center_list_indices', []) or [])
            seen = set()
            out = []
            for i in cur:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if not (0 <= ii < n):
                    continue
                if ii in seen:
                    continue
                seen.add(ii)
                out.append(ii)
            self.center_list_indices = out
            return out
        except Exception:
            self.center_list_indices = []
            return []

    def _shift_center_list_indices(self, pivot_idx, delta):
        """Shift explicit center-list indices when centroid indices are inserted/removed."""
        try:
            p = int(pivot_idx)
            d = int(delta)
        except Exception:
            return
        if d == 0:
            return
        try:
            cur = list(getattr(self, 'center_list_indices', []) or [])
            out = []
            for i in cur:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if d > 0:
                    if ii >= p:
                        ii += d
                else:
                    if ii == p:
                        continue
                    if ii > p:
                        ii += d
                out.append(ii)
            self.center_list_indices = out
        except Exception:
            pass
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            out_rows = []
            for r in rows:
                rr = dict(r)
                try:
                    ii = int(rr.get('source_idx', -1))
                except Exception:
                    ii = -1
                if d > 0:
                    if ii >= p:
                        ii += d
                else:
                    if ii == p:
                        continue
                    if ii > p:
                        ii += d
                rr['source_idx'] = int(ii)
                out_rows.append(rr)
            self.center_numeric_rows = out_rows
        except Exception:
            pass

    def _add_group_to_center_list(self, group_no):
        try:
            g = int(group_no)
        except Exception:
            return
        try:
            existing = self._get_explicit_center_list_indices()
            existing_set = set(existing)
            added = []
            for idx, c in enumerate(getattr(self, 'centroids', []) or []):
                try:
                    if int(c[0]) != g:
                        continue
                except Exception:
                    continue
                if idx in existing_set:
                    continue
                existing.append(int(idx))
                existing_set.add(int(idx))
                added.append(int(idx))
            self.center_list_indices = existing
            try:
                self._append_center_numeric_rows_from_indices(added)
            except Exception:
                pass
            if added and getattr(self, 'selected_index', None) not in set(existing):
                try:
                    self.selected_index = int(added[0])
                except Exception:
                    pass
        except Exception:
            return
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _group_centroid_indices(self, group_no):
        try:
            g = int(group_no)
        except Exception:
            return []
        out = []
        for idx, c in enumerate(getattr(self, 'centroids', []) or []):
            try:
                if int(c[0]) == g:
                    out.append(int(idx))
            except Exception:
                continue
        return out

    def _set_group_visible(self, group_no, visible):
        idxs = self._group_centroid_indices(group_no)
        if not idxs:
            return
        try:
            s = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            fv = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            if bool(visible):
                for ci in idxs:
                    s.discard(ci)
                    exp.discard(ci)
                    fv.add(ci)
            else:
                for ci in idxs:
                    s.add(ci)
                    exp.add(ci)
                    fv.discard(ci)
            self.excluded_centroid_indices = s
            self._explicit_excluded_centroid_indices = exp
            self._force_visible_centroid_indices = fv
            self._sanitize_excluded_indices()
        except Exception:
            return

    def _is_group_visible(self, group_no):
        idxs = self._group_centroid_indices(group_no)
        if not idxs:
            return True
        excl = set(getattr(self, 'excluded_centroid_indices', set()) or set())
        return any((ci not in excl) for ci in idxs)

    def _add_all_groups_to_center_list(self):
        try:
            existing = self._get_explicit_center_list_indices()
            existing_set = set(existing)
            added = []
            groups = self._available_group_numbers()
            for g in groups:
                for ci in self._group_centroid_indices(g):
                    if ci in existing_set:
                        continue
                    existing.append(int(ci))
                    existing_set.add(int(ci))
                    added.append(int(ci))
            self.center_list_indices = existing
            try:
                self._append_center_numeric_rows_from_indices(added)
            except Exception:
                pass
            if added and getattr(self, 'selected_index', None) not in existing_set:
                try:
                    self.selected_index = int(added[0])
                except Exception:
                    pass
        except Exception:
            return
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _set_all_groups_visible(self, visible):
        try:
            s = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            fv = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            for idx, c in enumerate(getattr(self, 'centroids', []) or []):
                try:
                    g = int(c[0])
                except Exception:
                    continue
                if g <= 0:
                    continue
                ci = int(idx)
                if bool(visible):
                    s.discard(ci)
                    exp.discard(ci)
                    fv.add(ci)
                else:
                    s.add(ci)
                    exp.add(ci)
                    fv.discard(ci)
            self.excluded_centroid_indices = s
            self._explicit_excluded_centroid_indices = exp
            self._force_visible_centroid_indices = fv
            self._sanitize_excluded_indices()
        except Exception:
            return
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _on_toggle_show_all_groups(self, idx):
        try:
            show_all = (int(idx) == 0)
        except Exception:
            show_all = True
        self._set_all_groups_visible(show_all)

    def _on_toggle_overlay_source(self, idx):
        try:
            self.overlay_point_source = 'center' if int(idx) == 1 else 'left'
        except Exception:
            self.overlay_point_source = 'left'
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _update_centroid_extraction_button(self):
        try:
            btn = getattr(self, 'btn_start_centroid_extraction', None)
            if btn is None:
                return
            if bool(getattr(self, 'centroid_extraction_mode', False)):
                btn.setText('Finish Centroid Extraction')
                btn.setStyleSheet(
                    'QPushButton { background-color: rgb(230,230,230); color: black; border: none; border-radius: 8px; }'
                    'QPushButton:hover { background-color: rgb(220,220,220); }'
                    'QPushButton:pressed { background-color: rgb(210,210,210); }'
                )
            else:
                btn.setText('START Centroid Extraction')
                btn.setStyleSheet(
                    'QPushButton { background-color: rgb(160,15,15); color: white; border: none; border-radius: 8px; }'
                    'QPushButton:hover { background-color: rgb(160,15,15); }'
                    'QPushButton:pressed { background-color: rgb(160,15,15); }'
                )
        except Exception:
            pass

    def _load_centroid_extraction_preferences(self):
        try:
            s = QSettings('PiXY', 'PiXY')
        except Exception:
            return
        try:
            mode = str(s.value('centroid_extraction/overlay_mode', 'Posterized') or 'Posterized')
            if mode not in ('Original', 'Posterized'):
                mode = 'Posterized'
            self._centroid_extraction_overlay_mode = mode
        except Exception:
            self._centroid_extraction_overlay_mode = 'Posterized'
        try:
            raw = s.value('centroid_extraction/show_boundaries', True)
            if isinstance(raw, str):
                raw_l = raw.strip().lower()
                show = raw_l in ('1', 'true', 'yes', 'on')
            else:
                show = bool(raw)
            self._centroid_extraction_show_boundaries = bool(show)
        except Exception:
            self._centroid_extraction_show_boundaries = True

    def _save_centroid_extraction_preferences(self):
        try:
            s = QSettings('PiXY', 'PiXY')
            s.setValue('centroid_extraction/overlay_mode', str(getattr(self, '_centroid_extraction_overlay_mode', 'Posterized')))
            s.setValue('centroid_extraction/show_boundaries', bool(getattr(self, '_centroid_extraction_show_boundaries', True)))
            s.sync()
        except Exception:
            pass

    def _apply_overlay_boundary_state(self, mode: str, show_boundaries: bool):
        try:
            mode_n = str(mode or 'Original')
            if mode_n not in ('Original', 'Posterized'):
                mode_n = 'Original'
        except Exception:
            mode_n = 'Original'
        self.overlay_mode = mode_n
        try:
            self.overlay_mix = {'Original': 0, 'Posterized': 100}.get(mode_n, 0)
        except Exception:
            pass
        self.show_boundaries = bool(show_boundaries)
        try:
            if getattr(self, 'btn_boundary_show', None) is not None and getattr(self, 'btn_boundary_hide', None) is not None:
                self.btn_boundary_show.setChecked(bool(self.show_boundaries))
                self.btn_boundary_hide.setChecked(not bool(self.show_boundaries))
        except Exception:
            pass
        try:
            tog = getattr(self, 'overlay_mode_toggle', None)
            if tog is not None:
                tog.setCheckedIndex(1 if mode_n == 'Posterized' else 0)
        except Exception:
            pass

    def _set_centroid_extraction_mode(self, active: bool):
        prev = bool(getattr(self, 'centroid_extraction_mode', False))
        try:
            self.centroid_extraction_mode = bool(active)
        except Exception:
            pass
        try:
            if (not bool(active)) and bool(prev):
                # Finish時の設定を次回CentroidExtraction開始の既定値として記憶
                mode_now = str(getattr(self, 'overlay_mode', 'Posterized') or 'Posterized')
                if mode_now not in ('Original', 'Posterized'):
                    mode_now = 'Posterized'
                self._centroid_extraction_overlay_mode = mode_now
                self._centroid_extraction_show_boundaries = bool(getattr(self, 'show_boundaries', True))
                self._save_centroid_extraction_preferences()
        except Exception:
            pass
        try:
            if bool(active):
                mode_pref = str(getattr(self, '_centroid_extraction_overlay_mode', 'Posterized') or 'Posterized')
                show_pref = bool(getattr(self, '_centroid_extraction_show_boundaries', True))
                self._apply_overlay_boundary_state(mode_pref, show_pref)
            else:
                # 通常モードは常に Boundary非表示 + Original 固定
                self._apply_overlay_boundary_state('Original', False)
        except Exception:
            pass
        try:
            show_adv = bool(active)
            bc = getattr(self, 'boundary_controls', None)
            bt = getattr(self, 'boundary_toggle', None)
            ovc = getattr(self, 'overlay_mode_controls', None)
            ovt = getattr(self, 'overlay_mode_toggle', None)
            ovsrc_lbl = getattr(self, 'lbl_overlay_source', None)
            ovsrc_tog = getattr(self, 'toggle_overlay_source', None)
            if bc is not None:
                bc.setVisible(show_adv)
            elif bt is not None:
                bt.setVisible(show_adv)
            if ovc is not None:
                ovc.setVisible(show_adv)
            elif ovt is not None:
                ovt.setVisible(show_adv)
            if ovsrc_lbl is not None:
                ovsrc_lbl.setVisible(not bool(active))
            if ovsrc_tog is not None:
                ovsrc_tog.setVisible(not bool(active))
        except Exception:
            pass
        try:
            tabs = getattr(self, 'left_tabs', None)
            if tabs is not None:
                tabs.setCurrentIndex(0 if bool(active) else 1)
        except Exception:
            pass
        try:
            if bool(active):
                self.overlay_point_source = 'left'
            else:
                self.overlay_point_source = 'center'
            tog = getattr(self, 'toggle_overlay_source', None)
            if tog is not None:
                tog.setCheckedIndex(0 if bool(active) else 1)
        except Exception:
            pass
        try:
            self._update_centroid_extraction_button()
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _on_toggle_centroid_extraction_mode(self):
        try:
            self._set_centroid_extraction_mode(not bool(getattr(self, 'centroid_extraction_mode', False)))
        except Exception:
            pass

    def _get_overlay_render_payload(self):
        """Return centroids/selection/index-sets for overlay rendering based on source toggle."""
        try:
            src = str(getattr(self, 'overlay_point_source', 'left') or 'left').lower()
        except Exception:
            src = 'left'

        base = list(getattr(self, 'centroids', []) or [])
        if src != 'center':
            # Left List: use current auto/offline display controls (Show/Hide, Filter).
            manual_src = set(self._manual_centroid_indices() or set())
            keep = [i for i in range(len(base)) if i not in manual_src]
            remap = {orig: j for j, orig in enumerate(keep)}
            cent = [base[i] for i in keep]
            sel_orig = getattr(self, 'selected_index', None)
            sel_new = remap.get(int(sel_orig)) if sel_orig is not None else None
            excl_src = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            fv_src = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            excl_new = {remap[i] for i in excl_src if i in remap}
            fv_new = {remap[i] for i in fv_src if i in remap}

            # Label format for Left List overlay: G{grp}-{index_in_group}
            label_texts = []
            grp_count = {}
            for g, _xp, _yp in cent:
                try:
                    gg = int(g)
                except Exception:
                    gg = 0
                n0 = int(grp_count.get(gg, 0)) + 1
                grp_count[gg] = n0
                label_texts.append(f"G{gg}-{n0}")

            return {
                'centroids': cent,
                'selected_index': sel_new,
                'manual_indices': set(),
                'excluded_indices': excl_new,
                'force_visible_indices': fv_new,
                'visible_groups': self._get_visible_groups_set(),
                'label_texts': label_texts,
            }

        # Center List: use persisted middle-table numeric rows (not centroid index linkage).
        try:
            self._append_center_numeric_rows_from_indices(self._get_center_list_indices())
        except Exception:
            pass
        rows = list(getattr(self, 'center_numeric_rows', []) or [])

        cent = []
        manual_new = set()
        source_to_local = {}
        for ridx, r in enumerate(rows):
            try:
                rd = dict(r or {})
            except Exception:
                continue
            try:
                g = int(round(float(rd.get('grp', 0.0))))
            except Exception:
                g = 0

            xp = rd.get('x_proc', float('nan'))
            yp = rd.get('y_proc', float('nan'))
            try:
                xp_f = float(xp)
                yp_f = float(yp)
                if np.isnan(xp_f) or np.isnan(yp_f):
                    raise ValueError('nan')
            except Exception:
                uv_proc = self._proc_from_center_uv(rd.get('u', float('nan')), rd.get('v', float('nan')))
                if uv_proc is None:
                    continue
                xp_f, yp_f = uv_proc

            cent.append((int(g), float(xp_f), float(yp_f)))

            try:
                if float(rd.get('manual', 0.0)) >= 0.5:
                    manual_new.add(int(len(cent) - 1))
            except Exception:
                pass
            try:
                src_i = int(rd.get('source_idx', -1))
                if src_i >= 0:
                    source_to_local[src_i] = int(len(cent) - 1)
            except Exception:
                pass

        sel_new = None
        try:
            row_sel = self._current_center_selected_row()
            if row_sel is not None and 0 <= int(row_sel) < len(cent):
                sel_new = int(row_sel)
        except Exception:
            pass
        if sel_new is None:
            try:
                sel_orig = getattr(self, 'selected_index', None)
                if sel_orig is not None:
                    sel_new = source_to_local.get(int(sel_orig))
            except Exception:
                pass

        return {
            'centroids': cent,
            'selected_index': sel_new,
            'manual_indices': manual_new,
            # Middle list visibility is independent from left-column Show/Hide state.
            'excluded_indices': set(),
            'force_visible_indices': set(),
            # center-list mode is explicit subset, so group visibility filter is unnecessary.
            'visible_groups': None,
            'label_texts': None,
        }

    def _get_visible_groups_set(self):
        vis = getattr(self, 'visible_groups', None)
        if vis is None:
            return None
        try:
            return {int(g) for g in vis}
        except Exception:
            return None

    def _set_visible_groups(self, groups):
        try:
            if groups is None:
                self.visible_groups = None
            else:
                vals = {int(g) for g in (groups or set())}
                all_groups = set(self._available_group_numbers())
                if vals == all_groups:
                    self.visible_groups = None
                else:
                    self.visible_groups = vals
        except Exception:
            self.visible_groups = None
        # Sync excluded_centroid_indices with visible groups
        try:
            self._sync_show_from_filter()
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _sync_show_from_filter(self):
        """Sync excluded_centroid_indices with current Filter state.

        IMPORTANT:
        - Filter-driven exclusions are temporary.
        - Explicit exclusions (per-point Show/Hide toggle, or Replace via Update u,v)
          must persist even when the group becomes visible again.
        """
        try:
            vis = self._get_visible_groups_set()
            centroids = getattr(self, 'centroids', None) or []
            s = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            explicit = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            fv = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            for i, c in enumerate(centroids):
                try:
                    g = int(c[0])
                except Exception:
                    continue
                if vis is None:
                    # All visible → un-exclude (except explicit)
                    if int(i) not in explicit:
                        s.discard(i)
                    # Group now visible, force_visible override no longer needed
                    fv.discard(i)
                elif g in vis:
                    if int(i) not in explicit:
                        s.discard(i)
                    # Group now visible, force_visible override no longer needed
                    fv.discard(i)
                else:
                    # Group is hidden — only add to excluded if NOT force_visible
                    if int(i) not in fv:
                        s.add(i)
            self.excluded_centroid_indices = s
            self._force_visible_centroid_indices = fv
            self._sanitize_excluded_indices()
        except Exception:
            pass
        try:
            self._refresh_transposed_views()
        except Exception:
            pass

    def _toggle_single_group_visibility(self, group_no, checked):
        try:
            all_groups = set(self._available_group_numbers())
            cur = self._get_visible_groups_set()
            if cur is None:
                cur = set(all_groups)
            if bool(checked):
                cur.add(int(group_no))
            else:
                cur.discard(int(group_no))
            self._set_visible_groups(cur)
        except Exception:
            pass

    def _show_group_filter_popup(self):
        try:
            groups = self._available_group_numbers()
            if not groups:
                return
            btn = getattr(self, 'btn_filter', None)
            if btn is None:
                return

            menu = QMenu(self)
            cur = self._get_visible_groups_set()
            if cur is None:
                cur = set(groups)

            for g in groups:
                act = menu.addAction(f"Group {int(g)}")
                act.setCheckable(True)
                act.setChecked(int(g) in cur)
                act.toggled.connect(lambda checked, gg=int(g): self._toggle_single_group_visibility(gg, checked))

            menu.addSeparator()
            act_all = menu.addAction("All")
            act_all.triggered.connect(lambda: self._set_visible_groups(None))
            act_none = menu.addAction("None")
            act_none.triggered.connect(lambda: self._set_visible_groups(set()))

            pos = btn.mapToGlobal(QPoint(0, btn.height()))
            menu.exec(pos)
        except Exception:
            pass

    def _compose_centroids_with_manual(self, auto_centroids):
        try:
            auto_list = list(auto_centroids or [])
        except Exception:
            auto_list = []
        mt = []
        for entry in (getattr(self, 'manual_targets', []) or []):
            try:
                _g, x, y = entry
                mt.append((int(_g), float(x), float(y)))
            except Exception:
                pass
        auto_g0 = []
        auto_other = []
        for c in auto_list:
            try:
                if int(c[0]) == 0:
                    auto_g0.append(c)
                else:
                    auto_other.append(c)
            except Exception:
                auto_other.append(c)
        return auto_g0 + mt + auto_other

    def _auto_centroids_from_current(self):
        cur = list(getattr(self, 'centroids', []) or [])
        mt = list(getattr(self, 'manual_targets', []) or [])
        if not mt:
            return cur
        mt_n = len(mt)
        if len(cur) < mt_n:
            return cur
        base = self._manual_target_base_index()
        if base < 0 or (base + mt_n) > len(cur):
            return cur
        blk = cur[base:base + mt_n]
        same = True
        for a, b in zip(blk, mt):
            try:
                if int(a[0]) != int(b[0]) or abs(float(a[1]) - float(b[1])) > 1e-9 or abs(float(a[2]) - float(b[2])) > 1e-9:
                    same = False
                    break
            except Exception:
                same = False
                break
        if not same:
            return cur
        return cur[:base] + cur[base + mt_n:]

    def _get_add_target_group(self):
        try:
            cb = getattr(self, 'combo_add_target_group', None)
            if cb is None:
                return 0
            try:
                d = cb.currentData()
                if d is not None:
                    return int(d)
            except Exception:
                pass
            txt = str(cb.currentText() or '').strip().lower()
            txt = txt.replace('grp', '').strip()
            return int(txt) if txt != '' else 0
        except Exception:
            return 0

    def _refresh_target_group_combo(self):
        try:
            cb = getattr(self, 'combo_add_target_group', None)
            if cb is None:
                return
            cur = self._get_add_target_group()

            left_max = 0
            try:
                for g in (self._available_group_numbers() or []):
                    gg = int(g)
                    if gg > left_max:
                        left_max = gg
            except Exception:
                pass

            center_max = 0
            try:
                for r in (getattr(self, 'center_numeric_rows', []) or []):
                    try:
                        gv = int(round(float(r.get('grp', 0.0))))
                    except Exception:
                        continue
                    if gv > center_max:
                        center_max = gv
            except Exception:
                pass

            upper = max(int(left_max), int(center_max)) + 1
            upper = max(1, int(upper))
            vals = list(range(0, upper + 1))

            old = False
            try:
                old = cb.blockSignals(True)
            except Exception:
                old = False
            try:
                cb.clear()
            except Exception:
                pass
            for g in vals:
                try:
                    cb.addItem(f"Grp {int(g)}", int(g))
                except Exception:
                    pass
            sel = int(cur) if int(cur) in set(vals) else int(max(0, min(int(cur), int(upper))))
            try:
                idx = cb.findData(int(sel))
                if idx < 0:
                    idx = 0
                cb.setCurrentIndex(int(idx))
            except Exception:
                pass
            try:
                cb.blockSignals(old)
            except Exception:
                pass
        except Exception:
            pass

    def _on_update_target_uv(self):
        # Toggle pick-mode（Update u,v）: update selected middle-row u,v only.
        if self.pick_mode == 'center_uv_update':
            self._end_pick_mode()
            return
        try:
            row = self._current_center_selected_row()
            if row is None:
                return
            self._start_pick_mode('center_uv_update', ref_index=int(row))
            self._move_cursor_to_image_center()
        except Exception:
            return

    def _on_clear_target(self):
        # Clear should remove the currently selected row in the middle list.
        try:
            sel_idx = self._current_center_selected_index()
            if sel_idx is not None:
                self.selected_index = int(sel_idx)
        except Exception:
            sel_idx = None

        try:
            if getattr(self, 'manual_targets', None) is None:
                self.manual_targets = []
        except Exception:
            pass

        # Non-manual row: remove only from explicit center list entries.
        try:
            mset = set(self._manual_centroid_indices() or set())
        except Exception:
            mset = set()
        try:
            if sel_idx is not None and int(sel_idx) not in mset:
                explicit = list(self._get_explicit_center_list_indices())
                tgt = int(sel_idx)
                new_explicit = [int(i) for i in explicit if int(i) != tgt]
                if len(new_explicit) != len(explicit):
                    self.center_list_indices = new_explicit
                    try:
                        self._remove_center_numeric_row_by_source_idx(tgt)
                    except Exception:
                        pass
                    self.selected_index = None
                    try:
                        self.schedule_update(force=True, recompute_centroids=False)
                    except Exception:
                        pass
                return
        except Exception:
            pass

        m_idx = self._selected_manual_target_index()
        if m_idx is None:
            return
        rem = int(m_idx)
        base = self._manual_target_base_index()
        rem_centroid_idx = int(base + rem)
        try:
            self.manual_targets.pop(rem)
        except Exception:
            return

        try:
            self._shift_center_list_indices(rem_centroid_idx, -1)
        except Exception:
            pass
        try:
            self._remove_center_numeric_row_by_source_idx(rem_centroid_idx)
        except Exception:
            pass

        try:
            old_excl = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            new_excl = set()
            for i in old_excl:
                ii = int(i)
                if ii == rem_centroid_idx:
                    continue
                if ii > rem_centroid_idx:
                    ii -= 1
                new_excl.add(ii)
            self.excluded_centroid_indices = new_excl
        except Exception:
            pass
        try:
            old_exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            new_exp = set()
            for i in old_exp:
                ii = int(i)
                if ii == rem_centroid_idx:
                    continue
                if ii > rem_centroid_idx:
                    ii -= 1
                new_exp.add(ii)
            self._explicit_excluded_centroid_indices = new_exp
        except Exception:
            pass
        try:
            old_force = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            new_force = set()
            for i in old_force:
                ii = int(i)
                if ii == rem_centroid_idx:
                    continue
                if ii > rem_centroid_idx:
                    ii -= 1
                new_force.add(ii)
            self._force_visible_centroid_indices = new_force
        except Exception:
            pass

        auto_only = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())
        self.centroids = self._compose_centroids_with_manual(auto_only)
        base = self._manual_target_base_index()
        if len(self.manual_targets) == 0:
            self.selected_index = None
        else:
            self.selected_index = int(base + min(rem, len(self.manual_targets) - 1))
        try:
            self._safe_populate_tables(
                self.table_ref, self.table,
                self.ref_points, self.ref_obs,
                self.centroids, self.selected_index,
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

    def _on_clear_target_all(self):
        """Clear all rows in middle list (explicit Add rows + manual targets)."""
        try:
            if getattr(self, 'manual_targets', None) is None:
                self.manual_targets = []
        except Exception:
            pass

        # Clear all explicit additions first.
        try:
            self.center_list_indices = []
        except Exception:
            pass
        try:
            self.center_numeric_rows = []
        except Exception:
            pass

        try:
            mt_n = int(len(getattr(self, 'manual_targets', []) or []))
        except Exception:
            mt_n = 0
        if mt_n > 0:
            try:
                base = int(self._manual_target_base_index())
            except Exception:
                base = 0

            def _remove_manual_block(idxs):
                out = set()
                start = int(base)
                end = int(base + mt_n)
                for i in (idxs or set()):
                    try:
                        ii = int(i)
                    except Exception:
                        continue
                    if ii < start:
                        out.add(ii)
                    elif ii >= end:
                        out.add(ii - mt_n)
                return out

            try:
                self.excluded_centroid_indices = _remove_manual_block(getattr(self, 'excluded_centroid_indices', set()) or set())
            except Exception:
                pass
            try:
                self._explicit_excluded_centroid_indices = _remove_manual_block(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
            except Exception:
                pass
            try:
                self._force_visible_centroid_indices = _remove_manual_block(getattr(self, '_force_visible_centroid_indices', set()) or set())
            except Exception:
                pass

            try:
                self.manual_targets = []
            except Exception:
                pass
            try:
                auto_only = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())
                self.centroids = self._compose_centroids_with_manual(auto_only)
            except Exception:
                pass

        self.selected_index = None
        try:
            self._sanitize_excluded_indices()
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _current_center_selected_index(self):
        """Return selected centroid index based on middle table selection mapping."""
        try:
            t = getattr(self, 'table_between', None)
            if t is not None:
                r = int(t.currentRow())
                if r >= 2:
                    row = int(r - 2)
                    idxs = list(getattr(self, '_table_between_row_indices', []) or [])
                    if 0 <= row < len(idxs):
                        return int(idxs[row])
        except Exception:
            pass
        try:
            idx = getattr(self, 'selected_index', None)
            return None if idx is None else int(idx)
        except Exception:
            return None

    def _current_center_selected_row(self):
        """Return selected data-row index in middle table (0-based, without header rows)."""
        try:
            t = getattr(self, 'table_between', None)
            if t is None:
                return None
            r = int(t.currentRow())
            if r < 2:
                return None
            rr = int(r - 2)
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not (0 <= rr < len(rows)):
                return None
            return rr
        except Exception:
            return None

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
        # 現在表示中の重心（自動 + 手動）をそのまま出力に使う
        centroids = list(getattr(self, 'centroids', []) or [])
        excluded = set(getattr(self, 'excluded_centroid_indices', set()) or set())
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
                out_no = 0
                for i, cent in enumerate(centroids):
                    if i in excluded:
                        continue
                    out_no += 1
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
                        f.write(f"{out_no},{g},{sx},{sy},{sz}\n")
                    except Exception:
                        try:
                            f.write(f"{out_no},,, ,\n")
                        except Exception:
                            pass
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", f"Saved centroids to:\n{outpath}")
        except Exception as e:
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Error", str(e))

    # ---- Save / Load Project -------------------------------------------

    def save_project(self):
        """Save entire project state to a .pixy file (JSON inside)."""
        import json
        from qt_compat.QtWidgets import QMessageBox
        try:
            last_path = load_last_image_path()
            start_dir = os.path.dirname(last_path) if last_path else os.getcwd()
        except Exception:
            start_dir = os.getcwd()
        outpath, _ = QFileDialog.getSaveFileName(
            self, "Save Project", start_dir,
            "PiXY Project (*.pixy);;JSON Files (*.json);;All Files (*)",
        )
        if not outpath:
            return
        try:
            data = self._collect_project_data()
            # Embed image bytes (always on)
            try:
                import base64, cv2
                img_bytes = None
                # Prefer in-memory image if available
                try:
                    if getattr(self, 'img_full', None) is not None:
                        ok, buf = cv2.imencode('.png', self.img_full)
                        if ok:
                            img_bytes = buf.tobytes()
                except Exception:
                    img_bytes = None
                # Fallback: read from image_path
                if img_bytes is None:
                    try:
                        img_path = data.get('image_path', '')
                        if img_path and os.path.isfile(img_path):
                            with open(img_path, 'rb') as _f:
                                img_bytes = _f.read()
                    except Exception:
                        img_bytes = None
                if img_bytes is not None:
                    # Warn if large
                    try:
                        max_warn = 20 * 1024 * 1024
                        if len(img_bytes) > max_warn:
                            QMessageBox.warning(self, 'Save Project', 'Image is large and will be embedded; this will create a large project file.')
                    except Exception:
                        pass
                    try:
                        data['image_embedded'] = True
                        data['image_filename'] = os.path.basename(data.get('image_path','')) or 'embedded.png'
                        data['image_mime'] = 'application/octet-stream'
                        data['image_data_b64'] = base64.b64encode(img_bytes).decode('ascii')
                    except Exception:
                        pass
            except Exception:
                pass

            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Save Project", f"Saved to:\n{outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_project(self):
        """Load project state from a .pixy / .json file."""
        import json
        from qt_compat.QtWidgets import QMessageBox
        try:
            last_path = load_last_image_path()
            start_dir = os.path.dirname(last_path) if last_path else os.getcwd()
        except Exception:
            start_dir = os.getcwd()
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Project", start_dir,
            "PiXY Project (*.pixy);;JSON Files (*.json);;All Files (*)",
        )
        if not fpath:
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_project_data(data)
            self._apply_load_project_defaults()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _apply_load_project_defaults(self):
        """Apply UI defaults after Load Project.

        - Show the normal On-line Alignment contents by default.
        - Set recalculation trigger to Manual by default.
        """
        try:
            self._set_centroid_extraction_mode(False)
        except Exception:
            pass

        # Force Manual recalculation mode after loading a project
        try:
            self._on_toggle_calc_mode(1)
        except Exception:
            try:
                self.calc_mode = 'manual'
                self.auto_update_mode = False
                self._manual_recompute_request = False
            except Exception:
                pass

    def _on_export_image_clicked(self):
        """Export full-resolution image with centroid markers and index labels."""
        from qt_compat.QtWidgets import QMessageBox
        if self.img_full is None or self.centroid_processor is None:
            try:
                QMessageBox.warning(self, "Export Image", "No image is loaded.")
            except Exception:
                pass
            return

        # Keep consistency with current detection result (cache first, else recompute)
        try:
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
        except Exception:
            centroids = getattr(self, 'centroids', []) or []

        # Output path
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"export_image_{dt_str}.png"
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
                "Export Image",
                start_path,
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;All Files (*)",
            )
        except Exception:
            outpath = ""
        if not outpath:
            return

        # Draw markers on full-res image with FIXED pixel size (independent of image dimensions)
        out_img = self.img_full.copy()
        h_full, w_full = out_img.shape[:2]
        radius_px = 5
        font_scale = 0.55
        text_thickness = 1
        outline_thickness = 3

        for i, c in enumerate(centroids or []):
            try:
                _, xp, yp = c
                xf = float(xp) * float(self.scale_proc_to_full)
                yf = float(yp) * float(self.scale_proc_to_full)
                x = int(round(xf))
                y = int(round(yf))
            except Exception:
                continue
            if not (0 <= x < w_full and 0 <= y < h_full):
                continue

            # centroid marker
            try:
                cv2.circle(out_img, (x, y), int(radius_px), (64, 64, 64), -1, lineType=cv2.LINE_AA)
                cv2.circle(out_img, (x, y), int(radius_px), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            except Exception:
                pass

            # centroid index label (1-based)
            label = str(int(i) + 1)
            tx = int(x + radius_px + 2)
            ty = int(y - radius_px - 2)
            # keep roughly on-screen
            tx = max(0, min(w_full - 1, tx))
            ty = max(0, min(h_full - 1, ty))
            try:
                cv2.putText(out_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(outline_thickness), cv2.LINE_AA)
                cv2.putText(out_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (240, 240, 240), int(text_thickness), cv2.LINE_AA)
            except Exception:
                pass

        # Save with unicode-safe path handling on Windows
        try:
            ext = os.path.splitext(outpath)[1].lower()
            if ext == "":
                outpath = outpath + ".png"
                ext = ".png"
            ok, buf = cv2.imencode(ext, out_img)
            if not ok:
                raise ValueError("Failed to encode output image")
            buf.tofile(outpath)
            QMessageBox.information(self, "Export Image", f"Saved image to:\n{outpath}")
        except Exception as e:
            try:
                QMessageBox.critical(self, "Export Image Error", str(e))
            except Exception:
                pass

    def _collect_project_data(self):
        """Gather all saveable state into a dict."""
        data = {"pixy_version": "1.0", "format": "pixy-project"}

        # 画像パス
        try:
            image_path = str(getattr(self, 'image_path', '') or '')
            if not image_path:
                image_path = str(load_last_image_path() or '')
            data["image_path"] = image_path
        except Exception:
            data["image_path"] = ""

        # スライダー / パラメータ
        try:
            # Use the same source as actual centroid computation.
            p = self._get_params()
            data["levels"] = int(p.get("levels", 4))
        except Exception:
            try:
                data["levels"] = int(getattr(self, 'slider_num_groups', None).value() if getattr(self, 'slider_num_groups', None) is not None else getattr(self, 'levels_value', 4))
            except Exception:
                data["levels"] = 4
        try:
            data["min_area"] = int(self.slider_min_area.value())
        except Exception:
            data["min_area"] = 50
        try:
            data["trim_px"] = int(self.slider_trim.value())
        except Exception:
            data["trim_px"] = 0
        try:
            data["shape_complexity"] = int(self.slider_shape_complex.value())
        except Exception:
            data["shape_complexity"] = 10
        try:
            data["neck_separation"] = int(self.slider_neck_sep.value())
        except Exception:
            data["neck_separation"] = 0

        # 表示設定
        data["overlay_mode"] = str(getattr(self, 'overlay_mode', 'Original'))
        data["show_boundaries"] = bool(getattr(self, 'show_boundaries', True))
        data["flip_mode"] = str(getattr(self, 'flip_mode', 'auto'))
        data["view_orientation"] = str(getattr(self, 'view_orientation', 'Image'))
        data["manual_image_rotation_deg"] = int(getattr(self, 'manual_image_rotation_deg', 0))
        data["grain_ident_mode"] = str(getattr(self, 'grain_ident_mode', 'basic'))
        data["calc_mode"] = str(getattr(self, 'calc_mode', 'auto'))

        # 参照点
        ref_list = []
        for i, pt in enumerate(self.ref_points):
            entry = {"index": i}
            if pt is not None:
                entry["x_proc"] = float(pt[0])
                entry["y_proc"] = float(pt[1])
            else:
                entry["x_proc"] = None
                entry["y_proc"] = None
            obs = self.ref_obs[i] if i < len(self.ref_obs) else {}
            entry["stage_x"] = str(obs.get("x", ""))
            entry["stage_y"] = str(obs.get("y", ""))
            entry["stage_z"] = str(obs.get("z", ""))
            ref_list.append(entry)
        data["ref_points"] = ref_list
        data["visible_ref_cols"] = int(getattr(self, 'visible_ref_cols', 3))

        # 重心リスト
        centroids_list = []
        for g, x, y in (self.centroids or []):
            centroids_list.append({"group": int(g), "x_proc": float(x), "y_proc": float(y)})
        data["centroids"] = centroids_list
        data["manual_target_mode"] = bool(getattr(self, 'manual_target_mode', False))
        mt_list = []
        for g, x, y in (getattr(self, 'manual_targets', []) or []):
            try:
                mt_list.append({"group": int(g), "x_proc": float(x), "y_proc": float(y)})
            except Exception:
                pass
        data["manual_targets"] = mt_list
        try:
            data["center_list_indices"] = [int(i) for i in (getattr(self, 'center_list_indices', []) or [])]
        except Exception:
            data["center_list_indices"] = []
        try:
            rows = []
            for r in (getattr(self, 'center_numeric_rows', []) or []):
                try:
                    rows.append({
                        "source_idx": int(r.get('source_idx', -1)),
                        "grp": float(r.get('grp', float('nan'))),
                        "u": float(r.get('u', float('nan'))),
                        "v": float(r.get('v', float('nan'))),
                        "x": float(r.get('x', float('nan'))),
                        "y": float(r.get('y', float('nan'))),
                        "z": float(r.get('z', float('nan'))),
                        "x_proc": float(r.get('x_proc', float('nan'))),
                        "y_proc": float(r.get('y_proc', float('nan'))),
                        "show": float(r.get('show', 1.0)),
                        "manual": float(r.get('manual', 0.0)),
                    })
                except Exception:
                    continue
            data["center_numeric_rows"] = rows
        except Exception:
            data["center_numeric_rows"] = []
        try:
            data["overlay_point_source"] = str(getattr(self, 'overlay_point_source', 'left'))
        except Exception:
            data["overlay_point_source"] = 'left'
        try:
            # Persist only explicit exclusions (not temporary Filter-driven group hiding)
            src = getattr(self, '_explicit_excluded_centroid_indices', None)
            if src is None:
                src = getattr(self, 'excluded_centroid_indices', set())
            data["excluded_centroid_indices"] = sorted([int(i) for i in (src or set())])
        except Exception:
            data["excluded_centroid_indices"] = []
        try:
            data["excluded_ref_indices"] = sorted([int(i) for i in (getattr(self, 'excluded_ref_indices', set()) or set())])
        except Exception:
            data["excluded_ref_indices"] = []
        data["selected_index"] = self.selected_index

        # スケール情報
        data["scale_proc_to_full"] = float(getattr(self, 'scale_proc_to_full', 1.0))
        data["proc_target_width"] = int(getattr(self, 'proc_target_width', 640))

        return data

    def _apply_project_data(self, data):
        """Restore state from a loaded project dict."""
        from qt_compat.QtWidgets import QMessageBox

        # 画像を復元（埋め込み画像を最優先）
        image_loaded = False
        embedded_load_error = None
        try:
            if bool(data.get("image_embedded", False)) and data.get("image_data_b64"):
                import base64
                raw = base64.b64decode(str(data.get("image_data_b64", "")))
                arr = np.frombuffer(raw, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError("embedded image decode failed")

                self.img_full = decoded
                try:
                    self.image_path = str(data.get("image_path", "") or "")
                except Exception:
                    self.image_path = ""
                self._build_processing_image()
                self._cache = {"img_id": id(self.proc_img), "levels": None, "min_area": None, "trim_px": None, "poster": None, "centroids": None}
                self._initial_center_done = False
                image_loaded = True
        except Exception as e:
            embedded_load_error = e

        # 画像パス参照でフォールバック
        img_path = data.get("image_path", "")
        if not image_loaded and img_path and os.path.isfile(img_path):
            self._open_image_from_path(img_path)
            image_loaded = True
        elif not image_loaded and img_path:
            QMessageBox.warning(self, "Load Project",
                f"Image not found (skipped):\n{img_path}")

        # 埋め込みがあったのに復元に失敗した場合は理由を通知（後方互換で処理は継続）
        if (not image_loaded) and bool(data.get("image_embedded", False)) and embedded_load_error is not None:
            try:
                QMessageBox.warning(
                    self,
                    "Load Project",
                    f"Embedded image restore failed.\n{embedded_load_error}"
                )
            except Exception:
                pass

        # スライダー復元
        try:
            lv = int(data.get("levels", 4))
            # Basic mode control (Number of Groups)
            if getattr(self, 'slider_num_groups', None) is not None:
                v_num = max(self.slider_num_groups.minimum(), min(self.slider_num_groups.maximum(), lv))
                self.slider_num_groups.setValue(v_num)
                try:
                    self.edit_num_groups.setText(str(int(v_num)))
                except Exception:
                    pass
            # Advanced mode control (PosterLevel)
            self.levels_value = lv
            if getattr(self, 'slider_levels', None) is not None:
                self.slider_levels.setValue(max(self.slider_levels.minimum(), min(self.slider_levels.maximum(), lv)))
            try:
                self.edit_levels.setText(str(lv))
            except Exception:
                pass
        except Exception:
            pass
        for attr, key, default in [
            ('slider_min_area', 'min_area', 50),
            ('slider_trim', 'trim_px', 0),
            ('slider_shape_complex', 'shape_complexity', 10),
            ('slider_neck_sep', 'neck_separation', 0),
        ]:
            try:
                sl = getattr(self, attr, None)
                if sl is not None:
                    v = int(data.get(key, default))
                    v = max(sl.minimum(), min(sl.maximum(), v))
                    sl.setValue(v)
            except Exception:
                pass

        # 表示設定復元
        try:
            self.overlay_mode = str(data.get("overlay_mode", "Original"))
        except Exception:
            pass
        try:
            self.show_boundaries = bool(data.get("show_boundaries", True))
        except Exception:
            pass
        try:
            self.flip_mode = str(data.get("flip_mode", "auto"))
        except Exception:
            pass
        try:
            self.view_orientation = str(data.get("view_orientation", "Image"))
        except Exception:
            pass
        # Coordinate(Image/Stage) の復元は、表示フラグだけでなく関連UI状態
        # (Rotate有効/無効、表示行の出し分け) まで同期する。
        try:
            vo = str(getattr(self, 'view_orientation', 'Image'))
            idx = 0 if vo.lower() == 'image' else 1
            if getattr(self, 'view_orientation_toggle', None) is not None:
                try:
                    self.view_orientation_toggle.setCheckedIndex(int(idx))
                except Exception:
                    pass
            self._on_toggle_coordinate(int(idx))
        except Exception:
            pass
        try:
            deg = int(data.get("manual_image_rotation_deg", 0))
            self.manual_image_rotation_deg = deg
            if getattr(self, 'slider_img_rotate', None) is not None:
                try:
                    self.slider_img_rotate.blockSignals(True)
                    self.slider_img_rotate.setValue(deg)
                finally:
                    self.slider_img_rotate.blockSignals(False)
            if getattr(self, 'lbl_rot_val', None) is not None:
                try:
                    self.lbl_rot_val.setText(f"{int(deg)}°")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.grain_ident_mode = 'advanced'
        except Exception:
            pass
        try:
            mode_raw = str(data.get("calc_mode", "auto")).strip().lower()
            mode = 'manual' if mode_raw == 'manual' else 'auto'
            self.calc_mode = mode
            self.auto_update_mode = (mode == 'auto')
            try:
                self._manual_recompute_request = False
            except Exception:
                pass
            # Sync segmented control UI with internal mode without triggering handler side effects.
            try:
                tcm = getattr(self, 'toggle_calc_mode', None)
                if tcm is not None:
                    tcm.setCheckedIndex(1 if mode == 'manual' else 0)
                    buttons = getattr(tcm, '_buttons', [None, None])
                    btn1 = buttons[1] if len(buttons) > 1 else None
                    if btn1 is not None:
                        try:
                            btn1.setProperty('pixy_calc_in_progress', False)
                        except Exception:
                            pass
                        btn1.setText('ReCalculate' if mode == 'manual' else 'Manual')
            except Exception:
                pass
            # Reset per-mode snapshots to avoid stale state from previous session.
            try:
                self._calc_params_by_mode = {'auto': None, 'manual': None}
            except Exception:
                pass
        except Exception:
            pass

        # 参照点復元
        ref_data = data.get("ref_points", [])
        for entry in ref_data:
            try:
                i = int(entry["index"])
                if i < 0 or i >= len(self.ref_points):
                    continue
                xp = entry.get("x_proc")
                yp = entry.get("y_proc")
                if xp is not None and yp is not None:
                    self.ref_points[i] = (float(xp), float(yp))
                else:
                    self.ref_points[i] = None
                if i < len(self.ref_obs):
                    self.ref_obs[i] = {
                        "x": str(entry.get("stage_x", "")),
                        "y": str(entry.get("stage_y", "")),
                        "z": str(entry.get("stage_z", "")),
                    }
            except Exception:
                pass
        try:
            self.visible_ref_cols = int(data.get("visible_ref_cols", 3))
        except Exception:
            pass

        # 重心復元
        clist = data.get("centroids", [])
        restored = []
        for c in clist:
            try:
                restored.append((int(c["group"]), float(c["x_proc"]), float(c["y_proc"])))
            except Exception:
                pass
        self.centroids = restored
        try:
            self._auto_centroids = list(restored)
        except Exception:
            self._auto_centroids = []
        try:
            self.manual_target_mode = bool(data.get("manual_target_mode", False))
        except Exception:
            self.manual_target_mode = False
        mt_restored = []
        for c in (data.get("manual_targets", []) or []):
            try:
                mt_restored.append((int(c["group"]), float(c["x_proc"]), float(c["y_proc"])))
            except Exception:
                pass
        self.manual_targets = mt_restored
        self.centroids = self._compose_centroids_with_manual(self._auto_centroids)
        try:
            self.center_list_indices = [int(i) for i in (data.get("center_list_indices", []) or [])]
        except Exception:
            self.center_list_indices = []
        try:
            rows = []
            for r in (data.get("center_numeric_rows", []) or []):
                try:
                    rows.append({
                        "source_idx": int(r.get('source_idx', -1)),
                        "grp": float(r.get('grp', float('nan'))),
                        "u": float(r.get('u', float('nan'))),
                        "v": float(r.get('v', float('nan'))),
                        "x": float(r.get('x', float('nan'))),
                        "y": float(r.get('y', float('nan'))),
                        "z": float(r.get('z', float('nan'))),
                        "x_proc": float(r.get('x_proc', float('nan'))),
                        "y_proc": float(r.get('y_proc', float('nan'))),
                        "show": float(r.get('show', 1.0)),
                        "manual": float(r.get('manual', 0.0)),
                    })
                except Exception:
                    continue
            self.center_numeric_rows = rows
        except Exception:
            self.center_numeric_rows = []
        try:
            if not (getattr(self, 'center_numeric_rows', None) or []):
                self._append_center_numeric_rows_from_indices(self._get_center_list_indices())
        except Exception:
            pass
        try:
            src = str(data.get("overlay_point_source", 'left') or 'left').lower()
            self.overlay_point_source = 'center' if src == 'center' else 'left'
        except Exception:
            self.overlay_point_source = 'left'
        try:
            tog = getattr(self, 'toggle_overlay_source', None)
            if tog is not None:
                tog.setCheckedIndex(1 if str(getattr(self, 'overlay_point_source', 'left')) == 'center' else 0)
        except Exception:
            pass
        try:
            self.excluded_centroid_indices = set(int(i) for i in (data.get("excluded_centroid_indices", []) or []))
        except Exception:
            self.excluded_centroid_indices = set()
        try:
            # Treat loaded exclusions as explicit (Filter state is not persisted)
            self._explicit_excluded_centroid_indices = set(self.excluded_centroid_indices)
        except Exception:
            self._explicit_excluded_centroid_indices = set()
        try:
            self.excluded_ref_indices = set(int(i) for i in (data.get("excluded_ref_indices", []) or []))
        except Exception:
            self.excluded_ref_indices = set()
        self._sanitize_excluded_indices()
        self.selected_index = data.get("selected_index")
        try:
            # Keep cache in sync so manual/no-recompute paths reuse loaded results.
            if isinstance(getattr(self, '_cache', None), dict):
                self._cache['img_id'] = id(getattr(self, 'proc_img', None))
                self._cache['levels'] = int(data.get("levels", self._cache.get('levels', 4)))
                self._cache['min_area'] = int(data.get("min_area", self._cache.get('min_area', 50)))
                self._cache['trim_px'] = int(data.get("trim_px", self._cache.get('trim_px', 0)))
                self._cache['centroids'] = list(self.centroids or [])
        except Exception:
            pass

        # テーブル・画面を更新
        try:
            self._safe_populate_tables(
                self.table_ref, self.table,
                self.ref_points, self.ref_obs,
                self.centroids, self.selected_index,
                self.ref_selected_index,
                flip_mode=self.flip_mode,
                visible_ref_cols=self.visible_ref_cols,
            )
        except Exception:
            pass
        # Load 後はポスター・boundary_mask を全パラメータで再計算する。
        # Advanced パラメータ (trim, neck, shape) が正しく反映された boundary を描画するため。
        self._manual_recompute_request = True
        self.schedule_update(force=True, recompute_centroids=True)

    # ---- end Save / Load Project -----------------------------------------

    def _on_table_current_changed(self, curRow, curCol, prevRow, prevCol):
        if curCol is None or curCol < 0:
            return
        # 右テーブルはデータ列のみ（オフセット無し）
        idx = curCol
        if self.selected_index != idx:
            self.selected_index = idx
            # Selection-only update: redraw highlight/table sync without heavy recomputation.
            self.schedule_update(force=True, recompute_centroids=False)
        try:
            self._center_on_centroid_index(idx)
        except Exception:
            pass

    def _on_table_between_current_changed(self, curRow, curCol, prevRow, prevCol):
        # transposed view row maps to original table column (selected centroid index)
        try:
            if bool(getattr(self, 'centroid_extraction_mode', False)):
                return
            if curRow is None or curRow < 0:
                return
            header_rows = 2
            if int(curRow) < header_rows:
                return
            row = int(curRow) - header_rows
            idxs = list(getattr(self, '_table_between_row_indices', []) or [])
            if not (0 <= row < len(idxs)):
                return
            idx = int(idxs[row])
            if self.selected_index != idx:
                self.selected_index = idx
                # Selection-only update: redraw highlight/table sync without heavy recomputation.
                self.schedule_update(force=True, recompute_centroids=False)
            try:
                self._center_on_centroid_index(idx)
            except Exception:
                pass
        except Exception:
            pass

    def _on_table_between_cell_clicked(self, row, col):
        # Show column toggle is handled by the cell widget (SegmentControl), not here.
        pass

    def eventFilter(self, obj, event):
        # UI 側ではイベント処理を行わず、標準の処理へ委譲
        return super().eventFilter(obj, event)

    def _on_ref_cell_clicked(self, row, col):
        # 左テーブルクリック時に、Stage.X/Y/Z(行2,3,4)なら即編集を開始する
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

        # Use unified mapping (rotation/flip + bbox shift) when available
        mapping = getattr(self, '_display_mapping', None)
        try:
            view_orient = getattr(self, 'view_orientation', 'Image')
        except Exception:
            view_orient = 'Image'

        if mapping is not None and mapping.get('mode') == view_orient:
            map_type = str(mapping.get('type', '')).strip().lower()
            # Preferred: use Qt transform mapping (exactly matches rendered pixmap)
            try:
                if map_type == 'qt' and mapping.get('qt') is not None and mapping.get('qt_inv') is not None:
                    from qt_compat.QtCore import QPointF
                    pad = float(mapping.get('pad', 0.0))
                    z_map = float(mapping.get('z', z))
                    draw_w = float(mapping.get('draw_w', 0.0))
                    draw_h = float(mapping.get('draw_h', 0.0))
                    shift_x = float(mapping.get('shift_x', 0.0))
                    shift_y = float(mapping.get('shift_y', 0.0))
                    inv = mapping.get('qt_inv')

                    # trueMatrix already incorporates the shift, so only subtract pad.
                    x_rot = float(pos.x()) - pad
                    y_rot = float(pos.y()) - pad

                    p0 = inv.map(QPointF(x_rot, y_rot))
                    x_pix = float(p0.x())
                    y_pix = float(p0.y())

                    # Allow clicks up to ~6 display pixels outside the pre-rotation
                    # image boundary.  Rotation preserves distances, so this is also
                    # ~6 pixels in label space — enough for SmoothTransformation
                    # anti-aliasing (1-2 px) and trueMatrix integer-rounding residual.
                    # Clicks in dark bounding-box corners are typically 50+ px outside,
                    # so they are still rejected.
                    max_outside = 6.0
                    dx_out = max(0.0, -x_pix, x_pix - draw_w)
                    dy_out = max(0.0, -y_pix, y_pix - draw_h)
                    if max(dx_out, dy_out) > max_outside:
                        # Outside image — do NOT fall through to rotation-unaware fallback.
                        return None

                    x_pix = min(max(x_pix, 0.0), draw_w)
                    y_pix = min(max(y_pix, 0.0), draw_h)
                    x_full = x_pix / max(1e-12, z_map)
                    y_full = y_pix / max(1e-12, z_map)
                    if 0 <= x_full <= img_w and 0 <= y_full <= img_h:
                        mxy = (x_full, y_full)
                    else:
                        mxy = None
                    if mxy is not None:
                        return mxy

                    # Qt mapping active but click is outside the rotated image area
                    # (e.g. dark corner). Do NOT fall through to the rotation-unaware
                    # fallback — that would place the point at a completely wrong position.
                    return None
            except Exception:
                # Qt mapping failed unexpectedly. Still do not fall through.
                if map_type == 'qt':
                    return None

            if map_type != 'qt':
                try:
                    import math
                    pad = float(mapping.get('pad', 0.0))
                    min_x = float(mapping.get('min_x', 0.0))
                    min_y = float(mapping.get('min_y', 0.0))
                    cx = float(mapping.get('cx', 0.0))
                    cy = float(mapping.get('cy', 0.0))
                    theta = float(mapping.get('theta', 0.0))
                    flip = bool(mapping.get('flip', False))
                    draw_w = float(mapping.get('draw_w', 0.0))
                    draw_h = float(mapping.get('draw_h', 0.0))

                    # label -> rotated canvas coords (inside image area)
                    x_rot = float(pos.x()) - pad
                    y_rot = float(pos.y()) - pad
                    if x_rot < 0 or y_rot < 0:
                        raise ValueError('outside')

                    # rotated bbox origin -> center-relative coords
                    xr = x_rot + min_x
                    yr = y_rot + min_y

                    c = math.cos(theta)
                    s = math.sin(theta)
                    # inverse rotate
                    x_rel = xr * c + yr * s
                    y_rel = -xr * s + yr * c
                    # inverse flip (horizontal)
                    if flip:
                        x_rel = -x_rel

                    x_pix = x_rel + cx
                    y_pix = y_rel + cy
                    if not (0.0 <= x_pix <= draw_w and 0.0 <= y_pix <= draw_h):
                        raise ValueError('outside')

                    x_full = x_pix / z
                    y_full = y_pix / z
                    if 0 <= x_full <= img_w and 0 <= y_full <= img_h:
                        return x_full, y_full
                except Exception:
                    pass

        # fallback (no rotation mapping)
        x_full = (pos.x() - off_x) / z
        y_full = (pos.y() - off_y) / z
        if not (0 <= x_full <= img_w and 0 <= y_full <= img_h):
            return None
        return x_full, y_full

    def _full_to_display(self, x_full, y_full):
        # フル画像座標からラベル座標へ（ズーム＋回転/反転を含む）
        if self._img_base_size is None:
            return None
        z = max(0.0001, float(getattr(self, '_display_scale', max(0.1, float(self.proc_zoom)))))
        off_x, off_y = self._display_offset

        mapping = getattr(self, '_display_mapping', None)
        try:
            view_orient = getattr(self, 'view_orientation', 'Image')
        except Exception:
            view_orient = 'Image'

        if mapping is not None and mapping.get('mode') == view_orient:
            map_type = str(mapping.get('type', '')).strip().lower()
            # Preferred: use Qt transform mapping (exactly matches rendered pixmap)
            try:
                if map_type == 'qt' and mapping.get('qt') is not None:
                    from qt_compat.QtCore import QPointF
                    pad = float(mapping.get('pad', 0.0))
                    z_map = float(mapping.get('z', z))
                    shift_x = float(mapping.get('shift_x', 0.0))
                    shift_y = float(mapping.get('shift_y', 0.0))
                    tr = mapping.get('qt')

                    x_pix = float(x_full) * max(1e-12, z_map)
                    y_pix = float(y_full) * max(1e-12, z_map)
                    p1 = tr.map(QPointF(x_pix, y_pix))
                    # trueMatrix already incorporates the shift; just add pad.
                    return float(p1.x()) + pad, float(p1.y()) + pad
            except Exception:
                if map_type == 'qt':
                    return None

            if map_type != 'qt':
                try:
                    import math
                    pad = float(mapping.get('pad', 0.0))
                    min_x = float(mapping.get('min_x', 0.0))
                    min_y = float(mapping.get('min_y', 0.0))
                    cx = float(mapping.get('cx', 0.0))
                    cy = float(mapping.get('cy', 0.0))
                    theta = float(mapping.get('theta', 0.0))
                    flip = bool(mapping.get('flip', False))

                    x_pix = float(x_full) * z
                    y_pix = float(y_full) * z
                    x_rel = x_pix - cx
                    y_rel = y_pix - cy
                    if flip:
                        x_rel = -x_rel

                    c = math.cos(theta)
                    s = math.sin(theta)
                    xr = x_rel * c - y_rel * s
                    yr = x_rel * s + y_rel * c

                    x_rot = xr - min_x
                    y_rot = yr - min_y
                    return x_rot + pad, y_rot + pad
                except Exception:
                    pass

        # fallback
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
            if str(mode) in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                # pick モード開始時は対応ボタンをキャンセル表示にする
                if str(mode) == 'add':
                    target_btn = self.btn_add_ref
                elif str(mode) == 'update':
                    target_btn = getattr(self, 'btn_update_xy', None)
                elif str(mode) == 'target_add':
                    target_btn = getattr(self, 'btn_add_target', None)
                else:
                    target_btn = getattr(self, 'btn_update_target_uv', None)
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

    def _end_pick_mode(self, redraw: bool = True):
        self.pick_mode = None
        self.pick_ref_index = None
        self._replace_target_source_index = None
        self._ref_add_has_added = False
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
            btn_tadd = getattr(self, 'btn_add_target', None)
            if btn_tadd is not None:
                btn_tadd.setText("Add Target")
        except Exception:
            pass
        try:
            btn_tup = getattr(self, 'btn_update_target_uv', None)
            if btn_tup is not None:
                btn_tup.setText("Update u, v")
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
        if redraw:
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

        if self.pick_mode == 'center_uv_update':
            try:
                if self.scale_proc_to_full == 0:
                    return
                x_proc = x_full / self.scale_proc_to_full
                y_proc = y_full / self.scale_proc_to_full
                u, v = self._center_uv_from_proc(float(x_proc), float(y_proc))

                row = self.pick_ref_index if self.pick_ref_index is not None else self._current_center_selected_row()
                if row is None:
                    return
                row = int(row)
                rows = list(getattr(self, 'center_numeric_rows', []) or [])
                if not (0 <= row < len(rows)):
                    return
                rd = dict(rows[row] or {})
                rd['u'] = float(u)
                rd['v'] = float(v)
                rd['x_proc'] = float(x_proc)
                rd['y_proc'] = float(y_proc)
                rows[row] = rd
                self.center_numeric_rows = rows
                try:
                    si = int(rd.get('source_idx', -1))
                    if si >= 0:
                        self.selected_index = si
                except Exception:
                    pass
                try:
                    self._end_pick_mode(redraw=False)
                except Exception:
                    pass
                try:
                    self.schedule_update(force=True, recompute_centroids=False)
                except Exception:
                    pass
            except Exception:
                pass
            return

        # ピックモード中は、マウス位置の座標をRefに保存して終了
        if self.pick_mode in ('target_add', 'target_update'):
            if self.scale_proc_to_full == 0:
                return
            x_proc = x_full / self.scale_proc_to_full
            y_proc = y_full / self.scale_proc_to_full
            center_full_before = None
            try:
                if getattr(self, 'proc_scroll', None) is not None:
                    vp = self.proc_scroll.viewport()
                    pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                    pos_label = self._viewport_pos_to_label_pos(pos_vp)
                    center_full_before = self._display_to_full(pos_label)
            except Exception:
                center_full_before = None
            try:
                if getattr(self, 'manual_targets', None) is None:
                    self.manual_targets = []
                auto_only = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())

                if self.pick_mode == 'target_add':
                    base = self._manual_target_base_index()
                    mt_old_n = len(self.manual_targets)
                    insert_idx = int(base + mt_old_n)
                    try:
                        add_grp = int(self._get_add_target_group())
                    except Exception:
                        add_grp = 0
                    try:
                        self._shift_center_list_indices(insert_idx, +1)
                    except Exception:
                        pass
                    self.manual_targets.append((int(add_grp), float(x_proc), float(y_proc)))
                    try:
                        if hasattr(self, '_log_info'):
                            self._log_info(
                                f"AddTarget: added idx={insert_idx} x_proc={float(x_proc):.3f} y_proc={float(y_proc):.3f}"
                            )
                    except Exception:
                        pass
                    try:
                        old_excl = set(getattr(self, 'excluded_centroid_indices', set()) or set())
                        old_exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
                        new_excl = set()
                        new_exp = set()
                        for i in old_excl:
                            ii = int(i)
                            if ii >= insert_idx:
                                ii += 1
                            new_excl.add(ii)
                        for i in old_exp:
                            ii = int(i)
                            if ii >= insert_idx:
                                ii += 1
                            new_exp.add(ii)
                        src_idx = getattr(self, '_replace_target_source_index', None)
                        if src_idx is not None:
                            try:
                                src_new = int(src_idx)
                                if src_new >= insert_idx:
                                    src_new += 1
                                new_excl.add(src_new)
                                new_exp.add(src_new)
                            except Exception:
                                pass
                        self.excluded_centroid_indices = new_excl
                        self._explicit_excluded_centroid_indices = new_exp
                    except Exception:
                        pass
                    try:
                        old_force = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
                        new_force = set()
                        for i in old_force:
                            ii = int(i)
                            if ii >= insert_idx:
                                ii += 1
                            new_force.add(ii)

                        src_idx = getattr(self, '_replace_target_source_index', None)
                        src_grp = getattr(self, '_replace_target_source_group', None)
                        vis = self._get_visible_groups_set()
                        hide_g0 = (vis is not None and 0 not in vis)
                        if src_idx is not None and src_grp is not None and int(src_grp) != 0 and hide_g0:
                            new_force.add(insert_idx)

                        self._force_visible_centroid_indices = new_force
                    except Exception:
                        pass
                    self.centroids = self._compose_centroids_with_manual(auto_only)
                    self.selected_index = insert_idx
                    try:
                        self._append_center_numeric_rows_from_indices([insert_idx])
                    except Exception:
                        pass
                else:
                    idx_t = self.pick_ref_index if self.pick_ref_index is not None else self._selected_manual_target_index()
                    if idx_t is None:
                        return
                    idx_t = int(idx_t)
                    if not (0 <= idx_t < len(self.manual_targets)):
                        return
                    self.manual_targets[idx_t] = (0, float(x_proc), float(y_proc))
                    self.centroids = self._compose_centroids_with_manual(auto_only)
                    base = self._manual_target_base_index()
                    self.selected_index = int(base + idx_t)

                self._safe_populate_tables(
                    self.table_ref, self.table,
                    self.ref_points, self.ref_obs,
                    self.centroids, self.selected_index,
                    self.ref_selected_index,
                    flip_mode=self.flip_mode,
                    visible_ref_cols=self.visible_ref_cols,
                )
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass
                try:
                    if self.pick_mode == 'target_update':
                        self._end_pick_mode(redraw=False)
                    elif self.pick_mode == 'target_add':
                        # Stay in add mode; will end when cursor leaves the image.
                        self._target_add_has_added = True
                except Exception:
                    pass
                self._replace_target_source_index = None
                self._replace_target_source_group = None
                try:
                    # Cancel pending wheel-settle recenter to avoid post-add viewport jumps.
                    ivc = getattr(self, 'interactions', None)
                    if ivc is not None:
                        try:
                            if getattr(ivc, '_wheel_settle_timer', None) is not None:
                                ivc._wheel_settle_timer.stop()
                        except Exception:
                            pass
                        try:
                            if getattr(ivc, '_wheel_zoom_timer', None) is not None:
                                ivc._wheel_zoom_timer.stop()
                        except Exception:
                            pass
                        try:
                            ivc._wheel_zoom_pending = False
                            ivc._wheel_zoom_target = None
                            ivc._wheel_zoom_anchor_full = None
                            ivc._wheel_zoom_anchor_vp = None
                        except Exception:
                            pass

                    self._apply_proc_zoom()
                    if center_full_before is not None:
                        try:
                            self._ensure_full_pos_visible(float(center_full_before[0]), float(center_full_before[1]))
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            return

        if self.pick_mode in ('add', 'update'):
            if self.scale_proc_to_full != 0:
                x_proc = x_full / self.scale_proc_to_full
                y_proc = y_full / self.scale_proc_to_full
                center_full_before = None
                try:
                    if getattr(self, 'proc_scroll', None) is not None:
                        vp = self.proc_scroll.viewport()
                        pos_vp = QPoint(int(vp.width() // 2), int(vp.height() // 2))
                        pos_label = self._viewport_pos_to_label_pos(pos_vp)
                        center_full_before = self._display_to_full(pos_label)
                except Exception:
                    center_full_before = None
                idx = self.pick_ref_index if self.pick_ref_index is not None else self.ref_selected_index
                if idx is not None and 0 <= idx < len(self.ref_points):
                    self.ref_points[idx] = (x_proc, y_proc)
                    # Add Fiducial は1点追加後も継続し、次の空きRefへ自動移動する。
                    if self.pick_mode == 'add':
                        try:
                            self._ref_add_has_added = True
                        except Exception:
                            pass
                        try:
                            next_idx = None
                            for i, pt in enumerate(self.ref_points):
                                if pt is None:
                                    next_idx = int(i)
                                    break
                            if next_idx is not None:
                                self.pick_ref_index = int(next_idx)
                                self.ref_selected_index = int(next_idx)
                        except Exception:
                            pass

                    # すぐに左表へ反映（populate_tables が遅延しても X/Y を見せる）
                    try:
                        from qt_compat.QtWidgets import QTableWidgetItem
                        from qt_compat.QtCore import Qt as _Qt

                        # Display convention (must match tables.populate_tables):
                        # u = x_full, v = (h_full - 1) - y_full  (origin bottom-left, +v upward)
                        try:
                            h_full = None
                            try:
                                if getattr(self, '_img_base_size', None) is not None:
                                    h_full = int(self._img_base_size[1])
                            except Exception:
                                h_full = None
                            if h_full is None or h_full <= 0:
                                try:
                                    if getattr(self, 'img_full', None) is not None:
                                        h_full = int(self.img_full.shape[0])
                                except Exception:
                                    h_full = None

                            u_disp = int(round(float(x_full)))
                            if h_full is not None and h_full > 0:
                                v_disp = int(round(float(h_full - 1) - float(y_full)))
                            else:
                                v_disp = int(round(-float(y_full)))
                        except Exception:
                            u_disp, v_disp = None, None

                        # Debug: log mapping chain for clicked point
                        try:
                            try:
                                px = int(pos.x()) if hasattr(pos, 'x') else pos[0]
                                py = int(pos.y()) if hasattr(pos, 'y') else pos[1]
                            except Exception:
                                px = py = None
                            self._dbg(f"CLICK debug: label_pos={px},{py} -> full={x_full:.3f},{y_full:.3f} -> proc={x_proc:.3f},{y_proc:.3f} -> u,v={u_disp},{v_disp} img_base={getattr(self,'_img_base_size',None)} scale_proc_to_full={getattr(self,'scale_proc_to_full',None)}")
                            try:
                                if hasattr(self, '_log_info'):
                                    self._log_info(f"CLICK ref[{idx}] label_pos={px},{py} full={x_full:.3f},{y_full:.3f} proc={x_proc:.3f},{y_proc:.3f} u,v={u_disp},{v_disp} img_base={getattr(self,'_img_base_size',None)} scale_proc_to_full={getattr(self,'scale_proc_to_full',None)}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        xi = str(u_disp) if u_disp is not None else ""
                        yi = str(v_disp) if v_disp is not None else ""

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
                    # End pick mode first so _apply_proc_zoom() won't re-draw the crosshair.
                    try:
                        if self.pick_mode == 'update':
                            self._end_pick_mode(redraw=False)
                    except Exception:
                        pass
                    # Redraw once to reflect the newly added/updated ref point.
                    try:
                        # Cancel pending wheel-settle recenter to avoid post-click viewport jumps.
                        ivc = getattr(self, 'interactions', None)
                        if ivc is not None:
                            try:
                                if getattr(ivc, '_wheel_settle_timer', None) is not None:
                                    ivc._wheel_settle_timer.stop()
                            except Exception:
                                pass
                            try:
                                if getattr(ivc, '_wheel_zoom_timer', None) is not None:
                                    ivc._wheel_zoom_timer.stop()
                            except Exception:
                                pass
                            try:
                                ivc._wheel_zoom_pending = False
                                ivc._wheel_zoom_target = None
                                ivc._wheel_zoom_anchor_full = None
                                ivc._wheel_zoom_anchor_vp = None
                            except Exception:
                                pass

                        self._apply_proc_zoom()
                        if center_full_before is not None:
                            try:
                                self._ensure_full_pos_visible(float(center_full_before[0]), float(center_full_before[1]))
                            except Exception:
                                pass
                    except Exception:
                        pass

    def _on_ref_item_changed(self, item):
        # 左テーブル（Ref）の Stage.* 行（2,3,4行目）入力を半角へ正規化し、内部配列に反映
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

                    # Update internal ref_obs immediately when editing Stage rows (2,3,4)
                    try:
                        # canonical rows: X,Y,StageX,StageY,StageZ,... start at src_row_offset
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
        # move to next editable cell: Stage X (col 2) -> Stage Y (col3), Stage Y -> Stage Z (col4),
        # Stage Z -> next row Stage X.
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
                            # editable Stage columns in transposed view are 2,3,4
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
            # Only Stage columns are editable in the view
            editable_cols = (2, 3, 4)  # StageX, StageY, StageZ in transposed view
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
            excluded = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            out_no = 0
            for c in range(tbl.columnCount()):
                if c in excluded:
                    continue
                out_no += 1
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
                    lines.append(f"{out_no}\t{grp}\t{sx}\t{sy}\t{sz}")
                except Exception:
                    lines.append(f"{out_no}\t\t\t\t")
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

            # Inject image coordinate context for u/v display (bottom-left origin) unless provided.
            try:
                if 'image_base_size' not in kwargs:
                    kwargs['image_base_size'] = getattr(self, '_img_base_size', None)
                if 'scale_proc_to_full' not in kwargs:
                    kwargs['scale_proc_to_full'] = getattr(self, 'scale_proc_to_full', 1.0)
                if 'excluded_ref_indices' not in kwargs:
                    kwargs['excluded_ref_indices'] = getattr(self, 'excluded_ref_indices', set()) or set()
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
            # Inject image coordinate context for u/v display (bottom-left origin) unless provided.
            try:
                if 'image_base_size' not in kwargs:
                    kwargs['image_base_size'] = getattr(self, '_img_base_size', None)
                if 'scale_proc_to_full' not in kwargs:
                    kwargs['scale_proc_to_full'] = getattr(self, 'scale_proc_to_full', 1.0)
                if 'excluded_ref_indices' not in kwargs:
                    kwargs['excluded_ref_indices'] = getattr(self, 'excluded_ref_indices', set()) or set()
            except Exception:
                pass
            populate_tables(*args, **kwargs)
            # Reinstall pseudo-headers after populate (data might overwrite them)
            try:
                self._setup_pseudo_headers_ref(self.table_ref)
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
            except Exception:
                pass
            # Auto-shrink fonts so long XYZ values (e.g., 5+ digits) don't clip.
            try:
                QTimer.singleShot(220, self._auto_fit_table_fonts)
            except Exception:
                pass
        except Exception:
            pass

    def _auto_fit_table_fonts(self):
        """Reduce table cell font size to avoid clipping long numeric strings.

        Applies to canonical and transposed tables. Header fonts are set explicitly
        elsewhere, so shrinking the table font mainly affects the items.
        """
        try:
            if getattr(self, '_auto_fit_fonts_running', False):
                return
            self._auto_fit_fonts_running = True
        except Exception:
            pass
        try:
            tables = []
            try:
                tables.append((getattr(self, 'table_ref', None), 2))
            except Exception:
                pass
            try:
                tables.append((getattr(self, 'table', None), 2))
            except Exception:
                pass
            try:
                tables.append((getattr(self, 'table_ref_view', None), 2))
            except Exception:
                pass
            try:
                tables.append((getattr(self, 'table_between', None), 2))
            except Exception:
                pass

            for tbl, row_start in tables:
                try:
                    self._auto_fit_table_font(tbl, row_start=row_start, min_pt=8, max_pt=12)
                except Exception:
                    pass
        finally:
            try:
                self._auto_fit_fonts_running = False
            except Exception:
                pass

    def _auto_fit_table_font(self, tbl, row_start=0, min_pt=8, max_pt=12):
        """Choose a font size so that item texts fit within each column."""
        try:
            if tbl is None:
                return
            rows = int(getattr(tbl, 'rowCount')())
            cols = int(getattr(tbl, 'columnCount')())
            if rows <= 0 or cols <= 0:
                return
        except Exception:
            return

        try:
            from qt_compat.QtGui import QFont, QFontMetrics
        except Exception:
            return

        try:
            base_font = QFont(tbl.font())
        except Exception:
            base_font = QFont()

        try:
            base_pt = int(base_font.pointSize()) if base_font is not None else int(max_pt)
        except Exception:
            base_pt = int(max_pt)
        try:
            base_pt = max(int(min_pt), min(int(max_pt), int(base_pt) if int(base_pt) > 0 else int(max_pt)))
        except Exception:
            base_pt = int(max_pt)

        # If content is empty or no potentially-wide strings, keep current.
        try:
            has_wide_text = False
            for r in range(int(row_start), rows):
                for c in range(cols):
                    it = tbl.item(r, c)
                    if it is None:
                        continue
                    t = str(it.text() or "").strip()
                    if not t:
                        continue
                    # focus on values that are likely to clip
                    if len(t) >= 5 or ('.' in t) or ('-' in t):
                        has_wide_text = True
                        break
                if has_wide_text:
                    break
            if not has_wide_text:
                return
        except Exception:
            pass

        # Determine best size by trying from current/max down to min.
        chosen = None
        try:
            # allow growing back up when values shrink
            start_pt = int(max_pt)
        except Exception:
            start_pt = int(max_pt)

        def _fits(pt):
            try:
                # conservative padding per cell
                pad = 10
                for c in range(cols):
                    try:
                        avail = int(tbl.columnWidth(c)) - pad
                    except Exception:
                        avail = None
                    if avail is None or avail <= 6:
                        continue
                    for r in range(int(row_start), rows):
                        it = tbl.item(r, c)
                        if it is None:
                            continue
                        t = str(it.text() or "")
                        if t.strip() == "":
                            continue
                        # Skip small texts to keep this light
                        try:
                            ts = t.strip()
                            if len(ts) < 5 and ('.' not in ts) and ('-' not in ts):
                                continue
                        except Exception:
                            pass
                        # Measure using the *item's* font (bold etc), falling back to table font.
                        try:
                            f_item = QFont(it.font())
                        except Exception:
                            f_item = QFont(base_font)
                        try:
                            f_item.setPointSize(int(pt))
                        except Exception:
                            pass
                        fm = QFontMetrics(f_item)
                        try:
                            w = int(fm.horizontalAdvance(t))
                        except Exception:
                            try:
                                w = int(fm.width(t))
                            except Exception:
                                w = 0
                        if w > avail:
                            return False
                return True
            except Exception:
                return False

        for pt in range(int(start_pt), int(min_pt) - 1, -1):
            if _fits(pt):
                chosen = int(pt)
                break
        if chosen is None:
            chosen = int(min_pt)

        try:
            prev = tbl.property('_auto_fit_font_pt')
        except Exception:
            prev = None
        try:
            if prev is not None and int(prev) == int(chosen):
                return
        except Exception:
            pass

        # Apply chosen size to items (preserve bold/italic per item).
        try:
            for r in range(int(row_start), rows):
                for c in range(cols):
                    it = tbl.item(r, c)
                    if it is None:
                        continue
                    try:
                        t = str(it.text() or "").strip()
                        if not t:
                            continue
                    except Exception:
                        pass
                    try:
                        f_item = QFont(it.font())
                    except Exception:
                        f_item = QFont(base_font)
                    try:
                        f_item.setPointSize(int(chosen))
                        it.setFont(f_item)
                    except Exception:
                        pass
            try:
                tbl.setProperty('_auto_fit_font_pt', int(chosen))
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

            # Display convention for Image columns:
            # u = x_full
            # v = (h_full - 1) - y_full   (origin bottom-left, +v upward)
            try:
                _spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
            except Exception:
                _spf = 1.0
            try:
                _h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else None
            except Exception:
                _h_full = None
            if not (_h_full is not None and _h_full > 0):
                try:
                    if getattr(self, 'img_full', None) is not None:
                        _h_full = int(self.img_full.shape[0])
                except Exception:
                    _h_full = None

            def _fmt_uv_from_proc_pt(pt):
                try:
                    if pt is None:
                        return "", ""
                    x_full = float(pt[0]) * _spf
                    y_full = float(pt[1]) * _spf
                    u = int(round(x_full))
                    if _h_full is not None and _h_full > 0:
                        v = int(round((_h_full - 1) - y_full))
                    else:
                        v = int(round(-y_full))
                    return str(u), str(v)
                except Exception:
                    return "", ""

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
                                            txt, _ = _fmt_uv_from_proc_pt(pt)
                                        elif src_r == (ref_src_row_offset + 1):
                                            pt = self.ref_points[src_c] if 0 <= src_c < len(self.ref_points) else None
                                            _, txt = _fmt_uv_from_proc_pt(pt)
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
                        tbl.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                        tbl.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
                    data_rows = int(len(getattr(self, 'ref_points', []) or []))
                    # Visible columns: u,v,Stage(XYZ),Residual(XYZ),Excl  (|R| removed)
                    sub_labels_ref = ["u", "v", "X", "Y", "Z", "X", "Y", "Z", ""]
                    data_cols = len(sub_labels_ref)
                    excl_col_idx = data_cols - 1
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
                                try:
                                    txt = ""
                                    if c == 0:
                                        pt = self.ref_points[r] if 0 <= r < len(self.ref_points) else None
                                        txt, _ = _fmt_uv_from_proc_pt(pt)
                                    elif c == 1:
                                        pt = self.ref_points[r] if 0 <= r < len(self.ref_points) else None
                                        _, txt = _fmt_uv_from_proc_pt(pt)
                                    elif c in (2, 3, 4):
                                        obs = self.ref_obs[r] if 0 <= r < len(self.ref_obs) else None
                                        if isinstance(obs, dict):
                                            key = 'x' if c == 2 else ('y' if c == 3 else 'z')
                                            txt = str(obs.get(key, "") or "")
                                            try:
                                                if txt.strip().lower() in {"image", "stage", "residual", "x", "y", "z", "u", "v", "|r|"}:
                                                    txt = ""
                                            except Exception:
                                                pass
                                        else:
                                            txt = ""
                                    elif c == excl_col_idx:
                                        # Excl column — handled as checkbox below
                                        txt = ""
                                    else:
                                        src_r = ref_src_row_offset + c
                                        src_item = src.item(src_r, r) if (0 <= src_r < src.rowCount()) else None
                                        txt = src_item.text() if src_item is not None else ""
                                except Exception:
                                    txt = ""
                                it = QTableWidgetItem(str(txt))
                                try:
                                    it.setTextAlignment(_Qt.AlignHCenter | _Qt.AlignVCenter)
                                except Exception:
                                    pass
                                # Make Stage columns (X/Y/Z) visually bold
                                try:
                                    if c in (2, 3, 4):
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                # Show column: toggle widget (not checkbox)
                                if c == excl_col_idx:
                                    try:
                                        it.setFlags(_Qt.ItemIsEnabled | _Qt.ItemIsSelectable)
                                    except Exception:
                                        pass
                                elif c in (2, 3, 4):
                                    # Stage columns remain editable
                                    pass
                                else:
                                    try:
                                        it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                    except Exception:
                                        pass
                                # Dim text + gray background for excluded ref rows
                                try:
                                    if self._is_ref_excluded(r):
                                        it.setForeground(QColor(180, 180, 180))
                                        it.setBackground(QColor(235, 235, 235))
                                except Exception:
                                    pass
                                dst.setItem(header_rows + r, c, it)
                                # Place Show/Hide toggle in the excl column
                                if c == excl_col_idx:
                                    try:
                                        tog = self._make_show_toggle_ref(r)
                                        if tog is not None:
                                            wrap = QWidget(dst)
                                            lay = QHBoxLayout(wrap)
                                            lay.setContentsMargins(0, 0, 0, 0)
                                            lay.setSpacing(0)
                                            lay.addWidget(tog)
                                            lay.setAlignment(Qt.AlignCenter)
                                            dst.setCellWidget(header_rows + r, c, wrap)
                                    except Exception:
                                        pass

                        # Style the row-number gutter (vertical header): bold + readable gray
                        try:
                            dst.verticalHeader().setStyleSheet(
                                'QHeaderView::section { background-color: lightgray; color: gray; font-weight: bold; border: none; }'
                            )
                        except Exception:
                            pass

                        # Apply in-cell 2-row header (Image/Stage/Residual/Excl)
                        group_configs = [(0, 2, "Image"), (2, 3, "Stage (input)"), (5, 3, "Residual"), (8, 1, "")]
                        sub_labels = ["u", "v", "X", "Y", "Z", "X", "Y", "Z", ""]
                        if len(sub_labels) != data_cols:
                            if len(sub_labels) > data_cols:
                                sub_labels = sub_labels[:data_cols]
                            else:
                                sub_labels = sub_labels + [""] * (data_cols - len(sub_labels))
                        _apply_incell_two_row_header(dst, group_configs, sub_labels)

                        # If a fixed header widget exists, always hide in-table header rows
                        # to avoid height jitter during refresh/rebuild timing.
                        try:
                            if getattr(self, 'table_ref_view_header', None) is not None:
                                dst.setRowHidden(0, True)
                                dst.setRowHidden(1, True)
                            else:
                                dst.setRowHidden(0, False)
                                dst.setRowHidden(1, False)
                        except Exception:
                            pass

                        # Keep row heights stable to reduce layout shifts
                        try:
                            vh = dst.verticalHeader()
                            vh.setSectionResizeMode(QHeaderView.Fixed)
                            vh.setDefaultSectionSize(TABLE_DEFAULT_ROW_HEIGHT)
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
                    from qt_compat.QtWidgets import QTableWidgetItem, QWidget, QHBoxLayout
                    from qt_compat.QtCore import Qt as _Qt
                    import Strings as STR
                    center_indices = list(self._get_center_list_indices())
                    try:
                        self._append_center_numeric_rows_from_indices(center_indices)
                    except Exception:
                        pass
                    center_rows = list(getattr(self, 'center_numeric_rows', []) or [])
                    data_rows = int(len(center_rows))
                    try:
                        self._table_between_row_indices = [int(r.get('source_idx', -1)) for r in center_rows]
                    except Exception:
                        self._table_between_row_indices = []
                    # Add one extra column at the left for Posterization Level (group_no)
                    base_cols = 0
                    try:
                        base_cols = len(getattr(STR, 'TABLE_RIGHT_ROW_LABELS', []) or [])
                    except Exception:
                        base_cols = 0
                    base_cols = max(0, int(base_cols))
                    data_cols = base_cols + 2
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

                        # Vertical header: blank for header rows, then 1..N (center-list order)
                        try:
                            hlabels = [str(i + 1) for i in range(data_rows)]
                            dst.setVerticalHeaderLabels(["", ""] + hlabels)
                        except Exception:
                            pass

                        for r in range(data_rows):
                            try:
                                rowd = dict(center_rows[r] or {})
                            except Exception:
                                continue
                            try:
                                cidx = int(rowd.get('source_idx', -1))
                            except Exception:
                                cidx = -1
                            for c in range(data_cols):
                                try:
                                    if c == 0:
                                        try:
                                            txt = str(int(round(float(rowd.get('grp', 0.0)))))
                                        except Exception:
                                            txt = ""
                                    elif c in (1, 2):
                                        try:
                                            vv = rowd.get('u', 0.0) if c == 1 else rowd.get('v', 0.0)
                                            txt = str(int(round(float(vv))))
                                        except Exception:
                                            txt = ""
                                    elif c == (data_cols - 1):
                                        # Exclude flag — will be set as checkbox below
                                        txt = ""
                                    else:
                                        try:
                                            sval = rowd.get('x', float('nan')) if c == 3 else (rowd.get('y', float('nan')) if c == 4 else rowd.get('z', float('nan')))
                                            txt = "" if np.isnan(float(sval)) else str(float(sval))
                                        except Exception:
                                            pass
                                except Exception:
                                    txt = ""
                                it = QTableWidgetItem(str(txt))
                                try:
                                    it.setTextAlignment(_Qt.AlignHCenter | _Qt.AlignVCenter)
                                except Exception:
                                    pass
                                # Show column: toggle widget (not checkbox)
                                if c == (data_cols - 1):
                                    try:
                                        it.setFlags(_Qt.ItemIsEnabled | _Qt.ItemIsSelectable)
                                    except Exception:
                                        pass
                                else:
                                    # All other cells: non-editable
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
                                    tmp_sub_labels = ["Grp", "u", "v", "X", "Y", "Z", ""]
                                    sub_lbl = tmp_sub_labels[c] if 0 <= c < len(tmp_sub_labels) else None
                                    if sub_lbl in ("X", "Y", "Z"):
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                dst.setItem(header_rows + r, c, it)
                                # Place Show/Hide toggle in the Show column
                                if c == (data_cols - 1):
                                    try:
                                        if 0 <= cidx < len(getattr(self, 'centroids', []) or []):
                                            tog = self._make_show_toggle_centroid(cidx)
                                        else:
                                            tog = None
                                        if tog is not None:
                                            wrap = QWidget(dst)
                                            lay = QHBoxLayout(wrap)
                                            lay.setContentsMargins(0, 0, 0, 0)
                                            lay.setSpacing(0)
                                            lay.addWidget(tog)
                                            lay.setAlignment(Qt.AlignCenter)
                                            dst.setCellWidget(header_rows + r, c, wrap)
                                    except Exception:
                                        pass

                        # Style the row-number gutter (vertical header): bold + readable gray
                        try:
                            dst.verticalHeader().setStyleSheet(
                                'QHeaderView::section { background-color: lightgray; color: gray; font-weight: bold; border: none; }'
                            )
                        except Exception:
                            pass

                        # In-cell header (Posterization/Image/Stage)
                        group_configs = [(0, 1, ""), (1, 2, "Image"), (3, 3, "Stage"), (6, 1, "")]
                        sub_labels = ["Grp", "u", "v", "X", "Y", "Z", ""]
                        _apply_incell_two_row_header(dst, group_configs, sub_labels)

                        # If a fixed header widget exists, always hide in-table header rows
                        # to avoid height jitter during refresh/rebuild timing.
                        try:
                            if getattr(self, 'table_between_header', None) is not None:
                                dst.setRowHidden(0, True)
                                dst.setRowHidden(1, True)
                            else:
                                dst.setRowHidden(0, False)
                                dst.setRowHidden(1, False)
                        except Exception:
                            pass

                        try:
                            vh = dst.verticalHeader()
                            vh.setSectionResizeMode(QHeaderView.Fixed)
                            vh.setDefaultSectionSize(TABLE_DEFAULT_ROW_HEIGHT)
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

            try:
                self._refresh_offline_group_lists()
            except Exception:
                pass

            try:
                self._apply_excl_checkbox_style()
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
                QTimer.singleShot(0, lambda: self._adjust_center_column_widths())
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
                        # すべて同じ幅に (Show列はトグル用に広め)
                        widths_ref = [50] * cnt
                        # Last column (Show) wider for toggle
                        if cnt >= 1:
                            widths_ref[-1] = 48
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
                        # --- 動的幅算出: 列合計 + VH + SB + margin ---
                        try:
                            col_total = sum(widths_ref)
                            vh_w = 30  # 安全なフォールバック
                            try:
                                vh = tbl.verticalHeader()
                                if vh is not None:
                                    vhw = vh.sizeHint().width()
                                    if vhw > 0:
                                        vh_w = vhw
                            except Exception:
                                pass
                            sb_w = 17  # AlwaysOn scrollbar (Windows default)
                            try:
                                sb = tbl.verticalScrollBar()
                                if sb is not None:
                                    sbw = sb.sizeHint().width()
                                    if sbw > 0:
                                        sb_w = sbw
                            except Exception:
                                pass
                            margin = 4  # frame border
                            new_w = col_total + vh_w + sb_w + margin
                            new_w = max(200, new_w)
                            tbl.setFixedWidth(new_w)
                            # ヘッダー・コンテナ・画像の幅も同期
                            lc = getattr(self, 'left_container', None)
                            if lc is not None:
                                try:
                                    lc.setFixedWidth(new_w)
                                except Exception:
                                    pass
                            img = getattr(self, 'left_top_image', None)
                            if img is not None:
                                try:
                                    img.setFixedWidth(new_w)
                                except Exception:
                                    pass
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
                        # and keep Show column wider for toggle.
                        ref_tbl = getattr(self, 'table_ref_view', None)
                        for i in range(cnt2):
                            try:
                                if i == (cnt2 - 1):
                                    w = 48
                                elif ref_tbl is not None and i < ref_tbl.columnCount():
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

    def resizeEvent(self, event):
        try:
            super().resizeEvent(event)
        except Exception:
            pass
        try:
            self._update_offline_group_table_height_cache()
        except Exception:
            pass
        try:
            QTimer.singleShot(0, self._refresh_offline_group_lists)
        except Exception:
            pass

    def _apply_excl_checkbox_style(self):
        # No longer using checkboxes — Show/Hide toggles are cell widgets
        pass

    def _compute_group_header_colors(self, poster_img=None):
        """Compute representative RGB colors per K-Means group using median pixel color."""
        try:
            poster = poster_img
            if poster is None:
                poster = getattr(self, '_cache', {}).get('poster')
            if poster is None:
                return {}

            try:
                unique_colors = np.unique(np.asarray(poster).reshape(-1, 3), axis=0)
            except Exception:
                return {}

            proc_img = getattr(self, 'proc_img', None)
            use_proc = False
            try:
                use_proc = (proc_img is not None and proc_img.shape[:2] == poster.shape[:2])
            except Exception:
                use_proc = False

            group_rgb = {}
            for idx, col in enumerate(unique_colors, 1):
                try:
                    bgr = np.asarray(col, dtype=np.uint8)
                    mask = np.all(poster == bgr, axis=2)
                    if not np.any(mask):
                        continue
                    src = proc_img if use_proc else poster
                    vals = np.asarray(src[mask], dtype=np.float32)
                    if vals.size == 0:
                        continue
                    med_bgr = np.median(vals, axis=0)
                    r = int(np.clip(round(float(med_bgr[2])), 0, 255))
                    g = int(np.clip(round(float(med_bgr[1])), 0, 255))
                    b = int(np.clip(round(float(med_bgr[0])), 0, 255))
                    group_rgb[int(idx)] = (r, g, b)
                except Exception:
                    continue

            try:
                self._cache['group_header_rgb'] = dict(group_rgb)
            except Exception:
                pass
            return group_rgb
        except Exception:
            return {}

    def _update_offline_group_table_height_cache(self):
        """Update cached offline group-table height from current viewport size."""
        try:
            sc = getattr(self, 'offline_group_scroll', None)
            if sc is None:
                return
            vp = sc.viewport()
            if vp is None:
                return
            avail = int(vp.height())
            h = int(avail - 34)
            h = max(140, min(560, h))
            self._offline_group_table_h_cached = int(h)
        except Exception:
            pass

    def _offline_group_table_height(self):
        """Return table height adapted to current visible area of the Offline tab."""
        try:
            cached = getattr(self, '_offline_group_table_h_cached', None)
            if cached is not None:
                return int(cached)
        except Exception:
            pass
        try:
            self._update_offline_group_table_height_cache()
            cached = getattr(self, '_offline_group_table_h_cached', None)
            if cached is not None:
                return int(cached)
        except Exception:
            pass
        try:
            sc = getattr(self, 'offline_group_scroll', None)
            if sc is None:
                return 190
            vp = sc.viewport()
            if vp is None:
                return 190
            avail = int(vp.height())
            # Reserve header/margins and keep a practical range.
            h = int(avail - 34)
            h = max(140, min(560, h))
            return h
        except Exception:
            return 190

    def _refresh_offline_group_lists(self):
        """Rebuild group-wise u/v lists shown at the bottom of Off-line Targeting tab."""
        try:
            try:
                self._refresh_target_group_combo()
            except Exception:
                pass
            host = getattr(self, 'offline_group_layout', None)
            if host is None:
                return

            while host.count() > 0:
                item = host.takeAt(0)
                if item is None:
                    continue
                w = item.widget()
                if w is not None:
                    try:
                        w.deleteLater()
                    except Exception:
                        pass

            centroids = list(getattr(self, 'centroids', []) or [])
            if not centroids:
                empty = QLabel("No detected centroids")
                try:
                    empty.setStyleSheet("color:#666; font-size:11px;")
                except Exception:
                    pass
                host.addWidget(empty)
                return

            table_h = self._offline_group_table_height()
            try:
                mid_tbl = getattr(self, 'table_between', None)
            except Exception:
                mid_tbl = None
            try:
                data_font = mid_tbl.font() if mid_tbl is not None else None
            except Exception:
                data_font = None
            try:
                mid_u_w = int(mid_tbl.columnWidth(1)) if mid_tbl is not None and mid_tbl.columnCount() > 1 else 0
            except Exception:
                mid_u_w = 0
            try:
                mid_v_w = int(mid_tbl.columnWidth(2)) if mid_tbl is not None and mid_tbl.columnCount() > 2 else 0
            except Exception:
                mid_v_w = 0
            col_u_w = max(32, mid_u_w if mid_u_w > 0 else 57)
            col_v_w = max(32, mid_v_w if mid_v_w > 0 else 57)
            try:
                header_h = 27
                grp_toggle_h = 27
                body_h = max(120, int(table_h))
                # Add List行を廃止した分、Show/Hideトグル上下の余白を詰める。
                panel_h = header_h + grp_toggle_h + body_h + 4
            except Exception:
                body_h = 190
                panel_h = 248

            grouped = {}
            cache_uv = None
            try:
                cache_uv = list(getattr(self, '_cache', {}).get('centroids_full_uv') or [])
            except Exception:
                cache_uv = None

            if cache_uv and len(cache_uv) == len(centroids):
                for g, u, v in cache_uv:
                    try:
                        gg = int(g)
                    except Exception:
                        continue
                    if gg <= 0:
                        continue
                    grouped.setdefault(gg, []).append((int(u), int(v)))
            else:
                try:
                    spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                except Exception:
                    spf = 1.0
                try:
                    h_full = int(getattr(self, '_img_base_size', None)[1]) if getattr(self, '_img_base_size', None) is not None else None
                except Exception:
                    h_full = None
                if h_full is None:
                    try:
                        h_full = int(self.img_full.shape[0]) if getattr(self, 'img_full', None) is not None else None
                    except Exception:
                        h_full = None

                for g, xp, yp in centroids:
                    try:
                        gg = int(g)
                    except Exception:
                        continue
                    if gg <= 0:
                        continue
                    try:
                        x_full = float(xp) * spf
                        y_full = float(yp) * spf
                        u = int(round(x_full))
                        if h_full is not None and h_full > 0:
                            v = int(round((h_full - 1) - y_full))
                        else:
                            v = int(round(-y_full))
                    except Exception:
                        continue
                    grouped.setdefault(gg, []).append((u, v))

            group_colors = {}
            try:
                group_colors = dict(getattr(self, '_cache', {}).get('group_header_rgb') or {})
            except Exception:
                group_colors = {}
            if not group_colors:
                try:
                    group_colors = self._compute_group_header_colors()
                except Exception:
                    group_colors = {}

            try:
                all_visible = True
                for g in sorted(grouped.keys()):
                    if not self._is_group_visible(int(g)):
                        all_visible = False
                        break
                tog_all = getattr(self, 'toggle_show_all_groups', None)
                if tog_all is not None:
                    tog_all.setCheckedIndex(0 if all_visible else 1)
            except Exception:
                pass

            for idx, grp in enumerate(sorted(grouped.keys())):
                panel = QWidget()
                pv = QVBoxLayout(panel)
                pv.setContentsMargins(0, 0, 0, 0)
                pv.setSpacing(0)

                try:
                    grp_visible = self._is_group_visible(int(grp))
                    grp_toggle = SegmentControl(["Show", "Hide"], checked_index=(0 if grp_visible else 1), btn_w=68, btn_h=27)
                    grp_toggle.set_on_changed(lambda i, g=int(grp): (self._set_group_visible(g, int(i) == 0), self.schedule_update(force=True, recompute_centroids=False)))
                    pv.addWidget(grp_toggle, 0)
                except Exception:
                    pass

                try:
                    cr, cg, cb = group_colors.get(int(grp), (128, 128, 128))
                except Exception:
                    cr, cg, cb = (128, 128, 128)
                try:
                    # Perceived luminance for contrast-aware text color.
                    lum = (0.299 * float(cr)) + (0.587 * float(cg)) + (0.114 * float(cb))
                    txt_color = 'black' if lum >= 150.0 else 'white'
                except Exception:
                    txt_color = 'white'
                head = QPushButton(f"Add Group{int(grp)}")
                try:
                    head.setFixedHeight(27)
                    head.setStyleSheet(
                        f"background-color: rgb({cr},{cg},{cb}); color: {txt_color};"
                        "font-weight: bold; border: none; border-radius: 4px;"
                    )
                    try:
                        head.clicked.connect(lambda _=False, g=int(grp): self._add_group_to_center_list(g))
                    except Exception:
                        pass
                except Exception:
                    pass
                pv.addWidget(head, 0)

                tbl = QTableWidget()
                tbl.setColumnCount(2)
                tbl.setRowCount(len(grouped[grp]))
                tbl.setHorizontalHeaderLabels(["u", "v"])
                try:
                    if data_font is not None:
                        tbl.setFont(data_font)
                except Exception:
                    pass
                try:
                    tbl.verticalHeader().setVisible(True)
                    tbl.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    vh = tbl.verticalHeader()
                    vh.setSectionResizeMode(QHeaderView.Fixed)
                    vh.setMinimumWidth(28)
                    vh.setMaximumWidth(28)
                except Exception:
                    pass
                try:
                    hh = tbl.horizontalHeader()
                    hh.setSectionResizeMode(QHeaderView.Fixed)
                    tbl.setColumnWidth(0, col_u_w)
                    tbl.setColumnWidth(1, col_v_w)
                except Exception:
                    pass
                try:
                    tbl.setSelectionMode(QAbstractItemView.NoSelection)
                    tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
                    tbl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    try:
                        row_hdr_w = int(tbl.verticalHeader().width() or 28)
                    except Exception:
                        row_hdr_w = 28
                    try:
                        frame_w = int(tbl.frameWidth() or 1) * 2
                    except Exception:
                        frame_w = 2
                    try:
                        scroll_w = int(tbl.verticalScrollBar().sizeHint().width() or 16)
                    except Exception:
                        scroll_w = 16
                    tbl.setFixedWidth(row_hdr_w + col_u_w + col_v_w + scroll_w + frame_w)
                    tbl.setFixedHeight(body_h)
                    panel.setFixedHeight(panel_h)
                except Exception:
                    pass

                for r, (u, v) in enumerate(grouped[grp]):
                    iu = QTableWidgetItem(str(int(u)))
                    iv = QTableWidgetItem(str(int(v)))
                    try:
                        iu.setTextAlignment(Qt.AlignCenter)
                        iv.setTextAlignment(Qt.AlignCenter)
                    except Exception:
                        pass
                    try:
                        if data_font is not None:
                            iu.setFont(data_font)
                            iv.setFont(data_font)
                    except Exception:
                        pass
                    tbl.setItem(r, 0, iu)
                    tbl.setItem(r, 1, iv)

                pv.addWidget(tbl, 1)
                host.addWidget(panel, 0)

            host.addStretch(1)
        except Exception:
            pass

    def _make_show_toggle_ref(self, ref_idx):
        """Create an iOS-style toggle switch for the ref table."""
        try:
            from qt_compat.QtWidgets import QWidget as _QW
            from qt_compat.QtCore import QRectF
            from qt_compat.QtGui import QPainter, QColor as _QC

            excluded = self._is_ref_excluded(ref_idx)

            class _Toggle(_QW):
                def __init__(self, checked=True, parent=None):
                    super().__init__(parent)
                    self._checked = checked
                    self.setFixedSize(32, 16)
                    self.setCursor(Qt.PointingHandCursor)
                    self._cb = None
                def paintEvent(self, ev):
                    p = QPainter(self)
                    p.setRenderHint(QPainter.Antialiasing)
                    bg = _QC('#757575') if self._checked else _QC('#ccc')
                    p.setBrush(bg)
                    p.setPen(Qt.NoPen)
                    p.drawRoundedRect(QRectF(0, 0, 32, 16), 8, 8)
                    p.setBrush(_QC('white'))
                    x = 18.0 if self._checked else 2.0
                    p.drawEllipse(QRectF(x, 2, 12, 12))
                    p.end()
                def mousePressEvent(self, ev):
                    self._checked = not self._checked
                    self.update()
                    if self._cb:
                        self._cb(not self._checked)

            def _apply_ref(is_excluded, ri=ref_idx):
                try:
                    s = set(getattr(self, 'excluded_ref_indices', set()) or set())
                    if is_excluded:
                        s.add(ri)
                    else:
                        s.discard(ri)
                    self.excluded_ref_indices = s
                    try:
                        self._safe_populate_tables(
                            self.table_ref, self.table,
                            self.ref_points, self.ref_obs,
                            self.centroids, self.selected_index,
                            self.ref_selected_index,
                            flip_mode=self.flip_mode,
                            visible_ref_cols=self.visible_ref_cols,
                            excluded_ref_indices=self.excluded_ref_indices,
                        )
                    except Exception:
                        pass
                    try:
                        self._refresh_transposed_views()
                    except Exception:
                        pass
                except Exception:
                    pass

            sw = _Toggle(checked=not excluded)
            sw._cb = _apply_ref
            return sw
        except Exception:
            return None

    def _make_show_toggle_centroid(self, centroid_idx):
        """Create an iOS-style toggle switch for the centroid table."""
        try:
            excluded = self._is_centroid_excluded(centroid_idx)

            def _apply(is_excluded, ci=centroid_idx):
                try:
                    s = set(getattr(self, 'excluded_centroid_indices', set()) or set())
                    exp = set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set())
                    fv = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
                    if is_excluded:
                        s.add(ci)
                        exp.add(ci)
                        fv.discard(ci)
                    else:
                        s.discard(ci)
                        exp.discard(ci)
                        # Manual toggle ON should always make the point visible,
                        # even if hidden by Filter or previously hidden by Update Target replace.
                        fv.add(ci)
                    self.excluded_centroid_indices = s
                    self._explicit_excluded_centroid_indices = exp
                    self._force_visible_centroid_indices = fv
                    self._sanitize_excluded_indices()
                    try:
                        self._refresh_transposed_views()
                    except Exception:
                        pass
                    try:
                        # START押下時(OFF->ON)のみ一度自動再計算を実行。
                        self.schedule_update(force=True, recompute_centroids=(bool(active) and (not bool(prev))))
                    except Exception:
                        pass
                except Exception:
                    pass

            from qt_compat.QtWidgets import QWidget as _QW
            from qt_compat.QtCore import QRectF
            from qt_compat.QtGui import QPainter, QColor as _QC

            class _Toggle(_QW):
                def __init__(self, checked=True, parent=None):
                    super().__init__(parent)
                    self._checked = checked
                    self.setFixedSize(32, 16)
                    self.setCursor(Qt.PointingHandCursor)
                    self._cb = None
                def paintEvent(self, ev):
                    p = QPainter(self)
                    p.setRenderHint(QPainter.Antialiasing)
                    bg = _QC('#757575') if self._checked else _QC('#ccc')
                    p.setBrush(bg)
                    p.setPen(Qt.NoPen)
                    p.drawRoundedRect(QRectF(0, 0, 32, 16), 8, 8)
                    p.setBrush(_QC('white'))
                    x = 18.0 if self._checked else 2.0
                    p.drawEllipse(QRectF(x, 2, 12, 12))
                    p.end()
                def mousePressEvent(self, ev):
                    self._checked = not self._checked
                    self.update()
                    if self._cb:
                        self._cb(not self._checked)

            sw = _Toggle(checked=not excluded)
            sw._cb = _apply
            return sw
        except Exception:
            return None

    def _adjust_center_column_widths(self, fixed_px: int = 0):
        """Compute the required width from actual column widths and apply it.

        If *fixed_px* > 0 it is used as a hard override; otherwise the width
        is calculated as:  sum(column widths) + vertical-header width
                          + scrollbar width + small margin.
        Both ``table_between`` and ``center_container`` are resized.
        """
        tbl = getattr(self, 'table_between', None)
        if tbl is None:
            return

        try:
            if fixed_px and int(fixed_px) > 0:
                new_w = max(64, int(fixed_px))
            else:
                # --- dynamically compute required width ---
                col_total = 0
                for i in range(tbl.columnCount()):
                    cw = tbl.columnWidth(i)
                    if cw <= 0:
                        cw = 50  # fallback per-column
                    col_total += cw
                vh_w = 0
                try:
                    vh = tbl.verticalHeader()
                    if vh is not None and vh.isVisible():
                        vh_w = vh.width()
                        if vh_w <= 0:
                            vh_w = vh.sizeHint().width()
                        if vh_w <= 0:
                            vh_w = 30  # safe fallback
                except Exception:
                    vh_w = 30
                sb_w = 0
                try:
                    sb = tbl.verticalScrollBar()
                    if sb is not None and sb.isVisible():
                        sb_w = sb.width()
                        if sb_w <= 0:
                            sb_w = 17
                except Exception:
                    sb_w = 17
                margin = 6  # frame / border padding
                new_w = col_total + vh_w + sb_w + margin
                new_w = max(64, new_w)
        except Exception:
            new_w = 350  # ultimate fallback

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
                idxs = list(getattr(self, '_table_between_row_indices', []) or [])
                if int(idx) not in idxs:
                    return
                view_r = int(idxs.index(int(idx))) + header_rows
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
            self._adjust_center_column_widths()  # auto-calculate
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
                sb_w = 0
                try:
                    sb = tbl.verticalScrollBar()
                    if sb is not None:
                        sb_w = int(sb.width() or 0)
                        if sb_w <= 0:
                            sb_w = 17
                except Exception:
                    sb_w = 17
                pad = 4
                w = content_w + vh + sb_w + pad
                # テーブルの実際の幅があればそちらに合わせる
                try:
                    tw = tbl.width()
                    if tw > 0:
                        w = tw
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
        try:
            key = event.key()
        except Exception:
            key = None

        # Global cancel: stop heavy calculation with Esc.
        if key == Qt.Key_Escape and bool(getattr(self, '_calc_in_progress', False)):
            self._request_calc_stop("esc")
            return

        # ピックモード中の操作
        if self.pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
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
            add_target_btn = getattr(self, 'btn_add_target', None)
            if add_btn is not None:
                try:
                    style_add = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    add_btn.setStyleSheet(style_add)
                except Exception:
                    pass
                try:
                    add_btn.setFixedWidth(int(round(base_w * 1.5)) + 15)
                except Exception:
                    pass
            if add_target_btn is not None:
                try:
                    style_add = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    add_target_btn.setStyleSheet(style_add)
                except Exception:
                    pass
                try:
                    add_target_btn.setFixedWidth(int(base_w) + 10)
                except Exception:
                    pass

            # Update XY + Clear: blue, same width as Export/Clipboard
            upd_btn = getattr(self, 'btn_update_xy', None)
            clr_btn = getattr(self, 'btn_clear_ref', None)
            upd_target_btn = getattr(self, 'btn_update_target_uv', None)
            clr_target_btn = getattr(self, 'btn_clear_target', None)
            clr_target_all_btn = getattr(self, 'btn_clear_target_all', None)
            for btn in (upd_btn, clr_btn, upd_target_btn, clr_target_btn, clr_target_all_btn):
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
            filter_btn = getattr(self, 'btn_filter', None)
            add_all_grp_btn = getattr(self, 'btn_add_all_grp_list', None)
            open_btn = getattr(self, 'btn_open', None)
            replace_img_btn = getattr(self, 'btn_replace_image', None)
            flip_btn = getattr(self, 'btn_flip_mode', None)
            combo_flip = getattr(self, 'combo_flip_mode', None)
            new_btn = getattr(self, 'btn_new_project', None)
            save_btn = getattr(self, 'btn_save_project', None)
            load_btn = getattr(self, 'btn_load_project', None)
            start_ce_btn = getattr(self, 'btn_start_centroid_extraction', None)
            for btn in (exp_btn, clip_btn, filter_btn, add_all_grp_btn, open_btn, replace_img_btn, flip_btn, save_btn, load_btn):
                if btn is not None:
                    try:
                        style_blue = f"QPushButton {{ background-color: {blue}; color: white; border: none; border-radius: {radius}px; }}"
                        btn.setStyleSheet(style_blue)
                    except Exception:
                        pass
                    try:
                        w = int(base_w) + 10
                        if btn is add_all_grp_btn:
                            w = int(w * 2)
                        btn.setFixedWidth(w)
                    except Exception:
                        pass

            # Open Image / Export / New Project: make them red like "Add Ref. Point"
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

            if new_btn is not None:
                try:
                    style_red = f"QPushButton {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    new_btn.setStyleSheet(style_red)
                except Exception:
                    pass
                try:
                    new_btn.setFixedWidth(int(base_w) + 10)
                except Exception:
                    pass

            if start_ce_btn is not None:
                try:
                    start_ce_btn.setFixedWidth(int(base_w) + 120)
                except Exception:
                    pass
                try:
                    # Keep mode-aware color/text in sync after base style pass.
                    self._update_centroid_extraction_button()
                except Exception:
                    pass

            if replace_img_btn is not None:
                try:
                    # Keep existing base width relation and make it +30pt wider.
                    replace_img_btn.setFixedWidth(int(base_w) + 40)
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
        """Enforce all buttons to have fixed height of 40px for better visibility.
        Skip buttons inside SegmentControl widgets (they have their own size)."""
        try:
            from qt_compat.QtWidgets import QPushButton
            for b in self.findChildren(QPushButton):
                try:
                    if (b.text() or "") in ("−", "▢", "✕"):
                        continue
                    # Skip buttons that live inside a SegmentControl
                    parent = b.parent()
                    if isinstance(parent, SegmentControl):
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
                        hdr_ref.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                        hdr_ref.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
                    hdr_ref.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                    hdr_ref.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
                        hdr_mid.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                        hdr_mid.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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
                    hdr_mid.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                    hdr_mid.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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

            try:
                ncols = int(tbl.columnCount() or 0)
            except Exception:
                ncols = 0

            # Row 0: Group labels (Image, Stage, Residual, Excl)
            if ncols >= 9:
                group_configs = [
                    (0, 2, "Image"),
                    (2, 3, "Stage (input)"),
                    (5, 3, "Residual"),
                    (8, 1, ""),
                ]
                sub_labels = ["u", "v", "X", "Y", "Z", "X", "Y", "Z", ""]
            else:
                group_configs = [
                    (0, 2, "Image"),
                    (2, 3, "Stage (input)"),
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
                tbl.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                tbl.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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

            if ncols >= 7:
                group_configs = [
                    (0, 1, ""),  # col 0: Grp
                    (1, 2, "Image"),          # cols 1-2
                    (3, 3, "Stage"),          # cols 3-5
                    (6, 1, ""),               # col 6: Excl
                ]
                sub_labels = ["Grp", "u", "v", "X", "Y", "Z", ""]
            elif ncols >= 6:
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
                tbl.setRowHeight(0, TABLE_HEADER_ROW0_HEIGHT)
                tbl.setRowHeight(1, TABLE_HEADER_ROW1_HEIGHT)
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