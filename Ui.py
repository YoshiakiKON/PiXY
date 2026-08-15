# -*- coding: utf-8 -*-
"""
Centroid Finder のメイン UI ウィンドウ実装。

主な機能:
- 画像の読み込みと表示
- 重心検出パラメータの調整
- 参照点の設定とフィッティング
- テーブル表示と編集
"""

import qt_compat
from qt_compat.QtWidgets import (
    QSlider, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QWidget,
    QFileDialog, QStyle, QSizePolicy, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QScrollArea, QApplication, QMenu, QComboBox, QTabWidget, QFrame, QGraphicsOpacityEffect,
    QStyledItemDelegate
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
    LOG_MODE,
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
import copy
import ctypes
from ctypes import wintypes


# Unified table row-height constants (single source of truth)
TABLE_HEADER_ROW0_HEIGHT = 24
TABLE_HEADER_ROW1_HEIGHT = 20
TABLE_DEFAULT_ROW_HEIGHT = 24
LEFT_COLUMN_MIN_WIDTH = 550


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


class CenterShowToggleDelegate(QStyledItemDelegate):
    """Lightweight paint/event delegate for transposed Show/Excl columns."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner

    def _is_checked(self, index, widget=None):
        try:
            owner = self._owner
            if owner is None:
                return True
            view_row = int(index.row()) - 2
            if view_row < 0:
                return True
            # Left table (ref transposed): checked means NOT excluded.
            try:
                rv = getattr(owner, 'table_ref_view', None)
                if rv is not None and widget is rv:
                    return (not bool(owner._is_ref_excluded(view_row)))
            except Exception:
                pass

            # Middle table (center transposed): checked means visible.
            row_keys = list(getattr(owner, '_table_between_row_keys', []) or [])
            if not (0 <= view_row < len(row_keys)):
                return True
            si = int(row_keys[view_row][0])
            pp = str(row_keys[view_row][1] or 'c').lower().strip()
            if pp not in ('c', 'r'):
                pp = 'c'
            for rr in (getattr(owner, 'center_numeric_rows', []) or []):
                try:
                    rd = dict(rr or {})
                    if int(rd.get('source_idx', -1)) == si and str(rd.get('pos', 'c') or 'c').lower().strip() == pp:
                        return bool(owner._is_center_row_visible(rd))
                except Exception:
                    continue
        except Exception:
            pass
        return True

    def paint(self, painter, option, index):
        try:
            if index.row() < 2:
                return super().paint(painter, option, index)
            checked = bool(self._is_checked(index, getattr(option, 'widget', None)))
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)

            rect = option.rect
            w, h = 32, 16
            x = int(rect.x() + (rect.width() - w) / 2)
            y = int(rect.y() + (rect.height() - h) / 2)
            track = QRect(x, y, w, h)
            knob = QRect(x + (18 if checked else 2), y + 2, 12, 12)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#757575') if checked else QColor('#cccccc'))
            painter.drawRoundedRect(track, 8, 8)
            painter.setBrush(QColor('white'))
            painter.drawEllipse(knob)
            painter.restore()
        except Exception:
            try:
                super().paint(painter, option, index)
            except Exception:
                pass

    def editorEvent(self, event, model, option, index):
        try:
            if index.row() < 2:
                return False
            if event is None:
                return False
            et = int(event.type())
            if et not in (int(QEvent.MouseButtonRelease), int(QEvent.MouseButtonDblClick)):
                return False
            owner = self._owner
            if owner is None:
                return False
            view_row = int(index.row()) - 2
            w = getattr(option, 'widget', None)
            try:
                rv = getattr(owner, 'table_ref_view', None)
            except Exception:
                rv = None
            if rv is not None and w is rv:
                owner._toggle_ref_row_excluded_by_view_row(view_row)
            else:
                owner._toggle_center_row_visible_by_view_row(view_row)
            return True
        except Exception:
            return False


class AreaHistogramWidget(QWidget):
    """軽量な面積ヒストグラム描画ウィジェット（Qtペイント、曲線接続、ログ軸）。"""

    rangeChanged = pyqtSignal(float, float)
    rangeCommitted = pyqtSignal(float, float)

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
        was_dragging = bool(self._dragging)
        self._dragging = None
        if not was_dragging:
            return
        try:
            if self._bins and len(self._bins) >= 2:
                b0 = float(self._bins[0]); b1 = float(self._bins[-1])
                self.rangeCommitted.emit(float(self._sel_min or b0), float(self._sel_max or b1))
        except Exception:
            pass

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
            painter.drawText(QRect(0, 0, max(10, w), max(10, int(margin_t - 6))), Qt.AlignLeft | Qt.AlignVCenter, "Particle Size Range (pix)")
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
            pal.setColor(QPalette.Window, self._stage_title_color())
            self.setPalette(pal)
        except Exception:
            self.setStyleSheet(f'#titleBar {{ background-color: {self._stage_title_color().name()}; }}')

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

    def _stage_title_color(self):
        try:
            stage_n = str(getattr(self.window(), 'workflow_stage', 'offline') or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        return QColor(24, 96, 80) if stage_n == 'online' else QColor(160, 15, 15)

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
        painter.fillRect(self.rect(), self._stage_title_color())

    def paintEvent(self, event):
        """Force paint background to ensure color is applied."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._stage_title_color())


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

                # Prefer TOML parser when available.
                try:
                    try:
                        import tomllib  # py3.11+
                    except Exception:
                        import tomli as tomllib  # type: ignore
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
            if DEBUG or LOG_MODE:
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

        def _log_warn(msg):
            """Always persist warnings, even when LOG_MODE is off."""
            try:
                import os as _os
                from datetime import datetime as _dt
                ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                pid = _os.getpid()
                line = f"[WARN {ts} pid={pid}] {msg}"
                print(line, flush=True)
                with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        self._log_warn = _log_warn
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
            """Write high-frequency run-time info only in log mode."""
            if not LOG_MODE:
                return
            try:
                import os as _os
                from datetime import datetime as _dt
                ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                pid = _os.getpid()
                line = f"[INFO {ts} pid={pid}] {msg}"
                print(line, flush=True)
                with open("debug_px2xy.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        self._log_info = _log_info


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
        self._center_name_max_len = 0    # cached max length of center Name strings
        self.center_add_core_enabled = True
        self.center_add_rim_enabled = True
        self.rim_offset_px = 3
        self._auto_rim_proc_points = []
        self._centroid_rim_proc_points = []
        self.centroid_generation = 0  # successful centroid-recompute generation counter
        self._center_row_uid_seq = 0  # stable unique id for center rows
        self._manual_name_seq = 0     # running number for manual-name assignment
        self._table_between_row_indices = []  # table_between行 -> centroidsインデックス
        self._table_between_row_keys = []     # table_between行 -> (source_idx, 'c')
        self._offline_group_tables = {}   # grp -> {table: QTableWidget, indices: list[int]}
        self._offline_group_sort_key = 'u'  # u|v
        self._offline_group_sort_desc = False
        self._offline_group_sort_state = {}  # grp -> {'key': 'u'|'v', 'desc': bool}
        self.center_group_name_overrides = {}  # grp -> manual display name (offline AddToList panel)
        self.swap_left_center_columns = False  # stage-driven layout: keep fiducial controls on left in online stage
        self.workflow_stage = 'offline'  # 'offline' | 'online'
        self.online_image_grid_mode = 'uv'  # Online+Image grid: 'uv' | 'xy'
        self._center_sort_key = 'no'      # no|u|v|grp|pno|cr|gen
        self._center_sort_desc = False
        self._center_sort_secondary_key = 'no'   # backward-compat (unused)
        self._center_sort_secondary_desc = False
        self.center_name_filter_text = ""
        self._center_undo_stack = []
        self._center_undo_stack_max = 30
        self.center_add_uv_similarity_px = 2  # treat +/-N px in u,v as similar on left->center add
        self.overlay_point_source = 'left'  # 'left': 左リスト(全検出), 'center': 中リスト(Add済み)
        self._btn_start_ce_fixed_width = None
        self._startup_image_retry_count = 0
        self._centroid_finish_blink_on = False
        self._replace_target_source_index = None
        self._replace_target_source_group = None
        self.selected_index = None     # 選択中の重心インデックス
        self.selected_point_pos = 'c'  # 選択中点の種別 ('c'|'r')
        self.selected_point_keys = set()  # 複数選択 {(source_idx, 'c'|'r'), ...}
        self._center_uv_update_queue = []  # [{'row': int, 'no': str, 'key': (src_idx, pos)}]
        self._center_uv_update_pos = 0
        self._center_uv_update_active_key = None
        self._stage_info_override_text = None
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
        self._normal_overlay_mode_before_centroid = 'Original'
        self._normal_show_boundaries_before_centroid = False
        try:
            self._load_centroid_extraction_preferences()
        except Exception:
            pass
        try:
            self._load_center_add_preferences()
        except Exception:
            pass
        # Display labels: editable display strings separate from internal keys
        # Internal keys should be code-safe identifiers; change display text here.
        self.display_labels = {
            'overlay_ratio': 'Display Mode',

            'min_area': 'Particle Size Range (pix)',
            'trim': 'Boundary Offset (pix)'
        }

        self.show_boundaries = False   # 通常モード既定: 境界線は非表示
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
            "rim_points_proc": None,  # 自動検出重心に対応するRim点(proc)
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

        # Online(Image) grid-mode toggle overlay (u,v / X,Y)
        self.toggle_online_grid_mode = None
        try:
            self.toggle_online_grid_mode = SegmentControl(
                ["u,v Grid", "X,Y Grid"],
                parent=self.proc_scroll.viewport(),
                checked_index=0,
                btn_w=92,
                btn_h=24,
            )
            self.toggle_online_grid_mode.set_on_changed(lambda idx: self._on_toggle_online_grid_mode(int(idx)))
            self.toggle_online_grid_mode.setVisible(False)
        except Exception:
            self.toggle_online_grid_mode = None

        # center_uv_update navigation buttons (shown only while override message is visible)
        self.btn_center_uv_next = QPushButton("Next", self.proc_scroll.viewport())
        self.btn_center_uv_back = QPushButton("Back", self.proc_scroll.viewport())
        self.btn_center_uv_clear = QPushButton("Clear", self.proc_scroll.viewport())
        self.btn_center_uv_finish = QPushButton("Finish Update", self.proc_scroll.viewport())
        for _b in (self.btn_center_uv_next, self.btn_center_uv_back, self.btn_center_uv_clear, self.btn_center_uv_finish):
            try:
                _b.setVisible(False)
                _b.setFixedHeight(24)
                _b.setStyleSheet(
                    "QPushButton {"
                    "background-color: rgba(185,185,185,230);"
                    "color: white;"
                    "border: none;"
                    "border-radius: 6px;"
                    "padding: 2px 8px;"
                    "font-weight: bold;"
                    "}"
                    "QPushButton:hover { background-color: rgba(175,175,175,235); }"
                    "QPushButton:pressed { background-color: rgba(165,165,165,240); }"
                )
            except Exception:
                pass
        try:
            # Make floating update buttons easier to click/read.
            self.btn_center_uv_next.setFixedWidth(84)
            self.btn_center_uv_back.setFixedWidth(84)
            self.btn_center_uv_clear.setFixedWidth(84)
            self.btn_center_uv_finish.setFixedWidth(132)
        except Exception:
            pass
        try:
            self.btn_center_uv_next.clicked.connect(self._on_center_uv_update_next)
            self.btn_center_uv_back.clicked.connect(self._on_center_uv_update_back)
            self.btn_center_uv_clear.clicked.connect(self._on_center_uv_update_clear)
            self.btn_center_uv_finish.clicked.connect(self._on_center_uv_update_finish)
        except Exception:
            pass

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
        try:
            QTimer.singleShot(0, self._update_online_stage_controls_overlay_visibility)
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
        try:
            self.table_ref.setHorizontalHeaderLabels([f"Fiducial {i + 1}" for i in range(self.table_ref.columnCount())])
        except Exception:
            pass

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
                self._action_btn_base_w = 120  # Default base width
                self._action_btn_base_h = 56   # Default base height (increased for better visibility)
                # Try to measure from Add button if available
                add_btn = getattr(self, 'btn_add_ref', None)
                if add_btn is not None:
                    try:
                        w = int(add_btn.width() or 0)
                        h = int(add_btn.height() or 0)
                        if w > 0:
                            self._action_btn_base_w = max(120, w)
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

        # 残すのは PosterLevel と Particle Size Range に加え、Trim(pix)
        # Use code-safe internal keys for widgets; display text comes from self.display_labels
        self.edit_min_area, self.slider_min_area = self._make_spin_slider('min_area', 50, 10, 5000, 1)
        self.edit_trim, self.slider_trim = self._make_spin_slider('trim', 0, 0, 10, 1)
        self.edit_neck_sep, self.slider_neck_sep = self._make_spin_slider('neck_separation', 0, 0, 10, 1)
        self.edit_shape_complex, self.slider_shape_complex = self._make_spin_slider('shape_complexity', 3, 0, 10, 1)



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
        self.btn_add_target = QPushButton("Add Target Point")
        self.btn_add_target.setFixedHeight(40)
        self.btn_add_target.clicked.connect(self._on_add_target_point)
        self.edit_add_target_name_prefix = QLineEdit()
        self.edit_add_target_name_prefix.setFixedHeight(40)
        self.edit_add_target_name_prefix.setPlaceholderText("Name")
        self.edit_add_target_name_prefix.setFixedWidth(260)
        self.combo_add_target_pos = QComboBox()
        try:
            self.combo_add_target_pos.addItems(["", "C", "R"])
            self.combo_add_target_pos.setFixedHeight(40)
            self.combo_add_target_pos.setFixedWidth(60)
            self.combo_add_target_pos.currentIndexChanged.connect(self._on_add_target_pos_changed)
            self.combo_add_target_pos.setCurrentIndex(0)
        except Exception:
            pass
        self.manual_add_target_pos = ''
        self.edit_add_target_name_seq = QLineEdit("001")
        self.edit_add_target_name_seq.setFixedHeight(40)
        self.edit_add_target_name_seq.setAlignment(Qt.AlignCenter)
        self.edit_add_target_name_seq.setReadOnly(True)
        self.edit_add_target_name_seq.setFixedWidth(46)
        self.lbl_add_target_name_sep = QLabel("-")
        self.lbl_add_target_name_sep.setAlignment(Qt.AlignCenter)
        self.btn_select_all = QPushButton("Select ALL")
        self.btn_select_all.setFixedHeight(40)
        self.btn_select_all.clicked.connect(self._on_center_select_all)
        self.btn_select_all.setVisible(False)
        self.btn_update_target_uv = QPushButton("Update u, v")
        self.btn_update_target_uv.setFixedHeight(40)
        self.btn_update_target_uv.clicked.connect(self._on_update_target_uv)
        self.btn_clear_target = QPushButton("Clear Selected")
        self.btn_clear_target.setFixedHeight(40)
        self.btn_clear_target.clicked.connect(self._on_clear_target)
        self.btn_center_undo = QPushButton("Undo")
        self.btn_center_undo.setFixedHeight(40)
        self.btn_center_undo.clicked.connect(self._on_center_undo)
        self.btn_clear_target_all = QPushButton("Clear ALL")
        self.btn_clear_target_all.setFixedHeight(40)
        self.btn_clear_target_all.clicked.connect(self._on_clear_target_all)
        self.btn_center_name_filter = QPushButton("Name Filter")
        self.btn_center_name_filter.setFixedHeight(40)
        self.btn_center_name_filter.clicked.connect(self._on_center_name_filter_button)

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
        self.show_boundaries = False
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
                    self.view_orientation_toggle = SegmentControl(["Image (u, v)", "Stage (X, Y)"], checked_index=0, btn_w=128, btn_h=27)
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

        # 1段目はボタン専用にして潰れを防ぐ
        img_header.addStretch(1)

        img_layout.addLayout(img_header, 0)

        # 2段目: Coordinate / Boundary / Display Mode（重心抽出中だけ表示する追加行）
        try:
            param_row_top = QWidget()
            self._image_param_top_row = param_row_top
            prt = QHBoxLayout(param_row_top)
            prt.setContentsMargins(6, 0, 6, 0)
            prt.setSpacing(10)
            try:
                if getattr(self, 'view_orientation_controls', None) is not None:
                    prt.addWidget(self.view_orientation_controls, 0, Qt.AlignVCenter)
            except Exception:
                pass
            try:
                if getattr(self, 'boundary_controls', None) is not None:
                    prt.addWidget(self.boundary_controls, 0, Qt.AlignVCenter)
                elif getattr(self, 'boundary_toggle', None) is not None:
                    prt.addWidget(self.boundary_toggle, 0, Qt.AlignVCenter)
            except Exception:
                pass
            try:
                if overlay_ctrl is not None:
                    prt.addWidget(overlay_ctrl, 0, Qt.AlignVCenter)
            except Exception:
                pass
            prt.addStretch(1)
            img_layout.addWidget(param_row_top, 0)
        except Exception:
            pass

        # 3段目: Image Rotate / Normal/Flip（重心抽出中だけ追加表示）
        midbar = QWidget()
        mb = QHBoxLayout(midbar)
        mb.setContentsMargins(6, 0, 6, 0)
        mb.setSpacing(10)

        # Build 3 groups so we can toggle visibility cleanly by Coordinate.
        self._mid_rotate_controls = QWidget()
        self._mid_flip_controls = QWidget()
        self._mid_axis_controls = QWidget()
        self._mid_stats_controls = QWidget()

        try:
            if getattr(self, 'view_orientation_controls', None) is not None:
                mb.addWidget(self.view_orientation_controls, 0, Qt.AlignVCenter)
        except Exception:
            pass

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
                self.slider_img_rotate.setPageStep(10)
                self.slider_img_rotate.setTickInterval(30)
                self.slider_img_rotate.setTickPosition(QSlider.TicksBelow)
                self.slider_img_rotate._wheel_wrap = True
                self.slider_img_rotate._use_custom_ticks = False
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
            slider_container = QWidget()
            try:
                svl = QVBoxLayout(slider_container)
                svl.setContentsMargins(0, 0, 0, 0)
                svl.setSpacing(0)
                svl.setAlignment(self.slider_img_rotate, Qt.AlignVCenter)
                svl.addWidget(self.slider_img_rotate)
            except Exception:
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

        # --- Flip group (Image only)
        try:
            fhl = QHBoxLayout(self._mid_flip_controls)
            fhl.setContentsMargins(0, 0, 0, 0)
            fhl.setSpacing(6)
            try:
                self.flip_toggle_image = SegmentControl(["Normal", "Flip"], checked_index=0, btn_w=77, btn_h=27)
                self.flip_toggle_image.set_on_changed(lambda idx: self._on_flip_changed('image', int(idx)))
            except Exception:
                self.flip_toggle_image = None
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

        # --- Axis sign controls (Stage only)
        try:
            ahl = QHBoxLayout(self._mid_axis_controls)
            ahl.setContentsMargins(0, 0, 0, 0)
            ahl.setSpacing(8)

            lbl_right = QLabel("Right")
            try:
                f = lbl_right.font(); f.setBold(True); lbl_right.setFont(f)
            except Exception:
                pass

            lbl_top = QLabel("Top")
            try:
                f = lbl_top.font(); f.setBold(True); lbl_top.setFont(f)
            except Exception:
                pass

            try:
                self.axis_toggle_x = SegmentControl(["+X", "-X"], checked_index=0, btn_w=44, btn_h=27)
                self.axis_toggle_x.set_on_changed(lambda idx: self._on_stage_axis_changed('x', int(idx)))
            except Exception:
                self.axis_toggle_x = None

            try:
                self.axis_toggle_y = SegmentControl(["+Y", "-Y"], checked_index=0, btn_w=44, btn_h=27)
                self.axis_toggle_y.set_on_changed(lambda idx: self._on_stage_axis_changed('y', int(idx)))
            except Exception:
                self.axis_toggle_y = None

            ahl.addWidget(lbl_right)
            if self.axis_toggle_x is not None:
                ahl.addWidget(self.axis_toggle_x)
            ahl.addSpacing(10)
            ahl.addWidget(lbl_top)
            if self.axis_toggle_y is not None:
                ahl.addWidget(self.axis_toggle_y)
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

        # Add groups to the bar in the desired order.
        mb.addWidget(self._mid_rotate_controls, 0, Qt.AlignVCenter)
        mb.addWidget(self._mid_flip_controls, 0, Qt.AlignVCenter)
        mb.addWidget(self._mid_axis_controls, 0, Qt.AlignVCenter)
        mb.addWidget(self._mid_stats_controls, 0, Qt.AlignVCenter)
        mb.addStretch(1)

        try:
            if getattr(self, 'view_orientation_controls', None) is not None:
                self.view_orientation_controls.setVisible(False)
            self._mid_rotate_controls.setVisible(True)
            self._mid_flip_controls.setVisible(True)
            if getattr(self, '_mid_axis_controls', None) is not None:
                self._mid_axis_controls.setVisible(False)
            self._mid_stats_controls.setVisible(False)
        except Exception:
            pass

        img_layout.insertWidget(1, midbar, 0)
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
            sliders_layout.setSpacing(6)
            sliders_layout.setContentsMargins(6, 6, 6, 6)
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
                    if str(key) in ('min_area',):
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
                self.edit_num_groups, self.slider_num_groups = self._make_spin_slider('num_groups', 2, 2, 10, 1)
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

        # Particle Size Range row (Common)
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
                self.area_hist.rangeCommitted.connect(self._on_area_hist_range_committed)
            except Exception:
                pass
            sliders_layout.addWidget(self.area_hist)
        except Exception:
            self.area_hist = None

        # MinAreaとテーブルの間にボタン行（左詰め）を追加
        actions_row = QHBoxLayout()
        # ここから「境界線」ボタンは削除（画像右上に移動済み）
        self.btn_start_centroid_extraction = QPushButton("START Centroid Extraction")
        self.btn_center_add_core = QPushButton("Core")
        self.btn_center_add_rim = QPushButton("Rim")
        self.btn_center_rim_offset_minus = QPushButton("-")
        self.btn_center_rim_offset_plus = QPushButton("+")
        self.lbl_center_rim_offset = QLabel("Rim Offset")
        self.edit_center_rim_offset = QLineEdit()
        self.slider_center_rim_offset = QSlider(Qt.Horizontal)
        self.btn_center_add_core.setCheckable(True)
        self.btn_center_add_rim.setCheckable(True)
        self.btn_center_add_core.setChecked(bool(getattr(self, 'center_add_core_enabled', True)))
        self.btn_center_add_rim.setChecked(bool(getattr(self, 'center_add_rim_enabled', True)))
        try:
            self.edit_center_rim_offset.setFixedWidth(48)
            self.edit_center_rim_offset.setFixedHeight(28)
            self.edit_center_rim_offset.setAlignment(Qt.AlignCenter)
            self.edit_center_rim_offset.setFont(ctrl_font)
        except Exception:
            pass
        try:
            self.btn_center_rim_offset_minus.setFixedSize(28, 23)
            self.btn_center_rim_offset_plus.setFixedSize(28, 23)
            self.btn_center_rim_offset_minus.setFont(ctrl_font)
            self.btn_center_rim_offset_plus.setFont(ctrl_font)
            self.btn_center_rim_offset_minus.setStyleSheet("padding:0px; margin:0px;")
            self.btn_center_rim_offset_plus.setStyleSheet("padding:0px; margin:0px;")
            self.lbl_center_rim_offset.setFont(ctrl_font)
            self.lbl_center_rim_offset.setFixedWidth(180)
            self.lbl_center_rim_offset.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.slider_center_rim_offset.setRange(0, 50)
            self.slider_center_rim_offset.setSingleStep(1)
            self.slider_center_rim_offset.setPageStep(5)
            self.slider_center_rim_offset.setFixedHeight(28)
            self.slider_center_rim_offset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.slider_center_rim_offset.setStyleSheet("margin:0px; padding:0px;")
        except Exception:
            pass
        self.btn_add_ref = QPushButton(STR.BUTTON_ADD_REF)
        self.btn_update_xy = QPushButton(STR.BUTTON_UPDATE_XY)
        self.btn_clear_ref = QPushButton(STR.BUTTON_CLEAR)
        self.btn_start_centroid_extraction.clicked.connect(self._on_toggle_centroid_extraction_mode)
        self.btn_center_add_core.toggled.connect(self._on_toggle_center_add_core)
        self.btn_center_add_rim.toggled.connect(self._on_toggle_center_add_rim)
        self.btn_center_rim_offset_minus.clicked.connect(lambda: self._nudge_rim_offset(-1))
        self.btn_center_rim_offset_plus.clicked.connect(lambda: self._nudge_rim_offset(+1))
        self.edit_center_rim_offset.returnPressed.connect(self._on_rim_offset_edit_finished)
        self.slider_center_rim_offset.valueChanged.connect(self._on_rim_offset_slider_changed)
        try:
            self._update_rim_offset_label()
        except Exception:
            pass
        try:
            self._update_rim_offset_enabled_state()
        except Exception:
            pass
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
        self.main_row_layout = main_row
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
            try:
                # We render a custom 2-row pseudo header; hide Qt native horizontal header.
                self.table_ref_view.horizontalHeader().setVisible(False)
            except Exception:
                pass
            self.table_ref_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.table_ref_view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.table_ref_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
            try:
                # We render a custom 2-row pseudo header; hide Qt native horizontal header.
                self.table_between.horizontalHeader().setVisible(False)
            except Exception:
                pass
            # Keep startup schema aligned with middle-table builder:
            # No, Name, u, v, Grp, No., C/R, Gen., Show.
            try:
                self.table_between.setColumnCount(9)
                self.table_between.setRowCount(2)
                self._setup_pseudo_headers_between(self.table_between)
            except Exception:
                pass
            # Keep scrollbar presence stable so the center column doesn't jitter
            self.table_between.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.table_between.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            try:
                sb_mid = self.table_between.verticalScrollBar()
                if sb_mid is not None:
                    sb_mid.setMinimumWidth(14)
                    try:
                        sb_mid.setSingleStep(1)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self.table_between.setStyleSheet(
                    "QScrollBar:vertical { width: 14px; background: #efefef; }"
                    "QScrollBar::handle:vertical { background: #9c9c9c; min-height: 24px; border-radius: 6px; }"
                    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
                )
            except Exception:
                pass
            try:
                self.table_between.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            except Exception:
                pass
            try:
                # make the transposed middle table selectable by rows so image<->table sync is easier
                self.table_between.setSelectionBehavior(QAbstractItemView.SelectRows)
                self.table_between.setSelectionMode(QAbstractItemView.ExtendedSelection)
                self.table_between.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
                self.table_between.setContextMenuPolicy(Qt.CustomContextMenu)
                self.table_between.currentCellChanged.connect(self._on_table_between_current_changed)
                self.table_between.cellClicked.connect(self._on_table_between_cell_clicked)
                self.table_between.cellDoubleClicked.connect(self._on_table_between_cell_double_clicked)
                self.table_between.itemSelectionChanged.connect(self._on_table_between_selection_changed)
                self.table_between.itemChanged.connect(self._on_table_between_item_changed)
                self.table_between.customContextMenuRequested.connect(self._on_table_between_context_menu)
            except Exception:
                pass
            try:
                self._center_show_toggle_delegate = CenterShowToggleDelegate(self, self.table_between)
            except Exception:
                self._center_show_toggle_delegate = None
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
        try:
            # Keep row selection clearly visible even when cells have custom gray background.
            self.table_between.setStyleSheet(
                "QTableWidget::item:selected {"
                "background-color: rgb(46, 132, 255);"
                "color: white;"
                "}"
            )
        except Exception:
            pass

        left_col = QVBoxLayout()
        self.left_col_layout = left_col
        try:
            left_col.setContentsMargins(0, 0, 0, 0)
            left_col.setSpacing(0)
        except Exception:
            pass
        left_col.addWidget(self.left_top_image, 0, Qt.AlignTop)

        # Project buttons are created here and relocated to the top image header.
        try:
            self.left_project_row = QWidget()
            try:
                self.left_project_row.setObjectName('leftProjectRow')
                self.left_project_row.setStyleSheet('#leftProjectRow { border: none; }')
            except Exception:
                pass
            _proj_row = QHBoxLayout(self.left_project_row)
            _proj_row.setContentsMargins(0, 0, 0, 0)
            _proj_row.setSpacing(6)
            self.btn_new_project = QPushButton("New Project")
            self.btn_save_project = QPushButton("Save Project")
            self.btn_load_project = QPushButton("Load Project")
            self.btn_left_settings = QPushButton("Setting")
            for _btn in (self.btn_new_project, self.btn_save_project, self.btn_load_project, self.btn_left_settings):
                _btn.setFixedHeight(40)
            self.btn_new_project.clicked.connect(self.open_image)
            self.btn_save_project.clicked.connect(self.save_project)
            self.btn_load_project.clicked.connect(self.load_project)
            self.btn_left_settings.clicked.connect(self._on_open_left_add_settings)

            # Place project buttons on the top bar:
            # New/Save/Load to the left of Export Image, Setting next to Replace Image.
            try:
                if 'img_header' in locals() and img_header is not None:
                    img_header.insertWidget(0, self.btn_new_project, 0, Qt.AlignLeft | Qt.AlignVCenter)
                    img_header.insertWidget(1, self.btn_save_project, 0, Qt.AlignLeft | Qt.AlignVCenter)
                    img_header.insertWidget(2, self.btn_load_project, 0, Qt.AlignLeft | Qt.AlignVCenter)
                    try:
                        _rep_idx = int(img_header.indexOf(self.btn_replace_image))
                    except Exception:
                        _rep_idx = -1
                    if _rep_idx >= 0:
                        img_header.insertWidget(_rep_idx + 1, self.btn_left_settings, 0, Qt.AlignLeft | Qt.AlignVCenter)
                    else:
                        img_header.addWidget(self.btn_left_settings, 0, Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass

            _proj_row.addStretch(1)
            self.left_project_row.setVisible(False)
            left_col.addWidget(self.left_project_row, 0)
        except Exception:
            self.left_project_row = None
            self.btn_new_project = None
            self.btn_save_project = None
            self.btn_load_project = None
            self.btn_left_settings = None

        # Workflow stage toggle: centered under the top-left logo.
        try:
            self.left_stage_controls = QWidget()
            _stage_row = QHBoxLayout(self.left_stage_controls)
            _stage_row.setContentsMargins(0, 2, 0, 0)
            _stage_row.setSpacing(6)
            _stage_row.addStretch(1)
            self.toggle_workflow_stage = SegmentControl(["Offline targeting", "Online alignment"], checked_index=0, btn_w=170, btn_h=28)
            try:
                self.toggle_workflow_stage.set_on_changed(lambda idx: self._on_toggle_workflow_stage(int(idx)))
            except Exception:
                pass
            _stage_row.addWidget(self.toggle_workflow_stage, 0)
            _stage_row.addStretch(1)
            left_col.addWidget(self.left_stage_controls, 0)
            try:
                left_col.insertWidget(1, self.left_stage_controls, 0)
            except Exception:
                pass

            # Stage-specific hints shown under the Offline/Online toggle.
            self.left_stage_hint = QWidget()
            _stage_hint_col = QVBoxLayout(self.left_stage_hint)
            _stage_hint_col.setContentsMargins(0, 0, 0, 0)
            _stage_hint_col.setSpacing(0)

            self.lbl_stage_hint_offline = QLabel('Create a "New Project" and "Add Target Points" on an image.')
            self.lbl_stage_hint_online = QLabel('"Add fiducial point" and enter stage XYZ coordinates.')
            for _lbl in (self.lbl_stage_hint_offline, self.lbl_stage_hint_online):
                try:
                    _lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    _lbl.setWordWrap(False)
                    try:
                        _f = _lbl.font()
                        _f.setBold(True)
                        _lbl.setFont(_f)
                    except Exception:
                        pass
                    try:
                        _lbl.setStyleSheet('font-weight: bold;')
                    except Exception:
                        pass
                    try:
                        _lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                        _lbl.setFixedHeight(max(20, int(_lbl.sizeHint().height())))
                    except Exception:
                        pass
                    _stage_hint_col.addWidget(_lbl, 0, Qt.AlignHCenter)
                except Exception:
                    pass
            try:
                self.left_stage_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            except Exception:
                pass
            try:
                self.lbl_stage_hint_online.setVisible(False)
            except Exception:
                pass
            left_col.addWidget(self.left_stage_hint, 0)
            try:
                left_col.insertWidget(2, self.left_stage_hint, 0)
            except Exception:
                pass
        except Exception:
            self.left_stage_controls = None
            self.left_stage_hint = None
            self.lbl_workflow_stage = None
            self.toggle_workflow_stage = None
            self.lbl_stage_hint_offline = None
            self.lbl_stage_hint_online = None

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
                self.left_tabs.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            try:
                # Improve readability: slightly wider tabs and stronger selected/inactive contrast.
                self.left_tabs.tabBar().setExpanding(False)
                self.left_tabs.tabBar().setElideMode(Qt.ElideNone)
            except Exception:
                pass
            self.left_tabs.setStyleSheet(
                """
                QTabWidget#leftWorkflowTabs::pane {
                    border: none;
                    top: 0px;
                    margin-top: 0px;
                    padding-top: 0px;
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
            offline_col.setContentsMargins(0, 0, 0, 0)
            offline_col.setSpacing(0)
            self.offline_col_layout = offline_col
            self.offline_overlay_controls = None
            self.lbl_overlay_source = None
            self.toggle_overlay_source = None
            if getattr(self, 'grain_section', None) is not None:
                offline_col.addWidget(self.grain_section, 0)
            try:
                self.offline_global_controls = QWidget()
                offline_global_row = QHBoxLayout(self.offline_global_controls)
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
                offline_col.addWidget(self.offline_global_controls, 0)
            except Exception:
                self.offline_global_controls = None
                self.btn_add_all_grp_list = None
                self.toggle_show_all_groups = None
            try:
                self.offline_group_scroll = QScrollArea()
                self.offline_group_scroll.setWidgetResizable(True)
                self.offline_group_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                self.offline_group_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.offline_group_scroll.setFrameShape(QScrollArea.NoFrame)
                self.offline_group_scroll.setMinimumHeight(320)
                try:
                    self.offline_group_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                except Exception:
                    pass

                self.offline_group_inner = QWidget()
                self.offline_group_layout = QHBoxLayout(self.offline_group_inner)
                self.offline_group_layout.setContentsMargins(0, 0, 0, 0)
                self.offline_group_layout.setSpacing(6)
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
            self.online_col_layout = online_col
            self.left_tabs.addTab(self.tab_online, 'On-line Alignment')
            try:
                tb = self.left_tabs.tabBar()
                tb.hide()
                tb.setFixedHeight(0)
                tb.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            # 起動時は Off-line を既定にする
            self.left_tabs.setCurrentIndex(0)
        except Exception:
            self.left_tabs = None
            self.tab_offline = None
            self.tab_online = None
            self.offline_col_layout = None
            self.online_col_layout = None
            self.offline_group_scroll = None
            self.offline_group_inner = None
            self.offline_group_layout = None
            self.offline_overlay_controls = None
            self.offline_global_controls = None
            offline_col = None
            online_col = None

        # Centroid Extraction controls:
        # - Start/Finish button stays in left-bottom.
        # - Core/Rim/Offset options stay in the left column.
        try:
            self.left_extract_controls = QWidget()
            start_extract_row = QHBoxLayout(self.left_extract_controls)
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
        except Exception:
            self.left_extract_controls = None
            pass

        try:
            self.extract_mode_options_controls = QWidget()
            ext_opts_col = QVBoxLayout(self.extract_mode_options_controls)
            ext_opts_col.setContentsMargins(0, 0, 0, 2)
            ext_opts_col.setSpacing(6)
            core_rim_row = QHBoxLayout()
            core_rim_row.setContentsMargins(0, 0, 0, 0)
            core_rim_row.setSpacing(10)
            try:
                self.lbl_core_rim_title = QLabel("Targeting Region")
                try:
                    ff = self.lbl_core_rim_title.font()
                    ff.setBold(True)
                    self.lbl_core_rim_title.setFont(ff)
                except Exception:
                    pass
                core_rim_row.addWidget(self.lbl_core_rim_title)
            except Exception:
                self.lbl_core_rim_title = None
            try:
                core_rim_row.addWidget(self.btn_center_add_core)
            except Exception:
                pass
            try:
                core_rim_row.addWidget(self.btn_center_add_rim)
            except Exception:
                pass
            try:
                core_rim_row.addStretch(1)
            except Exception:
                pass
            try:
                ext_opts_col.addLayout(core_rim_row)
            except Exception:
                pass

            rim_offset_row = QHBoxLayout()
            rim_offset_row.setContentsMargins(0, 0, 0, 0)
            rim_offset_row.setSpacing(6)
            try:
                rim_offset_row.addWidget(self.lbl_center_rim_offset)
            except Exception:
                pass
            rim_offset_box = QWidget()
            try:
                rim_offset_box.setFixedWidth(self.control_area_width)
            except Exception:
                pass
            rim_offset_box_l = QHBoxLayout(rim_offset_box)
            rim_offset_box_l.setContentsMargins(0, 0, 0, 0)
            rim_offset_box_l.setSpacing(0)
            try:
                rim_offset_box_l.addWidget(self.btn_center_rim_offset_minus)
            except Exception:
                pass
            rim_offset_box_l.addSpacing(5)
            try:
                rim_offset_box_l.addWidget(self.edit_center_rim_offset)
            except Exception:
                pass
            rim_offset_box_l.addSpacing(45)
            try:
                rim_offset_box_l.addWidget(self.btn_center_rim_offset_plus)
            except Exception:
                pass
            try:
                rim_offset_row.addWidget(rim_offset_box)
            except Exception:
                pass
            try:
                rim_offset_row.addWidget(self.slider_center_rim_offset, 1)
            except Exception:
                pass
            try:
                ext_opts_col.addLayout(rim_offset_row)
            except Exception:
                pass
        except Exception:
            self.extract_mode_options_controls = None
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
                # Fixed header exists, so keep duplicated in-table pseudo header rows hidden.
                self.table_ref_view.setRowHidden(0, True)
                self.table_ref_view.setRowHidden(1, True)
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
            online_col.addWidget(self.table_ref_view, 0)
            try:
                self.online_export_controls = QWidget()
                _on_exp_row = QHBoxLayout(self.online_export_controls)
                _on_exp_row.setContentsMargins(0, 0, 0, 0)
                _on_exp_row.setSpacing(6)
                self.btn_online_export = QPushButton("Export XYZ")
                self.btn_online_export.setFixedHeight(40)
                self.btn_online_export.clicked.connect(self.export_centroids)
                self.btn_online_clipboard = QPushButton("Clipboard")
                self.btn_online_clipboard.setFixedHeight(40)
                self.btn_online_clipboard.clicked.connect(self._copy_centroids_to_clipboard)
                _on_exp_row.addWidget(self.btn_online_export, 0)
                _on_exp_row.addWidget(self.btn_online_clipboard, 0)
                _on_exp_row.addStretch(1)
                online_col.addWidget(self.online_export_controls, 0)
            except Exception:
                self.online_export_controls = None
                self.btn_online_export = None
                self.btn_online_clipboard = None
        else:
            left_col.addWidget(self.table_ref_view, 1)

        if self.left_tabs is not None:
            left_col.addWidget(self.left_tabs, 1)
            try:
                self._set_centroid_extraction_mode(False)
            except Exception:
                pass
            try:
                # 起動時は常に Off-line で開始
                self._set_workflow_stage('offline', sync_toggle=True, allow_mode_side_effects=False)
            except Exception:
                pass
        try:
            if getattr(self, 'left_extract_controls', None) is not None:
                # START Centroid Extraction 行を左カラム内容の直下に配置
                left_col.addWidget(self.left_extract_controls, 0)
        except Exception:
            pass
        # Wrap left column layout in a QWidget and cap its maximum width so it doesn't grow too wide
        self.left_container = QWidget()
        self.left_container.setLayout(left_col)
        try:
            # 左カラム最小幅（必要時はこの値を調整）
            min_left_w = int(getattr(self, 'left_column_min_width', LEFT_COLUMN_MIN_WIDTH) or LEFT_COLUMN_MIN_WIDTH)
            self.left_column_min_width = min_left_w
            # 起動時は最小幅で開始し、その後は再計算側で必要に応じて拡張する
            self.left_container.setMinimumWidth(min_left_w)
            self.left_container.setFixedWidth(min_left_w)
        except Exception:
            try:
                self.left_container.setMinimumWidth(LEFT_COLUMN_MIN_WIDTH)
            except Exception:
                pass
        main_row.addWidget(self.left_container, 0)
        # Middle panel dedicated to centroid-extraction parameter controls.
        try:
            self.middle_extract_panel = QWidget()
            self.middle_extract_layout = QVBoxLayout(self.middle_extract_panel)
            self.middle_extract_layout.setContentsMargins(0, 0, 0, 0)
            self.middle_extract_layout.setSpacing(10)
            try:
                self.lbl_middle_extract_mode = QLabel("Crntroid Extraction Mode")
                try:
                    f = self.lbl_middle_extract_mode.font()
                    f.setBold(True)
                    try:
                        f.setPointSize(max(12, int(f.pointSize()) + 1))
                    except Exception:
                        pass
                    self.lbl_middle_extract_mode.setFont(f)
                except Exception:
                    pass
                self.lbl_middle_extract_mode.setFixedHeight(28)
            except Exception:
                self.lbl_middle_extract_mode = None
            self.middle_extract_panel.setVisible(False)
            self.middle_extract_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            main_row.addWidget(self.middle_extract_panel, 0)
        except Exception:
            self.middle_extract_panel = None
            self.middle_extract_layout = None
        # Center area: place the transposed bottom table between left and image
        # Create a center column layout for the table_between
        try:
            center_col = QVBoxLayout()
            try:
                # Remove default layout margins to avoid clipping the right scrollbar area.
                center_col.setContentsMargins(0, 0, 0, 0)
                # Keep vertical breathing room between button rows.
                center_col.setSpacing(6)
            except Exception:
                pass
            self.center_extract_controls_host = None
            self.center_extract_controls_layout = None
            # Fixed 2-row header (does not scroll vertically) for the middle transposed table.
            try:
                # Add Export/Clipboard buttons above center table (aligned vertically with Open Image)
                try:
                    center_btn_grid = QGridLayout()
                    center_btn_grid.setContentsMargins(0, 0, 0, 0)
                    center_btn_grid.setHorizontalSpacing(4)
                    center_btn_grid.setVerticalSpacing(6)
                    try:
                        self.btn_filter.setVisible(False)
                    except Exception:
                        pass
                    try:
                        # 1行目: Add Target Point + name fields
                        center_btn_grid.addWidget(self.btn_add_target, 0, 0)
                        add_prefix_wrap = QWidget()
                        add_prefix_wrap_l = QHBoxLayout(add_prefix_wrap)
                        add_prefix_wrap_l.setContentsMargins(24, 0, 0, 0)
                        add_prefix_wrap_l.setSpacing(0)
                        add_prefix_wrap_l.addStretch(1)
                        add_prefix_wrap_l.addWidget(self.edit_add_target_name_prefix, 0, Qt.AlignRight | Qt.AlignVCenter)
                        center_btn_grid.addWidget(add_prefix_wrap, 0, 1)
                        add_seq_row = QHBoxLayout()
                        add_seq_row.setContentsMargins(0, 0, 0, 0)
                        add_seq_row.setSpacing(0)
                        try:
                            self.lbl_add_target_name_sep.setFixedWidth(8)
                            self.lbl_add_target_name_sep.setStyleSheet("padding:0px; margin:0px;")
                        except Exception:
                            pass
                        add_seq_row.addWidget(self.lbl_add_target_name_sep, 0)
                        try:
                            self.edit_add_target_name_seq.setTextMargins(0, 0, 0, 0)
                        except Exception:
                            pass
                        add_seq_row.addWidget(self.edit_add_target_name_seq, 0)
                        add_seq_row.addSpacing(4)
                        try:
                            add_seq_row.addWidget(self.combo_add_target_pos, 0)
                        except Exception:
                            pass
                        center_btn_grid.addLayout(add_seq_row, 0, 2)
                        center_btn_grid.addWidget(self.btn_center_undo, 0, 3)

                        # 2行目: Name Filter / Update u, v / Clear Selected / Clear ALL
                        center_btn_grid.addWidget(self.btn_center_name_filter, 1, 0)
                        center_btn_grid.addWidget(self.btn_update_target_uv, 1, 1)
                        center_btn_grid.addWidget(self.btn_clear_target, 1, 2)
                        center_btn_grid.addWidget(self.btn_clear_target_all, 1, 3)
                    except Exception:
                        pass
                    try:
                        center_btn_grid.setColumnStretch(0, 0)
                        center_btn_grid.setColumnStretch(1, 0)
                        center_btn_grid.setColumnStretch(2, 0)
                        center_btn_grid.setColumnStretch(3, 1)
                    except Exception:
                        pass

                    # ManualInput controls belong to Offline targeting (left tab).
                    # Keep a wrapper widget so we can place the same controls in one place.
                    self.offline_manual_controls = QWidget()
                    self.offline_manual_controls.setLayout(center_btn_grid)
                    if offline_col is not None:
                        try:
                            offline_col.insertWidget(1, self.offline_manual_controls, 0)
                        except Exception:
                            offline_col.addWidget(self.offline_manual_controls, 0)
                    else:
                        center_col.addWidget(self.offline_manual_controls, 0)

                except Exception:
                    self.offline_manual_controls = None

                self.table_between_header = QTableWidget()
                hdr_mid = self.table_between_header
                hdr_mid.setRowCount(2)
                # Pre-allocate 9 columns to match current middle table layout:
                # No, Name, u, v, Grp, No., C/R, Gen., Show
                hdr_mid.setColumnCount(9)
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
                    pref_mid = max(9, int(getattr(self, 'table_between', None).columnCount() or 9))
                except Exception:
                    pref_mid = 9
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
                    # When a fixed header widget exists, hide the duplicate in-table
                    # header rows immediately as well, not only after the first refresh.
                    self.table_between.setRowHidden(0, True)
                    self.table_between.setRowHidden(1, True)
                except Exception:
                    pass
                try:
                    hdr_mid.cellClicked.connect(self._on_table_between_cell_clicked)
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

                                # Ensure header is wide enough so the last column (e.g., Z) isn't clipped.
                                # Center container width is adjusted only in _adjust_center_column_widths().
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
            if bool(getattr(self, 'swap_left_center_columns', False)):
                try:
                    # Keep logo/project row on left, and move old left content to middle.
                    self.center_swap_container = QWidget()
                    swap_col = QVBoxLayout(self.center_swap_container)
                    swap_col.setContentsMargins(0, 0, 0, 0)
                    swap_col.setSpacing(6)

                    try:
                        if getattr(self, 'left_extract_controls', None) is not None:
                            left_col.removeWidget(self.left_extract_controls)
                            swap_col.addWidget(self.left_extract_controls, 0)
                    except Exception:
                        pass
                    try:
                        if getattr(self, 'left_tabs', None) is not None:
                            left_col.removeWidget(self.left_tabs)
                            swap_col.addWidget(self.left_tabs, 1)
                    except Exception:
                        pass

                    # Place former center content under left logo/project area.
                    left_col.addWidget(self.center_container, 1)

                    try:
                        self.center_swap_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                        try:
                            ref_w = int(getattr(self, 'table_ref_view', None).width() or 0)
                        except Exception:
                            ref_w = 0
                        if ref_w <= 0:
                            ref_w = 500
                        self.center_swap_container.setFixedWidth(max(240, int(ref_w)))
                    except Exception:
                        pass
                    main_row.addWidget(self.center_swap_container, 0)
                except Exception:
                    main_row.addWidget(self.center_container, 0)
            else:
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
                levels = 2
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
            shape = int(getattr(self, 'slider_shape_complex', None).value() if hasattr(self, 'slider_shape_complex') else 3)
        except Exception:
            shape = 3

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
                sv = int(snap.get('shape_complexity', 0))
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
        # Keep left logo tied to workflow stage, not coordinate mode.
        try:
            self._update_workflow_stage_logo()
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
        try:
            self._update_online_grid_mode_toggle_visibility()
        except Exception:
            pass
        try:
            self._update_online_stage_controls_overlay_visibility()
        except Exception:
            pass
        try:
            self._update_online_stage_controls_overlay_visibility()
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

    def _on_toggle_online_grid_mode(self, idx):
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
            self.online_image_grid_mode = 'xy' if int(idx) == 1 else 'uv'
        except Exception:
            self.online_image_grid_mode = 'uv'

        try:
            self._update_online_grid_mode_toggle_visibility()
        except Exception:
            pass
        try:
            self._update_online_stage_controls_overlay_visibility()
        except Exception:
            pass
        try:
            self._apply_proc_zoom()
            if center_full is not None:
                try:
                    self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
                except Exception:
                    pass
        except Exception:
            try:
                self.schedule_update(force=True, recompute_centroids=False)
            except Exception:
                pass

    def _update_online_grid_mode_toggle_visibility(self):
        try:
            tog = getattr(self, 'toggle_online_grid_mode', None)
            if tog is None:
                return
            tog.setVisible(False)
            try:
                gm = str(getattr(self, 'online_image_grid_mode', 'uv') or 'uv').lower().strip()
            except Exception:
                gm = 'uv'
            if gm not in ('uv', 'xy'):
                gm = 'uv'
            self.online_image_grid_mode = gm
            try:
                tog.setCheckedIndex(1 if gm == 'xy' else 0)
            except Exception:
                pass
        except Exception:
            pass

    def _update_online_stage_controls_overlay_visibility(self):
        try:
            host = getattr(self, 'online_stage_controls_overlay', None)
            if host is None:
                return
            try:
                stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
            except Exception:
                stage_n = 'offline'
            show = bool(stage_n == 'online')
            host.setVisible(show)
            if show:
                try:
                    host.raise_()
                except Exception:
                    pass

            try:
                orient = str(getattr(self, 'view_orientation', 'Image') or 'Image').strip().lower()
            except Exception:
                orient = 'image'
            coord_idx = 1 if orient == 'stage' else 0

            try:
                if getattr(self, 'overlay_coord_toggle', None) is not None:
                    self.overlay_coord_toggle.setCheckedIndex(int(coord_idx))
            except Exception:
                pass

            try:
                x_sign = int(getattr(self, 'stage_axis_x_sign', 1) or 1)
            except Exception:
                x_sign = 1
            try:
                y_sign = int(getattr(self, 'stage_axis_y_sign', 1) or 1)
            except Exception:
                y_sign = 1

            try:
                if getattr(self, 'overlay_axis_toggle_x', None) is not None:
                    self.overlay_axis_toggle_x.setCheckedIndex(0 if x_sign > 0 else 1)
            except Exception:
                pass
            try:
                if getattr(self, 'overlay_axis_toggle_y', None) is not None:
                    self.overlay_axis_toggle_y.setCheckedIndex(0 if y_sign > 0 else 1)
            except Exception:
                pass

            try:
                if show:
                    self._reposition_viewport_overlays()
            except Exception:
                pass
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
        try:
            self._update_online_stage_controls_overlay_visibility()
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
        # Particle Size Range slider is hidden; selection is done on the histogram in both modes.
        try:
            if getattr(self, 'row_min_area', None) is not None:
                self.row_min_area.setVisible(False)
        except Exception:
            pass
        # Advanced-only
        for name in ('row_trim', 'row_neck_sep', 'row_shape_complex'):
            try:
                w = getattr(self, name, None)
                if w is not None:
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
        # name is expected to be a code-safe key (e.g. 'min_area')
        try:
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


        self.schedule_update(force=True)


    def _reset_group_visibility_on_group_count_change(self):
        """Reset per-group visibility filters when clustering group count changes."""
        try:
            self._set_all_groups_visible(True)
        except Exception:
            pass
        try:
            tog_all = getattr(self, 'toggle_show_all_groups', None)
            if tog_all is not None:
                tog_all.setCheckedIndex(0)
        except Exception:
            pass

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

        try:
            self._reset_group_visibility_on_group_count_change()
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
                    self._sync_center_numeric_rows_xyz_from_table()
                except Exception:
                    pass
                try:
                    # Also refresh TargetPoints (middle table) right after Fiducial XYZ edits.
                    self._refresh_transposed_views(refresh_center_view=True)
                except Exception:
                    pass
                try:
                    self._apply_proc_zoom()
                except Exception:
                    pass
                try:
                    # Fiducial(Stage XYZ) edits must immediately refresh transform outputs
                    # even when heavy centroid recompute is gated/manual.
                    self.schedule_update(force=True, recompute_centroids=False)
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
                self._sync_center_numeric_rows_xyz_from_table()
            except Exception:
                pass
            try:
                self._apply_proc_zoom()
            except Exception:
                pass
            try:
                # Keep behavior consistent with deferred path.
                self.schedule_update(force=True, recompute_centroids=False)
            except Exception:
                pass
            try:
                self._refresh_transposed_views(refresh_center_view=True)
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

        try:
            if key in ('num_groups',):
                self._reset_group_visibility_on_group_count_change()
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

        try:
            skey = getattr(slider, '_pixy_key', None)
        except Exception:
            skey = None
        try:
            if skey in ('num_groups',):
                self._reset_group_visibility_on_group_count_change()
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
            try:
                retry_count = int(getattr(self, '_startup_image_retry_count', 0) or 0)
            except Exception:
                retry_count = 0
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
            try:
                cwd = os.getcwd()
                if cwd and cwd not in base_dirs:
                    base_dirs.append(cwd)
            except Exception:
                pass

            # Prefer explicit demo files first.
            demo_names = [
                'DemoBSE.png',
                'DemoBMP.bmp',
                'DemoBMP.png',
            ]
            demo_candidates = [os.path.join(root, name) for root in base_dirs for name in demo_names]

            # Also try last successfully opened image as a robust fallback.
            try:
                last_path = load_last_image_path()
                if last_path:
                    demo_candidates.append(str(last_path))
            except Exception:
                pass

            tried = set()
            for p in demo_candidates:
                try:
                    if not p:
                        continue
                    pp = os.path.abspath(str(p))
                    key = pp.lower()
                    if key in tried:
                        continue
                    tried.add(key)
                    if not os.path.isfile(pp):
                        continue
                    ok = self._open_image_from_path(pp, reset_project_state=True)
                    if ok:
                        try:
                            self._startup_image_retry_count = 0
                        except Exception:
                            pass
                        return
                except Exception:
                    continue

            # If startup timing/order caused a transient failure, retry once shortly after.
            if retry_count < 1:
                try:
                    self._startup_image_retry_count = int(retry_count + 1)
                except Exception:
                    pass
                try:
                    QTimer.singleShot(300, self._open_startup_image)
                except Exception:
                    pass
                return

            # Final fallback: keep app open and show guidance instead of forcing dialog.
            try:
                self._show_open_image_prompt_message()
            except Exception:
                pass
        except Exception:
            try:
                self._show_open_image_prompt_message()
            except Exception:
                pass

    def _reset_project_coordinates(self):
        """Reset coordinate-related project state for New Project."""
        try:
            self._end_pick_mode(redraw=False)
        except Exception:
            pass
        # --- Fiducial points ---
        try:
            self.ref_points = [None] * 10
            self.ref_selected_index = 0
            self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]
            self.excluded_ref_indices = set()
        except Exception:
            pass
        # --- Target points / center list ---
        try:
            self.manual_targets = []
            self.centroids = []
            self._auto_centroids = []
            self._auto_rim_proc_points = []
            self._centroid_rim_proc_points = []
            self.center_list_indices = []
            self.center_numeric_rows = []
            self._table_between_row_indices = []
            self._table_between_row_keys = []
            self.overlay_point_source = 'left'
            self.selected_index = None
            self.selected_point_pos = 'c'
            self.selected_point_keys = set()
            self.excluded_centroid_indices = set()
            self._explicit_excluded_centroid_indices = set()
            self._force_visible_centroid_indices = set()
            self._replace_target_source_index = None
            self._replace_target_source_group = None
            self._target_add_has_added = False
            self.center_group_name_overrides = {}
            self._center_undo_stack = []
            self.center_name_filter_text = ""
            self._manual_name_seq = 0
        except Exception:
            pass
        # --- Detection parameters ---
        try:
            for attr, default in (('slider_trim', 0), ('slider_neck_sep', 0), ('slider_shape_complex', 3)):
                s = getattr(self, attr, None)
                if s is not None:
                    s.blockSignals(True)
                    s.setValue(default)
                    s.blockSignals(False)
            for attr, default in (('edit_trim', '0'), ('edit_neck_sep', '0'), ('edit_shape_complex', '3')):
                e = getattr(self, attr, None)
                if e is not None:
                    e.setText(str(default))
        except Exception:
            pass
        # --- Image orientation ---
        try:
            self.manual_image_rotation_deg = 0
            s = getattr(self, 'slider_img_rotate', None)
            if s is not None:
                s.blockSignals(True)
                s.setValue(0)
                s.blockSignals(False)
            lbl = getattr(self, 'lbl_rot_val', None)
            if lbl is not None:
                lbl.setText("0°")
            self.flip_mode_image = 'normal'
            ft = getattr(self, 'flip_toggle_image', None)
            if ft is not None:
                ft.setCheckedIndex(0)
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
        # Rebuild tables immediately so UI is cleared before the async redraw.
        try:
            self._safe_populate_tables(
                self.table_ref, self.table,
                self.ref_points, self.ref_obs,
                [], self.selected_index, self.ref_selected_index,
                flip_mode=self.flip_mode,
                visible_ref_cols=self.visible_ref_cols,
            )
        except Exception:
            pass
        try:
            self._refresh_transposed_views(refresh_center_view=True)
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
        # Preserve legacy detection behavior for startup/new project image loads.
        recompute_on_load = bool(auto_detect) or bool(reset_project_state) or bool(getattr(self, 'centroid_extraction_mode', False))
        self.schedule_update(force=True, recompute_centroids=bool(recompute_on_load))
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
            try:
                stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
            except Exception:
                stage_n = 'offline'
            if stage_n == 'online':
                caption_color = ctypes.c_uint(0x00506018)  # RGB(24,96,80)
                border_color = ctypes.c_uint(0x00506018)
            else:
                caption_color = ctypes.c_uint(0x000F0FA0)
                border_color = ctypes.c_uint(0x000F0FA0)
            text_color = ctypes.c_uint(0x00FFFFFF)
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
        try:
            self._rebuild_center_name_max_len_from_rows()
        except Exception:
            pass
            extra_row = getattr(self, '_image_param_extra_row', None)
            if extra_row is not None:
                extra_row.setVisible(bool(is_image and bool(getattr(self, 'centroid_extraction_mode', False))))
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
            rim_offset_px=int(getattr(self, 'rim_offset_px', 3) or 3),
            neck_separation=int(getattr(self, 'slider_neck_sep', None).value() if hasattr(self, 'slider_neck_sep') else 0),
            shape_complexity=int(getattr(self, 'slider_shape_complex', None).value() if hasattr(self, 'slider_shape_complex') else 3),
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

    def _on_area_hist_range_committed(self, sel_min, sel_max):
        # Recompute once after the drag gesture completes.
        try:
            self._on_area_hist_range_changed(sel_min, sel_max)
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
        center_state_snapshot = None
        try:
            try:
                if bool(getattr(self, 'centroid_extraction_mode', False)):
                    center_state_snapshot = {
                        'center_list_indices': copy.deepcopy(list(getattr(self, 'center_list_indices', []) or [])),
                        'center_numeric_rows': copy.deepcopy(list(getattr(self, 'center_numeric_rows', []) or [])),
                        '_table_between_row_indices': copy.deepcopy(list(getattr(self, '_table_between_row_indices', []) or [])),
                        '_table_between_row_keys': copy.deepcopy(list(getattr(self, '_table_between_row_keys', []) or [])),
                    }
            except Exception:
                center_state_snapshot = None

            if self.img_full is None:
                self.img_label_proc.clear()
                self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, [], self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
                self.centroids = []
                self._auto_centroids = []
                self._auto_rim_proc_points = []
                self._centroid_rim_proc_points = []
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
            rim_points_now = []
            # Ensure poster is always defined before use in downstream rendering.
            poster = None
            poster_dt = None
            did_centroid_recompute = False
            manual_visibility_before_recompute = None
            group_name_transfer_state = None
            try:
                if bool(recompute_centroids):
                    manual_visibility_before_recompute = self._capture_manual_target_visibility()
            except Exception:
                manual_visibility_before_recompute = None
            try:
                if bool(recompute_centroids):
                    group_name_transfer_state = self._capture_group_name_transfer_state()
            except Exception:
                group_name_transfer_state = None
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
                cache_rim_points = self._cache.get("rim_points_proc")
                cache_areas = self._cache.get("areas")
                areas_now = cache_areas
                rim_points_now = cache_rim_points
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
                        rim_points_now = cache_rim_points
                        areas_now = cache_areas
                        boundary_mask_now = self._cache.get("boundary_mask")
                    else:
                        centroids = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())
                        rim_points_now = list(getattr(self, '_auto_rim_proc_points', []) or [])
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
                        try:
                            self.centroid_generation = int(getattr(self, 'centroid_generation', 0) or 0) + 1
                        except Exception:
                            pass
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        rim_points_now = list(getattr(self.centroid_processor, 'last_rim_points_proc', []) or [])
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
                            "rim_points_proc": rim_points_now,
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
                        try:
                            self.centroid_generation = int(getattr(self, 'centroid_generation', 0) or 0) + 1
                        except Exception:
                            pass
                        areas_now = getattr(self.centroid_processor, 'last_component_areas', [])
                        rim_points_now = list(getattr(self.centroid_processor, 'last_rim_points_proc', []) or [])
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
                            "rim_points_proc": rim_points_now,
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
            try:
                self._auto_rim_proc_points = list(rim_points_now or [])
            except Exception:
                self._auto_rim_proc_points = []
            try:
                if bool(did_centroid_recompute):
                    self._transfer_group_name_overrides_after_recompute(group_name_transfer_state, self._auto_centroids)
            except Exception:
                pass

            # データ反映を先に行い、描画前に最新の点群を反映させる（灰色丸を即表示）
            # 手動ターゲットは常に自動重心へ加算（+α）する。
            self.centroids = self._compose_centroids_with_manual(self._auto_centroids)
            try:
                self._centroid_rim_proc_points = self._compose_rim_points_with_manual(self._auto_centroids, self._auto_rim_proc_points)
            except Exception:
                self._centroid_rim_proc_points = []
            try:
                if bool(did_centroid_recompute):
                    self._reset_visibility_after_recompute(manual_visibility_before_recompute)
            except Exception:
                pass
            self._sanitize_excluded_indices()

            # NOTE:
            # Do not write directly to `table_between` here.
            # Direct item writes can emit `itemChanged` and be interpreted as Name edits.
            # 選択インデックスが範囲外なら解除
            if self.selected_index is not None and not (0 <= self.selected_index < len(self.centroids)):
                self.selected_index = None

            # 右側オーバーレイ画像を保持（フル解像度）
            self._last_overlay_full = overlay_full

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
                self._refresh_transposed_views(refresh_center_view=False)
            except Exception:
                pass
            try:
                # ensure selection sync after refresh
                QTimer.singleShot(0, self._sync_table_selection)
            except Exception:
                pass

            # Extraction mode: keep center model immutable during left-side updates.
            try:
                if center_state_snapshot is not None and not bool(getattr(self, '_center_model_mutation_pending', False)):
                    before_rows = list(getattr(self, 'center_numeric_rows', []) or [])
                    after_rows = list(center_state_snapshot.get('center_numeric_rows', []) or [])
                    before_list = list(getattr(self, 'center_list_indices', []) or [])
                    after_list = list(center_state_snapshot.get('center_list_indices', []) or [])
                    changed = (str(before_rows) != str(after_rows)) or (str(before_list) != str(after_list))
                    if changed:
                        try:
                            _warn = getattr(self, '_log_warn', None)
                            if callable(_warn):
                                _warn("Center model changed during extraction-mode update; restoring snapshot.")
                        except Exception:
                            pass
                    self.center_list_indices = copy.deepcopy(list(center_state_snapshot.get('center_list_indices', []) or []))
                    self.center_numeric_rows = copy.deepcopy(list(center_state_snapshot.get('center_numeric_rows', []) or []))
                    self._table_between_row_indices = copy.deepcopy(list(center_state_snapshot.get('_table_between_row_indices', []) or []))
                    self._table_between_row_keys = copy.deepcopy(list(center_state_snapshot.get('_table_between_row_keys', []) or []))
            except Exception:
                pass
            finally:
                try:
                    self._center_model_mutation_pending = False
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
        try:
            if bool(str(os.environ.get('PIXY_ADD_DEBUG', '')).strip()):
                import traceback as _tb
                caller = '|'.join(l.strip() for l in _tb.format_stack(limit=5)[-4:-1])
                self._log_info(
                    f"[ADD_DEBUG] _apply_proc_zoom "
                    f"src={getattr(self,'overlay_point_source','?')} "
                    f"sel={getattr(self,'selected_index',None)} "
                    f"keys={getattr(self,'selected_point_keys',set())} "
                    f"rows={len(getattr(self,'center_numeric_rows',[]) or [])} "
                    f"caller={caller}"
                )
        except Exception:
            pass
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
            try:
                self._last_overlay_payload = dict(ov or {})
            except Exception:
                self._last_overlay_payload = ov
            pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
                source_img,
                self.proc_zoom,
                self.view_padding,
                ov['centroids'],
                None,
                self.ref_points,
                self.scale_proc_to_full,
                ref_selected_index=getattr(self, 'ref_selected_index', None),
                manual_indices=ov['manual_indices'],
                excluded_indices=ov['excluded_indices'],
                force_visible_indices=ov['force_visible_indices'],
                visible_groups=ov['visible_groups'],
                label_texts=ov.get('label_texts'),
                local_to_source=ov.get('local_to_source'),
                local_to_pos=ov.get('local_to_pos'),
                colors=None,
                debug_ref_coords=True,
                interp_mode=self.interp_mode,
                max_pixels=self._get_render_max_pixels(),
            )
        except Exception:
            pm = None
            off_x = off_y = 0
            new_w = new_h = 0

        center_full = self._capture_proc_view_center_full()

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
                excluded_ref = set(getattr(self, 'excluded_ref_indices', set()) or set())
                # Gather pairs where we have both image (proc->full) and numeric stage obs
                for i, rp in enumerate(getattr(self, 'ref_points', []) or []):
                    if i in excluded_ref:
                        continue
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
            None,  # always draw all dots at normal size; selection is overlaid separately
            self.ref_points,
            self.scale_proc_to_full,
            ref_selected_index=getattr(self, 'ref_selected_index', None),
            manual_indices=ov['manual_indices'],
            excluded_indices=ov['excluded_indices'],
            force_visible_indices=ov['force_visible_indices'],
            visible_groups=ov['visible_groups'],
            label_texts=ov.get('label_texts'),
            local_to_source=ov.get('local_to_source'),
            local_to_pos=ov.get('local_to_pos'),
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

            # Online: recompute target X,Y,Z from stage transform whenever info is available.
            try:
                if info is not None and str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower() == 'online':
                    if self._sync_center_xyz_from_stage_info(info):
                        try:
                            self._refresh_transposed_views(
                                update_ref_view=False,
                                refresh_offline_lists=False,
                                refresh_center_view=True,
                            )
                        except Exception:
                            pass
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
                        try:
                            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
                        except Exception:
                            stage_n = 'offline'
                        try:
                            grid_mode = str(getattr(self, 'online_image_grid_mode', 'uv') or 'uv').lower().strip()
                        except Exception:
                            grid_mode = 'uv'
                        use_stage_grid_on_image = bool(stage_n == 'online' and grid_mode == 'xy' and info is not None)

                        if use_stage_grid_on_image:
                            try:
                                import numpy as _np
                                import math

                                display_scale = getattr(self, '_display_scale', None)
                                if display_scale is None:
                                    display_scale = float(self.proc_zoom)

                                s_val = float(info.get('s', 1.0))
                                px_per_stage = float(display_scale) / max(1e-12, s_val)

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

                                w_full = int(self._img_base_size[0]) if getattr(self, '_img_base_size', None) else new_w
                                h_full = int(self._img_base_size[1]) if getattr(self, '_img_base_size', None) else new_h
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
                                font = p.font()
                                font.setPointSize(9)
                                p.setFont(font)

                                def _stage_to_disp_xy(sx, sy):
                                    try:
                                        stage = _np.asarray([float(sx), float(sy)], dtype=_np.float64)
                                        uv = (1.0 / max(1e-12, s_val)) * (R.T @ (stage - t))
                                        if reflect_fit:
                                            uv[0] = -uv[0]
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
                                use_stage_grid_on_image = False

                        if not use_stage_grid_on_image:
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

            try:
                pm_selected = self._compose_pixmap_with_selected_overlay(pm_to_show)
            except Exception:
                pm_selected = pm_to_show

            self.img_label_proc.setPixmap(pm_selected)
            try:
                self.img_label_proc.resize(pm_selected.width(), pm_selected.height())
            except Exception:
                pass
            self._restore_proc_view_center_full(center_full)
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

    def _compose_pixmap_with_selected_overlay(self, base_pm):
        """Return a pixmap with only selected centroid marker(s) overlaid.

        The base pixmap should already contain image, base centroids, ref points,
        grids/rotation, etc. This method draws selection emphasis only (larger size).
        """
        if base_pm is None:
            return None
        try:
            from qt_compat.QtGui import QPixmap, QPainter, QPen, QColor
        except Exception:
            return base_pm

        try:
            # Recompute payload on each selection draw so middle-table row selection
            # (especially center mode c/r rows) is reflected immediately.
            ov = self._get_overlay_render_payload()
        except Exception:
            ov = None
        if not isinstance(ov, dict):
            return base_pm

        try:
            cent = list(ov.get('centroids', []) or [])
            local_to_source = list(ov.get('local_to_source', []) or [])
            local_to_pos = list(ov.get('local_to_pos', []) or [])
            label_texts = list(ov.get('label_texts', []) or [])
            manual_locals = set(ov.get('manual_indices', set()) or set())
        except Exception:
            return base_pm

        selected_locals = []
        try:
            keys = set(getattr(self, 'selected_point_keys', set()) or set())
        except Exception:
            keys = set()

        if keys:
            for li in range(len(cent)):
                try:
                    si = int(local_to_source[li]) if li < len(local_to_source) else int(li)
                except Exception:
                    si = int(li)
                try:
                    ptag = str(local_to_pos[li] if li < len(local_to_pos) else 'c').lower().strip()
                except Exception:
                    ptag = 'c'
                if ptag not in ('c', 'r'):
                    ptag = 'c'
                if (si, ptag) in keys:
                    selected_locals.append(int(li))

        # Fallback to selected_index whenever key-matching produced no result (e.g. stale keys after add).
        if not selected_locals:
            try:
                sel = ov.get('selected_index', None)
                if sel is None:
                    sel = getattr(self, 'selected_index', None)
                if sel is not None:
                    sel = int(sel)
                    if 0 <= sel < len(cent):
                        selected_locals = [sel]
            except Exception:
                selected_locals = []

        do_log_overlay = False
        try:
            do_log_overlay = bool(getattr(self, '_selection_overlay_log_armed', False))
        except Exception:
            do_log_overlay = False

        if not selected_locals:
            if do_log_overlay:
                try:
                    self._log_overlay_selection_snapshot('compose:no_selected_locals', [], [], len(cent))
                except Exception:
                    pass
                try:
                    self._selection_overlay_log_armed = False
                except Exception:
                    pass
            try:
                if bool(str(os.environ.get('PIXY_ADD_DEBUG', '')).strip()):
                    self._log_info(
                        f"[ADD_DEBUG] overlay:no_selected_locals "
                        f"keys={getattr(self,'selected_point_keys',set())} "
                        f"ov_sel={ov.get('selected_index')} cent_n={len(cent)}"
                    )
            except Exception:
                pass
            return base_pm

        try:
            excluded = set(ov.get('excluded_indices', set()) or set())
            forced = set(ov.get('force_visible_indices', set()) or set())
            vis_groups = ov.get('visible_groups', None)
            vis_groups = None if vis_groups is None else {int(g) for g in (vis_groups or set())}
        except Exception:
            excluded = set()
            forced = set()
            vis_groups = None

        pm2 = QPixmap(base_pm)
        p = QPainter(pm2)
        drawn_keys = []
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            rs = 6
            for sel in selected_locals:
                # Hide/Show setting takes priority for image-side selected markers.
                try:
                    grp = int(cent[sel][0])
                except Exception:
                    grp = 0
                try:
                    if sel not in forced:
                        if sel in excluded:
                            continue
                        if vis_groups is not None and grp not in vis_groups:
                            continue
                except Exception:
                    pass
                try:
                    _g, xp, yp = cent[sel]
                    xf = float(xp) * float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                    yf = float(yp) * float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
                    dxy = self._full_to_display(float(xf), float(yf))
                    if dxy is None:
                        continue
                    xd = int(round(float(dxy[0])))
                    yd = int(round(float(dxy[1])))
                    fill_color = QColor(64, 64, 64)
                    try:
                        # pos_list takes priority: 'r' -> blue, 'c' -> red; label is fallback.
                        ptag_sel = str(local_to_pos[sel] if sel < len(local_to_pos) else '').lower().strip()
                        if ptag_sel == 'r':
                            fill_color = QColor(0, 102, 220)
                        elif ptag_sel == 'c' or int(sel) in manual_locals:
                            fill_color = QColor(220, 50, 50)
                        elif label_texts and 0 <= int(sel) < len(label_texts):
                            lbl_lower = str(label_texts[int(sel)]).strip().lower()
                            if lbl_lower.endswith('r'):
                                fill_color = QColor(0, 102, 220)
                            elif lbl_lower.endswith('c'):
                                fill_color = QColor(220, 50, 50)
                    except Exception:
                        pass
                    p.setBrush(fill_color)
                    p.drawEllipse(xd - rs, yd - rs, rs * 2, rs * 2)
                    try:
                        si = int(local_to_source[sel]) if sel < len(local_to_source) else int(sel)
                    except Exception:
                        si = int(sel)
                    try:
                        ptag = str(local_to_pos[sel] if sel < len(local_to_pos) else 'c').lower().strip()
                    except Exception:
                        ptag = 'c'
                    if ptag not in ('c', 'r'):
                        ptag = 'c'
                    drawn_keys.append((int(si), str(ptag)))
                except Exception:
                    continue
        finally:
            try:
                p.end()
            except Exception:
                pass
        if do_log_overlay:
            try:
                self._log_overlay_selection_snapshot('compose:draw_selected_overlay', selected_locals, drawn_keys, len(cent))
            except Exception:
                pass
            try:
                self._selection_overlay_log_armed = False
            except Exception:
                pass
        return pm2

    def _refresh_selected_overlay_only(self):
        """Redraw selection highlight on current base (all dots already in _display_pm_base)."""
        try:
            base = getattr(self, '_display_pm_base', None)
            if base is None:
                self._apply_proc_zoom()
                return
            pm = self._compose_pixmap_with_selected_overlay(base)
            if pm is None:
                return
            self.img_label_proc.setPixmap(pm)
        except Exception:
            try:
                self._apply_proc_zoom()
            except Exception:
                pass

    def _refresh_list_and_selection(self):
        """Redraw when overlay list changed: rebuild base with all current dots, then highlight selection.

        Call this instead of _apply_proc_zoom() when only the point list changed
        (e.g. target point added/removed) and the base image/zoom/rotation are unchanged.
        Falls back to full _apply_proc_zoom() when the cached base is unavailable.
        """
        try:
            base = getattr(self, '_display_pm_base', None)
            if base is None:
                self._apply_proc_zoom()
                return
            # _display_pm_base now always has all dots; re-render from scratch to include new/removed dots.
            self._apply_proc_zoom()
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
                    try:
                        has_override = bool(str(getattr(self, '_stage_info_override_text', '') or '').strip())
                    except Exception:
                        has_override = False
                    x = int(margin) if has_override else int(max(margin, vp.width() - ov_tl.width() - margin))
                    y = int(margin)
                    ov_tl.move(x, y)
                except Exception:
                    ov_tl.move(int(margin), int(margin))

            tog_grid = getattr(self, 'toggle_online_grid_mode', None)
            if tog_grid is not None:
                try:
                    if bool(tog_grid.isVisible()):
                        tx = int(max(margin, (vp.width() - tog_grid.width()) // 2))
                        ty = int(margin)
                        tog_grid.move(tx, ty)
                        try:
                            tog_grid.raise_()
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                btn_next = getattr(self, 'btn_center_uv_next', None)
                btn_back = getattr(self, 'btn_center_uv_back', None)
                btn_clear = getattr(self, 'btn_center_uv_clear', None)
                btn_finish = getattr(self, 'btn_center_uv_finish', None)
                if ov_tl is not None and btn_next is not None and btn_back is not None and btn_clear is not None and btn_finish is not None:
                    if bool(btn_next.isVisible()) or bool(btn_back.isVisible()) or bool(btn_clear.isVisible()) or bool(btn_finish.isVisible()):
                        bx = int(ov_tl.x() + ov_tl.width() + 8)
                        by = int(ov_tl.y())
                        if bool(btn_back.isVisible()):
                            btn_back.move(bx, by)
                            bx += int(btn_back.width() + 6)
                        if bool(btn_next.isVisible()):
                            btn_next.move(bx, by)
                            bx += int(btn_next.width() + 6)
                        if bool(btn_clear.isVisible()):
                            btn_clear.move(bx, by)
                            bx += int(btn_clear.width() + 6)
                        if bool(btn_finish.isVisible()):
                            btn_finish.move(bx, by)
            except Exception:
                pass

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

    def _stage_input_decimal_digits_xy_z(self):
        try:
            max_xy_digits = 0
            max_z_digits = 0
            for ro in (getattr(self, 'ref_obs', None) or []):
                if not ro:
                    continue
                for key in ('x', 'y'):
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
                            frac = s.split('.', 1)[1]
                            digits = max(0, len(frac))
                        except Exception:
                            digits = 0
                    else:
                        digits = 0
                    if digits > max_xy_digits:
                        max_xy_digits = digits

                try:
                    raw = ro.get('z', '')
                except Exception:
                    raw = ''
                if raw is not None:
                    s = str(raw).strip().replace(',', '')
                    if s:
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
                                frac = s.split('.', 1)[1]
                                digits = max(0, len(frac))
                            except Exception:
                                digits = 0
                        else:
                            digits = 0
                        if digits > max_z_digits:
                            max_z_digits = digits
            return int(max_xy_digits), int(max_z_digits)
        except Exception:
            return 0, 0

    def _stage_input_decimal_digits(self):
        try:
            xy_digits, z_digits = self._stage_input_decimal_digits_xy_z()
            return int(max(int(xy_digits), int(z_digits)))
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

            try:
                ov_text = str(getattr(self, '_stage_info_override_text', '') or '').strip()
            except Exception:
                ov_text = ''
            if ov_text:
                try:
                    f = overlay.font()
                    f.setPointSize(20)
                    f.setBold(True)
                    overlay.setFont(f)
                except Exception:
                    pass
                overlay.setText(ov_text)
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
                return

            try:
                stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
            except Exception:
                stage_n = 'offline'
            if not (stage_n == 'online' and info is not None):
                try:
                    self._set_center_uv_nav_visible(False)
                except Exception:
                    pass
                overlay.hide()
                return

            try:
                f = overlay.font()
                f.setPointSize(10)
                f.setBold(True)
                overlay.setFont(f)
            except Exception:
                pass

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
                dec_xy, _dec_z = self._stage_input_decimal_digits_xy_z()
                tx = self._format_stage_numeric(t_xy[0] if t_xy is not None else None, dec_xy)
                ty = self._format_stage_numeric(t_xy[1] if t_xy is not None else None, dec_xy)
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

            dec_xy, dec_z = self._stage_input_decimal_digits_xy_z()
            def _fmt_stage(v):
                return self._format_stage_numeric(v, dec_xy)

            line1 = f"Image (u, v) = ({u_img}, {v_img})"
            stage_z_txt = self._format_stage_numeric(stage_z, dec_z)
            line2 = f"Stage (X, Y, Z) = ({_fmt_stage(stage_x)}, {_fmt_stage(stage_y)}, {stage_z_txt})"
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

    def _capture_proc_view_center_full(self):
        try:
            if getattr(self, 'proc_scroll', None) is None:
                return None
            vp = self.proc_scroll.viewport()
            if vp is None:
                return None
            hsb = self.proc_scroll.horizontalScrollBar()
            vsb = self.proc_scroll.verticalScrollBar()
            center_pos = QPoint(
                int(round(float(hsb.value()) + float(vp.width()) / 2.0)),
                int(round(float(vsb.value()) + float(vp.height()) / 2.0)),
            )
            center_full = self._display_to_full(center_pos)
            if center_full is None:
                return None
            return (float(center_full[0]), float(center_full[1]))
        except Exception:
            return None

    def _restore_proc_view_center_full(self, center_full):
        try:
            if center_full is None:
                return
            self._ensure_full_pos_visible(float(center_full[0]), float(center_full[1]))
        except Exception:
            pass

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
                        self._refresh_transposed_views(refresh_center_view=False)
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
        # Excl column toggle is handled by the delegate, not here.
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
                    self._refresh_transposed_views(refresh_center_view=False)
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
                self._refresh_transposed_views(refresh_center_view=False)
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

    def _on_center_select_all(self):
        try:
            tbl = getattr(self, 'table_between', None)
            if tbl is None:
                return
            try:
                tbl.selectAll()
            except Exception:
                pass
            try:
                tbl.setFocus()
            except Exception:
                pass
        except Exception:
            pass

    def _on_center_name_filter_changed(self, text):
        try:
            self.center_name_filter_text = str(text or "")
        except Exception:
            self.center_name_filter_text = ""
        try:
            btn = getattr(self, 'btn_center_name_filter', None)
            if btn is not None:
                if str(self.center_name_filter_text or '').strip():
                    btn.setText("Name Filter*")
                else:
                    btn.setText("Name Filter")
        except Exception:
            pass
        try:
            self._refresh_transposed_views()
        except Exception:
            pass

    def _on_center_name_filter_button(self):
        try:
            from qt_compat.QtWidgets import QInputDialog
            current = str(getattr(self, 'center_name_filter_text', '') or '')
            text, ok = QInputDialog.getText(self, "Name Filter", "Contains:", text=current)
            if not ok:
                return
            self._on_center_name_filter_changed(str(text or ''))
        except Exception:
            pass

    def _capture_center_undo_state(self):
        try:
            try:
                _rows = [dict(r or {}) for r in (getattr(self, 'center_numeric_rows', []) or [])]
            except Exception:
                _rows = list(getattr(self, 'center_numeric_rows', []) or [])
            st = {
                'manual_targets': list(getattr(self, 'manual_targets', []) or []),
                'centroids': list(getattr(self, 'centroids', []) or []),
                '_centroid_rim_proc_points': list(getattr(self, '_centroid_rim_proc_points', []) or []),
                'center_list_indices': list(getattr(self, 'center_list_indices', []) or []),
                'center_numeric_rows': _rows,
                'excluded_centroid_indices': set(getattr(self, 'excluded_centroid_indices', set()) or set()),
                '_explicit_excluded_centroid_indices': set(getattr(self, '_explicit_excluded_centroid_indices', set()) or set()),
                '_force_visible_centroid_indices': set(getattr(self, '_force_visible_centroid_indices', set()) or set()),
                'selected_index': getattr(self, 'selected_index', None),
                'selected_point_keys': set(getattr(self, 'selected_point_keys', set()) or set()),
                'selected_point_pos': str(getattr(self, 'selected_point_pos', 'c') or 'c'),
                '_manual_name_seq': int(getattr(self, '_manual_name_seq', 0) or 0),
                '_center_row_uid_seq': int(getattr(self, '_center_row_uid_seq', 0) or 0),
                '_center_name_max_len': int(getattr(self, '_center_name_max_len', 0) or 0),
            }
            return st
        except Exception:
            return None

    def _push_center_undo_state(self):
        try:
            st = self._capture_center_undo_state()
            if st is None:
                return
            stack = list(getattr(self, '_center_undo_stack', []) or [])
            stack.append(st)
            limit = int(getattr(self, '_center_undo_stack_max', 30) or 30)
            if limit > 0 and len(stack) > limit:
                stack = stack[-limit:]
            self._center_undo_stack = stack
        except Exception:
            pass

    def _restore_center_undo_state(self, st):
        try:
            if not isinstance(st, dict):
                return False
            try:
                self._center_model_mutation_pending = True
            except Exception:
                pass
            self.manual_targets = copy.deepcopy(list(st.get('manual_targets', []) or []))
            self.centroids = copy.deepcopy(list(st.get('centroids', []) or []))
            self._centroid_rim_proc_points = copy.deepcopy(list(st.get('_centroid_rim_proc_points', []) or []))
            self.center_list_indices = copy.deepcopy(list(st.get('center_list_indices', []) or []))
            self.center_numeric_rows = copy.deepcopy(list(st.get('center_numeric_rows', []) or []))
            self.excluded_centroid_indices = copy.deepcopy(set(st.get('excluded_centroid_indices', set()) or set()))
            self._explicit_excluded_centroid_indices = copy.deepcopy(set(st.get('_explicit_excluded_centroid_indices', set()) or set()))
            self._force_visible_centroid_indices = copy.deepcopy(set(st.get('_force_visible_centroid_indices', set()) or set()))
            self.selected_index = st.get('selected_index', None)
            self.selected_point_keys = copy.deepcopy(set(st.get('selected_point_keys', set()) or set()))
            self.selected_point_pos = str(st.get('selected_point_pos', 'c') or 'c')
            self._manual_name_seq = int(st.get('_manual_name_seq', getattr(self, '_manual_name_seq', 0) or 0) or 0)
            self._center_row_uid_seq = int(st.get('_center_row_uid_seq', getattr(self, '_center_row_uid_seq', 0) or 0) or 0)
            self._center_name_max_len = int(st.get('_center_name_max_len', getattr(self, '_center_name_max_len', 0) or 0) or 0)
            try:
                self._sanitize_excluded_indices()
            except Exception:
                pass
            try:
                self.schedule_update(force=True, recompute_centroids=False)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _on_center_undo(self):
        try:
            stack = list(getattr(self, '_center_undo_stack', []) or [])
            if not stack:
                return
            st = stack.pop()
            self._center_undo_stack = stack
            self._restore_center_undo_state(st)
            try:
                self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False, refresh_center_view=True)
            except Exception:
                pass
        except Exception:
            pass

    def _mark_center_model_mutation(self):
        try:
            self._center_model_mutation_pending = True
        except Exception:
            pass

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
            # Re-apply group visibility (Show/Hide) by group number after index refresh.
            try:
                self._sync_show_from_filter()
            except Exception:
                pass
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
            # Prefer actual K-Means poster groups so empty-yet-classified groups are shown.
            try:
                poster = getattr(self, '_cache', {}).get('poster') if isinstance(getattr(self, '_cache', None), dict) else None
            except Exception:
                poster = None
            if poster is not None:
                try:
                    n_groups = int(len(np.unique(np.asarray(poster).reshape(-1, 3), axis=0)))
                except Exception:
                    n_groups = 0
                if n_groups > 0:
                    groups.update(range(1, int(n_groups) + 1))

            for c in (getattr(self, 'centroids', []) or []):
                try:
                    groups.add(int(c[0]))
                except Exception:
                    pass

            # Fallback before first compute/cache build.
            if not groups:
                try:
                    n_cfg = int(getattr(self, 'slider_num_groups', None).value() if getattr(self, 'slider_num_groups', None) is not None else 0)
                except Exception:
                    n_cfg = 0
                if n_cfg > 0:
                    groups.update(range(1, int(n_cfg) + 1))

            try:
                groups = {int(g) for g in groups if int(g) > 0}
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

    def _next_center_row_id(self):
        try:
            self._center_row_uid_seq = int(getattr(self, '_center_row_uid_seq', 0) or 0) + 1
            return int(self._center_row_uid_seq)
        except Exception:
            return int(np.random.randint(1, 2**31 - 1))

    def _next_manual_name_seq(self):
        try:
            self._manual_name_seq = int(getattr(self, '_manual_name_seq', 0) or 0) + 1
            return int(self._manual_name_seq)
        except Exception:
            return 1

    def _add_target_name_state(self):
        try:
            prefix_widget = getattr(self, 'edit_add_target_name_prefix', None)
            prefix = str(prefix_widget.text() if prefix_widget is not None else '').strip()
        except Exception:
            prefix = ''
        if not prefix:
            prefix = 'Name'
        try:
            seq_widget = getattr(self, 'edit_add_target_name_seq', None)
            seq_text = str(seq_widget.text() if seq_widget is not None else '').strip()
        except Exception:
            seq_text = ''
        try:
            seq = int(seq_text)
        except Exception:
            seq = 1
        if seq <= 0:
            seq = 1
        return prefix, seq

    def _add_target_name_text(self):
        try:
            prefix, seq = self._add_target_name_state()
        except Exception:
            prefix, seq = 'Name', 1
        try:
            p = str(getattr(self, 'manual_add_target_pos', '') or '').lower().strip()
        except Exception:
            p = ''
        if p not in ('', 'c', 'r'):
            p = ''
        suffix = {'c': '_C', 'r': '_R'}.get(p, '')
        return f"{prefix}-{seq:03d}{suffix}", int(seq)

    def _on_add_target_pos_changed(self, index: int):
        try:
            self.manual_add_target_pos = {0: '', 1: 'c', 2: 'r'}.get(int(index), '')
        except Exception:
            self.manual_add_target_pos = ''

    def _current_add_target_pos(self):
        try:
            p = str(getattr(self, 'manual_add_target_pos', '') or '').lower().strip()
        except Exception:
            p = ''
        if p not in ('', 'c', 'r'):
            p = ''
        return p

    def _advance_add_target_name_seq(self):
        try:
            _, seq = self._add_target_name_state()
            seq = int(seq) + 1
            seq_widget = getattr(self, 'edit_add_target_name_seq', None)
            if seq_widget is not None:
                seq_widget.setText(f"{seq:03d}")
        except Exception:
            pass

    def _center_group_rank_map(self):
        """Return source_idx -> rank-in-group map based on current left-list ordering."""
        out = {}
        try:
            groups = list(self._available_group_numbers() or [])
        except Exception:
            groups = []
        for g in groups:
            try:
                gg = int(g)
            except Exception:
                continue
            if gg <= 0:
                continue
            try:
                rows_sorted = list(self._sorted_group_entries(gg) or [])
            except Exception:
                rows_sorted = []
            for rank, (_u, _v, src_i) in enumerate(rows_sorted, start=1):
                try:
                    out[int(src_i)] = int(rank)
                except Exception:
                    continue
        return out

    def _build_center_row_name(self, *, manual, group_no, group_rank, pos_tag, generation, manual_seq):
        try:
            if bool(manual):
                ms = int(manual_seq) if manual_seq is not None else 0
                if ms <= 0:
                    ms = self._next_manual_name_seq()
                return f"Manual-{ms:03d}"
        except Exception:
            pass
        try:
            g = int(group_no)
        except Exception:
            g = 0
        try:
            rk = int(group_rank)
        except Exception:
            rk = 0
        if rk <= 0:
            rk = 1
        try:
            p = str(pos_tag or 'c').lower().strip()
        except Exception:
            p = 'c'
        if p not in ('c', 'r'):
            p = 'c'
        try:
            suffix = {'c': '_C', 'r': '_R'}.get(p, '')
        except Exception:
            suffix = ''
        try:
            custom_group_name = str(dict(getattr(self, 'center_group_name_overrides', {}) or {}).get(int(g), '') or '').strip()
        except Exception:
            custom_group_name = ''
        if custom_group_name:
            # When a manual group name exists, keep the visible name compact.
            return f"{custom_group_name}-{rk:03d}{suffix}"
        try:
            gen = int(generation)
        except Exception:
            gen = int(getattr(self, 'centroid_generation', 0) or 0)
        base = f"Group{g}"
        return f"{base}-{rk:03d}{suffix}"

    def _update_center_name_max_len(self, name_text):
        """Update cached max length of center Name strings without scanning rows."""
        try:
            txt = str(name_text or '').strip()
        except Exception:
            txt = ''
        if not txt:
            return
        try:
            cur = int(getattr(self, '_center_name_max_len', 0) or 0)
        except Exception:
            cur = 0
        try:
            n = int(len(txt))
        except Exception:
            n = 0
        if n > cur:
            try:
                self._center_name_max_len = int(n)
            except Exception:
                pass

    def _rebuild_center_name_max_len_from_rows(self):
        """Recompute cached Name length once after load/restore."""
        try:
            max_len = 0
            for rr in (getattr(self, 'center_numeric_rows', []) or []):
                try:
                    nm = str(dict(rr or {}).get('name', '') or '').strip()
                except Exception:
                    nm = ''
                if len(nm) > max_len:
                    max_len = len(nm)
        except Exception:
            self._center_name_max_len = 0
        try:
            self._center_name_max_len = int(max_len)
        except Exception:
            pass
    def _center_name_column_width_hint(self, table=None):
        """Return a cached width hint for the middle-table Name column."""
        try:
            n = int(getattr(self, '_center_name_max_len', 0) or 0)
        except Exception:
            n = 0
        if n <= 0:
            return 0
        try:
            from qt_compat.QtGui import QFontMetrics
        except Exception:
            return 0
        try:
            src_tbl = table if table is not None else getattr(self, 'table_between', None)
            fnt = src_tbl.font() if src_tbl is not None else None
        except Exception:
            fnt = None
        if fnt is None:
            return 0
        try:
            fm = QFontMetrics(fnt)
            sample = 'W' * int(n)
            try:
                w = int(fm.horizontalAdvance(sample))
            except Exception:
                w = int(fm.width(sample)) if hasattr(fm, 'width') else 0
            return max(72, min(132, int(w + 18))) if w > 0 else 0
        except Exception:
            return 0

    def _fit_item_font_to_cell(self, tbl, item, col, min_pt=7, padding=8):
        """Shrink item font only when text would be elided in the target column."""
        try:
            if tbl is None or item is None:
                return
            txt = str(item.text() or "")
            if txt == "":
                return
            try:
                avail = int(tbl.columnWidth(int(col))) - int(padding)
            except Exception:
                avail = 0
            if avail <= 4:
                return
            f = item.font()
            cur_pt = int(f.pointSize() or 0)
            if cur_pt <= 0:
                cur_pt = 10
            try:
                from qt_compat.QtGui import QFontMetrics
            except Exception:
                return
            fm = QFontMetrics(f)
            try:
                text_w = int(fm.horizontalAdvance(txt))
            except Exception:
                text_w = int(fm.width(txt)) if hasattr(fm, 'width') else 0
            if text_w <= avail:
                return
            chosen = cur_pt
            for pt in range(cur_pt - 1, int(min_pt) - 1, -1):
                ff = item.font()
                ff.setPointSize(int(pt))
                fmm = QFontMetrics(ff)
                try:
                    w = int(fmm.horizontalAdvance(txt))
                except Exception:
                    w = int(fmm.width(txt)) if hasattr(fmm, 'width') else 0
                if w <= avail:
                    chosen = int(pt)
                    break
            if chosen < cur_pt:
                f.setPointSize(int(chosen))
                item.setFont(f)
        except Exception:
            pass

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

    def _snapshot_center_row_from_centroid(self, cidx, pos='c', group_rank_map=None, manual_name_override=None, manual_seq_override=None):
        try:
            idx = int(cidx)
            cents = list(getattr(self, 'centroids', []) or [])
            if not (0 <= idx < len(cents)):
                return None
            _g, xp, yp = cents[idx]
            ptag = str(pos or 'c').lower().strip()
            if ptag not in ('c', 'r'):
                ptag = 'c'
            if ptag == 'r':
                rim_list = list(getattr(self, '_centroid_rim_proc_points', []) or [])
                rim_pt = rim_list[idx] if 0 <= idx < len(rim_list) else None
                if rim_pt is not None:
                    try:
                        xp = float(rim_pt[0])
                        yp = float(rim_pt[1])
                    except Exception:
                        return None
                else:
                    # Manual targets have no separate rim coord; use core position as the Rim point.
                    try:
                        if idx not in set(self._manual_centroid_indices() or set()):
                            return None
                        # xp, yp already set from centroids[idx] — use as-is
                    except Exception:
                        return None
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

            try:
                gno = int(_g)
            except Exception:
                gno = 0

            try:
                grm = dict(group_rank_map or {})
            except Exception:
                grm = {}
            try:
                rank = int(grm.get(int(idx), 0))
            except Exception:
                rank = 0
            if rank <= 0:
                rank = int(idx) + 1

            gen_now = int(getattr(self, 'centroid_generation', 0) or 0)
            manual_seq = float('nan')
            if bool(is_manual >= 0.5):
                try:
                    if manual_seq_override is not None:
                        manual_seq = float(int(manual_seq_override))
                    else:
                        manual_seq = float(self._next_manual_name_seq())
                except Exception:
                    manual_seq = float('nan')
            if bool(is_manual >= 0.5) and manual_name_override:
                name_txt = str(manual_name_override)
            else:
                name_txt = self._build_center_row_name(
                    manual=bool(is_manual >= 0.5),
                    group_no=gno,
                    group_rank=rank,
                    pos_tag=ptag,
                    generation=gen_now,
                    manual_seq=manual_seq,
                )
            try:
                self._update_center_name_max_len(name_txt)
            except Exception:
                pass

            return {
                'row_id': int(self._next_center_row_id()),
                'source_idx': int(idx),
                'grp': float(gno),
                'group_no': float(gno),
                'group_rank': float(rank),
                'generation': float(gen_now),
                'manual_seq': float(manual_seq),
                'name': str(name_txt),
                'custom_name': '',
                'u': float(u),
                'v': float(v),
                'x': float(x_val),
                'y': float(y_val),
                'z': float(z_val),
                'x_proc': float(xp),
                'y_proc': float(yp),
                'no': float('nan'),
                'pos': str(ptag),
                'show': 1.0,
                'manual': float(is_manual),
            }
        except Exception:
            return None

    def _append_center_numeric_rows_from_indices(self, indices, manual_name_override=None, manual_seq_override=None, refresh_existing=False, source='unknown'):
        try:
            self._mark_center_model_mutation()
            try:
                src_tag = str(source or 'unknown').strip().lower()
            except Exception:
                src_tag = 'unknown'
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            existing = set()
            existing_sources = set()
            max_no = 0
            rank_map = None
            requested_pairs = []
            for i in (indices or []):
                raw_ptag = 'c'
                if isinstance(i, (tuple, list)) and len(i) >= 2:
                    try:
                        ii = int(i[0])
                    except Exception:
                        continue
                    try:
                        raw_ptag = str(i[1] or '').lower().strip()
                    except Exception:
                        raw_ptag = ''
                else:
                    try:
                        ii = int(i)
                    except Exception:
                        continue
                ptag = raw_ptag if raw_ptag in ('c', 'r') else 'c'
                requested_pairs.append((ii, ptag))

            allowed_sources = {
                'add_to_list',
                'add_all_to_list',
                'add_target',
                'load_project_recover',
            }

            def _get_rank_map():
                nonlocal rank_map
                if rank_map is None:
                    try:
                        rank_map = self._center_group_rank_map()
                    except Exception:
                        rank_map = {}
                return rank_map

            for r in rows:
                try:
                    rd = dict(r or {})
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                    if pp not in ('c', 'r'):
                        pp = 'c'
                    existing.add((si, pp))
                    existing_sources.add(si)

                    if not bool(refresh_existing):
                        try:
                            if int(rd.get('row_id', 0) or 0) <= 0:
                                rd['row_id'] = int(self._next_center_row_id())
                                r.update(rd)
                        except Exception:
                            pass
                        try:
                            self._update_center_name_max_len(rd.get('name', ''))
                        except Exception:
                            pass
                        try:
                            nv = int(round(float(r.get('no', float('nan')))))
                            if nv > max_no:
                                max_no = nv
                        except Exception:
                            pass
                        continue

                    if int(rd.get('row_id', 0) or 0) <= 0:
                        rd['row_id'] = int(self._next_center_row_id())

                    try:
                        is_manual = bool(float(rd.get('manual', 0.0)) >= 0.5)
                    except Exception:
                        is_manual = False

                    try:
                        gno = int(rd.get('group_no', rd.get('grp', 0) or 0))
                    except Exception:
                        gno = 0
                    if gno == 0:
                        try:
                            cents = list(getattr(self, 'centroids', []) or [])
                            if 0 <= si < len(cents):
                                gno = int(cents[si][0])
                        except Exception:
                            pass
                    rd['grp'] = float(gno)
                    rd['group_no'] = float(gno)

                    try:
                        rank = int(rd.get('group_rank', 0) or 0)
                    except Exception:
                        rank = 0
                    if rank <= 0:
                        try:
                            rank = int(_get_rank_map().get(si, 0) or 0)
                        except Exception:
                            rank = 0
                    if rank <= 0:
                        rank = int(si) + 1
                    rd['group_rank'] = float(rank)

                    if is_manual:
                        try:
                            ms = int(float(rd.get('manual_seq', float('nan'))))
                        except Exception:
                            ms = 0
                        # Avoid bumping sequence repeatedly during table refresh.
                        # Only allocate a new manual sequence when we truly need a new/generated name.
                        if ms <= 0 and (manual_name_override or (not str(rd.get('name', '') or '').strip())):
                            ms = int(self._next_manual_name_seq())
                        if ms > 0:
                            rd['manual_seq'] = float(ms)
                    else:
                        rd['manual_seq'] = float('nan')
                        rd['generation'] = float(int(getattr(self, 'centroid_generation', 0) or 0))

                    try:
                        cnam = str(rd.get('custom_name', '') or '').strip()
                    except Exception:
                        cnam = ''
                    if cnam:
                        rd['name'] = str(cnam)
                        rd['custom_name'] = str(cnam)
                    else:
                        if is_manual:
                            cur_name = str(rd.get('name', '') or '').strip()
                            if cur_name:
                                rd['name'] = str(cur_name)
                            else:
                                rd['name'] = str(self._build_center_row_name(
                                    manual=True,
                                    group_no=int(rd.get('group_no', 0) or 0),
                                    group_rank=int(rd.get('group_rank', 0) or 0),
                                    pos_tag=pp,
                                    generation=int(rd.get('generation', getattr(self, 'centroid_generation', 0) or 0) or 0),
                                    manual_seq=(int(ms) if int(ms) > 0 else None),
                                ))
                                rd['custom_name'] = ''
                        else:
                            rd['name'] = str(self._build_center_row_name(
                                manual=False,
                                group_no=int(rd.get('group_no', 0) or 0),
                                group_rank=int(rd.get('group_rank', 0) or 0),
                                pos_tag=pp,
                                generation=int(rd.get('generation', getattr(self, 'centroid_generation', 0) or 0) or 0),
                                manual_seq=None,
                            ))
                            rd['custom_name'] = ''
                    try:
                        self._update_center_name_max_len(rd.get('name', ''))
                    except Exception:
                        pass
                    r.update(rd)
                except Exception:
                    pass
                try:
                    nv = int(round(float(r.get('no', float('nan')))))
                    if nv > max_no:
                        max_no = nv
                except Exception:
                    pass
            missing_pairs = []
            try:
                for ii, ptag in requested_pairs:
                    if (ii, ptag) not in existing:
                        # Integer-origin additions should not recreate implicit 'c' when same source already exists.
                        if ptag == 'c' and ii in existing_sources:
                            continue
                        missing_pairs.append((ii, ptag))
            except Exception:
                missing_pairs = []

            would_mutate = bool(refresh_existing) or bool(missing_pairs)
            if would_mutate and src_tag not in allowed_sources:
                try:
                    warn = getattr(self, '_log_warn', None)
                    if callable(warn):
                        preview = ", ".join([f"{a}:{b}" for (a, b) in missing_pairs[:6]])
                        warn(
                            "Blocked unexpected center row mutation "
                            f"source={src_tag} refresh_existing={bool(refresh_existing)} "
                            f"missing={len(missing_pairs)} preview=[{preview}]"
                        )
                except Exception:
                    pass
                return

            for i in (indices or []):
                raw_ptag = 'c'
                if isinstance(i, (tuple, list)) and len(i) >= 2:
                    try:
                        ii = int(i[0])
                    except Exception:
                        continue
                    try:
                        raw_ptag = str(i[1] or '').lower().strip()
                    except Exception:
                        raw_ptag = ''
                else:
                    try:
                        ii = int(i)
                    except Exception:
                        continue
                    # Integer entries come from explicit center list indices.
                    # If the source already has any row (e.g. only 'r' remains after Clear),
                    # do not auto-recreate an implicit 'c' row.
                    if ii in existing_sources:
                        continue
                # Store the user-chosen pos (c or r) directly — no forced override.
                disp_ptag = raw_ptag if raw_ptag in ('', 'c', 'r') else ''
                ptag = raw_ptag if raw_ptag in ('c', 'r') else 'c'
                if (ii, ptag) in existing:
                    continue
                try:
                    snap_rank_map = _get_rank_map()
                except Exception:
                    snap_rank_map = None
                snap = self._snapshot_center_row_from_centroid(ii, pos=ptag, group_rank_map=snap_rank_map, manual_name_override=manual_name_override, manual_seq_override=manual_seq_override)
                if snap is None:
                    continue
                try:
                    snap['pos_display'] = str(disp_ptag)
                except Exception:
                    pass
                max_no += 1
                snap['no'] = float(max_no)
                rows.append(snap)
                existing.add((ii, ptag))
                existing_sources.add(ii)
            self.center_numeric_rows = rows
        except Exception:
            pass

    def _center_rows_need_append(self, indices):
        """Return True when any requested center entry is missing from cached rows."""
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            existing = set()
            existing_sources = set()
            for r in rows:
                try:
                    rd = dict(r or {})
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                existing.add((si, pp))
                existing_sources.add(si)

            for i in (indices or []):
                ptag = 'c'
                if isinstance(i, (tuple, list)) and len(i) >= 2:
                    try:
                        ii = int(i[0])
                    except Exception:
                        continue
                    try:
                        ptag = str(i[1] or 'c').lower().strip()
                    except Exception:
                        ptag = 'c'
                    if ptag not in ('c', 'r'):
                        ptag = 'c'
                    if (ii, ptag) not in existing:
                        return True
                else:
                    try:
                        ii = int(i)
                    except Exception:
                        continue
                    if ii not in existing_sources:
                        return True
            return False
        except Exception:
            return True

    def _sync_center_xyz_from_stage_info(self, info):
        """Compute stage X,Y,Z for each center row using the fitted transform; return True if any value changed."""
        try:
            import numpy as _np
            s = float(info.get('s', 1.0))
            R = _np.asarray(info.get('R'), dtype=_np.float64)
            t = _np.asarray(info.get('t'), dtype=_np.float64)
            reflect = bool(info.get('reflect', False))
            z_plane = info.get('z_plane', None)  # (a, b, c) where Z = a*X + b*Y + c
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
            h_full = None
            try:
                if getattr(self, '_img_base_size', None) is not None:
                    h_full = int(self._img_base_size[1])
            except Exception:
                pass
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not rows:
                return False
            changed = False
            for r in rows:
                try:
                    xp = float(r.get('x_proc', float('nan')))
                    yp = float(r.get('y_proc', float('nan')))
                    if _np.isnan(xp) or _np.isnan(yp):
                        uv = self._proc_from_center_uv(r.get('u', float('nan')), r.get('v', float('nan')))
                        if uv is None:
                            continue
                        xp, yp = uv
                    x_full = xp * spf
                    y_full = yp * spf
                    u = float(x_full)
                    v = float((h_full - 1) - y_full) if (h_full is not None and h_full > 0) else float(-y_full)
                    if reflect:
                        u = -u
                    stage = s * (R @ _np.asarray([u, v], dtype=_np.float64)) + t
                    new_x = float(stage[0])
                    new_y = float(stage[1])
                    old_x = float(r.get('x', float('nan')))
                    old_y = float(r.get('y', float('nan')))
                    if _np.isnan(old_x) or abs(new_x - old_x) > 0.001 or _np.isnan(old_y) or abs(new_y - old_y) > 0.001:
                        r['x'] = new_x
                        r['y'] = new_y
                        changed = True
                    if z_plane is not None:
                        try:
                            a, b, c = float(z_plane[0]), float(z_plane[1]), float(z_plane[2])
                            new_z = a * new_x + b * new_y + c
                            old_z = float(r.get('z', float('nan')))
                            if _np.isnan(old_z) or abs(new_z - old_z) > 0.001:
                                r['z'] = new_z
                                changed = True
                        except Exception:
                            pass
                except Exception:
                    continue
            return changed
        except Exception:
            return False

    def _sync_center_numeric_rows_xyz_from_table(self):
        """Refresh center-row XYZ values from the canonical right table when available."""
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not rows:
                return
            src = getattr(self, 'table', None)
            if src is None:
                return
            changed = False
            for i, r in enumerate(rows):
                try:
                    rd = dict(r or {})
                except Exception:
                    continue
                try:
                    idx = int(rd.get('source_idx', -1))
                except Exception:
                    idx = -1
                if not (0 <= idx < src.columnCount()):
                    continue
                try:
                    x_it = src.item(4, idx)
                    y_it = src.item(5, idx)
                    z_it = src.item(6, idx)
                    x_val = self._safe_float_or_nan(x_it.text() if x_it is not None else "")
                    y_val = self._safe_float_or_nan(y_it.text() if y_it is not None else "")
                    z_val = self._safe_float_or_nan(z_it.text() if z_it is not None else "")
                except Exception:
                    continue
                prev = (rd.get('x', float('nan')), rd.get('y', float('nan')), rd.get('z', float('nan')))
                rd['x'] = float(x_val)
                rd['y'] = float(y_val)
                rd['z'] = float(z_val)
                rows[i] = rd
                try:
                    p0 = float(prev[0]); p1 = float(prev[1]); p2 = float(prev[2])
                    if (not np.isfinite(p0) and np.isfinite(x_val)) or (not np.isfinite(p1) and np.isfinite(y_val)) or (not np.isfinite(p2) and np.isfinite(z_val)):
                        changed = True
                except Exception:
                    changed = True
            if changed:
                self.center_numeric_rows = rows
            else:
                # Keep latest numeric values even when they changed among finite values.
                self.center_numeric_rows = rows
        except Exception:
            pass

    def _is_center_row_visible(self, rowd):
        """Return True when a center row should be visible in overlay/table."""
        try:
            rd = dict(rowd or {})
        except Exception:
            rd = {}
        try:
            if float(rd.get('show', 1.0)) < 0.5:
                return False
        except Exception:
            pass
        return True

    def _set_center_row_visible(self, row_idx, visible, source_idx=None):
        """Apply Show/Hide from middle table row and refresh overlay/table."""
        try:
            self._mark_center_model_mutation()
            rr = int(row_idx)
        except Exception:
            return
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not (0 <= rr < len(rows)):
                return
            rd = dict(rows[rr] or {})
            try:
                prev_vis = bool(float(rd.get('show', 1.0)) >= 0.5)
            except Exception:
                prev_vis = True
            if bool(prev_vis) != bool(visible):
                self._push_center_undo_state()
            rd['show'] = 1.0 if bool(visible) else 0.0
            rows[rr] = rd
            self.center_numeric_rows = rows
        except Exception:
            pass

        try:
            # Keep Show/Hide interaction responsive: refresh middle/overlay only.
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False)
        except Exception:
            pass
        try:
            if str(getattr(self, 'overlay_point_source', 'left') or 'left') == 'center':
                self._apply_proc_zoom()
            else:
                self._refresh_selected_overlay_only()
        except Exception:
            try:
                self._apply_proc_zoom()
            except Exception:
                pass

    def _toggle_center_row_visible_by_view_row(self, view_row):
        """Toggle Show/Hide by middle-table view row index (excluding 2 header rows)."""
        try:
            vr = int(view_row)
        except Exception:
            return
        try:
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            if not (0 <= vr < len(row_keys)):
                return
            si = int(row_keys[vr][0])
            pp = str(row_keys[vr][1] or 'c').lower().strip()
        except Exception:
            return
        if pp not in ('c', 'r'):
            pp = 'c'

        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            for i, rr in enumerate(rows):
                try:
                    rd = dict(rr or {})
                    r_si = int(rd.get('source_idx', -1))
                    r_pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                except Exception:
                    continue
                if r_pp not in ('c', 'r'):
                    r_pp = 'c'
                if r_si == si and r_pp == pp:
                    cur = bool(self._is_center_row_visible(rd))
                    self._set_center_row_visible(i, (not cur), si)
                    break
        except Exception:
            pass

    def _make_show_toggle_center_row(self, row_idx, rowd):
        """Create an iOS-style Show/Hide toggle for the middle table row."""
        try:
            rr = int(row_idx)
        except Exception:
            rr = -1
        try:
            rd = dict(rowd or {})
        except Exception:
            rd = {}
        try:
            src_i = int(rd.get('source_idx', -1))
        except Exception:
            src_i = -1
        checked = bool(self._is_center_row_visible(rd))

        try:
            from qt_compat.QtWidgets import QWidget as _QW
            from qt_compat.QtCore import QRectF
            from qt_compat.QtGui import QPainter, QColor as _QC

            class _Toggle(_QW):
                def __init__(self, checked=True, parent=None):
                    super().__init__(parent)
                    self._checked = bool(checked)
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
                        self._cb(bool(self._checked))

            sw = _Toggle(checked=checked)
            sw._cb = lambda is_visible, r=rr, si=src_i: self._set_center_row_visible(r, bool(is_visible), si)
            return sw
        except Exception:
            return None

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

    def _remove_center_numeric_row_by_key(self, source_idx, pos_tag):
        """Remove middle-table row(s) matching exact key (source_idx, pos)."""
        try:
            self._mark_center_model_mutation()
            si = int(source_idx)
        except Exception:
            return False
        try:
            ptag = str(pos_tag or 'c').lower().strip()
        except Exception:
            ptag = 'c'
        if ptag not in ('c', 'r'):
            ptag = 'c'

        changed = False
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
        except Exception:
            rows = []

        kept = []
        for rr in rows:
            try:
                rd = dict(rr or {})
            except Exception:
                rd = {}
            try:
                r_si = int(rd.get('source_idx', -1))
            except Exception:
                r_si = -1
            try:
                r_pos = str(rd.get('pos', 'c') or 'c').lower().strip()
            except Exception:
                r_pos = 'c'
            if r_pos not in ('c', 'r'):
                r_pos = 'c'

            if r_si == si and r_pos == ptag:
                changed = True
                continue
            kept.append(rd)

        if not changed:
            return False

        self.center_numeric_rows = kept

        # Keep explicit source index only while any row for that source remains.
        try:
            still_has_source = False
            for rr in kept:
                try:
                    if int(dict(rr or {}).get('source_idx', -1)) == si:
                        still_has_source = True
                        break
                except Exception:
                    continue
            if not still_has_source:
                explicit = list(self._get_explicit_center_list_indices())
                new_explicit = [int(i) for i in explicit if int(i) != si]
                if len(new_explicit) != len(explicit):
                    self.center_list_indices = new_explicit
        except Exception:
            pass
        return True

    def _remove_center_target_by_source_idx(self, cidx):
        """Remove one middle-table target represented by source_idx.

        If the target belongs to the manual-target block, remove the underlying
        manual target and shift centroid indices accordingly. Otherwise, remove
        the explicit middle-list entry and its rendered rows.
        """
        try:
            tgt = int(cidx)
        except Exception:
            return False

        changed = False
        try:
            mt_n = len(getattr(self, 'manual_targets', []) or [])
        except Exception:
            mt_n = 0
        try:
            base = int(self._manual_target_base_index())
        except Exception:
            base = 0

        # Manual-target block: remove the underlying point.
        if mt_n > 0 and base <= tgt < (base + mt_n):
            rem = int(tgt - base)
            try:
                self.manual_targets.pop(rem)
                changed = True
            except Exception:
                return False
            try:
                self._shift_center_list_indices(tgt, -1)
            except Exception:
                pass
            try:
                old_excl = set(getattr(self, 'excluded_centroid_indices', set()) or set())
                new_excl = set()
                for i in old_excl:
                    ii = int(i)
                    if ii == tgt:
                        continue
                    if ii > tgt:
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
                    if ii == tgt:
                        continue
                    if ii > tgt:
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
                    if ii == tgt:
                        continue
                    if ii > tgt:
                        ii -= 1
                    new_force.add(ii)
                self._force_visible_centroid_indices = new_force
            except Exception:
                pass
            auto_only = list(getattr(self, '_auto_centroids', []) or self._auto_centroids_from_current())
            self.centroids = self._compose_centroids_with_manual(auto_only)
            try:
                self._centroid_rim_proc_points = self._compose_rim_points_with_manual(auto_only, getattr(self, '_auto_rim_proc_points', []))
            except Exception:
                self._centroid_rim_proc_points = []
            return changed

        # Explicit center-list row: remove the row snapshot and its source entry.
        try:
            explicit = list(self._get_explicit_center_list_indices())
            new_explicit = [int(i) for i in explicit if int(i) != tgt]
            if len(new_explicit) != len(explicit):
                self.center_list_indices = new_explicit
                changed = True
        except Exception:
            pass
        try:
            self._remove_center_numeric_row_by_source_idx(tgt)
        except Exception:
            pass
        return changed

    def _center_sort_click_key(self, col):
        try:
            c = int(col)
        except Exception:
            return None
        try:
            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        if stage_n == 'online':
            return {0: 'no', 2: 'u', 3: 'v', 4: 'x', 5: 'y', 6: 'z'}.get(c)
        return {0: 'no', 2: 'u', 3: 'v', 4: 'grp', 5: 'pno', 6: 'cr', 7: 'gen'}.get(c)

    def _toggle_center_sort(self, key):
        try:
            k = str(key or '').lower().strip()
        except Exception:
            return
        if k not in ('no', 'u', 'v', 'grp', 'pno', 'cr', 'gen', 'x', 'y', 'z'):
            return
        try:
            cur = str(getattr(self, '_center_sort_key', 'no') or 'no').lower().strip()
        except Exception:
            cur = 'no'
        try:
            desc = bool(getattr(self, '_center_sort_desc', False))
        except Exception:
            desc = False
        if cur == k:
            desc = not desc
        else:
            cur = k
            desc = False
        self._center_sort_key = cur
        self._center_sort_desc = bool(desc)

    def _sort_center_rows(self, rows):
        try:
            src = list(rows or [])
            if len(src) <= 1:
                return src
            key = str(getattr(self, '_center_sort_key', 'no') or 'no').lower().strip()
            desc = bool(getattr(self, '_center_sort_desc', False))

            if key == 'no':
                def _no_val(rr):
                    try:
                        rd = dict(rr or {})
                    except Exception:
                        rd = {}
                    try:
                        nv = float(rd.get('no', float('nan')))
                        if np.isnan(nv):
                            raise ValueError('nan')
                        return int(round(nv))
                    except Exception:
                        try:
                            return int(rd.get('source_idx', -1)) + 1
                        except Exception:
                            return 0
                indexed = list(enumerate(src))
                if not desc:
                    indexed.sort(key=lambda it: (_no_val(it[1]), it[0]))
                else:
                    indexed.sort(key=lambda it: (-_no_val(it[1]), it[0]))
                return [r for _i, r in indexed]

            def _num(rr):
                try:
                    rd = dict(rr or {})
                except Exception:
                    rd = {}
                k2 = key
                if k2 == 'grp':
                    k2 = 'group_no'
                elif k2 == 'pno':
                    k2 = 'group_rank'
                elif k2 == 'gen':
                    k2 = 'generation'
                vv = rd.get(k2, float('nan'))
                try:
                    f = float(vv)
                    if np.isnan(f):
                        return None
                    return f
                except Exception:
                    return None

            def _txt(rr):
                try:
                    rd = dict(rr or {})
                except Exception:
                    rd = {}
                if key == 'cr':
                    try:
                        pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                    except Exception:
                        pp = 'c'
                    if pp not in ('c', 'r'):
                        pp = 'c'
                    return ('R' if pp == 'r' else 'C')
                return ''

            if key == 'cr':
                indexed = list(enumerate(src))
                if not desc:
                    indexed.sort(key=lambda it: (_txt(it[1]), it[0]))
                else:
                    indexed.sort(key=lambda it: (_txt(it[1]) != 'R', it[0]))
                return [r for _i, r in indexed]

            indexed = list(enumerate(src))
            if not desc:
                indexed.sort(key=lambda it: ((_num(it[1]) is None), (_num(it[1]) if _num(it[1]) is not None else 0.0), it[0]))
            else:
                indexed.sort(key=lambda it: ((_num(it[1]) is None), -(_num(it[1]) if _num(it[1]) is not None else 0.0), it[0]))
            return [r for _i, r in indexed]
        except Exception:
            return list(rows or [])

    def _center_label_with_sort(self, key, base_text):
        try:
            cur = str(getattr(self, '_center_sort_key', 'no') or 'no').lower().strip()
            desc = bool(getattr(self, '_center_sort_desc', False))
        except Exception:
            cur = 'no'
            desc = False
        if str(key) == str(cur):
            return f"{base_text} {'▼' if desc else '▲'}"
        return str(base_text)

    def _sorted_group_entries(self, group_no):
        """Return [(u, v, source_idx), ...] for a group sorted by current left-panel sort state."""
        out = []
        try:
            g = int(group_no)
        except Exception:
            return out
        centroids = list(getattr(self, 'centroids', []) or [])
        cache_uv = None
        try:
            cache_uv = list(getattr(self, '_cache', {}).get('centroids_full_uv') or [])
        except Exception:
            cache_uv = None

        if cache_uv and len(cache_uv) == len(centroids):
            for src_i, (gg, u, v) in enumerate(cache_uv):
                try:
                    if int(gg) != g:
                        continue
                    out.append((int(u), int(v), int(src_i)))
                except Exception:
                    continue
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

            for src_i, (gg, xp, yp) in enumerate(centroids):
                try:
                    if int(gg) != g:
                        continue
                    x_full = float(xp) * spf
                    y_full = float(yp) * spf
                    u = int(round(x_full))
                    if h_full is not None and h_full > 0:
                        v = int(round((h_full - 1) - y_full))
                    else:
                        v = int(round(-y_full))
                    out.append((int(u), int(v), int(src_i)))
                except Exception:
                    continue

        try:
            st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
            rec = dict(st.get(int(g), {}) or {})
            key = str(rec.get('key', getattr(self, '_offline_group_sort_key', 'u')) or 'u').lower().strip()
        except Exception:
            key = 'u'
        if key not in ('u', 'v'):
            key = 'u'
        try:
            st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
            rec = dict(st.get(int(g), {}) or {})
            desc = bool(rec.get('desc', getattr(self, '_offline_group_sort_desc', False)))
        except Exception:
            desc = False
        kidx = 0 if key == 'u' else 1
        out.sort(key=lambda t: (int(t[kidx]), int(t[1 - kidx]), int(t[2])), reverse=bool(desc))
        return out

    def _on_offline_group_header_clicked(self, group_no, section):
        try:
            sec = int(section)
        except Exception:
            return
        if sec not in (0, 1):
            return
        key = 'u' if sec == 0 else 'v'
        try:
            g = int(group_no)
        except Exception:
            return
        try:
            st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
            rec = dict(st.get(int(g), {}) or {})
            cur_key = str(rec.get('key', getattr(self, '_offline_group_sort_key', 'u')) or 'u').lower().strip()
            cur_desc = bool(rec.get('desc', getattr(self, '_offline_group_sort_desc', False)))
        except Exception:
            cur_key = 'u'
            cur_desc = False
        if key == cur_key:
            cur_desc = not cur_desc
        else:
            cur_key = key
            cur_desc = False
        try:
            st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
            st[int(g)] = {'key': str(cur_key), 'desc': bool(cur_desc)}
            self._offline_group_sort_state = st
        except Exception:
            pass
        try:
            self._refresh_offline_group_lists()
        except Exception:
            pass
        try:
            # Left-list sort order affects overlay label numbering (Gx-y),
            # so force a lightweight redraw to refresh labels immediately.
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _center_group_display_name(self, group_no):
        """Return display label for a group in centroid-extraction offline list."""
        try:
            g = int(group_no)
        except Exception:
            g = 0
        try:
            raw = str(dict(getattr(self, 'center_group_name_overrides', {}) or {}).get(g, '') or '').strip()
        except Exception:
            raw = ''
        if raw:
            return str(raw)
        return f"Group {int(g)}"

    def _on_center_group_name_text_changed(self, group_no, text):
        """Store manual group-name text and refresh matching auto row names."""
        try:
            g = int(group_no)
        except Exception:
            return
        try:
            t = str(text or '').strip()
        except Exception:
            t = ''
        try:
            names = dict(getattr(self, 'center_group_name_overrides', {}) or {})
        except Exception:
            names = {}
        if t:
            names[g] = str(t)
        else:
            try:
                names.pop(g, None)
            except Exception:
                pass
        self.center_group_name_overrides = names

        try:
            self._refresh_center_auto_names_from_group_overrides(target_groups={int(g)})
        except Exception:
            pass

    def _refresh_center_auto_names_from_group_overrides(self, target_groups=None, refresh_views=True):
        """Refresh auto-generated center row names for specified groups (or all groups)."""
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not rows:
                return False
        except Exception:
            return False

        gset = None
        if target_groups is not None:
            try:
                gset = {int(x) for x in (target_groups or set())}
            except Exception:
                gset = None

        changed = False
        for i, rr in enumerate(rows):
            try:
                rd = dict(rr or {})
            except Exception:
                continue
            try:
                is_manual = bool(float(rd.get('manual', 0.0)) >= 0.5)
            except Exception:
                is_manual = False
            if is_manual:
                continue
            try:
                cnam = str(rd.get('custom_name', '') or '').strip()
            except Exception:
                cnam = ''
            if cnam:
                continue
            try:
                gno = int(rd.get('group_no', rd.get('grp', 0) or 0))
            except Exception:
                gno = 0
            if gset is not None and int(gno) not in gset:
                continue
            try:
                pp = str(rd.get('pos', 'c') or 'c').lower().strip()
            except Exception:
                pp = 'c'
            if pp not in ('c', 'r'):
                pp = 'c'
            try:
                rk = int(rd.get('group_rank', 0) or 0)
            except Exception:
                rk = 0
            if rk <= 0:
                try:
                    rk = int(rd.get('source_idx', -1)) + 1
                except Exception:
                    rk = 1
            if rk <= 0:
                rk = 1
            try:
                gen = int(rd.get('generation', getattr(self, 'centroid_generation', 0) or 0) or 0)
            except Exception:
                gen = int(getattr(self, 'centroid_generation', 0) or 0)
            new_name = str(self._build_center_row_name(
                manual=False,
                group_no=gno,
                group_rank=rk,
                pos_tag=pp,
                generation=gen,
                manual_seq=None,
            ))
            if str(rd.get('name', '') or '') == new_name:
                continue
            rd['name'] = new_name
            rows[i] = rd
            changed = True

        if not changed:
            return False
        try:
            self._mark_center_model_mutation()
        except Exception:
            pass
        self.center_numeric_rows = rows
        try:
            self._rebuild_center_name_max_len_from_rows()
        except Exception:
            pass
        if bool(refresh_views):
            try:
                self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False, refresh_center_view=True)
            except Exception:
                pass
            try:
                if str(getattr(self, 'overlay_point_source', 'left') or 'left') == 'center':
                    self._apply_proc_zoom()
                else:
                    self._refresh_selected_overlay_only()
            except Exception:
                pass
        return True

    def _count_groups_from_centroids(self, centroids):
        out = {}
        for c in (centroids or []):
            try:
                g = int(c[0])
            except Exception:
                continue
            if g <= 0:
                continue
            out[g] = int(out.get(g, 0) + 1)
        return out

    def _capture_group_name_transfer_state(self):
        """Capture current named-group color/count state for post-recompute transfer."""
        try:
            names = {}
            for k, v in dict(getattr(self, 'center_group_name_overrides', {}) or {}).items():
                try:
                    g = int(k)
                except Exception:
                    continue
                if g <= 0:
                    continue
                try:
                    nm = str(v or '').strip()
                except Exception:
                    nm = ''
                if nm:
                    names[g] = nm
        except Exception:
            names = {}
        if not names:
            return None

        try:
            raw_colors = dict(getattr(self, '_cache', {}).get('group_header_rgb') or {})
        except Exception:
            raw_colors = {}
        colors = {}
        for g, rgb in raw_colors.items():
            try:
                gi = int(g)
            except Exception:
                continue
            if gi <= 0:
                continue
            try:
                r = int(rgb[0]); gg = int(rgb[1]); b = int(rgb[2])
            except Exception:
                continue
            colors[gi] = (r, gg, b)

        try:
            counts = self._count_groups_from_centroids(list(getattr(self, '_auto_centroids', []) or []))
        except Exception:
            counts = {}

        return {
            'names': dict(names),
            'colors': dict(colors),
            'counts': dict(counts),
        }

    def _transfer_group_name_overrides_after_recompute(self, prev_state, new_centroids):
        """Transfer group-name overrides by nearest color; resolve collisions by larger old-group count."""
        if not isinstance(prev_state, dict):
            return False
        try:
            old_names = dict(prev_state.get('names', {}) or {})
        except Exception:
            old_names = {}
        if not old_names:
            return False

        try:
            old_colors = dict(prev_state.get('colors', {}) or {})
        except Exception:
            old_colors = {}
        try:
            old_counts = dict(prev_state.get('counts', {}) or {})
        except Exception:
            old_counts = {}

        try:
            new_counts = dict(self._count_groups_from_centroids(new_centroids))
        except Exception:
            new_counts = {}
        if not new_counts:
            return False

        try:
            raw_new_colors = dict(getattr(self, '_cache', {}).get('group_header_rgb') or {})
        except Exception:
            raw_new_colors = {}
        new_colors = {}
        for g, rgb in raw_new_colors.items():
            try:
                gi = int(g)
            except Exception:
                continue
            if gi <= 0 or gi not in new_counts:
                continue
            try:
                new_colors[gi] = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
            except Exception:
                continue

        candidates = []
        for old_g, nm in old_names.items():
            try:
                og = int(old_g)
            except Exception:
                continue
            try:
                name_txt = str(nm or '').strip()
            except Exception:
                name_txt = ''
            if not name_txt:
                continue

            old_cnt = int(old_counts.get(og, 0) or 0)
            best_g = None
            best_d = float('inf')

            if og in old_colors and new_colors:
                try:
                    orr, org, orb = [float(x) for x in old_colors[og]]
                except Exception:
                    orr = org = orb = None
                if orr is not None:
                    for ng, nrgb in new_colors.items():
                        try:
                            dr = float(orr - nrgb[0])
                            dg = float(org - nrgb[1])
                            db = float(orb - nrgb[2])
                            dd = float((dr * dr) + (dg * dg) + (db * db))
                        except Exception:
                            continue
                        if (dd < best_d) or (dd == best_d and int(new_counts.get(ng, 0)) > int(new_counts.get(best_g, 0) if best_g is not None else -1)):
                            best_d = dd
                            best_g = int(ng)

            if best_g is None:
                if og in new_counts:
                    best_g = int(og)
                    best_d = 0.0
                else:
                    try:
                        # Fallback when color cannot be compared: choose nearest group id.
                        best_g = min(new_counts.keys(), key=lambda k: (abs(int(k) - int(og)), -int(new_counts.get(int(k), 0)), int(k)))
                        best_d = float('inf')
                    except Exception:
                        continue

            candidates.append((int(best_g), int(og), float(best_d), int(old_cnt), str(name_txt)))

        if not candidates:
            return False

        chosen = {}
        for new_g, old_g, dist, old_cnt, nm in candidates:
            prev = chosen.get(int(new_g), None)
            cur = (int(old_cnt), -float(dist), -int(old_g), str(nm))
            if prev is None:
                chosen[int(new_g)] = (cur, str(nm))
                continue
            if cur > prev[0]:
                chosen[int(new_g)] = (cur, str(nm))

        new_names = {}
        for g, rec in chosen.items():
            try:
                txt = str(rec[1] or '').strip()
            except Exception:
                txt = ''
            if txt:
                new_names[int(g)] = txt

        try:
            prev_names = dict(getattr(self, 'center_group_name_overrides', {}) or {})
        except Exception:
            prev_names = {}
        if str(prev_names) == str(new_names):
            return False
        self.center_group_name_overrides = dict(new_names)
        try:
            self._refresh_center_auto_names_from_group_overrides(refresh_views=False)
        except Exception:
            pass
        return True

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
            pre_existing_set = set(existing_set)
            added_entries = []
            core_on = bool(getattr(self, 'center_add_core_enabled', True))
            rim_on = bool(getattr(self, 'center_add_rim_enabled', True))
            if (not core_on) and (not rim_on):
                core_on = True
            for _u, _v, idx in (self._sorted_group_entries(g) or []):
                if idx not in existing_set:
                    existing.append(int(idx))
                    existing_set.add(int(idx))
                if core_on:
                    added_entries.append((int(idx), 'c'))
                if rim_on:
                    added_entries.append((int(idx), 'r'))
            try:
                added_entries = list(self._filter_add_entries_by_uv_similarity(added_entries) or [])
            except Exception:
                pass
            try:
                accepted_sources = {int(e[0]) for e in (added_entries or [])}
                existing = [int(si) for si in (existing or []) if (int(si) in pre_existing_set) or (int(si) in accepted_sources)]
                existing_set = set(existing)
            except Exception:
                pass
            self.center_list_indices = existing
            try:
                self._append_center_numeric_rows_from_indices(added_entries, refresh_existing=False, source='add_to_list')
            except Exception:
                pass
            # Rebuild canonical names for auto rows in the just-added sources.
            # Keep explicit user renames (custom_name) untouched.
            try:
                touched_sources = set(int(e[0]) for e in (added_entries or []))
                if touched_sources:
                    rank_map = self._center_group_rank_map()
                    rows = list(getattr(self, 'center_numeric_rows', []) or [])
                    cents = list(getattr(self, 'centroids', []) or [])
                    gen_now = int(getattr(self, 'centroid_generation', 0) or 0)
                    changed = False
                    for i, rr in enumerate(rows):
                        try:
                            rd = dict(rr or {})
                            si = int(rd.get('source_idx', -1))
                        except Exception:
                            continue
                        if si not in touched_sources:
                            continue
                        try:
                            cnam = str(rd.get('custom_name', '') or '').strip()
                        except Exception:
                            cnam = ''
                        if cnam:
                            continue
                        try:
                            is_manual = bool(float(rd.get('manual', 0.0)) >= 0.5)
                        except Exception:
                            is_manual = False
                        if is_manual:
                            continue
                        try:
                            pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                        except Exception:
                            pp = 'c'
                        if pp not in ('c', 'r'):
                            pp = 'c'
                        try:
                            gno = int(rd.get('group_no', rd.get('grp', 0) or 0))
                        except Exception:
                            gno = 0
                        if gno <= 0:
                            try:
                                if 0 <= si < len(cents):
                                    gno = int(cents[si][0])
                            except Exception:
                                pass
                        try:
                            rank = int(rank_map.get(si, 0) or 0)
                        except Exception:
                            rank = 0
                        if rank <= 0:
                            rank = int(si) + 1
                        rd['grp'] = float(gno)
                        rd['group_no'] = float(gno)
                        rd['group_rank'] = float(rank)
                        rd['generation'] = float(gen_now)
                        rd['name'] = str(self._build_center_row_name(
                            manual=False,
                            group_no=gno,
                            group_rank=rank,
                            pos_tag=pp,
                            generation=gen_now,
                            manual_seq=None,
                        ))
                        rows[i] = rd
                        changed = True
                    if changed:
                        self.center_numeric_rows = rows
            except Exception:
                pass
            if added_entries and getattr(self, 'selected_index', None) not in set(existing):
                try:
                    self.selected_index = int(added_entries[0][0])
                except Exception:
                    pass
        except Exception:
            return
        try:
            # Keep Add fast: refresh middle list only (skip left/offline rebuild).
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False)
        except Exception:
            pass
        try:
            if str(getattr(self, 'overlay_point_source', 'left') or 'left') == 'center':
                self._apply_proc_zoom()
            else:
                self._refresh_selected_overlay_only()
        except Exception:
            pass
        try:
            if added_entries:
                QTimer.singleShot(0, self._scroll_center_table_to_bottom)
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

    def _center_uv_from_source_key(self, source_idx, pos_tag='c'):
        """Return integer (u, v) for a centroid key (source_idx, 'c'|'r'), or None."""
        try:
            si = int(source_idx)
        except Exception:
            return None
        try:
            ptag = str(pos_tag or 'c').lower().strip()
        except Exception:
            ptag = 'c'
        if ptag not in ('c', 'r'):
            ptag = 'c'
        try:
            cents = list(getattr(self, 'centroids', []) or [])
            if not (0 <= si < len(cents)):
                return None
            _g, xp, yp = cents[si]
            x_proc = float(xp)
            y_proc = float(yp)
            if ptag == 'r':
                rim_list = list(getattr(self, '_centroid_rim_proc_points', []) or [])
                rim_pt = rim_list[si] if 0 <= si < len(rim_list) else None
                if rim_pt is None:
                    return None
                x_proc = float(rim_pt[0])
                y_proc = float(rim_pt[1])
            u, v = self._center_uv_from_proc(x_proc, y_proc)
            return (int(round(float(u))), int(round(float(v))))
        except Exception:
            return None

    def _filter_add_entries_by_uv_similarity(self, entries):
        """Filter (source_idx,pos) entries by configured UV similarity threshold."""
        try:
            cand = list(entries or [])
        except Exception:
            return []
        if not cand:
            return []

        try:
            th = int(getattr(self, 'center_add_uv_similarity_px', 2) or 2)
        except Exception:
            th = 2
        th = max(0, int(th))

        existing_uv = []
        try:
            for rr in (getattr(self, 'center_numeric_rows', []) or []):
                try:
                    rd = dict(rr or {})
                    uu = float(rd.get('u', float('nan')))
                    vv = float(rd.get('v', float('nan')))
                    if np.isfinite(uu) and np.isfinite(vv):
                        existing_uv.append((int(round(uu)), int(round(vv))))
                except Exception:
                    continue
        except Exception:
            existing_uv = []

        accepted = []
        accepted_uv = list(existing_uv)
        for e in cand:
            try:
                si = int(e[0])
                pp = str(e[1] if len(e) > 1 else 'c').lower().strip()
            except Exception:
                continue
            if pp not in ('c', 'r'):
                pp = 'c'
            uv = self._center_uv_from_source_key(si, pp)
            if uv is None:
                continue
            similar = False
            try:
                cu, cv = uv
                for eu, ev in accepted_uv:
                    if abs(int(cu) - int(eu)) <= th and abs(int(cv) - int(ev)) <= th:
                        similar = True
                        break
            except Exception:
                similar = False
            if similar:
                continue
            accepted.append((si, pp))
            accepted_uv.append(uv)
        return accepted

    def _set_group_visible(self, group_no, visible):
        idxs = self._group_centroid_indices(group_no)
        if not idxs:
            return
        try:
            all_groups = set(self._available_group_numbers())
            cur = self._get_visible_groups_set()
            if cur is None:
                cur = set(all_groups)
            if bool(visible):
                cur.add(int(group_no))
            else:
                cur.discard(int(group_no))
            self.visible_groups = None if cur == all_groups else cur
        except Exception:
            pass
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
        try:
            vis = self._get_visible_groups_set()
            if vis is not None:
                return int(group_no) in vis
        except Exception:
            pass
        idxs = self._group_centroid_indices(group_no)
        if not idxs:
            return True
        excl = set(getattr(self, 'excluded_centroid_indices', set()) or set())
        return any((ci not in excl) for ci in idxs)

    def _add_all_groups_to_center_list(self):
        try:
            existing = self._get_explicit_center_list_indices()
            existing_set = set(existing)
            pre_existing_set = set(existing_set)
            added_entries = []
            core_on = bool(getattr(self, 'center_add_core_enabled', True))
            rim_on = bool(getattr(self, 'center_add_rim_enabled', True))
            groups = self._available_group_numbers()
            for g in groups:
                for ci in self._group_centroid_indices(g):
                    if ci not in existing_set:
                        existing.append(int(ci))
                        existing_set.add(int(ci))
                    if core_on:
                        added_entries.append((int(ci), 'c'))
                    if rim_on:
                        added_entries.append((int(ci), 'r'))
            try:
                added_entries = list(self._filter_add_entries_by_uv_similarity(added_entries) or [])
            except Exception:
                pass
            try:
                accepted_sources = {int(e[0]) for e in (added_entries or [])}
                existing = [int(si) for si in (existing or []) if (int(si) in pre_existing_set) or (int(si) in accepted_sources)]
                existing_set = set(existing)
            except Exception:
                pass
            self.center_list_indices = existing
            try:
                self._append_center_numeric_rows_from_indices(added_entries, refresh_existing=False, source='add_all_to_list')
            except Exception:
                pass
            try:
                touched_sources = set(int(e[0]) for e in (added_entries or []))
                if touched_sources:
                    rank_map = self._center_group_rank_map()
                    rows = list(getattr(self, 'center_numeric_rows', []) or [])
                    cents = list(getattr(self, 'centroids', []) or [])
                    gen_now = int(getattr(self, 'centroid_generation', 0) or 0)
                    changed = False
                    for i, rr in enumerate(rows):
                        try:
                            rd = dict(rr or {})
                            si = int(rd.get('source_idx', -1))
                        except Exception:
                            continue
                        if si not in touched_sources:
                            continue
                        try:
                            cnam = str(rd.get('custom_name', '') or '').strip()
                        except Exception:
                            cnam = ''
                        if cnam:
                            continue
                        try:
                            is_manual = bool(float(rd.get('manual', 0.0)) >= 0.5)
                        except Exception:
                            is_manual = False
                        if is_manual:
                            continue
                        try:
                            pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                        except Exception:
                            pp = 'c'
                        if pp not in ('c', 'r'):
                            pp = 'c'
                        try:
                            gno = int(rd.get('group_no', rd.get('grp', 0) or 0))
                        except Exception:
                            gno = 0
                        if gno <= 0:
                            try:
                                if 0 <= si < len(cents):
                                    gno = int(cents[si][0])
                            except Exception:
                                pass
                        try:
                            rank = int(rank_map.get(si, 0) or 0)
                        except Exception:
                            rank = 0
                        if rank <= 0:
                            rank = int(si) + 1
                        rd['grp'] = float(gno)
                        rd['group_no'] = float(gno)
                        rd['group_rank'] = float(rank)
                        rd['generation'] = float(gen_now)
                        rd['name'] = str(self._build_center_row_name(
                            manual=False,
                            group_no=gno,
                            group_rank=rank,
                            pos_tag=pp,
                            generation=gen_now,
                            manual_seq=None,
                        ))
                        rows[i] = rd
                        changed = True
                    if changed:
                        self.center_numeric_rows = rows
            except Exception:
                pass
            if added_entries and getattr(self, 'selected_index', None) not in existing_set:
                try:
                    self.selected_index = int(added_entries[0][0])
                except Exception:
                    pass
        except Exception:
            return
        try:
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False)
        except Exception:
            pass
        try:
            if str(getattr(self, 'overlay_point_source', 'left') or 'left') == 'center':
                self._apply_proc_zoom()
            else:
                self._refresh_selected_overlay_only()
        except Exception:
            pass
        try:
            if added_entries:
                QTimer.singleShot(0, self._scroll_center_table_to_bottom)
        except Exception:
            pass

    def _scroll_center_table_to_key(self, key):
        """Scroll middle table so the row mapped to key=(source_idx,pos) becomes visible."""
        try:
            if key is None:
                return
            si = int(key[0])
            try:
                pos = str(key[1] or 'c').lower().strip()
            except Exception:
                pos = 'c'
            if pos not in ('c', 'r'):
                pos = 'c'

            t = getattr(self, 'table_between', None)
            if t is None:
                return
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            if not row_keys:
                return
            data_row = -1
            for i, rk in enumerate(row_keys):
                try:
                    rsi = int(rk[0])
                    rpos = str(rk[1] or 'c').lower().strip()
                except Exception:
                    continue
                if rpos not in ('c', 'r'):
                    rpos = 'c'
                if rsi == si and rpos == pos:
                    data_row = int(i)
                    break
            if data_row < 0:
                return
            view_row = int(data_row + 2)
            if not (0 <= view_row < t.rowCount()):
                return
            it = t.item(view_row, 0)
            if it is not None:
                try:
                    t.scrollToItem(it, QAbstractItemView.PositionAtCenter)
                except Exception:
                    t.scrollToItem(it)
            try:
                if int(t.selectionMode()) != int(QAbstractItemView.NoSelection):
                    t.setCurrentCell(view_row, 0)
            except Exception:
                pass
        except Exception:
            pass

    def _scroll_center_table_to_bottom(self):
        """Scroll middle table to the last data row (if any)."""
        try:
            t = getattr(self, 'table_between', None)
            if t is None:
                return
            last_row = int(t.rowCount()) - 1
            if last_row < 2:
                return
            it = t.item(last_row, 0)
            if it is not None:
                try:
                    t.scrollToItem(it, QAbstractItemView.PositionAtBottom)
                except Exception:
                    t.scrollToItem(it)
            else:
                try:
                    vsb = t.verticalScrollBar()
                    if vsb is not None:
                        vsb.setValue(vsb.maximum())
                except Exception:
                    pass
            try:
                if int(t.selectionMode()) != int(QAbstractItemView.NoSelection):
                    t.setCurrentCell(last_row, 0)
            except Exception:
                pass
        except Exception:
            pass

    def _set_all_groups_visible(self, visible):
        try:
            self.visible_groups = None if bool(visible) else set()
        except Exception:
            pass
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

    def _on_toggle_center_add_core(self, checked):
        try:
            core_on = bool(checked)
            rim_btn = getattr(self, 'btn_center_add_rim', None)
            rim_on = bool(rim_btn.isChecked()) if rim_btn is not None else bool(getattr(self, 'center_add_rim_enabled', True))
            if (not core_on) and (not rim_on):
                btn = getattr(self, 'btn_center_add_core', None)
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(True)
                    btn.blockSignals(False)
                core_on = True
            self.center_add_core_enabled = bool(core_on)
            self.center_add_rim_enabled = bool(rim_on)
        except Exception:
            self.center_add_core_enabled = True
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _on_toggle_center_add_rim(self, checked):
        try:
            rim_on = bool(checked)
            core_btn = getattr(self, 'btn_center_add_core', None)
            core_on = bool(core_btn.isChecked()) if core_btn is not None else bool(getattr(self, 'center_add_core_enabled', True))
            if (not core_on) and (not rim_on):
                btn = getattr(self, 'btn_center_add_rim', None)
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(True)
                    btn.blockSignals(False)
                rim_on = True
            self.center_add_core_enabled = bool(core_on)
            self.center_add_rim_enabled = bool(rim_on)
        except Exception:
            self.center_add_rim_enabled = True
        try:
            self._update_rim_offset_enabled_state()
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _update_rim_offset_enabled_state(self):
        """Enable Rim Offset controls only when Rim selection is active."""
        try:
            rim_btn = getattr(self, 'btn_center_add_rim', None)
            rim_on = bool(rim_btn.isChecked()) if rim_btn is not None else bool(getattr(self, 'center_add_rim_enabled', True))
        except Exception:
            rim_on = True
        for name in (
            'lbl_center_rim_offset',
            'btn_center_rim_offset_minus',
            'edit_center_rim_offset',
            'btn_center_rim_offset_plus',
            'slider_center_rim_offset',
        ):
            try:
                w = getattr(self, name, None)
                if w is not None:
                    w.setEnabled(bool(rim_on))
            except Exception:
                pass

    def _update_centroid_extraction_button(self):
        try:
            btn = getattr(self, 'btn_start_centroid_extraction', None)
            if btn is None:
                return
            target_w = 0
            try:
                new_btn = getattr(self, 'btn_new_project', None)
                if new_btn is not None:
                    target_w = int(new_btn.width() or 0)
                    if target_w <= 0:
                        target_w = int(new_btn.sizeHint().width() or 0)
            except Exception:
                target_w = 0
            if bool(getattr(self, 'centroid_extraction_mode', False)):
                btn.setText('FINISH Centroid Extraction')
                try:
                    self._ensure_start_ce_button_width()
                    fw = int(getattr(self, '_btn_start_ce_fixed_width', 0) or 0)
                    if fw > 0:
                        btn.setFixedWidth(int(fw))
                except Exception:
                    pass
                try:
                    blink_on = bool(getattr(self, '_centroid_finish_blink_on', False))
                except Exception:
                    blink_on = False
                if blink_on:
                    btn.setStyleSheet(
                        'QPushButton { background-color: rgb(160,15,15); color: white; border: none; border-radius: 8px; }'
                        'QPushButton:hover { background-color: rgb(160,15,15); }'
                        'QPushButton:pressed { background-color: rgb(145,10,10); }'
                    )
                else:
                    btn.setStyleSheet(
                        'QPushButton { background-color: rgb(225,120,120); color: white; border: none; border-radius: 8px; }'
                        'QPushButton:hover { background-color: rgb(220,110,110); }'
                        'QPushButton:pressed { background-color: rgb(210,100,100); }'
                    )
            else:
                btn.setText('START Centroid Extraction')
                try:
                    self._ensure_start_ce_button_width()
                except Exception:
                    pass
                try:
                    fw = int(getattr(self, '_btn_start_ce_fixed_width', 0) or 0)
                    if fw > 0:
                        btn.setFixedWidth(int(fw))
                except Exception:
                    pass
                btn.setStyleSheet(
                    'QPushButton { background-color: rgb(160,15,15); color: white; border: none; border-radius: 8px; }'
                    'QPushButton:hover { background-color: rgb(160,15,15); }'
                    'QPushButton:pressed { background-color: rgb(160,15,15); }'
                )
        except Exception:
            pass

    def _set_centroid_finish_blink_active(self, active: bool):
        """Blink Finish button every second while centroid extraction mode is active."""
        try:
            en = bool(active)
        except Exception:
            en = False

        t = getattr(self, '_centroid_finish_blink_timer', None)
        if t is None:
            try:
                t = QTimer(self)
                t.setSingleShot(False)
                t.setInterval(1000)
                t.timeout.connect(self._on_centroid_finish_blink_timer)
                self._centroid_finish_blink_timer = t
            except Exception:
                t = None

        if en:
            try:
                self._centroid_finish_blink_on = False
            except Exception:
                pass
            try:
                if t is not None and not t.isActive():
                    t.start()
            except Exception:
                pass
        else:
            try:
                if t is not None and t.isActive():
                    t.stop()
            except Exception:
                pass
            try:
                self._centroid_finish_blink_on = False
            except Exception:
                pass

        try:
            self._update_centroid_extraction_button()
        except Exception:
            pass

    def _on_centroid_finish_blink_timer(self):
        try:
            if not bool(getattr(self, 'centroid_extraction_mode', False)):
                self._set_centroid_finish_blink_active(False)
                return
            self._centroid_finish_blink_on = not bool(getattr(self, '_centroid_finish_blink_on', False))
        except Exception:
            pass
        try:
            self._update_centroid_extraction_button()
        except Exception:
            pass

    def _update_rim_offset_label(self):
        try:
            edit = getattr(self, 'edit_center_rim_offset', None)
            if edit is None:
                return
            try:
                off = int(getattr(self, 'rim_offset_px', 3) or 3)
            except Exception:
                off = 3
            edit.setText(str(int(off)))
            try:
                sld = getattr(self, 'slider_center_rim_offset', None)
                if sld is not None:
                    sld.blockSignals(True)
                    sld.setValue(int(off))
                    sld.blockSignals(False)
            except Exception:
                pass
        except Exception:
            pass

    def _on_rim_offset_edit_finished(self):
        try:
            edit = getattr(self, 'edit_center_rim_offset', None)
            if edit is None:
                return
            raw = str(edit.text() or '').strip()
            v = int(raw)
        except Exception:
            try:
                self._update_rim_offset_label()
            except Exception:
                pass
            return
        v = max(0, min(50, int(v)))
        try:
            cur = int(getattr(self, 'rim_offset_px', 3) or 3)
        except Exception:
            cur = 3
        if int(v) == int(cur):
            try:
                self._update_rim_offset_label()
            except Exception:
                pass
            return
        try:
            self.rim_offset_px = int(v)
        except Exception:
            return
        try:
            self._update_rim_offset_label()
        except Exception:
            pass
        try:
            self._manual_recompute_request = True
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=True)
        except Exception:
            pass

    def _on_rim_offset_slider_changed(self, value):
        try:
            v = int(value)
        except Exception:
            return
        v = max(0, min(50, int(v)))
        try:
            cur = int(getattr(self, 'rim_offset_px', 3) or 3)
        except Exception:
            cur = 3
        if int(v) == int(cur):
            return
        try:
            self.rim_offset_px = int(v)
        except Exception:
            return
        try:
            self._update_rim_offset_label()
        except Exception:
            pass
        try:
            self._manual_recompute_request = True
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=True)
        except Exception:
            pass

    def _nudge_rim_offset(self, delta):
        try:
            d = int(delta)
        except Exception:
            d = 0
        if d == 0:
            return
        try:
            cur = int(getattr(self, 'rim_offset_px', 3) or 3)
        except Exception:
            cur = 3
        new_v = max(0, min(50, int(cur + d)))
        if new_v == int(cur):
            return
        try:
            self.rim_offset_px = int(new_v)
        except Exception:
            return
        try:
            self._update_rim_offset_label()
        except Exception:
            pass
        try:
            self._manual_recompute_request = True
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=True)
        except Exception:
            pass

    def _update_core_rim_controls_visibility(self):
        try:
            show = bool(getattr(self, 'centroid_extraction_mode', False))
        except Exception:
            show = False
        for name in (
            'btn_center_add_core',
            'btn_center_add_rim',
            'btn_center_rim_offset_minus',
            'btn_center_rim_offset_plus',
            'lbl_center_rim_offset',
            'edit_center_rim_offset',
            'slider_center_rim_offset',
        ):
            try:
                w = getattr(self, name, None)
                if w is not None:
                    w.setVisible(bool(show))
            except Exception:
                pass
        try:
            self._update_rim_offset_enabled_state()
        except Exception:
            pass

    def _ensure_start_ce_button_width(self):
        """Compute and keep a stable width for Start/Finish Centroid Extraction button."""
        try:
            btn = getattr(self, 'btn_start_centroid_extraction', None)
            if btn is None:
                return
            fw = int(getattr(self, '_btn_start_ce_fixed_width', 0) or 0)
            if fw <= 0:
                try:
                    fm = btn.fontMetrics()
                    w_start = int(fm.horizontalAdvance('START Centroid Extraction'))
                    w_finish = int(fm.horizontalAdvance('FINISH Centroid Extraction'))
                    fw = max(w_start, w_finish) + 36
                except Exception:
                    fw = 240
                fw = max(220, int(fw))
                self._btn_start_ce_fixed_width = int(fw)
            btn.setFixedWidth(int(fw))
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

    def _load_center_add_preferences(self):
        try:
            s = QSettings('PiXY', 'PiXY')
        except Exception:
            return
        try:
            raw = s.value('center_add/uv_similarity_px', 2)
            try:
                v = int(raw)
            except Exception:
                v = int(float(raw))
            self.center_add_uv_similarity_px = max(0, int(v))
        except Exception:
            self.center_add_uv_similarity_px = 2

    def _save_center_add_preferences(self):
        try:
            s = QSettings('PiXY', 'PiXY')
            s.setValue('center_add/uv_similarity_px', int(max(0, int(getattr(self, 'center_add_uv_similarity_px', 2) or 2))))
            s.sync()
        except Exception:
            pass

    def _on_open_left_add_settings(self):
        """Open settings for left->center add behavior."""
        try:
            from qt_compat.QtWidgets import QInputDialog
            cur = int(max(0, int(getattr(self, 'center_add_uv_similarity_px', 2) or 2)))
            val, ok = QInputDialog.getInt(
                self,
                'Setting',
                'UV similarity threshold (px):\nSkip adding points when both |du| and |dv| are <= this value.',
                cur,
                0,
                999,
                1,
            )
            if not ok:
                return
            self.center_add_uv_similarity_px = int(max(0, int(val)))
            try:
                self._save_center_add_preferences()
            except Exception:
                pass
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
            if bool(active) and (not bool(prev)):
                try:
                    self._normal_overlay_mode_before_centroid = str(getattr(self, 'overlay_mode', 'Original') or 'Original')
                    self._normal_show_boundaries_before_centroid = bool(getattr(self, 'show_boundaries', True))
                except Exception:
                    self._normal_overlay_mode_before_centroid = 'Original'
                    self._normal_show_boundaries_before_centroid = True
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
                # Finish後は抽出前の通常オーバーレイへ戻す
                mode_prev = str(getattr(self, '_normal_overlay_mode_before_centroid', 'Original') or 'Original')
                if mode_prev not in ('Original', 'Posterized'):
                    mode_prev = 'Original'
                show_prev = bool(getattr(self, '_normal_show_boundaries_before_centroid', True))
                self._apply_overlay_boundary_state(mode_prev, show_prev)
        except Exception:
            pass
        try:
            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        try:
            show_left_block = bool(stage_n == 'offline' and not bool(active))
            for name in ('left_stage_controls', 'offline_manual_controls', 'table_ref_view', 'table_ref_view_header'):
                w = getattr(self, name, None)
                if w is not None:
                    w.setVisible(show_left_block)
        except Exception:
            pass
        try:
            center_container = getattr(self, 'center_container', None)
            if center_container is not None:
                center_container.setVisible(bool(not active))
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
            if bool(active):
                self._set_workflow_stage('offline', sync_toggle=True, allow_mode_side_effects=False)
            else:
                stage_now = str(getattr(self, 'workflow_stage', 'online') or 'online').lower().strip()
                if stage_now not in ('offline', 'online'):
                    stage_now = 'online'
                self._set_workflow_stage(stage_now, sync_toggle=True, allow_mode_side_effects=False)
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
            # In centroid-extraction mode, lock middle column interactions.
            self._set_center_column_interaction_enabled(not bool(active))
        except Exception:
            pass
        try:
            self._update_centroid_extraction_button()
        except Exception:
            pass
        try:
            self._set_centroid_finish_blink_active(bool(active))
        except Exception:
            pass
        try:
            self._update_rim_offset_label()
        except Exception:
            pass
        try:
            self._update_core_rim_controls_visibility()
        except Exception:
            pass
        try:
            if bool(active):
                # Entering extraction mode: avoid unconditional heavy recompute.
                # Recompute only when cache is missing/stale; otherwise keep current center names stable.
                try:
                    cache = getattr(self, '_cache', {}) if isinstance(getattr(self, '_cache', None), dict) else {}
                    cache_cent = cache.get('centroids', None)
                    cache_img_id = cache.get('img_id', None)
                    cur_img_id = id(getattr(self, 'proc_img', None))
                    need_recompute = (cache_cent is None) or (cache_img_id != cur_img_id)
                except Exception:
                    need_recompute = True
                self.schedule_update(force=True, recompute_centroids=bool(need_recompute))
            else:
                # Finishing extraction: keep only overlay redraw.
                # Middle-table rebuild is intentionally skipped.
                try:
                    self._last_overlay_full = None
                    self._last_overlay_full_poster = None
                    self._last_overlay_mode = ''
                    self._last_show_boundaries = False
                except Exception:
                    pass
                try:
                    self.schedule_update(force=True, recompute_centroids=False)
                except Exception:
                    pass
                try:
                    self._apply_proc_zoom()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_toggle_centroid_extraction_mode(self):
        try:
            self._set_centroid_extraction_mode(not bool(getattr(self, 'centroid_extraction_mode', False)))
        except Exception:
            pass

    def _set_middle_column_visible(self, visible: bool):
        try:
            vis = bool(visible)
        except Exception:
            vis = True
        try:
            host = getattr(self, 'center_container', None)
            if host is not None:
                host.setVisible(vis)
        except Exception:
            pass

    def _set_online_fiducial_rows_fixed(self, rows: int = 5):
        try:
            n = max(1, int(rows))
        except Exception:
            n = 5
        try:
            tbl = getattr(self, 'table_ref_view', None)
            if tbl is None:
                return
            base_row = 2 if tbl.rowCount() > 2 else 0
            try:
                rh = int(tbl.rowHeight(base_row) or 0)
            except Exception:
                rh = 0
            if rh <= 0:
                try:
                    rh = int(tbl.verticalHeader().defaultSectionSize() or TABLE_DEFAULT_ROW_HEIGHT)
                except Exception:
                    rh = TABLE_DEFAULT_ROW_HEIGHT
            try:
                frame_w = int(tbl.frameWidth() or 1) * 2
            except Exception:
                frame_w = 2
            target_h = int((rh * n) + frame_w + 2)
            try:
                tbl.setFixedHeight(max(80, target_h))
            except Exception:
                pass
        except Exception:
            pass

    def _sync_table_ref_view_height(self):
        """Set the left TargetPoint table height to its visible content height."""
        try:
            tbl = getattr(self, 'table_ref_view', None)
            if tbl is None:
                return

            total_h = 0
            try:
                vh = tbl.verticalHeader()
                default_rh = int(vh.defaultSectionSize() or TABLE_DEFAULT_ROW_HEIGHT) if vh is not None else TABLE_DEFAULT_ROW_HEIGHT
            except Exception:
                default_rh = TABLE_DEFAULT_ROW_HEIGHT

            try:
                for r in range(tbl.rowCount()):
                    try:
                        if tbl.isRowHidden(r):
                            continue
                    except Exception:
                        pass
                    try:
                        rh = int(tbl.rowHeight(r) or 0)
                    except Exception:
                        rh = 0
                    if rh <= 0:
                        rh = default_rh
                    total_h += rh
            except Exception:
                total_h = 0

            try:
                hh = tbl.horizontalHeader()
                header_h = int(hh.height() or 0) if hh is not None and hh.isVisible() else 0
            except Exception:
                header_h = 0

            try:
                frame_h = int(tbl.frameWidth() or 1) * 2
            except Exception:
                frame_h = 2

            target_h = int(total_h + header_h + frame_h + 4)
            target_h = max(120, target_h)
            try:
                tbl.setFixedHeight(target_h)
            except Exception:
                try:
                    tbl.setMinimumHeight(target_h)
                    tbl.setMaximumHeight(target_h)
                except Exception:
                    pass
        except Exception:
            pass

    def _move_center_container_to_online(self):
        try:
            center = getattr(self, 'center_container', None)
            online_col = getattr(self, 'online_col_layout', None)
            main_row = getattr(self, 'main_row_layout', None)
            if center is None or online_col is None:
                return
            try:
                if main_row is not None:
                    main_row.removeWidget(center)
            except Exception:
                pass
            try:
                online_col.removeWidget(center)
            except Exception:
                pass
            try:
                online_col.addWidget(center, 1)
            except Exception:
                pass
            try:
                center.setVisible(True)
            except Exception:
                pass
        except Exception:
            pass

    def _move_center_container_to_offline(self):
        try:
            center = getattr(self, 'center_container', None)
            offline_col = getattr(self, 'offline_col_layout', None)
            online_col = getattr(self, 'online_col_layout', None)
            main_row = getattr(self, 'main_row_layout', None)
            if center is None or offline_col is None:
                return
            try:
                if main_row is not None:
                    main_row.removeWidget(center)
            except Exception:
                pass
            try:
                if online_col is not None:
                    online_col.removeWidget(center)
            except Exception:
                pass
            try:
                offline_col.removeWidget(center)
            except Exception:
                pass
            try:
                offline_col.addWidget(center, 1)
            except Exception:
                pass
            try:
                center.setVisible(True)
            except Exception:
                pass
        except Exception:
            pass

    def _move_center_container_to_main(self):
        try:
            center = getattr(self, 'center_container', None)
            offline_col = getattr(self, 'offline_col_layout', None)
            online_col = getattr(self, 'online_col_layout', None)
            main_row = getattr(self, 'main_row_layout', None)
            if center is None or main_row is None:
                return
            try:
                if offline_col is not None:
                    offline_col.removeWidget(center)
            except Exception:
                pass
            try:
                if online_col is not None:
                    online_col.removeWidget(center)
            except Exception:
                pass
            try:
                if main_row.indexOf(center) < 0:
                    main_row.insertWidget(1, center, 0)
            except Exception:
                pass
            try:
                center.setVisible(True)
            except Exception:
                pass
        except Exception:
            pass

    def _update_workflow_stage_logo(self):
        try:
            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        if stage_n not in ('offline', 'online'):
            stage_n = 'offline'

        img = getattr(self, 'left_top_image', None)
        if img is None:
            return

        try:
            if stage_n == 'offline':
                preferred = [
                    r"C:\Python\PiX\PiXY_Pix.png",
                    r"C:\Python\PiXY\PiXY_Pix.png",
                    os.path.join(os.path.dirname(__file__), 'PiXY_Pix.png'),
                    os.path.join(os.path.dirname(__file__), 'PiXY.png'),
                ]
            else:
                preferred = [
                    r"C:\Python\PiXY\PiXY_XY.png",
                    os.path.join(os.path.dirname(__file__), 'PiXY_XY.png'),
                    os.path.join(os.path.dirname(__file__), 'PiXY.png'),
                ]
            pm = None
            for pth in preferred:
                try:
                    cand = QPixmap(pth)
                    if cand is not None and not cand.isNull():
                        pm = cand
                        break
                except Exception:
                    continue
            if pm is not None:
                try:
                    target_w, target_h = 450, 200
                    self._left_top_pix = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img.setPixmap(self._left_top_pix)
                except Exception:
                    self._left_top_pix = pm
                    img.setPixmap(pm)
        except Exception:
            pass

    def _update_workflow_layout_visibility(self):
        try:
            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        if stage_n not in ('offline', 'online'):
            stage_n = 'offline'

        extraction_on = bool(getattr(self, 'centroid_extraction_mode', False))
        is_online = bool(stage_n == 'online')
        center_extract_mode = bool((not is_online) and extraction_on)

        try:
            self._relocate_extraction_controls_to_center(center_extract_mode)
        except Exception:
            pass
        try:
            # Relocated controls may require a wider center column.
            QTimer.singleShot(0, lambda: self._adjust_center_column_widths())
        except Exception:
            pass

        if is_online:
            try:
                self._set_online_fiducial_rows_fixed(5)
            except Exception:
                pass
            try:
                self._move_center_container_to_online()
            except Exception:
                pass
            try:
                self._set_middle_column_visible(True)
            except Exception:
                pass
            try:
                mp = getattr(self, 'middle_extract_panel', None)
                if mp is not None:
                    mp.setVisible(False)
            except Exception:
                pass
        else:
            # Keep TargetPoint table on the left (Offline tab) in both normal/extraction modes.
            try:
                self._move_center_container_to_offline()
            except Exception:
                pass
            try:
                # During centroid extraction, TargetPoint table is not used.
                self._set_middle_column_visible(not bool(extraction_on))
            except Exception:
                pass
            try:
                mp = getattr(self, 'middle_extract_panel', None)
                if mp is not None:
                    mp.setVisible(False)
                    try:
                        mp.setMinimumHeight(0)
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            extra_row = getattr(self, '_image_param_extra_row', None)
            if extra_row is not None:
                # Keep the top image control rows available in normal workflow.
                # Axis (+/-) controls live in this row and must remain accessible in Stage view.
                show_param_rows = bool(not extraction_on)
                extra_row.setVisible(show_param_rows)
        except Exception:
            pass
        try:
            top_row = getattr(self, '_image_param_top_row', None)
            if top_row is not None:
                top_row.setVisible(bool((not is_online) and extraction_on))
        except Exception:
            pass
        try:
            w = getattr(self, 'view_orientation_controls', None)
            if w is not None:
                # Coordinate belongs in the unified Online row.
                w.setVisible(bool(is_online and (not extraction_on)))
        except Exception:
            pass

        try:
            tabs = getattr(self, 'left_tabs', None)
            if tabs is not None:
                tabs.setCurrentIndex(0 if stage_n == 'offline' else 1)
        except Exception:
            pass

        try:
            w = getattr(self, 'left_extract_controls', None)
            if w is not None:
                w.setVisible(stage_n == 'offline')
        except Exception:
            pass

        try:
            w = getattr(self, 'left_stage_controls', None)
            if w is not None:
                # Keep stage toggle available in both Offline/Online.
                # Hide only while centroid extraction mode is active.
                w.setVisible(bool(not extraction_on))
        except Exception:
            pass
        try:
            hint_box = getattr(self, 'left_stage_hint', None)
            if hint_box is not None:
                hint_box.setVisible(bool(not extraction_on))
            h_off = getattr(self, 'lbl_stage_hint_offline', None)
            h_on = getattr(self, 'lbl_stage_hint_online', None)
            if h_off is not None:
                h_off.setVisible(bool((not extraction_on) and (not is_online)))
            if h_on is not None:
                h_on.setVisible(bool((not extraction_on) and is_online))
            if hint_box is not None:
                hh = 0
                try:
                    if h_off is not None and h_off.isVisible():
                        hh = max(hh, int(h_off.sizeHint().height()))
                except Exception:
                    pass
                try:
                    if h_on is not None and h_on.isVisible():
                        hh = max(hh, int(h_on.sizeHint().height()))
                except Exception:
                    pass
                hint_box.setFixedHeight(max(0, int(hh)))
        except Exception:
            pass

        try:
            w = getattr(self, 'left_top_image', None)
            if w is not None:
                # In centroid extraction mode, free left-column space for controls.
                w.setVisible(bool(not extraction_on))
        except Exception:
            pass

        # Offline tab internals (centroid-extraction contents) are shown only while
        # centroid extraction mode is active.
        show_offline_extract_details = bool((not is_online) and extraction_on)
        # Hide TargetPoint controls and table while extraction mode is active.
        show_offline_targetpoint = bool((not is_online) and (not extraction_on))
        # Fiducial table must be visible in Online mode.
        show_fiducial_table = bool(is_online or show_offline_targetpoint)
        try:
            w = getattr(self, 'offline_manual_controls', None)
            if w is not None:
                # TargetPoint buttons are not needed during centroid extraction.
                w.setVisible(show_offline_targetpoint)
        except Exception:
            pass
        try:
            w = getattr(self, 'table_ref_view', None)
            if w is not None:
                w.setVisible(show_fiducial_table)
        except Exception:
            pass
        try:
            w = getattr(self, 'table_ref_view_header', None)
            if w is not None:
                w.setVisible(show_fiducial_table)
        except Exception:
            pass
        try:
            w = getattr(self, 'grain_section', None)
            if w is not None:
                w.setVisible(show_offline_extract_details)
        except Exception:
            pass
        try:
            w = getattr(self, 'offline_group_scroll', None)
            if w is not None:
                w.setVisible(show_offline_extract_details)
        except Exception:
            pass
        try:
            w = getattr(self, 'extract_mode_options_controls', None)
            if w is not None:
                w.setVisible(show_offline_extract_details)
        except Exception:
            pass
        try:
            w = getattr(self, 'btn_add_all_grp_list', None)
            if w is not None:
                w.setVisible(show_offline_extract_details)
        except Exception:
            pass
        try:
            w = getattr(self, 'toggle_show_all_groups', None)
            if w is not None:
                w.setVisible(show_offline_extract_details)
        except Exception:
            pass
        try:
            oc = getattr(self, 'offline_col_layout', None)
            if oc is not None:
                oc.setSpacing(4 if show_offline_extract_details else 0)
        except Exception:
            pass

        try:
            online_ctrl = getattr(self, 'online_export_controls', None)
            if online_ctrl is not None:
                online_ctrl.setVisible(is_online)
        except Exception:
            pass
        try:
            self._update_online_grid_mode_toggle_visibility()
        except Exception:
            pass
        try:
            self._update_online_stage_controls_overlay_visibility()
        except Exception:
            pass

    def _set_workflow_stage(self, stage: str, sync_toggle: bool = True, allow_mode_side_effects: bool = True):
        try:
            stage_n = str(stage or 'offline').lower().strip()
        except Exception:
            stage_n = 'offline'
        if stage_n not in ('offline', 'online'):
            stage_n = 'offline'

        try:
            self.workflow_stage = stage_n
        except Exception:
            pass

        try:
            tog = getattr(self, 'toggle_workflow_stage', None)
            if sync_toggle and tog is not None:
                tog.setCheckedIndex(0 if stage_n == 'offline' else 1)
        except Exception:
            pass

        try:
            self._update_workflow_stage_logo()
        except Exception:
            pass
        try:
            QTimer.singleShot(0, self._apply_button_styles)
        except Exception:
            pass
        try:
            QTimer.singleShot(0, self._apply_windows_titlebar_style)
        except Exception:
            pass

        # Offline targeting always uses Image coordinates.
        # This avoids leaving Stage coordinate/tool state active when coming back
        # from Online Stage view.
        try:
            if stage_n == 'offline':
                self._on_toggle_coordinate(0)
        except Exception:
            try:
                self.coordinate = 'Image'
                self.view_orientation = 'Image'
            except Exception:
                pass

        try:
            self._update_workflow_layout_visibility()
        except Exception:
            pass
        try:
            # Repaint transposed ref view immediately on stage toggle so Online stage-input
            # tint is visible before any Add Fiducial action.
            self._refresh_transposed_views(update_ref_view=True, refresh_offline_lists=False, refresh_center_view=False)
        except Exception:
            pass

        try:
            # Going to On-line should always finish centroid extraction mode.
            if bool(allow_mode_side_effects) and stage_n == 'online' and bool(getattr(self, 'centroid_extraction_mode', False)):
                self._set_centroid_extraction_mode(False)
        except Exception:
            pass

    def _on_toggle_workflow_stage(self, idx):
        try:
            ii = int(idx)
        except Exception:
            ii = 1
        if ii == 0:
            self._set_workflow_stage('offline', sync_toggle=False, allow_mode_side_effects=True)
        else:
            self._set_workflow_stage('online', sync_toggle=False, allow_mode_side_effects=True)

    def _relocate_extraction_controls_to_center(self, to_center: bool):
        """Keep centroid-extraction parameter widgets in the offline/left column."""
        try:
            enable_center = bool(to_center)
        except Exception:
            enable_center = False

        offline_col = getattr(self, 'offline_col_layout', None)
        left_col = getattr(self, 'left_col_layout', None)
        center_host = getattr(self, 'middle_extract_panel', None)
        center_layout = getattr(self, 'middle_extract_layout', None)
        if offline_col is None or center_host is None or center_layout is None:
            return

        left_extract = getattr(self, 'left_extract_controls', None)

        # Keep everything in the left/offline column; center panel is no longer used.
        widgets = [
            getattr(self, 'grain_section', None),
            getattr(self, 'extract_mode_options_controls', None),
            getattr(self, 'offline_global_controls', None),
            getattr(self, 'offline_group_scroll', None),
        ]

        try:
            while center_layout.count() > 0:
                it = center_layout.takeAt(0)
                if it is None:
                    continue
                try:
                    ww = it.widget()
                except Exception:
                    ww = None
                if ww is not None:
                    try:
                        ww.setParent(None)
                    except Exception:
                        pass
        except Exception:
            pass

        for w in widgets:
            if w is None:
                continue
            try:
                center_layout.removeWidget(w)
            except Exception:
                pass
            try:
                offline_col.removeWidget(w)
            except Exception:
                pass
            try:
                offline_col.addWidget(w, 0)
            except Exception:
                pass

        try:
            if left_extract is not None and left_col is not None:
                center_layout.removeWidget(left_extract)
                offline_col.removeWidget(left_extract)
                left_col.removeWidget(left_extract)
                left_col.addWidget(left_extract, 0)
        except Exception:
            pass

        try:
            center_host.setVisible(False)
        except Exception:
            pass
        try:
            if left_extract is not None and left_col is not None:
                left_extract.setVisible(True)
        except Exception:
            pass

    def _set_center_column_interaction_enabled(self, enabled: bool):
        """Enable/disable middle-column controls and table selection together."""
        try:
            en = bool(enabled)
        except Exception:
            en = True

        center_widgets = [
            'table_between', 'table_between_header',
            'btn_export', 'btn_clipboard',
            'btn_add_target', 'btn_select_all',
            'combo_add_target_pos',
            'btn_center_name_filter', 'btn_update_target_uv',
            'btn_clear_target', 'btn_center_undo', 'btn_clear_target_all',
        ]
        for nm in center_widgets:
            try:
                w = getattr(self, nm, None)
                if w is not None:
                    w.setEnabled(en)
            except Exception:
                pass

        try:
            host = getattr(self, 'center_container', None)
            if host is not None:
                eff = host.graphicsEffect()
                if not isinstance(eff, QGraphicsOpacityEffect):
                    try:
                        eff = QGraphicsOpacityEffect(host)
                        host.setGraphicsEffect(eff)
                    except Exception:
                        eff = None
                if eff is not None:
                    try:
                        # In centroid-extraction offline mode, center area now hosts
                        # extraction controls, so keep it fully opaque.
                        center_extract_mode = bool(
                            (not bool(en))
                            and bool(getattr(self, 'centroid_extraction_mode', False))
                            and str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip() == 'offline'
                        )
                        eff.setOpacity(1.0 if (en or center_extract_mode) else 0.42)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            tbl = getattr(self, 'table_between', None)
            if tbl is not None:
                from qt_compat.QtWidgets import QAbstractItemView
                if en:
                    tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
                else:
                    tbl.clearSelection()
                    tbl.setCurrentCell(-1, -1)
                    tbl.setSelectionMode(QAbstractItemView.NoSelection)
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
            core_on = bool(getattr(self, 'center_add_core_enabled', True))
            rim_on = bool(getattr(self, 'center_add_rim_enabled', True))

            rank_map = {}
            try:
                for g in sorted(self._available_group_numbers() or []):
                    try:
                        gg = int(g)
                    except Exception:
                        continue
                    if gg <= 0:
                        continue
                    rows_sorted = list(self._sorted_group_entries(gg) or [])
                    for rank, (_u, _v, src_i) in enumerate(rows_sorted, start=1):
                        try:
                            rank_map[int(src_i)] = int(rank)
                        except Exception:
                            continue
            except Exception:
                rank_map = {}

            cent = []
            label_texts = []
            source_kind_to_local = {}
            local_to_source = []
            local_to_pos = []
            rim_list = list(getattr(self, '_centroid_rim_proc_points', []) or [])
            grp_count = {}
            for oi in keep:
                try:
                    gg = int(base[oi][0])
                except Exception:
                    gg = 0
                rank = rank_map.get(int(oi))
                if rank is None:
                    n0 = int(grp_count.get(gg, 0)) + 1
                    grp_count[gg] = n0
                    rank = n0

                if core_on:
                    cent.append(base[oi])
                    label_texts.append(f"G{gg}-{int(rank)}c")
                    source_kind_to_local[(int(oi), 'c')] = int(len(cent) - 1)
                    local_to_source.append(int(oi))
                    local_to_pos.append('c')

                if rim_on:
                    rim_pt = rim_list[oi] if 0 <= oi < len(rim_list) else None
                    if rim_pt is not None:
                        try:
                            rx, ry = float(rim_pt[0]), float(rim_pt[1])
                            cent.append((int(gg), float(rx), float(ry)))
                            label_texts.append(f"G{gg}-{int(rank)}r")
                            source_kind_to_local[(int(oi), 'r')] = int(len(cent) - 1)
                            local_to_source.append(int(oi))
                            local_to_pos.append('r')
                        except Exception:
                            pass

            sel_new = None
            sel_orig = getattr(self, 'selected_index', None)
            if sel_orig is not None:
                try:
                    so = int(sel_orig)
                    try:
                        spos = str(getattr(self, 'selected_point_pos', 'c') or 'c').lower().strip()
                    except Exception:
                        spos = 'c'
                    if spos == 'r' and rim_on and (so, 'r') in source_kind_to_local:
                        sel_new = source_kind_to_local.get((so, 'r'))
                    elif core_on and (so, 'c') in source_kind_to_local:
                        sel_new = source_kind_to_local.get((so, 'c'))
                    elif rim_on and (so, 'r') in source_kind_to_local:
                        sel_new = source_kind_to_local.get((so, 'r'))
                except Exception:
                    sel_new = None

            excl_src = set(getattr(self, 'excluded_centroid_indices', set()) or set())
            fv_src = set(getattr(self, '_force_visible_centroid_indices', set()) or set())
            excl_new = {i for i, oi in enumerate(local_to_source) if int(oi) in excl_src}
            fv_new = {i for i, oi in enumerate(local_to_source) if int(oi) in fv_src}

            return {
                'centroids': cent,
                'selected_index': sel_new,
                'manual_indices': set(),
                'excluded_indices': excl_new,
                'force_visible_indices': fv_new,
                'visible_groups': self._get_visible_groups_set(),
                'label_texts': label_texts,
                'local_to_source': list(local_to_source),
                'local_to_pos': list(local_to_pos),
            }

        # Center List: use persisted middle-table numeric rows (not centroid index linkage).
        try:
            _ci = self._get_center_list_indices()
            if self._center_rows_need_append(_ci):
                try:
                    _warn = getattr(self, '_log_warn', None)
                    if callable(_warn):
                        _warn("Detected missing center rows during overlay payload build; skipped auto-append to protect middle-table state.")
                except Exception:
                    pass
        except Exception:
            pass
        rows = list(getattr(self, 'center_numeric_rows', []) or [])

        cent = []
        label_texts = []
        manual_new = set()
        source_to_local = {}
        excluded_new = set()
        local_to_source = []
        local_to_pos = []
        for ridx, r in enumerate(rows):
            try:
                rd = dict(r or {})
            except Exception:
                continue
            try:
                g = 0
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
                nm = str(rd.get('name', '') or '').strip()
                if nm:
                    label_texts.append(str(nm))
                else:
                    raise ValueError('no-name')
            except Exception:
                try:
                    nv = float(rd.get('no', float('nan')))
                    if np.isnan(nv):
                        raise ValueError('nan')
                    label_texts.append(str(int(round(nv))))
                except Exception:
                    try:
                        label_texts.append(str(int(rd.get('source_idx', -1)) + 1))
                    except Exception:
                        label_texts.append(str(int(len(cent))))
            try:
                local_to_source.append(int(rd.get('source_idx', -1)))
            except Exception:
                local_to_source.append(-1)
            try:
                ptag = str(rd.get('pos', 'c') or 'c').lower().strip()
            except Exception:
                ptag = 'c'
            if ptag not in ('c', 'r'):
                ptag = 'c'
            local_to_pos.append(ptag)

            try:
                if not self._is_center_row_visible(rd):
                    excluded_new.add(int(len(cent) - 1))
            except Exception:
                pass

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
            sel_orig = getattr(self, 'selected_index', None)
            if sel_orig is not None:
                so = int(sel_orig)
                try:
                    spos = str(getattr(self, 'selected_point_pos', 'c') or 'c').lower().strip()
                except Exception:
                    spos = 'c'
                if spos not in ('c', 'r'):
                    spos = 'c'
                for li, si in enumerate(local_to_source):
                    try:
                        if int(si) == so and str(local_to_pos[li] or 'c').lower().strip() == spos:
                            sel_new = int(li)
                            break
                    except Exception:
                        continue
                if sel_new is None:
                    sel_new = source_to_local.get(so)
        except Exception:
            pass
        if sel_new is None:
            try:
                row_sel = self._current_center_selected_row()
                if row_sel is not None and 0 <= int(row_sel) < len(cent):
                    sel_new = int(row_sel)
            except Exception:
                pass

        return {
            'centroids': cent,
            'selected_index': sel_new,
            'manual_indices': manual_new,
            # Middle list visibility follows per-row Show/Hide (and source exclusion when linked).
            'excluded_indices': excluded_new,
            'force_visible_indices': set(),
            # center-list mode is explicit subset, so group visibility filter is unnecessary.
            'visible_groups': None,
            'label_texts': label_texts,
            'local_to_source': list(local_to_source),
            'local_to_pos': list(local_to_pos),
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
            self._refresh_transposed_views(refresh_center_view=False)
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

    def _compose_rim_points_with_manual(self, auto_centroids, auto_rim_points):
        """Align auto rim-point list with the centroid order after manual-target insertion."""
        try:
            auto_list = list(auto_centroids or [])
        except Exception:
            auto_list = []
        try:
            rim_list = list(auto_rim_points or [])
        except Exception:
            rim_list = []
        if len(rim_list) < len(auto_list):
            rim_list = rim_list + [None] * (len(auto_list) - len(rim_list))

        auto_g0_idx = []
        auto_other_idx = []
        for i, c in enumerate(auto_list):
            try:
                if int(c[0]) == 0:
                    auto_g0_idx.append(int(i))
                else:
                    auto_other_idx.append(int(i))
            except Exception:
                auto_other_idx.append(int(i))

        mt_n = 0
        try:
            mt_n = len(getattr(self, 'manual_targets', []) or [])
        except Exception:
            mt_n = 0

        out = []
        for i in auto_g0_idx:
            out.append(rim_list[i] if 0 <= i < len(rim_list) else None)
        out.extend([None] * int(mt_n))
        for i in auto_other_idx:
            out.append(rim_list[i] if 0 <= i < len(rim_list) else None)
        return out

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

    def _on_update_target_uv(self):
        # Toggle pick-mode（Update u,v）: single or multiple middle rows.
        if self.pick_mode == 'center_uv_update':
            self._end_pick_mode()
            return
        try:
            if bool(getattr(self, 'centroid_extraction_mode', False)):
                self._set_centroid_extraction_mode(False)
        except Exception:
            pass
        try:
            queue = self._collect_center_uv_update_queue()
            if not queue:
                return
            self._center_uv_update_queue = list(queue)
            self._center_uv_update_pos = 0
            self._start_pick_mode('center_uv_update', ref_index=int(queue[0].get('row', -1)))
            self._activate_center_uv_update_target()
        except Exception:
            return

    def _collect_center_uv_update_queue(self):
        """Build ordered target queue (top-to-bottom) for center UV update."""
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            if not rows:
                return []
        except Exception:
            return []

        sel_rows = []
        try:
            t = getattr(self, 'table_between', None)
            if t is not None:
                try:
                    sel_rows = list(self._selected_center_row_numbers() or [])
                except Exception:
                    sel_rows = []
                if not sel_rows:
                    try:
                        rr = int(t.currentRow()) - 2
                        if 0 <= rr < len(rows):
                            sel_rows = [rr]
                    except Exception:
                        pass
        except Exception:
            sel_rows = []

        # Fallback: derive from current multi-selected key set.
        if not sel_rows:
            try:
                keys = set(getattr(self, 'selected_point_keys', set()) or set())
                row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
                if keys and row_keys:
                    for ri, rk in enumerate(row_keys):
                        try:
                            si = int(rk[0])
                            pp = str(rk[1] or 'c').lower().strip()
                        except Exception:
                            continue
                        if pp not in ('c', 'r'):
                            pp = 'c'
                        if (si, pp) in keys:
                            sel_rows.append(int(ri))
            except Exception:
                pass

        if not sel_rows:
            try:
                rr = self._current_center_selected_row()
                if rr is not None and 0 <= int(rr) < len(rows):
                    sel_rows = [int(rr)]
            except Exception:
                pass

        sel_rows = sorted({int(r) for r in sel_rows if 0 <= int(r) < len(rows)})
        out = []
        for rr in sel_rows:
            try:
                rd = dict(rows[rr] or {})
                si = int(rd.get('source_idx', -1))
                pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                if pp not in ('c', 'r'):
                    pp = 'c'
                try:
                    nv = float(rd.get('no', float('nan')))
                    no_txt = str(int(round(nv))) if np.isfinite(nv) else str(int(rr) + 1)
                except Exception:
                    no_txt = str(int(rr) + 1)
                out.append({'row': int(rr), 'no': str(no_txt), 'key': (int(si), pp)})
            except Exception:
                continue
        return out

    def _center_point_full_from_key(self, source_idx, pos_tag):
        """Return (x_full, y_full) for centroid key (source_idx, pos)."""
        try:
            i = int(source_idx)
        except Exception:
            return None
        try:
            ptag = str(pos_tag or 'c').lower().strip()
        except Exception:
            ptag = 'c'
        if ptag not in ('c', 'r'):
            ptag = 'c'
        try:
            cents = list(getattr(self, 'centroids', []) or [])
            if not (0 <= i < len(cents)):
                return None
            _g, xp, yp = cents[i]
            if ptag == 'r':
                rim = list(getattr(self, '_centroid_rim_proc_points', []) or [])
                rp = rim[i] if 0 <= i < len(rim) else None
                if rp is not None:
                    xp, yp = float(rp[0]), float(rp[1])
            spf = float(getattr(self, 'scale_proc_to_full', 1.0) or 1.0)
            return float(xp) * spf, float(yp) * spf
        except Exception:
            return None

    def _zoom_to_max_on_point_key(self, source_idx, pos_tag):
        """Zoom to max configured scale and center viewport on specified point."""
        try:
            p = self._center_point_full_from_key(source_idx, pos_tag)
            if p is None:
                return
            max_zoom = 1024.0
            try:
                target_px = float(getattr(self, 'max_zoom_target_visible_px', 220) or 220)
                if target_px > 0:
                    vp = self.proc_scroll.viewport()
                    vw = max(1.0, float(vp.width()))
                    vh = max(1.0, float(vp.height()))
                    z_target = max(vw, vh) / target_px
                    max_zoom = min(max_zoom, max(0.01, float(z_target)))
            except Exception:
                pass
            self.proc_zoom = float(max_zoom)
            self._apply_proc_zoom()
            self._ensure_full_pos_visible(float(p[0]), float(p[1]))
        except Exception:
            pass

    def _activate_center_uv_update_target(self):
        """Activate current target in center_uv_update queue."""
        updates_disabled = False
        try:
            q = list(getattr(self, '_center_uv_update_queue', []) or [])
            if not q:
                return
            try:
                pos = int(getattr(self, '_center_uv_update_pos', 0) or 0)
            except Exception:
                pos = 0
            if pos < 0:
                pos = 0
            if pos >= len(q):
                try:
                    self._end_pick_mode(redraw=False)
                except Exception:
                    pass
                return

            cur = dict(q[pos] or {})
            row = int(cur.get('row', -1))
            si, pp = cur.get('key', (-1, 'c'))
            si = int(si)
            pp = str(pp or 'c').lower().strip()
            if pp not in ('c', 'r'):
                pp = 'c'

            self.pick_ref_index = int(row)
            self.selected_index = int(si)
            self.selected_point_pos = str(pp)
            self.selected_point_keys = {(int(si), str(pp))}
            self._center_uv_update_active_key = (int(si), str(pp))

            try:
                self.setUpdatesEnabled(False)
                updates_disabled = True
            except Exception:
                pass

            no_txt = str(cur.get('no', '?'))
            total = max(1, int(len(q)))
            if total <= 1:
                self._stage_info_override_text = f"Select new coordinates for point No. {no_txt}"
            else:
                self._stage_info_override_text = f"Select new coordinates for point No. {no_txt} ({int(pos)+1}/{total})"

            try:
                self._sync_table_selection()
            except Exception:
                pass
            try:
                self._zoom_to_max_on_point_key(int(si), str(pp))
            except Exception:
                pass
            try:
                self._refresh_selected_overlay_only()
            except Exception:
                pass
            try:
                self._set_center_uv_update_ui_lock(True)
            except Exception:
                pass
            try:
                self._apply_pick_mode_wait_button_style()
            except Exception:
                pass
            try:
                self._set_center_uv_nav_visible(True)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            if updates_disabled:
                try:
                    self.setUpdatesEnabled(True)
                except Exception:
                    pass

    def _set_center_uv_nav_visible(self, visible):
        try:
            v = bool(visible)
            b1 = getattr(self, 'btn_center_uv_next', None)
            b2 = getattr(self, 'btn_center_uv_back', None)
            b0 = getattr(self, 'btn_center_uv_clear', None)
            b3 = getattr(self, 'btn_center_uv_finish', None)
            if b1 is not None:
                b1.setVisible(v)
            if b2 is not None:
                b2.setVisible(v)
            if b0 is not None:
                b0.setVisible(v)
            if b3 is not None:
                b3.setVisible(v)
            if v:
                try:
                    self._reposition_viewport_overlays()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_center_uv_update_finish(self):
        try:
            if str(getattr(self, 'pick_mode', '') or '') == 'center_uv_update':
                self._end_pick_mode()
        except Exception:
            pass

    def _on_center_uv_update_clear(self):
        """Remove current target point and continue update workflow."""
        try:
            if str(getattr(self, 'pick_mode', '') or '') != 'center_uv_update':
                return
            q = list(getattr(self, '_center_uv_update_queue', []) or [])
            if not q:
                self._end_pick_mode()
                return
            try:
                pos = int(getattr(self, '_center_uv_update_pos', 0) or 0)
            except Exception:
                pos = 0
            pos = max(0, min(pos, len(q) - 1))

            # Clear currently focused target point from middle list using existing flow.
            try:
                self._on_clear_target()
            except Exception:
                pass

            # Remove current queue entry and continue.
            try:
                del q[pos]
            except Exception:
                q = q[:pos] + q[pos + 1:]
            self._center_uv_update_queue = list(q)

            if not q:
                self._end_pick_mode()
                return
            if pos >= len(q):
                pos = len(q) - 1
            self._center_uv_update_pos = int(pos)
            self._activate_center_uv_update_target()
        except Exception:
            pass

    def _on_center_uv_update_next(self):
        try:
            if str(getattr(self, 'pick_mode', '') or '') != 'center_uv_update':
                return
            q = list(getattr(self, '_center_uv_update_queue', []) or [])
            if not q:
                return
            if len(q) <= 1:
                self._move_center_uv_update_adjacent_row(+1)
                return
            try:
                pos = int(getattr(self, '_center_uv_update_pos', 0) or 0)
            except Exception:
                pos = 0
            if pos < len(q) - 1:
                self._center_uv_update_pos = int(pos + 1)
                self._activate_center_uv_update_target()
        except Exception:
            pass

    def _on_center_uv_update_back(self):
        try:
            if str(getattr(self, 'pick_mode', '') or '') != 'center_uv_update':
                return
            q = list(getattr(self, '_center_uv_update_queue', []) or [])
            if not q:
                return
            if len(q) <= 1:
                self._move_center_uv_update_adjacent_row(-1)
                return
            try:
                pos = int(getattr(self, '_center_uv_update_pos', 0) or 0)
            except Exception:
                pos = 0
            if pos > 0:
                self._center_uv_update_pos = int(pos - 1)
                self._activate_center_uv_update_target()
        except Exception:
            pass

    def _move_center_uv_update_adjacent_row(self, step):
        """When a single row is selected, Back/Next moves to visible upper/lower rows."""
        try:
            step_i = int(step)
        except Exception:
            step_i = 0
        if step_i == 0:
            return

        try:
            q = list(getattr(self, '_center_uv_update_queue', []) or [])
            if not q:
                return
            cur = dict(q[0] or {})
            ckey = cur.get('key', None)
            if not isinstance(ckey, (tuple, list)) or len(ckey) < 2:
                return
            try:
                csi = int(ckey[0])
                cpp = str(ckey[1] or 'c').lower().strip()
            except Exception:
                return
            if cpp not in ('c', 'r'):
                cpp = 'c'

            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            if not row_keys:
                return
            cur_i = None
            for i, rk in enumerate(row_keys):
                try:
                    si = int(rk[0])
                    pp = str(rk[1] or 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                if si == csi and pp == cpp:
                    cur_i = int(i)
                    break
            if cur_i is None:
                return

            nxt_i = int(cur_i + step_i)
            if not (0 <= nxt_i < len(row_keys)):
                return
            try:
                nsi = int(row_keys[nxt_i][0])
                npp = str(row_keys[nxt_i][1] or 'c').lower().strip()
            except Exception:
                return
            if npp not in ('c', 'r'):
                npp = 'c'

            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            row_idx = None
            no_txt = str(int(nxt_i) + 1)
            for ri, rr in enumerate(rows):
                try:
                    rd = dict(rr or {})
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                if si == nsi and pp == npp:
                    row_idx = int(ri)
                    try:
                        nv = float(rd.get('no', float('nan')))
                        if np.isfinite(nv):
                            no_txt = str(int(round(nv)))
                    except Exception:
                        pass
                    break
            if row_idx is None:
                return

            self._center_uv_update_queue = [{'row': int(row_idx), 'no': str(no_txt), 'key': (int(nsi), str(npp))}]
            self._center_uv_update_pos = 0
            self._activate_center_uv_update_target()
        except Exception:
            pass

    def _set_center_uv_update_ui_lock(self, active):
        """Lock most buttons during center_uv_update, keeping specified toggles usable."""
        try:
            is_active = bool(active)
            if not is_active:
                locked = list(getattr(self, '_center_uv_locked_buttons', []) or [])
                for b in locked:
                    try:
                        prev_style = str(b.property('_pixy_prev_style') or '')
                    except Exception:
                        prev_style = ''
                    try:
                        prev_enabled = b.property('_pixy_prev_enabled')
                        prev_enabled = True if prev_enabled is None else bool(prev_enabled)
                    except Exception:
                        prev_enabled = True
                    try:
                        b.setEnabled(prev_enabled)
                    except Exception:
                        pass
                    try:
                        b.setStyleSheet(prev_style)
                    except Exception:
                        pass
                    try:
                        b.setProperty('_pixy_prev_style', None)
                    except Exception:
                        pass
                    try:
                        b.setProperty('_pixy_prev_enabled', None)
                    except Exception:
                        pass
                self._center_uv_locked_buttons = []
                return

            # Reset previous lock snapshot before taking a new one.
            try:
                self._set_center_uv_update_ui_lock(False)
            except Exception:
                pass

            allow_btns = set()
            for nm in (
                'btn_center_uv_next',
                'btn_center_uv_back',
                'btn_center_uv_clear',
                'btn_center_uv_finish',
            ):
                try:
                    b = getattr(self, nm, None)
                    if b is not None:
                        allow_btns.add(b)
                except Exception:
                    pass

            allow_containers = []
            for nm in (
                'view_orientation_toggle',
                'flip_toggle_image',
                'flip_toggle_stage',
                'axis_toggle_x',
                'axis_toggle_y',
                'toggle_online_grid_mode',
                'online_stage_controls_overlay',
            ):
                try:
                    w = getattr(self, nm, None)
                    if w is not None:
                        allow_containers.append(w)
                except Exception:
                    pass

            def _allowed(btn):
                try:
                    if btn in allow_btns:
                        return True
                except Exception:
                    pass
                p = btn
                while p is not None:
                    try:
                        if p in allow_containers:
                            return True
                    except Exception:
                        pass
                    try:
                        p = p.parent()
                    except Exception:
                        p = None
                return False

            accent_btns = set()
            for nm in (
                'btn_add_ref',
                'btn_export',
                'btn_online_export',
                'btn_open',
                'btn_new_project',
            ):
                try:
                    b = getattr(self, nm, None)
                    if b is not None:
                        accent_btns.add(b)
                except Exception:
                    pass

            red_btns = set()
            for nm in (
                'btn_add_target',
                'btn_start_centroid_extraction',
            ):
                try:
                    b = getattr(self, nm, None)
                    if b is not None:
                        red_btns.add(b)
                except Exception:
                    pass

            locked = []
            for b in list(self.findChildren(QPushButton) or []):
                try:
                    if _allowed(b):
                        b.setEnabled(True)
                        continue
                    try:
                        b.setProperty('_pixy_prev_style', b.styleSheet() or '')
                    except Exception:
                        pass
                    try:
                        b.setProperty('_pixy_prev_enabled', bool(b.isEnabled()))
                    except Exception:
                        pass
                    b.setEnabled(False)
                    if b in accent_btns:
                        try:
                            stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
                        except Exception:
                            stage_n = 'offline'
                        if stage_n == 'online':
                            b.setStyleSheet(
                                "QPushButton {"
                                "background-color: rgb(176,220,212);"
                                "color: rgb(250,250,250);"
                                "border: none;"
                                "border-radius: 8px;"
                                "}"
                            )
                        else:
                            b.setStyleSheet(
                                "QPushButton {"
                                "background-color: rgb(236,176,176);"
                                "color: rgb(250,250,250);"
                                "border: none;"
                                "border-radius: 8px;"
                                "}"
                            )
                    elif b in red_btns:
                        b.setStyleSheet(
                            "QPushButton {"
                            "background-color: rgb(236,176,176);"
                            "color: rgb(250,250,250);"
                            "border: none;"
                            "border-radius: 8px;"
                            "}"
                        )
                    else:
                        b.setStyleSheet(
                            "QPushButton {"
                            "background-color: rgb(210,210,210);"
                            "color: rgb(245,245,245);"
                            "border: none;"
                            "border-radius: 8px;"
                            "}"
                        )
                    locked.append(b)
                except Exception:
                    continue
            self._center_uv_locked_buttons = locked
        except Exception:
            pass

    def _on_clear_target(self):
        # Clear removes the currently selected middle-table row(s).
        try:
            sel_keys = list(self._selected_center_row_keys() or [])
        except Exception:
            sel_keys = []
        if not sel_keys:
            return

        try:
            self._push_center_undo_state()
        except Exception:
            pass

        try:
            if getattr(self, 'manual_targets', None) is None:
                self.manual_targets = []
        except Exception:
            pass

        changed = False
        for si, ptag in sel_keys:
            try:
                changed = bool(self._remove_center_numeric_row_by_key(int(si), str(ptag))) or changed
            except Exception:
                continue

        if not changed:
            return
        try:
            self.selected_index = None
            self.selected_point_keys = set()
            self.selected_point_pos = 'c'
        except Exception:
            pass
        try:
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False, refresh_center_view=True)
        except Exception:
            pass
        try:
            self.schedule_update(force=True, recompute_centroids=False)
        except Exception:
            pass

    def _on_clear_target_all(self):
        """Clear all rows in middle list (explicit Add rows + manual targets)."""
        try:
            from qt_compat.QtWidgets import QMessageBox
            res = QMessageBox.question(
                self,
                "Clear All",
                "This will delete all rows in the middle table. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return
        except Exception:
            pass
        try:
            self._push_center_undo_state()
        except Exception:
            pass
        try:
            if getattr(self, 'manual_targets', None) is None:
                self.manual_targets = []
        except Exception:
            pass

        # Clear all explicit additions first.
        try:
            self._mark_center_model_mutation()
        except Exception:
            pass
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
                try:
                    self._centroid_rim_proc_points = self._compose_rim_points_with_manual(auto_only, getattr(self, '_auto_rim_proc_points', []))
                except Exception:
                    self._centroid_rim_proc_points = []
            except Exception:
                pass

        self.selected_index = None
        try:
            self.selected_point_keys = set()
            self.selected_point_pos = 'c'
        except Exception:
            pass
        try:
            self._sanitize_excluded_indices()
        except Exception:
            pass
        try:
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False, refresh_center_view=True)
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

    def _selected_center_source_indices(self):
        """Return unique source_idx values for currently selected middle-table rows."""
        try:
            t = getattr(self, 'table_between', None)
            if t is None:
                return []
            header_rows = 2
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            rows = []
            try:
                sm = t.selectionModel()
                if sm is not None:
                    for mi in (sm.selectedRows() or []):
                        try:
                            rr = int(mi.row()) - header_rows
                        except Exception:
                            continue
                        if 0 <= rr < len(row_keys):
                            rows.append(int(rr))
            except Exception:
                rows = []
            out = []
            seen = set()
            for rr in rows:
                try:
                    if 0 <= int(rr) < len(row_keys):
                        si = int(row_keys[int(rr)][0])
                    else:
                        continue
                    if si in seen:
                        continue
                    seen.add(si)
                    out.append(si)
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _selected_center_row_numbers(self):
        """Return selected middle-table data-row numbers (0-based, header excluded)."""
        try:
            t = getattr(self, 'table_between', None)
            if t is None:
                return []
            header_rows = 2
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            nrows = int(len(row_keys))
            rows = set()

            # Prefer selection ranges so Shift-range selection includes all rows in the range.
            try:
                ranges = list(t.selectedRanges() or [])
            except Exception:
                ranges = []
            for rg in ranges:
                try:
                    top = int(rg.topRow())
                    bot = int(rg.bottomRow())
                except Exception:
                    continue
                if bot < top:
                    top, bot = bot, top
                for vr in range(top, bot + 1):
                    rr = int(vr - header_rows)
                    if 0 <= rr < nrows:
                        rows.add(int(rr))

            # Fallback: selectedRows for cases where range info is unavailable.
            if not rows:
                try:
                    sm = t.selectionModel()
                    if sm is not None:
                        for mi in (sm.selectedRows() or []):
                            try:
                                rr = int(mi.row()) - header_rows
                            except Exception:
                                continue
                            if 0 <= rr < nrows:
                                rows.add(int(rr))
                except Exception:
                    pass

            return sorted(rows)
        except Exception:
            return []

    def _selected_center_row_keys(self):
        """Return selected middle-table row keys [(source_idx, pos), ...] in row order."""
        try:
            t = getattr(self, 'table_between', None)
            if t is None:
                return []
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            sel_rows = list(self._selected_center_row_numbers() or [])
            out = []
            seen = set()
            for rr in sel_rows:
                if not (0 <= int(rr) < len(row_keys)):
                    continue
                try:
                    si = int(row_keys[int(rr)][0])
                except Exception:
                    continue
                try:
                    ptag = str(row_keys[int(rr)][1] or 'c').lower().strip()
                except Exception:
                    ptag = 'c'
                if ptag not in ('c', 'r'):
                    ptag = 'c'
                key = (si, ptag)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out
        except Exception:
            return []

    def _log_center_selection_snapshot(self, reason=''):
        """Log current middle-table selection state in LOG_MODE for troubleshooting."""
        try:
            if not hasattr(self, '_log_info'):
                return
            max_items = 8
            try:
                sel_rows = [int(x) for x in (self._selected_center_row_numbers() or [])]
            except Exception:
                sel_rows = []
            try:
                sel_keys_rows = []
                for k in (self._selected_center_row_keys() or []):
                    try:
                        sel_keys_rows.append((int(k[0]), str(k[1] if len(k) > 1 else 'c').lower().strip()))
                    except Exception:
                        continue
            except Exception:
                sel_keys_rows = []
            try:
                spk = []
                for k in (set(getattr(self, 'selected_point_keys', set()) or set())):
                    try:
                        spk.append((int(k[0]), str(k[1] if len(k) > 1 else 'c').lower().strip()))
                    except Exception:
                        continue
            except Exception:
                spk = []
            try:
                sidx = getattr(self, 'selected_index', None)
            except Exception:
                sidx = None
            try:
                spos = str(getattr(self, 'selected_point_pos', 'c') or 'c').lower().strip()
            except Exception:
                spos = 'c'
            if spos not in ('c', 'r'):
                spos = 'c'

            try:
                sel_rows_preview = list(sel_rows[:max_items])
            except Exception:
                sel_rows_preview = []
            try:
                sel_keys_preview = list(sel_keys_rows[:max_items])
            except Exception:
                sel_keys_preview = []
            try:
                spk_preview = list(spk[:max_items])
            except Exception:
                spk_preview = []

            sig = (
                str(reason or ''),
                int(len(sel_rows)),
                int(len(sel_keys_rows)),
                int(len(spk)),
                tuple(sel_rows_preview),
                tuple(sel_keys_preview),
                tuple(spk_preview),
                None if sidx is None else int(sidx),
                str(spos),
            )
            if sig == getattr(self, '_last_center_sel_log_sig', None):
                return
            self._last_center_sel_log_sig = sig

            self._log_info(
                "CENTER_SEL "
                f"reason={str(reason or '')} "
                f"rows_count={len(sel_rows)} rows_head={sel_rows_preview} "
                f"row_keys_count={len(sel_keys_rows)} row_keys_head={sel_keys_preview} "
                f"selected_point_keys_count={len(spk)} selected_point_keys_head={spk_preview} "
                f"selected_index={sidx} "
                f"selected_point_pos={spos}"
            )
            # Arm one-shot overlay log so heavy draw-loop logging does not flood.
            try:
                self._selection_overlay_log_armed = True
            except Exception:
                pass
        except Exception:
            pass

    def _log_overlay_selection_snapshot(self, reason, selected_locals, drawn_keys, total_centroids):
        """Log which local points were selected and actually drawn as blue markers."""
        try:
            if not hasattr(self, '_log_info'):
                return
            max_items = 8
            try:
                spk = [
                    (int(k[0]), str(k[1] if len(k) > 1 else 'c').lower().strip())
                    for k in (set(getattr(self, 'selected_point_keys', set()) or set()))
                ]
            except Exception:
                spk = []

            try:
                sel_locals = [int(x) for x in (selected_locals or [])]
            except Exception:
                sel_locals = []
            try:
                dkeys = [(int(k[0]), str(k[1])) for k in (drawn_keys or [])]
            except Exception:
                dkeys = []

            try:
                spk_head = list(spk[:max_items])
            except Exception:
                spk_head = []
            try:
                sel_locals_head = list(sel_locals[:max_items])
            except Exception:
                sel_locals_head = []
            try:
                dkeys_head = list(dkeys[:max_items])
            except Exception:
                dkeys_head = []

            sig = (
                str(reason or ''),
                int(len(spk)),
                int(len(sel_locals)),
                int(len(dkeys)),
                tuple(spk_head),
                tuple(sel_locals_head),
                tuple(dkeys_head),
                int(total_centroids or 0),
            )
            if sig == getattr(self, '_last_overlay_sel_log_sig', None):
                return
            self._last_overlay_sel_log_sig = sig

            self._log_info(
                "OVERLAY_SEL "
                f"reason={str(reason or '')} "
                f"selected_point_keys_count={len(spk)} selected_point_keys_head={spk_head} "
                f"selected_local_count={len(sel_locals)} selected_local_head={sel_locals_head} "
                f"drawn_marker_count={len(dkeys)} drawn_marker_head={dkeys_head} "
                f"centroids={int(total_centroids or 0)}"
            )
        except Exception:
            pass

    def _on_table_between_context_menu(self, pos):
        """Open context menu for middle table selection."""
        try:
            tbl = getattr(self, 'table_between', None)
            if tbl is None:
                return
            try:
                if not bool(tbl.isEnabled()):
                    return
            except Exception:
                pass

            # Right-click on a non-selected row should move focus to that row first.
            try:
                it = tbl.itemAt(pos)
                if it is not None:
                    rr = int(it.row())
                    if rr >= 2:
                        is_sel = False
                        try:
                            is_sel = bool(tbl.selectionModel().isRowSelected(rr, tbl.rootIndex()))
                        except Exception:
                            is_sel = False
                        if not is_sel:
                            try:
                                tbl.clearSelection()
                                tbl.setCurrentCell(rr, 0)
                                tbl.selectRow(rr)
                            except Exception:
                                pass
            except Exception:
                pass

            keys = list(self._selected_center_row_keys() or [])
            has_sel = bool(keys)

            menu = QMenu(tbl)
            act_show = menu.addAction("Show")
            act_hide = menu.addAction("Hide")
            menu.addSeparator()
            act_clear = menu.addAction("Clear")
            act_update = menu.addAction("Update u, v")
            act_rename = menu.addAction("Rename...")

            for a in (act_show, act_hide, act_clear, act_update, act_rename):
                try:
                    a.setEnabled(bool(has_sel))
                except Exception:
                    pass

            action = menu.exec(tbl.viewport().mapToGlobal(pos))
            if action is None:
                return
            if action == act_show:
                self._set_selected_center_rows_visible(True)
                return
            if action == act_hide:
                self._set_selected_center_rows_visible(False)
                return
            if action == act_clear:
                self._on_clear_target()
                return
            if action == act_update:
                self._on_update_target_uv()
                return
            if action == act_rename:
                self._on_center_rename_selected()
                return
        except Exception:
            pass

    def _set_selected_center_rows_visible(self, visible):
        """Set Show/Hide state for currently selected middle-table rows."""
        try:
            keys = list(self._selected_center_row_keys() or [])
            if not keys:
                return
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            changed = False
            key_set = set()
            for k in keys:
                try:
                    si = int(k[0])
                    pp = str(k[1] if len(k) > 1 else 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                key_set.add((si, pp))
            if not key_set:
                return

            for i, rr in enumerate(rows):
                try:
                    rd = dict(rr or {})
                except Exception:
                    continue
                try:
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                if (si, pp) not in key_set:
                    continue
                try:
                    prev = bool(float(rd.get('show', 1.0)) >= 0.5)
                except Exception:
                    prev = True
                if bool(prev) == bool(visible):
                    continue
                rd['show'] = 1.0 if bool(visible) else 0.0
                rows[i] = rd
                changed = True

            if not changed:
                return
            try:
                self._push_center_undo_state()
            except Exception:
                pass
            self.center_numeric_rows = rows
            try:
                # Context-menu Show/Hide should avoid full update path.
                self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False)
            except Exception:
                pass
            try:
                if str(getattr(self, 'overlay_point_source', 'left') or 'left') == 'center':
                    self._apply_proc_zoom()
                else:
                    self._refresh_selected_overlay_only()
            except Exception:
                try:
                    self._apply_proc_zoom()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_center_rename_selected(self):
        """Rename selected middle-table rows (single direct name / multi serial names)."""
        try:
            from qt_compat.QtWidgets import QInputDialog

            keys = list(self._selected_center_row_keys() or [])
            if not keys:
                return

            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            key_to_idx = {}
            for i, rr in enumerate(rows):
                try:
                    rd = dict(rr or {})
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                    if pp not in ('c', 'r'):
                        pp = 'c'
                    key_to_idx[(si, pp)] = int(i)
                except Exception:
                    continue

            target_idx = []
            for k in keys:
                try:
                    si = int(k[0])
                    pp = str(k[1] if len(k) > 1 else 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                idx = key_to_idx.get((si, pp), None)
                if idx is None:
                    continue
                target_idx.append(int(idx))
            if not target_idx:
                return

            if len(target_idx) == 1:
                i0 = int(target_idx[0])
                rd0 = dict(rows[i0] or {})
                cur_name = str(rd0.get('name', '') or '')
                txt, ok = QInputDialog.getText(self, 'Rename', 'Name:', text=cur_name)
                if not ok:
                    return
                new_name = str(txt or '').strip()
                if not new_name:
                    return
                try:
                    self._push_center_undo_state()
                except Exception:
                    pass
                rd0['name'] = str(new_name)
                rd0['custom_name'] = str(new_name)
                try:
                    self._update_center_name_max_len(new_name)
                except Exception:
                    pass
                rows[i0] = rd0
            else:
                base, ok = QInputDialog.getText(self, 'Rename Multiple', 'Base name:', text='Name')
                if not ok:
                    return
                base = str(base or '').strip()
                if not base:
                    return
                try:
                    suffix, ok3 = QInputDialog.getItem(self, 'Rename Multiple', 'Suffix (optional):', ['', 'C', 'R'], 0, False)
                    if not ok3:
                        return
                    suffix = str(suffix or '').strip().upper()
                except Exception:
                    suffix = ''
                if suffix not in ('', 'C', 'R'):
                    suffix = ''
                start_no, ok2 = QInputDialog.getInt(self, 'Rename Multiple', 'Start number:', 1, 0, 999999, 1)
                if not ok2:
                    return
                try:
                    self._push_center_undo_state()
                except Exception:
                    pass
                for off, idx in enumerate(target_idx):
                    try:
                        rd = dict(rows[int(idx)] or {})
                    except Exception:
                        continue
                    new_name = f"{base}-{int(start_no + off):03d}"
                    if suffix:
                        new_name = f"{new_name}_{suffix}"
                    rd['name'] = str(new_name)
                    rd['custom_name'] = str(new_name)
                    try:
                        self._update_center_name_max_len(new_name)
                    except Exception:
                        pass
                    rows[int(idx)] = rd

            self._mark_center_model_mutation()
            self.center_numeric_rows = rows
            try:
                self.schedule_update(force=True, recompute_centroids=False)
            except Exception:
                pass
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
                self._refresh_transposed_views(refresh_center_view=False)
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


    def _build_center_xyz_export_tsv(self, include_header=True):
        """Build shared TSV text from middle-table data rows: NO/Name/X/Y/Z (Hide excluded)."""
        try:
            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            try:
                rows = list(self._sort_center_rows(rows))
            except Exception:
                pass
            try:
                q = str(getattr(self, 'center_name_filter_text', '') or '').strip().lower()
            except Exception:
                q = ''
            if q:
                filtered = []
                for rr in rows:
                    try:
                        nm = str(dict(rr or {}).get('name', '') or '').lower()
                    except Exception:
                        nm = ''
                    if q in nm:
                        filtered.append(rr)
                rows = filtered

            lines = []
            if bool(include_header):
                lines.append("No\tName\tX\tY\tZ")
            for rr in rows:
                try:
                    rd = dict(rr or {})
                except Exception:
                    rd = {}
                try:
                    if not bool(self._is_center_row_visible(rd)):
                        continue
                except Exception:
                    pass
                try:
                    nv = float(rd.get('no', float('nan')))
                    no_txt = str(int(round(nv))) if np.isfinite(nv) else str(int(rd.get('source_idx', -1)) + 1)
                except Exception:
                    try:
                        no_txt = str(int(rd.get('source_idx', -1)) + 1)
                    except Exception:
                        no_txt = ""
                try:
                    name_txt = str(rd.get('name', '') or '')
                except Exception:
                    name_txt = ""
                try:
                    sx = str(rd.get('x', '') if np.isfinite(float(rd.get('x', float('nan')))) else '')
                except Exception:
                    sx = ""
                try:
                    sy = str(rd.get('y', '') if np.isfinite(float(rd.get('y', float('nan')))) else '')
                except Exception:
                    sy = ""
                try:
                    sz = str(rd.get('z', '') if np.isfinite(float(rd.get('z', float('nan')))) else '')
                except Exception:
                    sz = ""
                lines.append(f"{no_txt}\t{name_txt}\t{sx}\t{sy}\t{sz}")
            return "\n".join(lines)
        except Exception:
            return ""

    def export_centroids(self):
        if self.img_full is None or self.centroid_processor is None:
            return
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
            txt = str(self._build_center_xyz_export_tsv(include_header=True) or "")
            if not txt.strip():
                txt = "No\tName\tX\tY\tZ"
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(txt)
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", f"Saved text export to:\n{outpath}")
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
        data["centroid_generation"] = int(getattr(self, 'centroid_generation', 0) or 0)
        data["manual_name_seq"] = int(getattr(self, '_manual_name_seq', 0) or 0)
        data["center_row_uid_seq"] = int(getattr(self, '_center_row_uid_seq', 0) or 0)
        try:
            names_out = {}
            for k, v in dict(getattr(self, 'center_group_name_overrides', {}) or {}).items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                try:
                    vv = str(v or '').strip()
                except Exception:
                    vv = ''
                if vv:
                    names_out[str(kk)] = str(vv)
            data["center_group_name_overrides"] = names_out
        except Exception:
            data["center_group_name_overrides"] = {}

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
                data["levels"] = int(getattr(self, 'slider_num_groups', None).value() if getattr(self, 'slider_num_groups', None) is not None else 4)
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
            data["shape_complexity"] = 3
        try:
            data["neck_separation"] = int(self.slider_neck_sep.value())
        except Exception:
            data["neck_separation"] = 0

        # 表示設定
        data["overlay_mode"] = str(getattr(self, 'overlay_mode', 'Original'))
        data["show_boundaries"] = bool(getattr(self, 'show_boundaries', True))
        data["flip_mode"] = str(getattr(self, 'flip_mode', 'auto'))
        data["view_orientation"] = str(getattr(self, 'view_orientation', 'Image'))
        data["online_image_grid_mode"] = str(getattr(self, 'online_image_grid_mode', 'uv'))
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
            data["auto_rim_points_proc"] = list(getattr(self, '_auto_rim_proc_points', []) or [])
        except Exception:
            data["auto_rim_points_proc"] = []
        try:
            data["center_add_core_enabled"] = bool(getattr(self, 'center_add_core_enabled', True))
            data["center_add_rim_enabled"] = bool(getattr(self, 'center_add_rim_enabled', True))
        except Exception:
            pass
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
                        "row_id": int(r.get('row_id', 0) or 0),
                        "grp": 0.0,
                        "group_no": float(r.get('group_no', r.get('grp', 0.0))),
                        "group_rank": float(r.get('group_rank', float('nan'))),
                        "generation": float(r.get('generation', float('nan'))),
                        "manual_seq": float(r.get('manual_seq', float('nan'))),
                        "name": str(r.get('name', '') or ''),
                        "custom_name": str(r.get('custom_name', '') or ''),
                        "u": float(r.get('u', float('nan'))),
                        "v": float(r.get('v', float('nan'))),
                        "x": float(r.get('x', float('nan'))),
                        "y": float(r.get('y', float('nan'))),
                        "z": float(r.get('z', float('nan'))),
                        "x_proc": float(r.get('x_proc', float('nan'))),
                        "y_proc": float(r.get('y_proc', float('nan'))),
                        "no": float(r.get('no', float('nan'))),
                        "pos": str(r.get('pos', 'c') or 'c'),
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
            # Advanced mode control (PosterLevel) deprecated; num_groups is the sole source.
        except Exception:
            pass
        for attr, key, default in [
            ('slider_min_area', 'min_area', 50),
            ('slider_trim', 'trim_px', 0),
            ('slider_shape_complex', 'shape_complexity', 3),
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
            self.show_boundaries = bool(data.get("show_boundaries", False))
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
        try:
            gm = str(data.get("online_image_grid_mode", "uv") or "uv").lower().strip()
            self.online_image_grid_mode = 'xy' if gm == 'xy' else 'uv'
        except Exception:
            self.online_image_grid_mode = 'uv'
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
            self._update_online_grid_mode_toggle_visibility()
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
        try:
            self.centroid_generation = int(data.get("centroid_generation", getattr(self, 'centroid_generation', 0) or 0))
        except Exception:
            self.centroid_generation = int(getattr(self, 'centroid_generation', 0) or 0)
        try:
            self._manual_name_seq = int(data.get("manual_name_seq", getattr(self, '_manual_name_seq', 0) or 0))
        except Exception:
            self._manual_name_seq = int(getattr(self, '_manual_name_seq', 0) or 0)
        try:
            self._center_row_uid_seq = int(data.get("center_row_uid_seq", getattr(self, '_center_row_uid_seq', 0) or 0))
        except Exception:
            self._center_row_uid_seq = int(getattr(self, '_center_row_uid_seq', 0) or 0)
        try:
            raw_names = dict(data.get("center_group_name_overrides", {}) or {})
            names = {}
            for k, v in raw_names.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                try:
                    vv = str(v or '').strip()
                except Exception:
                    vv = ''
                if vv:
                    names[int(kk)] = str(vv)
            self.center_group_name_overrides = names
        except Exception:
            self.center_group_name_overrides = {}

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
            self._auto_rim_proc_points = list(data.get("auto_rim_points_proc", []) or [])
        except Exception:
            self._auto_rim_proc_points = []
        try:
            self.manual_target_mode = bool(data.get("manual_target_mode", False))
        except Exception:
            self.manual_target_mode = False
        try:
            self.center_add_core_enabled = bool(data.get("center_add_core_enabled", True))
            self.center_add_rim_enabled = bool(data.get("center_add_rim_enabled", True))
        except Exception:
            self.center_add_core_enabled = True
            self.center_add_rim_enabled = True
        try:
            if (not bool(self.center_add_core_enabled)) and (not bool(self.center_add_rim_enabled)):
                self.center_add_core_enabled = True
        except Exception:
            self.center_add_core_enabled = True
            self.center_add_rim_enabled = True
        mt_restored = []
        for c in (data.get("manual_targets", []) or []):
            try:
                mt_restored.append((int(c["group"]), float(c["x_proc"]), float(c["y_proc"])))
            except Exception:
                pass
        self.manual_targets = mt_restored
        self.centroids = self._compose_centroids_with_manual(self._auto_centroids)
        try:
            self._centroid_rim_proc_points = self._compose_rim_points_with_manual(self._auto_centroids, self._auto_rim_proc_points)
        except Exception:
            self._centroid_rim_proc_points = []
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
                        "row_id": int(r.get('row_id', 0) or 0),
                        "grp": 0.0,
                        "group_no": float(r.get('group_no', r.get('grp', 0.0))),
                        "group_rank": float(r.get('group_rank', float('nan'))),
                        "generation": float(r.get('generation', float('nan'))),
                        "manual_seq": float(r.get('manual_seq', float('nan'))),
                        "name": str(r.get('name', '') or ''),
                        "custom_name": str(r.get('custom_name', '') or ''),
                        "u": float(r.get('u', float('nan'))),
                        "v": float(r.get('v', float('nan'))),
                        "x": float(r.get('x', float('nan'))),
                        "y": float(r.get('y', float('nan'))),
                        "z": float(r.get('z', float('nan'))),
                        "x_proc": float(r.get('x_proc', float('nan'))),
                        "y_proc": float(r.get('y_proc', float('nan'))),
                        "no": float(r.get('no', float('nan'))),
                        "pos": str(r.get('pos', 'c') or 'c').lower().strip(),
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
                self._append_center_numeric_rows_from_indices(
                    self._get_center_list_indices(),
                    refresh_existing=False,
                    source='load_project_recover',
                )
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
        try:
            # Load Project is one of the few flows that should rebuild middle view.
            self._refresh_transposed_views(update_ref_view=False, refresh_offline_lists=False, refresh_center_view=True)
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
        self.selected_index = idx
        try:
            self.selected_point_pos = 'c'
            self.selected_point_keys = {(int(idx), 'c')}
        except Exception:
            pass
        try:
            self._sync_table_selection()
        except Exception:
            pass
        try:
            self._refresh_selected_overlay_only()
        except Exception:
            self.schedule_update(force=True, recompute_centroids=False)
        try:
            self._center_on_centroid_index(idx)
        except Exception:
            pass

    def _on_table_between_current_changed(self, curRow, curCol, prevRow, prevCol):
        # transposed view row maps to original table column (selected centroid index)
        try:
            if curRow is None or curRow < 0:
                return
            header_rows = 2
            row = int(curRow)
            if row < header_rows:
                return
            row = int(row - header_rows)
            idxs = list(getattr(self, '_table_between_row_indices', []) or [])
            if not (0 <= row < len(idxs)):
                return
            idx = int(idxs[row])
            self.selected_index = idx
            try:
                row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
                if 0 <= row < len(row_keys):
                    ptag = str(row_keys[row][1] or 'c').lower().strip()
                    if ptag in ('c', 'r'):
                        self.selected_point_pos = ptag
                        self.selected_point_keys = {(int(idx), ptag)}
                try:
                    # Prefer actual current multi-row selection to avoid stale/single-key drift.
                    sel_keys = list(self._selected_center_row_keys() or [])
                    if sel_keys:
                        self.selected_point_keys = {
                            (int(kk[0]), str(kk[1] if len(kk) > 1 else 'c').lower().strip())
                            for kk in sel_keys
                        }
                except Exception:
                    pass
            except Exception:
                pass
            try:
                self._sync_table_selection()
            except Exception:
                pass
            try:
                self._refresh_selected_overlay_only()
            except Exception:
                self.schedule_update(force=True, recompute_centroids=False)
            try:
                self._log_center_selection_snapshot('table_between_current_changed')
            except Exception:
                pass
            try:
                self._center_on_centroid_index(idx)
            except Exception:
                pass
        except Exception:
            pass

    def _on_table_between_selection_changed(self):
        """Sync multi-row selection in middle table to overlay selected points."""
        try:
            tbl = getattr(self, 'table_between', None)
            if tbl is None:
                return
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])

            sel_rows = list(self._selected_center_row_numbers() or [])

            if not sel_rows:
                try:
                    self.selected_point_keys = set()
                    self.selected_index = None
                    self.selected_point_pos = 'c'
                except Exception:
                    pass
                try:
                    self._refresh_selected_overlay_only()
                except Exception:
                    self.schedule_update(force=True, recompute_centroids=False)
                try:
                    self._log_center_selection_snapshot('table_between_selection_cleared')
                except Exception:
                    pass
                return

            try:
                sel_rows = sorted({int(r) for r in sel_rows})
            except Exception:
                pass

            keys = set()
            for rr in sel_rows:
                try:
                    si = int(row_keys[rr][0])
                    ptag = str(row_keys[rr][1] or 'c').lower().strip()
                except Exception:
                    continue
                if ptag not in ('c', 'r'):
                    ptag = 'c'
                keys.add((si, ptag))
            self.selected_point_keys = keys

            try:
                cur_row = None
                try:
                    cur_idx = tbl.currentIndex()
                    if cur_idx is not None and cur_idx.isValid():
                        rr = int(cur_idx.row()) - 2
                        if 0 <= rr < len(row_keys):
                            cur_row = int(rr)
                except Exception:
                    cur_row = None
                if cur_row is None:
                    cur_row = int(sel_rows[-1])
                cur = row_keys[int(cur_row)]
                self.selected_index = int(cur[0])
                ptag = str(cur[1] or 'c').lower().strip()
                self.selected_point_pos = ptag if ptag in ('c', 'r') else 'c'
            except Exception:
                pass

            try:
                self._refresh_selected_overlay_only()
            except Exception:
                self.schedule_update(force=True, recompute_centroids=False)
            try:
                self._log_center_selection_snapshot('table_between_selection_changed')
            except Exception:
                pass
        except Exception:
            pass

    def _on_table_between_cell_clicked(self, row, col):
        # Header click: sort No./u/v/X/Y/Z (asc/desc toggle)
        try:
            if row is None or col is None:
                return
            rr = int(row)
            cc = int(col)
            if rr not in (0, 1):
                return
            key = self._center_sort_click_key(cc)
            if key is None:
                return
            self._toggle_center_sort(key)
            try:
                self._refresh_transposed_views()
            except Exception:
                pass
            try:
                # Keep middle pseudo-header text/spans stable after sort toggles.
                self._setup_pseudo_headers_between(getattr(self, 'table_between', None))
            except Exception:
                pass
            try:
                # Ensure fixed header follows the rebuilt middle header immediately.
                self._rebuild_fixed_headers()
            except Exception:
                pass
        except Exception:
            pass

    def _on_table_between_cell_double_clicked(self, row, col):
        """Start inline edit on middle-table Name column by double click."""
        try:
            rr = int(row)
            cc = int(col)
        except Exception:
            return
        if rr < 2 or cc != 1:
            return
        try:
            tbl = getattr(self, 'table_between', None)
            if tbl is None or (not bool(tbl.isEnabled())):
                return
            it = tbl.item(rr, cc)
            if it is None:
                return
            try:
                if not (it.flags() & getattr(Qt, 'ItemIsEditable', 0)):
                    return
            except Exception:
                pass
            tbl.setCurrentCell(rr, cc)
            tbl.editItem(it)
            try:
                self._log_center_selection_snapshot('table_between_name_dblclick_edit')
            except Exception:
                pass
        except Exception:
            pass

    def _on_table_between_item_changed(self, item):
        """Persist Name-column edits from middle table into center_numeric_rows."""
        try:
            if item is None:
                return
            try:
                rr = int(item.row())
                cc = int(item.column())
            except Exception:
                return
            # data rows only; name column only
            if rr < 2 or cc != 1:
                return

            try:
                new_name = str(item.text() or '').strip()
            except Exception:
                new_name = ''
            if not new_name:
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass
                return

            row_i = int(rr - 2)
            row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
            if not (0 <= row_i < len(row_keys)):
                return
            try:
                key_si = int(row_keys[row_i][0])
                key_pp = str(row_keys[row_i][1] or 'c').lower().strip()
            except Exception:
                return
            if key_pp not in ('c', 'r'):
                key_pp = 'c'

            rows = list(getattr(self, 'center_numeric_rows', []) or [])
            tgt = None
            prev = ''
            for i, rr0 in enumerate(rows):
                try:
                    rd = dict(rr0 or {})
                    si = int(rd.get('source_idx', -1))
                    pp = str(rd.get('pos', 'c') or 'c').lower().strip()
                except Exception:
                    continue
                if pp not in ('c', 'r'):
                    pp = 'c'
                if si == key_si and pp == key_pp:
                    tgt = int(i)
                    prev = str(rd.get('name', '') or '')
                    break

            if tgt is None:
                return
            if str(prev) == str(new_name):
                return

            try:
                self._push_center_undo_state()
            except Exception:
                pass

            rd = dict(rows[tgt] or {})
            rd['name'] = str(new_name)
            rd['custom_name'] = str(new_name)
            try:
                self._update_center_name_max_len(new_name)
            except Exception:
                pass
            rows[tgt] = rd
            self.center_numeric_rows = rows

            try:
                self._refresh_transposed_views()
            except Exception:
                pass
        except Exception:
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
            if str(mode or '') != 'center_uv_update':
                self._apply_pick_mode_wait_button_style()
        except Exception:
            pass

    def _apply_pick_mode_wait_button_style(self):
        """Apply persistent wait-state style for pick-mode buttons (light red)."""
        try:
            radius = 8
            mode = str(getattr(self, 'pick_mode', '') or '')
            if mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                # pick モード開始時は対応ボタンをキャンセル表示にする
                if mode == 'add':
                    target_btn = self.btn_add_ref
                    wait_text = 'Finish'
                    wait_style = (
                        f"QPushButton {{ background-color: rgb(24,96,80); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(40,112,96); }}"
                        f"QPushButton:pressed {{ background-color: rgb(8,80,64); }}"
                    )
                elif mode == 'update':
                    target_btn = getattr(self, 'btn_update_xy', None)
                    wait_text = STR.BUTTON_ADD_REF_CANCEL
                    wait_style = (
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
                elif mode == 'target_add':
                    target_btn = getattr(self, 'btn_add_target', None)
                    wait_text = 'Finish'
                    wait_style = (
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
                elif mode == 'target_update':
                    target_btn = getattr(self, 'btn_update_target_uv', None)
                    wait_text = STR.BUTTON_ADD_REF_CANCEL
                    wait_style = (
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
                else:
                    target_btn = getattr(self, 'btn_update_target_uv', None)
                    wait_text = 'Finish'
                    wait_style = (
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
                if target_btn is not None:
                    # ボタンのテキストを「Cancel」に変更
                    try:
                        target_btn.setText(wait_text)
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

                    # Also lock height in QSS to avoid style-driven recalculation.
                    try:
                        target_btn.setStyleSheet(wait_style)
                    except Exception:
                        pass
        except Exception:
            pass

    def _flash_button_feedback(self, btn, role='gray', ms=500):
        """Flash button with a lighter tint for a short duration."""
        try:
            if btn is None:
                return
            radius = 8
            try:
                original_style = str(btn.styleSheet() or '')
            except Exception:
                original_style = ''
            try:
                h0 = int(btn.height() or 0)
            except Exception:
                h0 = 0
            try:
                w0 = int(btn.width() or 0)
            except Exception:
                w0 = 0

            role_l = str(role).lower()
            try:
                stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
            except Exception:
                stage_n = 'offline'
            if role_l == 'accent':
                if stage_n == 'online':
                    style = (
                        f"QPushButton {{ background-color: rgb(24,96,80); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(40,112,96); }}"
                        f"QPushButton:pressed {{ background-color: rgb(8,80,64); }}"
                    )
                else:
                    style = (
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
            elif role_l == 'green':
                style = (
                    f"QPushButton {{ background-color: rgb(24,96,80); color: white; border: none; border-radius: {radius}px; }}"
                    f"QPushButton:hover {{ background-color: rgb(40,112,96); }}"
                    f"QPushButton:pressed {{ background-color: rgb(8,80,64); }}"
                )
            elif str(role).lower() == 'red':
                style = (
                    f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                    f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                    f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                )
            else:
                style = (
                    f"QPushButton {{ background-color: rgb(185,185,185); color: white; border: none; border-radius: {radius}px; }}"
                    f"QPushButton:hover {{ background-color: rgb(175,175,175); }}"
                    f"QPushButton:pressed {{ background-color: rgb(165,165,165); }}"
                )

            try:
                btn.setStyleSheet(style)
            except Exception:
                pass

            try:
                def _restore_this_button(b=btn, s=original_style, hh=h0, ww=w0):
                    try:
                        b.setStyleSheet(s)
                    except Exception:
                        pass
                    try:
                        if int(hh) > 0:
                            b.setFixedHeight(int(hh))
                    except Exception:
                        pass
                    try:
                        if int(ww) > 0:
                            b.setFixedWidth(int(ww))
                    except Exception:
                        pass
                    try:
                        if str(getattr(self, 'pick_mode', '') or '') in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                            self._apply_pick_mode_wait_button_style()
                    except Exception:
                        pass

                QTimer.singleShot(int(ms), _restore_this_button)
            except Exception:
                try:
                    btn.setStyleSheet(original_style)
                except Exception:
                    pass
        except Exception:
            pass

    def _restore_button_visual_states(self):
        """Restore standard button styles and re-apply mode-aware active styles."""
        try:
            self._apply_button_styles()
        except Exception:
            pass
        try:
            if str(getattr(self, 'pick_mode', '') or '') in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                self._apply_pick_mode_wait_button_style()
        except Exception:
            pass

    def _wire_click_feedback_once(self):
        """Wire click feedback handlers only once."""
        try:
            if bool(getattr(self, '_click_feedback_wired', False)):
                return
        except Exception:
            pass

        def _bind(btn, role):
            try:
                if btn is None:
                    return
                btn.clicked.connect(lambda _checked=False, b=btn, r=str(role): self._flash_button_feedback(b, r, 500))
            except Exception:
                pass

        # Accent buttons: red offline / green online
        _bind(getattr(self, 'btn_add_ref', None), 'accent')
        _bind(getattr(self, 'btn_add_target', None), 'red')
        _bind(getattr(self, 'btn_start_centroid_extraction', None), 'red')
        _bind(getattr(self, 'btn_open', None), 'accent')
        _bind(getattr(self, 'btn_export', None), 'accent')
        _bind(getattr(self, 'btn_online_export', None), 'accent')
        _bind(getattr(self, 'btn_new_project', None), 'accent')

        # Gray-themed buttons
        _bind(getattr(self, 'btn_update_xy', None), 'gray')
        _bind(getattr(self, 'btn_clear_ref', None), 'gray')
        _bind(getattr(self, 'btn_update_target_uv', None), 'gray')
        _bind(getattr(self, 'btn_clear_target', None), 'gray')
        _bind(getattr(self, 'btn_clear_target_all', None), 'gray')
        _bind(getattr(self, 'btn_select_all', None), 'gray')
        _bind(getattr(self, 'btn_center_undo', None), 'gray')
        _bind(getattr(self, 'btn_center_name_filter', None), 'gray')
        _bind(getattr(self, 'btn_clipboard', None), 'gray')
        _bind(getattr(self, 'btn_online_clipboard', None), 'gray')
        _bind(getattr(self, 'btn_filter', None), 'gray')
        _bind(getattr(self, 'btn_replace_image', None), 'gray')
        _bind(getattr(self, 'btn_save_project', None), 'gray')
        _bind(getattr(self, 'btn_load_project', None), 'gray')
        _bind(getattr(self, 'btn_left_settings', None), 'gray')

        try:
            self._click_feedback_wired = True
        except Exception:
            pass

    def _end_pick_mode(self, redraw: bool = True):
        self.pick_mode = None
        self.pick_ref_index = None
        self._replace_target_source_index = None
        self._center_uv_update_queue = []
        self._center_uv_update_pos = 0
        self._center_uv_update_active_key = None
        self._stage_info_override_text = None
        self._ref_add_has_added = False
        try:
            self._set_center_uv_nav_visible(False)
        except Exception:
            pass
        try:
            self._set_center_uv_update_ui_lock(False)
        except Exception:
            pass
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
                btn_tadd.setText("Add Target Point")
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
        try:
            # Ensure override text disappears immediately after finishing update.
            self._update_stage_info_overlay(getattr(self, '_last_stage_info', None), getattr(self, 'view_orientation', 'Image'))
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
                try:
                    self._mark_center_model_mutation()
                except Exception:
                    pass
                self.center_numeric_rows = rows
                try:
                    si = int(rd.get('source_idx', -1))
                    if si >= 0:
                        self.selected_index = si
                        try:
                            ptag = str(rd.get('pos', 'c') or 'c').lower().strip()
                        except Exception:
                            ptag = 'c'
                        if ptag not in ('c', 'r'):
                            ptag = 'c'
                        self.selected_point_pos = ptag
                        self.selected_point_keys = {(int(si), str(ptag))}
                except Exception:
                    pass

                has_next = False
                try:
                    q = list(getattr(self, '_center_uv_update_queue', []) or [])
                    if q:
                        try:
                            pos = int(getattr(self, '_center_uv_update_pos', 0) or 0)
                        except Exception:
                            pos = 0
                        pos = max(0, min(pos, len(q) - 1))
                        self._center_uv_update_pos = int(pos + 1)
                        has_next = bool((pos + 1) < len(q))
                except Exception:
                    has_next = False

                try:
                    self.schedule_update(force=True, recompute_centroids=False)
                except Exception:
                    pass

                if has_next:
                    try:
                        # Show the just-picked point briefly before jumping to the next target.
                        try:
                            self._refresh_selected_overlay_only()
                        except Exception:
                            pass
                        def _go_next_center_uv():
                            try:
                                if str(getattr(self, 'pick_mode', '') or '') == 'center_uv_update':
                                    self._activate_center_uv_update_target()
                            except Exception:
                                pass
                        QTimer.singleShot(220, _go_next_center_uv)
                    except Exception:
                        pass
                else:
                    try:
                        self._end_pick_mode(redraw=False)
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
            try:
                self._push_center_undo_state()
            except Exception:
                pass
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
                t_add0 = monotonic()
                perf_enabled = bool(str(os.environ.get('PIXY_PERF', '')).strip())

                def _log_add_target_perf(stage):
                    if not perf_enabled:
                        return
                    try:
                        if hasattr(self, '_log_info'):
                            self._log_info(f"[PERF][AddTarget] {stage} dt={float(monotonic() - t_add0):.3f}s")
                    except Exception:
                        pass

                if self.pick_mode == 'target_add':
                    base = self._manual_target_base_index()
                    mt_old_n = len(self.manual_targets)
                    insert_idx = int(base + mt_old_n)
                    try:
                        self._shift_center_list_indices(insert_idx, +1)
                    except Exception:
                        pass
                    self.manual_targets.append((0, float(x_proc), float(y_proc)))
                    try:
                        if hasattr(self, '_log_info'):
                            self._log_info(
                                f"AddTarget: added idx={insert_idx} x_proc={float(x_proc):.3f} y_proc={float(y_proc):.3f}"
                            )
                    except Exception:
                        pass
                    _log_add_target_perf('manual_targets')
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
                    _log_add_target_perf('visibility_maps')
                    self.centroids = self._compose_centroids_with_manual(auto_only)
                    _log_add_target_perf('compose_centroids')
                    try:
                        self._centroid_rim_proc_points = self._compose_rim_points_with_manual(auto_only, getattr(self, '_auto_rim_proc_points', []))
                    except Exception:
                        self._centroid_rim_proc_points = []
                    _log_add_target_perf('compose_rim')
                    self.selected_index = insert_idx
                    add_pos = self._current_add_target_pos()
                    try:
                        _ptag_add = str(add_pos or 'c').lower().strip()
                        if _ptag_add not in ('c', 'r'):
                            _ptag_add = 'c'
                        self.selected_point_pos = _ptag_add
                        self.selected_point_keys = {(int(insert_idx), _ptag_add)}
                        # Switch early so all signals/renders triggered below use center mode.
                        if str(getattr(self, 'overlay_point_source', 'left') or 'left') != 'center':
                            self.overlay_point_source = 'center'
                            try:
                                tog = getattr(self, 'toggle_overlay_source', None)
                                if tog is not None:
                                    tog.setCheckedIndex(1)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        add_name, add_seq = self._add_target_name_text()
                    except Exception:
                        add_name, add_seq = 'Name-001', 1
                    try:
                        self._append_center_numeric_rows_from_indices(
                            [(insert_idx, add_pos)],
                            manual_name_override=add_name,
                            manual_seq_override=add_seq,
                            refresh_existing=False,
                            source='add_target',
                        )
                    except Exception:
                        pass
                    _log_add_target_perf('append_center_rows')
                    try:
                        self._advance_add_target_name_seq()
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
                    try:
                        self._centroid_rim_proc_points = self._compose_rim_points_with_manual(auto_only, getattr(self, '_auto_rim_proc_points', []))
                    except Exception:
                        self._centroid_rim_proc_points = []
                    base = self._manual_target_base_index()
                    self.selected_index = int(base + idx_t)

                self._safe_populate_tables(
                    self.table_ref, self.table,
                    self.ref_points, self.ref_obs,
                    self.centroids, self.selected_index,
                    self.ref_selected_index,
                    flip_mode=self.flip_mode,
                    visible_ref_cols=self.visible_ref_cols,
                    _auto_fit_tables=False,
                )
                _log_add_target_perf('safe_populate_tables')
                try:
                    self._refresh_transposed_views()
                except Exception:
                    pass
                _log_add_target_perf('refresh_transposed_views')
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
                    _log_add_target_perf('apply_proc_zoom')
                    if center_full_before is not None:
                        try:
                            self._ensure_full_pos_visible(float(center_full_before[0]), float(center_full_before[1]))
                        except Exception:
                            pass
                    _log_add_target_perf('done')
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
                                    t.setHorizontalHeaderLabels([f"Fiducial {i + 1}" for i in range(t.columnCount())])
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
                        self._refresh_transposed_views(refresh_center_view=False)
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
                                self.table_ref.setHorizontalHeaderLabels([f"Fiducial {i + 1}" for i in range(self.table_ref.columnCount())])
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
                    self._refresh_transposed_views(refresh_center_view=False)
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
        """Copy middle-table No/Name/X/Y/Z (visible rows only) to clipboard as TSV."""
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
            txt = str(self._build_center_xyz_export_tsv(include_header=True) or "")
            if not txt.strip():
                txt = "No\tName\tX\tY\tZ"
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
            perf_enabled = bool(str(os.environ.get('PIXY_PERF', '')).strip())
            t0 = monotonic() if perf_enabled else None
            try:
                do_auto_fit = bool(kwargs.pop('_auto_fit_tables', True))
            except Exception:
                do_auto_fit = True
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
                        # Keep core/rim toggle state consistent with internal flags
                        try:
                            b_core = getattr(self, 'btn_center_add_core', None)
                            b_rim = getattr(self, 'btn_center_add_rim', None)
                            if b_core is not None:
                                b_core.blockSignals(True)
                                b_core.setChecked(bool(getattr(self, 'center_add_core_enabled', True)))
                                b_core.blockSignals(False)
                            if b_rim is not None:
                                b_rim.blockSignals(True)
                                b_rim.setChecked(bool(getattr(self, 'center_add_rim_enabled', True)))
                                b_rim.blockSignals(False)
                        except Exception:
                            pass
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
                        self._refresh_transposed_views(refresh_center_view=False)
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
            if perf_enabled and t0 is not None:
                try:
                    if hasattr(self, '_log_info'):
                        self._log_info(f"[PERF][safe_populate_tables] populate_tables dt={float(monotonic() - t0):.3f}s")
                except Exception:
                    pass
            # Reinstall pseudo-headers after populate (data might overwrite them)
            try:
                self._setup_pseudo_headers_ref(self.table_ref)
            except Exception:
                pass
            # Keep the left TargetPoint table tight to its content so it doesn't leave a blank block.
            try:
                self._sync_table_ref_view_height()
            except Exception:
                pass
            try:
                self._sync_table_ref_view_height()
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
            if do_auto_fit:
                try:
                    QTimer.singleShot(220, self._auto_fit_table_fonts)
                except Exception:
                    pass
            if perf_enabled and t0 is not None:
                try:
                    if hasattr(self, '_log_info'):
                        self._log_info(f"[PERF][safe_populate_tables] total dt={float(monotonic() - t0):.3f}s")
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

    def _refresh_transposed_views(self, update_ref_view=True, refresh_offline_lists=True, refresh_center_view=True):
        # Create/update transposed copies of `self.table_ref` and `self.table`.
        try:
            perf_enabled = bool(str(os.environ.get('PIXY_PERF', '')).strip())
            t0 = monotonic() if perf_enabled else None
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
                    try:
                        stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
                    except Exception:
                        stage_n = 'offline'
                    is_online_stage = bool(stage_n == 'online')
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
                            dgl = getattr(self, '_center_show_toggle_delegate', None)
                            if dgl is not None and data_cols > 0:
                                dst.setItemDelegateForColumn(int(data_cols - 1), dgl)
                        except Exception:
                            pass

                        # Keep scrollbar presence stable to avoid width/layout shifts
                        try:
                            dst.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
                            dst.setHorizontalScrollBarPolicy(_Qt.ScrollBarAlwaysOff)
                            try:
                                sb = dst.verticalScrollBar()
                                if sb is not None:
                                    sb.setMinimumWidth(14)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Vertical header: blank for header rows, then 1..N (source horizontal headers)
                        try:
                            hlabels = []
                            for i in range(src.columnCount()):
                                hi = src.horizontalHeaderItem(i)
                                hlabels.append(hi.text() if hi is not None else f"Fiducial {i + 1}")

                            # Ensure label count matches data_rows exactly.
                            # If labels are short, Qt keeps stale/numeric leftovers, which scrambles numbering.
                            fid_labels = []
                            for i in range(data_rows):
                                if i < len(hlabels) and str(hlabels[i] or '').strip() != '':
                                    fid_labels.append(str(hlabels[i]))
                                else:
                                    fid_labels.append(f"Fiducial {i + 1}")
                            dst.setVerticalHeaderLabels(["", ""] + fid_labels)
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
                                # Online table: highlight Stage(input) XYZ fields.
                                try:
                                    if is_online_stage and c in (2, 3, 4):
                                        it.setBackground(QColor(226, 245, 226))
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
                    center_rows = list(getattr(self, 'center_numeric_rows', []) or [])
                    try:
                        center_rows = self._sort_center_rows(center_rows)
                    except Exception:
                        pass
                    try:
                        q = str(getattr(self, 'center_name_filter_text', '') or '').strip().lower()
                    except Exception:
                        q = ''
                    if q:
                        filtered = []
                        for rr in (center_rows or []):
                            try:
                                nm = str(dict(rr or {}).get('name', '') or '').lower()
                            except Exception:
                                nm = ''
                            if q in nm:
                                filtered.append(rr)
                        center_rows = filtered
                    try:
                        selected_key_set = set()
                        for kk in (getattr(self, 'selected_point_keys', set()) or set()):
                            try:
                                si = int(kk[0])
                                pp = str(kk[1] if len(kk) > 1 else 'c').lower().strip()
                            except Exception:
                                continue
                            if pp not in ('c', 'r'):
                                pp = 'c'
                            selected_key_set.add((si, pp))
                    except Exception:
                        selected_key_set = set()
                    data_rows = int(len(center_rows))
                    try:
                        mapped = []
                        mapped_keys = []
                        for r in (center_rows or []):
                            try:
                                rd = dict(r or {})
                            except Exception:
                                rd = {}
                            try:
                                mapped.append(int(rd.get('source_idx', -1)))
                            except Exception:
                                mapped.append(-1)
                            try:
                                ptag = str(rd.get('pos', 'c') or 'c').lower().strip()
                                if ptag not in ('c', 'r'):
                                    ptag = 'c'
                                mapped_keys.append((int(rd.get('source_idx', -1)), ptag))
                            except Exception:
                                mapped_keys.append((-1, 'c'))
                        self._table_between_row_indices = mapped
                        self._table_between_row_keys = mapped_keys
                    except Exception:
                        self._table_between_row_indices = []
                        self._table_between_row_keys = []
                    # Middle table layout:
                    # Offline: ID, Name, u, v, Grp, No., C/R, Gen., Show
                    # Online : ID, Name, u, v, X,   Y,   Z,   Show
                    base_cols = 0
                    try:
                        base_cols = len(getattr(STR, 'TABLE_RIGHT_ROW_LABELS', []) or [])
                    except Exception:
                        base_cols = 0
                    base_cols = max(0, int(base_cols))
                    try:
                        stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
                    except Exception:
                        stage_n = 'offline'
                    is_online_center = bool(stage_n == 'online')
                    # +4 metadata columns (Grp/No./C-R/Gen.) in offline mode.
                    # Online mode keeps XYZ columns.
                    data_cols = 8 if is_online_center else max(9, base_cols + 4)
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
                            dgl = getattr(self, '_center_show_toggle_delegate', None)
                            if dgl is not None and data_cols > 0:
                                dst.setItemDelegateForColumn(int(data_cols - 1), dgl)
                        except Exception:
                            pass

                        try:
                            dst.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
                            dst.setHorizontalScrollBarPolicy(_Qt.ScrollBarAlwaysOff)
                        except Exception:
                            pass
                        # Vertical header: keep blank (ID column is the single source of point IDs)
                        try:
                            dst.setVerticalHeaderLabels([""] * int(data_rows + header_rows))
                        except Exception:
                            pass

                        for r in range(data_rows):
                            try:
                                rowd = dict(center_rows[r] or {})
                            except Exception:
                                continue
                            try:
                                row_visible = bool(self._is_center_row_visible(rowd))
                            except Exception:
                                row_visible = True
                            try:
                                cidx = int(rowd.get('source_idx', -1))
                            except Exception:
                                cidx = -1
                            for c in range(data_cols):
                                try:
                                    if c == 0:
                                        try:
                                            nv = float(rowd.get('no', float('nan')))
                                            if np.isnan(nv):
                                                raise ValueError('nan')
                                            txt = str(int(round(nv)))
                                        except Exception:
                                            try:
                                                txt = str(int(rowd.get('source_idx', -1)) + 1)
                                            except Exception:
                                                txt = str(int(r) + 1)
                                    elif c == 1:
                                        txt = str(rowd.get('name', '') or '')
                                    elif c in (2, 3):
                                        try:
                                            vv = rowd.get('u', 0.0) if c == 2 else rowd.get('v', 0.0)
                                            txt = str(int(round(float(vv))))
                                        except Exception:
                                            txt = ""
                                    elif c == 4:
                                        if is_online_center:
                                            try:
                                                xv = rowd.get('x', float('nan'))
                                                xf = float(xv)
                                                txt = "" if np.isnan(xf) else str(int(round(xf)))
                                            except Exception:
                                                txt = ""
                                        else:
                                            try:
                                                gv = rowd.get('group_no', rowd.get('grp', float('nan')))
                                                txt = str(int(round(float(gv))))
                                            except Exception:
                                                txt = ""
                                    elif c == 5:
                                        if is_online_center:
                                            try:
                                                yv = rowd.get('y', float('nan'))
                                                yf = float(yv)
                                                txt = "" if np.isnan(yf) else str(int(round(yf)))
                                            except Exception:
                                                txt = ""
                                        else:
                                            try:
                                                pv = rowd.get('group_rank', float('nan'))
                                                txt = str(int(round(float(pv))))
                                            except Exception:
                                                txt = ""
                                    elif c == 6:
                                        if is_online_center:
                                            try:
                                                zv = rowd.get('z', float('nan'))
                                                zf = float(zv)
                                                txt = "" if np.isnan(zf) else str(int(round(zf)))
                                            except Exception:
                                                txt = ""
                                        else:
                                            try:
                                                pp = str(rowd.get('pos_display', rowd.get('pos', '')) or '').lower().strip()
                                            except Exception:
                                                pp = ''
                                            if pp == 'r':
                                                txt = 'R'
                                            elif pp == 'c':
                                                txt = 'C'
                                            else:
                                                txt = ''
                                    elif c == 7:
                                        if is_online_center:
                                            txt = ""
                                        else:
                                            try:
                                                if float(rowd.get('manual', 0.0)) >= 0.5:
                                                    txt = '0'
                                                else:
                                                    gv = rowd.get('generation', float('nan'))
                                                    txt = str(int(round(float(gv))))
                                            except Exception:
                                                txt = ""
                                    elif c == (data_cols - 1):
                                        # Exclude flag — will be set as checkbox below
                                        txt = ""
                                    else:
                                        txt = ""
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
                                elif c == 1:
                                    # Name column: editable by user.
                                    try:
                                        it.setFlags(it.flags() | getattr(_Qt, 'ItemIsEditable', 0))
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
                                        f = it.font()
                                        f.setBold(True)
                                        try:
                                            f.setPointSize(max(7, int(f.pointSize()) - 1))
                                        except Exception:
                                            pass
                                        it.setFont(f)
                                except Exception:
                                    pass
                                # Bold important columns for readability
                                try:
                                    tmp_sub_labels = (["ID", "Name", "u", "v", "X", "Y", "Z", ""] if is_online_center else ["ID", "Name", "u", "v", "Grp", "No.", "C/R", "Gen.", ""])
                                    sub_lbl = tmp_sub_labels[c] if 0 <= c < len(tmp_sub_labels) else None
                                    if (is_online_center and sub_lbl in ("X", "Y", "Z")) or ((not is_online_center) and sub_lbl in ("Grp", "No.", "C/R", "Gen.")):
                                        f = it.font(); f.setBold(True); it.setFont(f)
                                except Exception:
                                    pass
                                # Keep rebuild lightweight: skip per-cell font fitting here.
                                # Dim text + gray background for hidden center rows.
                                try:
                                    try:
                                        k_pp = str(rowd.get('pos', 'c') or 'c').lower().strip()
                                    except Exception:
                                        k_pp = 'c'
                                    if k_pp not in ('c', 'r'):
                                        k_pp = 'c'
                                    if not bool(row_visible):
                                        # Reflect Hide immediately even for selected rows.
                                        it.setForeground(QColor(150, 150, 150))
                                        it.setBackground(QColor(238, 238, 238))
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

                        # In-cell header (top ID/Name area is intentionally blank)
                        name_label = 'Name'
                        if is_online_center:
                            group_configs = [(0, 1, ""), (1, 1, ""), (2, 2, "Image"), (4, 3, "Stage"), (7, 1, "")]
                            sub_labels = [
                                self._center_label_with_sort('no', 'ID'),
                                name_label,
                                self._center_label_with_sort('u', 'u'),
                                self._center_label_with_sort('v', 'v'),
                                self._center_label_with_sort('x', 'X'),
                                self._center_label_with_sort('y', 'Y'),
                                self._center_label_with_sort('z', 'Z'),
                                "",
                            ]
                        else:
                            group_configs = [(0, 1, ""), (1, 1, ""), (2, 2, "Image"), (4, 4, "Centroid Extraction"), (8, 1, "")]
                            sub_labels = [
                                self._center_label_with_sort('no', 'ID'),
                                name_label,
                                self._center_label_with_sort('u', 'u'),
                                self._center_label_with_sort('v', 'v'),
                                self._center_label_with_sort('grp', 'Grp'),
                                self._center_label_with_sort('pno', 'No.'),
                                self._center_label_with_sort('cr', 'C/R'),
                                self._center_label_with_sort('gen', 'Gen.'),
                                "",
                            ]
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
            if bool(update_ref_view) and (not editing_left):
                try:
                    _build_ref_transposed_view()
                except Exception:
                    pass
                try:
                    self._pending_ref_view_refresh = False
                except Exception:
                    pass

            # update bottom/transposed table_between
            if bool(refresh_center_view):
                try:
                    _build_mid_transposed_view()
                except Exception:
                    pass
            if perf_enabled and t0 is not None:
                try:
                    if hasattr(self, '_log_info'):
                        self._log_info(f"[PERF][_refresh_transposed_views] total dt={float(monotonic() - t0):.3f}s")
                except Exception:
                    pass

            if bool(refresh_offline_lists):
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
            try:
                QTimer.singleShot(0, lambda: self._set_online_fiducial_rows_fixed(5))
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
                            min_left_w = int(getattr(self, 'left_column_min_width', LEFT_COLUMN_MIN_WIDTH) or LEFT_COLUMN_MIN_WIDTH)
                            new_w = max(min_left_w, max(370, new_w))
                            tbl.setFixedWidth(new_w)
                            # ヘッダー・コンテナ幅の同期先はスワップ有無で切り替える
                            if bool(getattr(self, 'swap_left_center_columns', False)):
                                lc = getattr(self, 'center_swap_container', None)
                            else:
                                lc = getattr(self, 'left_container', None)
                            if lc is not None:
                                try:
                                    lc.setMinimumWidth(min_left_w)
                                    lc.setFixedWidth(new_w)
                                except Exception:
                                    pass
                            # 左ロゴ幅は通常レイアウト時のみ left table に同期する。
                            if not bool(getattr(self, 'swap_left_center_columns', False)):
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
                    name_col_width = int(self._center_name_column_width_hint(tbl2))
                except Exception:
                    name_col_width = 0
                try:
                    tbl2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    tbl2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                except Exception:
                    pass
                try:
                    hdr2 = tbl2.horizontalHeader()
                    hdr2.setSectionResizeMode(QHeaderView.Fixed)
                except Exception:
                    hdr2 = None
                cnt2 = tbl2.columnCount()
                if cnt2 > 0:
                    # Match widths to the left transposed reference view when possible
                    # and keep Show column wider for toggle.
                    ref_tbl = getattr(self, 'table_ref_view', None)
                    for i in range(cnt2):
                        try:
                            if i == 0:
                                w = 64
                            elif i == (cnt2 - 1):
                                w = 48
                            elif i == 1 and int(name_col_width) > 0:
                                w = max(72, min(132, int(name_col_width)))
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
                        if hdr2 is not None:
                            hdr2.setDefaultSectionSize(max(8, 40))
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
            # Single source of truth for center width.
            try:
                self._adjust_center_column_widths()
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
            h = int(avail - 18)
            h = max(120, min(520, h))
            try:
                if bool(getattr(self, 'centroid_extraction_mode', False)):
                    h = max(96, min(220, h))
            except Exception:
                pass
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
            h = int(avail - 18)
            h = max(120, min(520, h))
            try:
                if bool(getattr(self, 'centroid_extraction_mode', False)):
                    h = max(96, min(220, h))
            except Exception:
                pass
            return h
        except Exception:
            return 190

    def _refresh_offline_group_lists(self):
        """Rebuild group-wise u/v lists shown at the bottom of Off-line Targeting tab."""
        try:
            host = getattr(self, 'offline_group_layout', None)
            if host is None:
                return
            self._offline_group_tables = {}

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
                mid_u_w = int(mid_tbl.columnWidth(2)) if mid_tbl is not None and mid_tbl.columnCount() > 2 else 0
            except Exception:
                mid_u_w = 0
            try:
                mid_v_w = int(mid_tbl.columnWidth(3)) if mid_tbl is not None and mid_tbl.columnCount() > 3 else 0
            except Exception:
                mid_v_w = 0
            col_u_w = max(32, mid_u_w if mid_u_w > 0 else 57)
            col_v_w = max(32, mid_v_w if mid_v_w > 0 else 57)
            try:
                header_h = 27
                grp_toggle_h = 27
                body_h = max(96, int(table_h))
                # Add List行を廃止した分、Show/Hideトグル上下の余白を詰める。
                panel_h = header_h + grp_toggle_h + body_h + 10
            except Exception:
                body_h = 190
                panel_h = 248

            grouped = {}
            try:
                for g in sorted(self._available_group_numbers() or []):
                    gg = int(g)
                    if gg <= 0:
                        continue
                    grouped[gg] = list(self._sorted_group_entries(gg) or [])
            except Exception:
                grouped = {}

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
                pv.setSpacing(4)

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
                head = QPushButton(f"Add Group {int(grp)}\nto Target List")
                try:
                    head.setFixedHeight(44)
                    head.setStyleSheet(
                        f"background-color: rgb({cr},{cg},{cb}); color: {txt_color};"
                        "font-weight: bold; border: none; border-radius: 4px;"
                    )
                    try:
                        head.clicked.connect(lambda _=False, g=int(grp): self._add_group_to_center_list(g))
                    except Exception:
                        pass
                    try:
                        head.clicked.connect(lambda _=False, b=head: self._flash_button_feedback(b, 'gray', 450))
                    except Exception:
                        pass
                except Exception:
                    pass
                pv.addWidget(head, 0)

                try:
                    edit_group_name = QLineEdit()
                    edit_group_name.setFixedHeight(24)
                    edit_group_name.setPlaceholderText("Group name")
                    try:
                        cur_name = str(dict(getattr(self, 'center_group_name_overrides', {}) or {}).get(int(grp), '') or '')
                    except Exception:
                        cur_name = ''
                    if cur_name:
                        edit_group_name.setText(cur_name)
                    try:
                        edit_group_name.textChanged.connect(
                            lambda txt, g=int(grp): self._on_center_group_name_text_changed(g, txt)
                        )
                    except Exception:
                        pass
                    pv.addWidget(edit_group_name, 0)
                except Exception:
                    pass

                tbl = QTableWidget()
                tbl.setColumnCount(2)
                tbl.setRowCount(len(grouped[grp]))
                try:
                    st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
                    rec = dict(st.get(int(grp), {}) or {})
                    k = str(rec.get('key', getattr(self, '_offline_group_sort_key', 'u')) or 'u').lower().strip()
                except Exception:
                    k = 'u'
                if k not in ('u', 'v'):
                    k = 'u'
                try:
                    st = dict(getattr(self, '_offline_group_sort_state', {}) or {})
                    rec = dict(st.get(int(grp), {}) or {})
                    desc = bool(rec.get('desc', getattr(self, '_offline_group_sort_desc', False)))
                except Exception:
                    desc = False
                su = f"u {'▼' if (k == 'u' and desc) else ('▲' if k == 'u' else '')}".rstrip()
                sv = f"v {'▼' if (k == 'v' and desc) else ('▲' if k == 'v' else '')}".rstrip()
                tbl.setHorizontalHeaderLabels([su, sv])
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
                    try:
                        hh.sectionClicked.connect(lambda sec, g=int(grp): self._on_offline_group_header_clicked(g, sec))
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
                    tbl.setSelectionMode(QAbstractItemView.SingleSelection)
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

                row_src_indices = []
                for r, (u, v, src_i) in enumerate(grouped[grp]):
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
                    row_src_indices.append(int(src_i))

                try:
                    tbl.cellClicked.connect(lambda row, col, t=tbl: self._on_offline_group_table_clicked(t, row, col))
                except Exception:
                    pass

                try:
                    self._offline_group_tables[int(grp)] = {
                        'panel': panel,
                        'table': tbl,
                        'indices': list(row_src_indices),
                    }
                except Exception:
                    pass

                pv.addWidget(tbl, 1)
                host.addWidget(panel, 0)

            host.addStretch(1)
            try:
                self._sync_offline_group_selection()
            except Exception:
                pass
        except Exception:
            pass

    def _on_offline_group_table_clicked(self, table, row, _col):
        """Left group-table row selection -> centroid selection + image highlight."""
        try:
            if table is None or row is None:
                return
            idx = None
            for _g, rec in dict(getattr(self, '_offline_group_tables', {}) or {}).items():
                try:
                    if rec.get('table') is table:
                        ids = list(rec.get('indices', []) or [])
                        if 0 <= int(row) < len(ids):
                            idx = int(ids[int(row)])
                        break
                except Exception:
                    continue
            if idx is None:
                return
            if self.selected_index != idx:
                self.selected_index = idx
                self.schedule_update(force=True, recompute_centroids=False)
            try:
                self._center_on_centroid_index(idx)
            except Exception:
                pass
        except Exception:
            pass

    def _sync_offline_group_selection(self):
        """Sync selected centroid index to left offline group-table row highlight."""
        try:
            sel = getattr(self, 'selected_index', None)
            mapping = dict(getattr(self, '_offline_group_tables', {}) or {})
            if not mapping:
                return

            target_tbl = None
            target_panel = None
            target_row_final = -1

            for _g, rec in mapping.items():
                try:
                    panel = rec.get('panel')
                    tbl = rec.get('table')
                    ids = list(rec.get('indices', []) or [])
                except Exception:
                    continue
                if tbl is None:
                    continue

                target_row = -1
                try:
                    if sel is not None:
                        target_row = int(ids.index(int(sel)))
                except Exception:
                    target_row = -1

                if target_row >= 0:
                    target_tbl = tbl
                    target_panel = panel
                    target_row_final = int(target_row)

                try:
                    tbl.blockSignals(True)
                    if 0 <= target_row < tbl.rowCount():
                        tbl.setCurrentCell(int(target_row), 0)
                        tbl.selectRow(int(target_row))
                    else:
                        tbl.clearSelection()
                finally:
                    try:
                        tbl.blockSignals(False)
                    except Exception:
                        pass

            try:
                self._ensure_offline_group_row_visible(target_panel, target_tbl, target_row_final)
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_offline_group_row_visible(self, panel, table, row):
        """Scroll outer group area and inner group table so selected row is visible."""
        try:
            if panel is not None:
                sc = getattr(self, 'offline_group_scroll', None)
                if sc is not None:
                    try:
                        sc.ensureWidgetVisible(panel, 12, 0)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if table is None or row is None:
                return
            rr = int(row)
            if rr < 0 or rr >= table.rowCount():
                return
            it = table.item(rr, 0)
            if it is not None:
                try:
                    table.scrollToItem(it, QAbstractItemView.PositionAtCenter)
                except Exception:
                    try:
                        table.scrollToItem(it)
                    except Exception:
                        pass
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
                        self._refresh_transposed_views(refresh_center_view=False)
                    except Exception:
                        pass
                except Exception:
                    pass

            sw = _Toggle(checked=not excluded)
            sw._cb = _apply_ref
            return sw
        except Exception:
            return None

    def _toggle_ref_row_excluded_by_view_row(self, view_row):
        """Toggle Excl for a ref transposed data row (0-based, header excluded)."""
        try:
            ri = int(view_row)
        except Exception:
            return
        try:
            if ri < 0:
                return
            if ri >= int(len(getattr(self, 'ref_points', []) or [])):
                return
        except Exception:
            return

        try:
            s = set(getattr(self, 'excluded_ref_indices', set()) or set())
            if int(ri) in s:
                s.discard(int(ri))
            else:
                s.add(int(ri))
            self.excluded_ref_indices = s
        except Exception:
            return

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
                excluded_ref_indices=self.excluded_ref_indices,
            )
        except Exception:
            pass
        try:
            self._refresh_transposed_views(refresh_center_view=False)
        except Exception:
            pass
        try:
            # Fiducial exclusion changes the stage transform → recompute target XYZ immediately.
            info = getattr(self, '_last_stage_info', None)
            if info is not None and self._sync_center_xyz_from_stage_info(info):
                self._refresh_transposed_views(
                    update_ref_view=False,
                    refresh_offline_lists=False,
                    refresh_center_view=True,
                )
        except Exception:
            pass
        try:
            self._apply_proc_zoom()
        except Exception:
            pass

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
                        self._refresh_transposed_views(refresh_center_view=False)
                    except Exception:
                        pass
                    try:
                        # Show/Hide is a visibility update only.
                        self.schedule_update(force=True, recompute_centroids=False)
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
                    if sb is not None:
                        # table_between is configured with ScrollBarAlwaysOn;
                        # reserve scrollbar width even before first paint to avoid clipping.
                        sb_w = sb.sizeHint().width()
                        if sb_w <= 0:
                            sb_w = sb.width()
                        if sb_w <= 0:
                            sb_w = 17
                        sb_w = max(16, int(sb_w))
                except Exception:
                    sb_w = 17
                margin = 4  # frame / border padding (+extra for scrollbar clipping safety)
                new_w = col_total + vh_w + sb_w + margin
                new_w = max(64, new_w)

                # Keep enough width for the 4-column offline control grid at startup.
                try:
                    def _w(widget):
                        if widget is None:
                            return 0
                        try:
                            ww = int(widget.width() or 0)
                        except Exception:
                            ww = 0
                        if ww <= 0:
                            try:
                                ww = int(widget.sizeHint().width() or 0)
                            except Exception:
                                ww = 0
                        return max(0, int(ww))

                    # Row-1 actual composition:
                    # AddTarget | (left margin 24 + Name prefix) | ('-' + seq + C/R, inner spacing 4) | Undo
                    w_add = _w(getattr(self, 'btn_add_target', None))
                    w_prefix = _w(getattr(self, 'edit_add_target_name_prefix', None))
                    w_pos = _w(getattr(self, 'combo_add_target_pos', None))
                    w_sep = _w(getattr(self, 'lbl_add_target_name_sep', None))
                    w_seq = _w(getattr(self, 'edit_add_target_name_seq', None))
                    w_undo = _w(getattr(self, 'btn_center_undo', None))
                    grid_gap = 4
                    first_row_min = w_add + grid_gap + (24 + w_prefix) + grid_gap + (w_sep + 4 + w_seq + 4 + w_pos) + grid_gap + w_undo

                    # Row-2: Name Filter / Update u,v / Clear Selected / Clear ALL
                    w_name_filter = _w(getattr(self, 'btn_center_name_filter', None))
                    w_update_t = _w(getattr(self, 'btn_update_target_uv', None))
                    w_clear_t = _w(getattr(self, 'btn_clear_target', None))
                    w_clear_all_t = _w(getattr(self, 'btn_clear_target_all', None))
                    second_row_min = w_name_filter + grid_gap + w_update_t + grid_gap + w_clear_t + grid_gap + w_clear_all_t

                    grid_min_w = max(int(first_row_min), int(second_row_min))
                    new_w = max(int(new_w), int(grid_min_w), 430)
                except Exception:
                    new_w = max(int(new_w), 430)

                # In centroid-extraction offline mode, keep enough width for relocated
                # extraction-parameter controls hosted above the center table.
                try:
                    stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
                except Exception:
                    stage_n = 'offline'
                try:
                    extraction_on = bool(getattr(self, 'centroid_extraction_mode', False))
                except Exception:
                    extraction_on = False
                if (stage_n == 'offline') and extraction_on:
                    try:
                        host = getattr(self, 'middle_extract_panel', None)
                        if host is not None and host.isVisible():
                            host_w = int(host.sizeHint().width() or 0)
                            if host_w <= 0:
                                host_w = int(host.minimumSizeHint().width() or 0)
                            if host_w > 0:
                                new_w = max(int(new_w), int(host_w) + 8)
                    except Exception:
                        pass
                    try:
                        new_w = max(int(new_w), int(getattr(self, 'left_column_min_width', LEFT_COLUMN_MIN_WIDTH) or LEFT_COLUMN_MIN_WIDTH))
                    except Exception:
                        new_w = max(int(new_w), LEFT_COLUMN_MIN_WIDTH)
        except Exception:
            new_w = 350  # ultimate fallback

        try:
            tbl_target_w = int(new_w)
            tbl.setFixedWidth(tbl_target_w)
        except Exception:
            try:
                tbl_target_w = int(new_w)
                tbl.setMinimumWidth(tbl_target_w)
            except Exception:
                pass
                tbl_target_w = int(new_w)

        col = getattr(self, 'center_container', None)
        if col is not None:
            extra_layout_w = 0
            try:
                lay = col.layout()
                if lay is not None:
                    m = lay.contentsMargins()
                    extra_layout_w = int(m.left()) + int(m.right()) + int(max(0, lay.spacing()))
            except Exception:
                extra_layout_w = 0
            container_target_w = int(tbl_target_w + extra_layout_w)
            try:
                col.setFixedWidth(container_target_w)
            except Exception:
                try:
                    col.setMinimumWidth(container_target_w)
                except Exception:
                    pass

            # Left/Center スワップ時は、左ロゴ列の幅を center table 側に合わせる。
            if bool(getattr(self, 'swap_left_center_columns', False)):
                try:
                    lc = getattr(self, 'left_container', None)
                    if lc is not None:
                        lc.setFixedWidth(container_target_w)
                except Exception:
                    pass
                try:
                    img = getattr(self, 'left_top_image', None)
                    if img is not None:
                        img.setFixedWidth(container_target_w)
                except Exception:
                    pass

    def _sync_table_selection(self):
        """Sync selected_index to visible transposed table selection and canonical table selection."""
        try:
            idx = getattr(self, 'selected_index', None)
            if idx is None:
                return
            skip_middle_sync = False
            try:
                fw = QApplication.focusWidget()
                tb = getattr(self, 'table_between', None)
                if tb is not None and fw is not None:
                    skip_middle_sync = bool(fw is tb) or bool(tb.isAncestorOf(fw))
            except Exception:
                skip_middle_sync = False
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
                row_keys = list(getattr(self, '_table_between_row_keys', []) or [])
                try:
                    spos = str(getattr(self, 'selected_point_pos', 'c') or 'c').lower().strip()
                except Exception:
                    spos = 'c'
                if spos not in ('c', 'r'):
                    spos = 'c'
                view_r = None
                try:
                    for r_i, rk in enumerate(row_keys):
                        try:
                            si = int(rk[0])
                            pk = str(rk[1] or 'c').lower().strip()
                        except Exception:
                            continue
                        if si == int(idx) and pk == spos:
                            view_r = int(r_i) + header_rows
                            break
                except Exception:
                    view_r = None
                try:
                    if view_r is None and int(idx) in idxs:
                        view_r = int(idxs.index(int(idx))) + header_rows
                except Exception:
                    view_r = None

                # Fallback: rebuild mapping from current center rows when cached mapping is stale.
                if view_r is None:
                    try:
                        rows = list(getattr(self, 'center_numeric_rows', []) or [])
                        for r_i, rr in enumerate(rows):
                            try:
                                rd = dict(rr or {})
                            except Exception:
                                rd = {}
                            try:
                                if int(rd.get('source_idx', -1)) == int(idx):
                                    view_r = int(r_i) + header_rows
                                    break
                            except Exception:
                                continue
                    except Exception:
                        view_r = None

                # Final fallback: treat selected_index as transposed row-local index.
                if view_r is None:
                    try:
                        rows = list(getattr(self, 'center_numeric_rows', []) or [])
                        if 0 <= int(idx) < len(rows):
                            view_r = int(idx) + header_rows
                    except Exception:
                        view_r = None

                if (not bool(skip_middle_sync)) and hasattr(self, 'table_between') and 0 <= view_r < self.table_between.rowCount():
                    try:
                        self.table_between.blockSignals(True)
                        rows_multi = []
                        try:
                            keys = set(getattr(self, 'selected_point_keys', set()) or set())
                        except Exception:
                            keys = set()
                        if keys and row_keys:
                            for r_i, rk in enumerate(row_keys):
                                try:
                                    si = int(rk[0])
                                    pk = str(rk[1] or 'c').lower().strip()
                                except Exception:
                                    continue
                                if pk not in ('c', 'r'):
                                    pk = 'c'
                                if (si, pk) in keys:
                                    rows_multi.append(int(r_i) + header_rows)
                        # choose column 0 for current cell; selection behavior is rows
                        self.table_between.setCurrentCell(view_r, 0)
                        self.table_between.clearSelection()
                        if rows_multi:
                            for rr in rows_multi:
                                if 0 <= int(rr) < self.table_between.rowCount():
                                    self.table_between.selectRow(int(rr))
                        else:
                            self.table_between.selectRow(view_r)
                    finally:
                        try:
                            self.table_between.blockSignals(False)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self._sync_offline_group_selection()
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
            cnt = int(tbl.columnCount() or 0)
            content_w = 0
            for i in range(cnt):
                try:
                    content_w += int(tbl.columnWidth(i))
                except Exception:
                    try:
                        content_w += int(tbl.horizontalHeader().defaultSectionSize() or 16)
                    except Exception:
                        content_w += 16
            try:
                vh = int(tbl.verticalHeader().width() or 0)
            except Exception:
                vh = 0
            try:
                sb = tbl.verticalScrollBar()
                sb_w = int(sb.width() or 17) if sb is not None else 17
            except Exception:
                sb_w = 17
            w = max(0, int(content_w + vh + sb_w + 4))
            try:
                img.setFixedWidth(int(w))
            except Exception:
                pass
            try:
                img.setMaximumWidth(int(w))
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
        try:
            self._wire_click_feedback_once()
        except Exception:
            pass

        red = "rgb(160,15,15)"    # Add Target / extraction warning color
        green = "rgb(24,96,80)"   # Logo-derived green from PiXY_XY.png
        green_hover = "rgb(40,112,96)"
        green_pressed = "rgb(8,80,64)"
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
        
        base_w = max(120, prev_base_w)

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
            # Add Ref. Point: logo green, wide width (1.5x)
            add_btn = getattr(self, 'btn_add_ref', None)
            add_target_btn = getattr(self, 'btn_add_target', None)
            if add_btn is not None:
                try:
                    style_add = (
                        f"QPushButton {{ background-color: {green}; color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: {green_hover}; }}"
                        f"QPushButton:pressed {{ background-color: {green_pressed}; }}"
                    )
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
                    add_target_btn.setFixedWidth(int(base_w) + 40)
                except Exception:
                    pass

            # Update XY + Clear: blue, same width as Export/Clipboard
            upd_btn = getattr(self, 'btn_update_xy', None)
            clr_btn = getattr(self, 'btn_clear_ref', None)
            upd_target_btn = getattr(self, 'btn_update_target_uv', None)
            clr_target_btn = getattr(self, 'btn_clear_target', None)
            clr_target_all_btn = getattr(self, 'btn_clear_target_all', None)
            select_all_btn = getattr(self, 'btn_select_all', None)
            undo_btn = getattr(self, 'btn_center_undo', None)
            name_filter_btn = getattr(self, 'btn_center_name_filter', None)
            nav_next_btn = getattr(self, 'btn_center_uv_next', None)
            nav_back_btn = getattr(self, 'btn_center_uv_back', None)
            nav_clear_btn = getattr(self, 'btn_center_uv_clear', None)
            nav_finish_btn = getattr(self, 'btn_center_uv_finish', None)
            for btn in (upd_btn, clr_btn, upd_target_btn, clr_target_btn, clr_target_all_btn, select_all_btn, undo_btn, name_filter_btn):
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

            # Update u,v navigation overlay buttons
            for btn in (nav_next_btn, nav_back_btn, nav_clear_btn):
                if btn is not None:
                    try:
                        style_blue = f"QPushButton {{ background-color: {blue}; color: white; border: none; border-radius: {radius}px; }}"
                        btn.setStyleSheet(style_blue)
                    except Exception:
                        pass
            if nav_finish_btn is not None:
                try:
                    # Requested: Finish Update should be light red in Update u,v mode.
                    nav_finish_btn.setStyleSheet(
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: rgb(220,110,110); }}"
                        f"QPushButton:pressed {{ background-color: rgb(210,100,100); }}"
                    )
                except Exception:
                    pass

            # Export + Clipboard + Open Image (+ Flip): base style (blue) and widths
            exp_btn = getattr(self, 'btn_export', None)
            clip_btn = getattr(self, 'btn_clipboard', None)
            online_exp_btn = getattr(self, 'btn_online_export', None)
            online_clip_btn = getattr(self, 'btn_online_clipboard', None)
            filter_btn = getattr(self, 'btn_filter', None)
            add_all_grp_btn = getattr(self, 'btn_add_all_grp_list', None)
            open_btn = getattr(self, 'btn_open', None)
            replace_img_btn = getattr(self, 'btn_replace_image', None)
            flip_btn = getattr(self, 'btn_flip_mode', None)
            combo_flip = getattr(self, 'combo_flip_mode', None)
            new_btn = getattr(self, 'btn_new_project', None)
            save_btn = getattr(self, 'btn_save_project', None)
            load_btn = getattr(self, 'btn_load_project', None)
            left_settings_btn = getattr(self, 'btn_left_settings', None)
            start_ce_btn = getattr(self, 'btn_start_centroid_extraction', None)
            core_btn = getattr(self, 'btn_center_add_core', None)
            rim_btn = getattr(self, 'btn_center_add_rim', None)
            rim_off_minus_btn = getattr(self, 'btn_center_rim_offset_minus', None)
            rim_off_plus_btn = getattr(self, 'btn_center_rim_offset_plus', None)
            for btn in (clip_btn, online_clip_btn, filter_btn, add_all_grp_btn, open_btn, replace_img_btn, flip_btn, save_btn, load_btn, left_settings_btn, core_btn, rim_btn):
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
                        if btn in (core_btn, rim_btn):
                            w = 64
                        btn.setFixedWidth(w)
                    except Exception:
                        pass
                    try:
                        if btn in (core_btn, rim_btn, add_all_grp_btn):
                            btn.setFixedHeight(27)
                    except Exception:
                        pass

            for btn in (rim_off_minus_btn, rim_off_plus_btn):
                if btn is not None:
                    try:
                        # Match Boundary Offset stepper buttons (- / +) appearance.
                        btn.setStyleSheet("padding:0px; margin:0px;")
                    except Exception:
                        pass
                    try:
                        btn.setFixedSize(28, 23)
                    except Exception:
                        pass

            try:
                stage_n = str(getattr(self, 'workflow_stage', 'offline') or 'offline').lower().strip()
            except Exception:
                stage_n = 'offline'
            accent_is_green = (stage_n == 'online')
            accent_bg = green if accent_is_green else red
            accent_hover_bg = green_hover if accent_is_green else 'rgb(220,110,110)'
            accent_pressed_bg = green_pressed if accent_is_green else 'rgb(210,100,100)'

            # Open Image / New Project / Export: keep one accent palette that flips with stage.
            if open_btn is not None:
                try:
                    open_btn.setStyleSheet(
                        f"QPushButton {{ background-color: {accent_bg}; color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: {accent_hover_bg}; }}"
                        f"QPushButton:pressed {{ background-color: {accent_pressed_bg}; }}"
                    )
                except Exception:
                    pass

            if exp_btn is not None:
                try:
                    style_green = (
                        f"QPushButton {{ background-color: {green}; color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: {green_hover}; }}"
                        f"QPushButton:pressed {{ background-color: {green_pressed}; }}"
                    )
                    exp_btn.setStyleSheet(style_green)
                except Exception:
                    pass
                try:
                    exp_btn.setFixedWidth(int(base_w) + 10)
                except Exception:
                    pass

            if online_exp_btn is not None:
                try:
                    style_green = (
                        f"QPushButton {{ background-color: {green}; color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: {green_hover}; }}"
                        f"QPushButton:pressed {{ background-color: {green_pressed}; }}"
                    )
                    online_exp_btn.setStyleSheet(style_green)
                except Exception:
                    pass
                try:
                    online_exp_btn.setFixedWidth(int(base_w) + 10)
                except Exception:
                    pass

            if new_btn is not None:
                try:
                    new_btn.setStyleSheet(
                        f"QPushButton {{ background-color: {accent_bg}; color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:hover {{ background-color: {accent_hover_bg}; }}"
                        f"QPushButton:pressed {{ background-color: {accent_pressed_bg}; }}"
                    )
                except Exception:
                    pass
                try:
                    new_btn.setFixedWidth(int(base_w) + 10)
                except Exception:
                    pass

            if start_ce_btn is not None:
                try:
                    self._ensure_start_ce_button_width()
                except Exception:
                    pass
                try:
                    # Keep mode-aware color/text in sync after base style pass.
                    self._update_centroid_extraction_button()
                except Exception:
                    pass

            if core_btn is not None:
                try:
                    # Core: ON(checked)=red, OFF(unchecked)=light red
                    core_btn.setStyleSheet(
                        f"QPushButton {{ background-color: rgb(225,120,120); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:checked {{ background-color: {red}; color: white; border: none; border-radius: {radius}px; }}"
                    )
                except Exception:
                    pass

            if rim_btn is not None:
                try:
                    # Rim: ON(checked)=dark blue, OFF(unchecked)=light blue
                    rim_btn.setStyleSheet(
                        f"QPushButton {{ background-color: rgb(120,175,230); color: white; border: none; border-radius: {radius}px; }}"
                        f"QPushButton:checked {{ background-color: rgb(0,100,200); color: white; border: none; border-radius: {radius}px; }}"
                    )
                except Exception:
                    pass

            # Keep top image-header action buttons aligned to identical size.
            try:
                top_btn_w = int(base_w) + 10
            except Exception:
                top_btn_w = 120
            for btn in (new_btn, save_btn, load_btn, open_btn, replace_img_btn, left_settings_btn):
                if btn is None:
                    continue
                try:
                    btn.setFixedWidth(int(top_btn_w))
                except Exception:
                    pass
                try:
                    btn.setFixedHeight(40)
                except Exception:
                    pass

            # Store the unscaled base width for future calls.
            # NOTE: Do not add padding here; _apply_button_styles may run many times
            # (e.g. after Update/Add pick-mode), and adding here would grow widths
            # cumulatively on every call.
            try:
                self._action_btn_base_w = int(raw_base_w)
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
                    # Keep small stepper buttons (-/+) compact like slider controls.
                    if (b.text() or "") in ("-", "+"):
                        try:
                            if int(b.width() or 0) <= 32:
                                continue
                        except Exception:
                            pass
                    # Skip buttons that live inside a SegmentControl
                    parent = b.parent()
                    if isinstance(parent, SegmentControl):
                        continue
                    # Keep Core/Rim/Add ALL Group to List at the same height as Add GroupN (27px).
                    try:
                        if b in (
                            getattr(self, 'btn_center_add_core', None),
                            getattr(self, 'btn_center_add_rim', None),
                            getattr(self, 'btn_add_all_grp_list', None),
                        ):
                            b.setFixedHeight(27)
                            continue
                    except Exception:
                        pass
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
                    font = item.font(); font.setBold(True); item.setFont(font)
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

            # Row 0: Group labels for middle transposed table.
            # Current layout (9 cols): ID, Name, u, v, Grp, No., C/R, Gen., Show
            try:
                ncols = int(tbl.columnCount() or 0)
            except Exception:
                ncols = 0

            if ncols >= 9:
                name_label = 'Name'
                group_configs = [
                    (0, 1, ""),
                    (1, 1, ""),
                    (2, 2, "Image"),
                    (4, 4, "Centroid Extraction"),
                    (8, 1, ""),
                ]
                sub_labels = [
                    self._center_label_with_sort('no', 'ID'),
                    name_label,
                    self._center_label_with_sort('u', 'u'),
                    self._center_label_with_sort('v', 'v'),
                    self._center_label_with_sort('grp', 'Grp'),
                    self._center_label_with_sort('pno', 'No.'),
                    self._center_label_with_sort('cr', 'C/R'),
                    self._center_label_with_sort('gen', 'Gen.'),
                    "",
                ]
            elif ncols == 8:
                name_label = 'Name'
                group_configs = [
                    (0, 1, ""),
                    (1, 1, ""),
                    (2, 2, "Image"),
                    (4, 3, "Stage"),
                    (7, 1, ""),
                ]
                sub_labels = [
                    self._center_label_with_sort('no', 'ID'),
                    name_label,
                    self._center_label_with_sort('u', 'u'),
                    self._center_label_with_sort('v', 'v'),
                    self._center_label_with_sort('x', 'X'),
                    self._center_label_with_sort('y', 'Y'),
                    self._center_label_with_sort('z', 'Z'),
                    "",
                ]
            elif ncols >= 7:
                group_configs = [
                    (0, 1, "No."),
                    (1, 2, "Image"),
                    (3, 3, "Stage"),
                    (6, 1, ""),
                ]
                sub_labels = [
                    self._center_label_with_sort('no', 'ID'),
                    "Name",
                    self._center_label_with_sort('u', 'u'),
                    self._center_label_with_sort('v', 'v'),
                    self._center_label_with_sort('grp', 'Grp'),
                    self._center_label_with_sort('pno', 'No.'),
                    self._center_label_with_sort('cr', 'C/R'),
                    self._center_label_with_sort('gen', 'Gen.'),
                    "",
                ]
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