"""
Centroid Finder のメイン UI ウィンドウ実装。

このモジュールは PyQt5 を使用した GUI ウィンドウを定義します。
主な機能:
- 画像の読み込みと表示
- 重心検出パラメータの調整 (PosterLevel, Min Area, Trim)
- 参照点の設定とフィッティング
- テーブル表示と編集
- 自動デバッグモード対応

依存関係:
- tables.py: テーブル操作
- interactions.py: マウス/キーボード操作
- rendering.py: 画像描画
- CalcCentroid.py: 重心計算
- Util.py: ユーティリティ
- Strings.py: UI 文字列定数
"""

import qt_compat
from qt_compat.QtWidgets import QSlider, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget, QFileDialog, QStyle, QSizePolicy, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QScrollArea, QApplication
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


class PatchWorker(QThread):
    """Background worker that extracts a patch from a full image and resizes it.

    Emits (patch_array, left_label, top_label, request_id) where patch_array is a numpy BGR image or None on error.
    """
    finished = pyqtSignal(object, int, int, int)

    def __init__(self, full_img, lf, tf, wf, hf, tgt_w, tgt_h, interp, left_label, top_label, request_id):
        super().__init__()
        self.full_img = full_img
        self.lf = lf
        self.tf = tf
        self.wf = wf
        self.hf = hf
        self.tgt_w = tgt_w
        self.tgt_h = tgt_h
        self.interp = interp
        self.left_label = left_label
        self.top_label = top_label
        self.request_id = request_id
        self._cancel = False

    def cancel(self):
        try:
            self._cancel = True
        except Exception:
            pass

    def run(self):
        try:
            tf = int(self.tf); lf = int(self.lf); wf = int(self.wf); hf = int(self.hf)
            if getattr(self, '_cancel', False):
                self.finished.emit(None, self.left_label, self.top_label, self.request_id)
                return
            patch = self.full_img[tf:tf+hf, lf:lf+wf]
            if getattr(self, '_cancel', False):
                self.finished.emit(None, self.left_label, self.top_label, self.request_id)
                return
            if patch is None or patch.size == 0:
                self.finished.emit(None, self.left_label, self.top_label, self.request_id)
                return
            patch_resized = cv2.resize(patch, (max(1, int(self.tgt_w)), max(1, int(self.tgt_h))), interpolation=self.interp)
            if getattr(self, '_cancel', False):
                self.finished.emit(None, self.left_label, self.top_label, self.request_id)
                return
            self.finished.emit(patch_resized, self.left_label, self.top_label, self.request_id)
        except Exception:
            try:
                self.finished.emit(None, self.left_label, self.top_label, self.request_id)
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
                    print(f"[DEBUG] {msg}")
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
        self.select_radius_display = 10.0  # 画像上の選択半径 (px)

        # 参照点関連
        self.ref_points = [None] * 10  # 参照点リスト [(x_proc, y_proc) or None]
        self.ref_selected_index = 0     # 選択中の参照点インデックス
        self.ref_obs = [{"x": "", "y": "", "z": ""} for _ in range(10)]  # 参照点の観測値

        # UI 状態
        self.visible_ref_cols = 3      # 表示する参照点列数
        self.flip_mode = 'auto'        # 左右反転モード ('auto', 'normal', 'flip')
        self.overlay_mix = 50          # オーバーレイ混合率 (%)
        # Display labels: editable display strings separate from internal keys
        # Internal keys should be code-safe identifiers; change display text here.
        self.display_labels = {
            'overlay_ratio': 'Posterization overlay',
            'poster_level': 'Number of colors',
            'min_area': 'Min grain Area (px)',
            'trim': 'Boundary offset (px)'
        }
        self.levels_value = 4          # PosterLevel の内部値
        self.show_boundaries = True    # 境界線表示フラグ

        # 画像表示関連
        # 仮想キャンバス関連: 実際の表示はビューポート分のみだが、スクロール範囲は仮想的に拡張する
        self._virtual_canvas_size = (0, 0)  # 仮想キャンバス幅,高さ (px)
        # パッチ生成の安全弁 (パッチのピクセル数上限)
        self.MAX_PATCH_PIXELS = 4096 * 4096  # 大きなパッチ作成を防ぐ

        self._img_base_size = None     # ベース画像サイズ (w, h)
        self.proc_zoom = 1.0           # 処理画像のズーム倍率
        self.view_padding = 200        # 表示パディング
        self._display_offset = (0, 0)  # 表示オフセット
        self._display_img_size = (0, 0) # 表示画像サイズ
        self._display_pm_base = None   # ベース Pixmap
        self._initial_center_done = False  # 初期センタリング完了フラグ

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
            "trim_px": None,     # Trim (px)
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

        # 残すのは PosterLevel と Min Area に加え、Trim(px)
        # Use code-safe internal keys for widgets; display text comes from self.display_labels
        self.edit_levels, self.slider_levels = self._make_spin_slider('poster_level', 4, 2, 20, 1)
        self.edit_min_area, self.slider_min_area = self._make_spin_slider('min_area', 50, 10, 5000, 1)
        self.edit_trim, self.slider_trim = self._make_spin_slider('trim', 0, 0, 10, 1)

        # PosterLevelの内部値（スライダー上限20を超えても保持）
        self.levels_value = self.slider_levels.value()

        # ボタン（画像開く / エクスポート）を作る（配置は後で画像ヘッダ等へ移動する）
        open_btn = QPushButton(STR.BUTTON_OPEN_IMAGE)
        open_btn.setFixedWidth(100)
        open_btn.clicked.connect(self.open_image)
        # Export ボタンは短くして隣に Clipboard を追加
        export_btn = QPushButton("Export")
        export_btn.setFixedWidth(100)
        export_btn.clicked.connect(self.export_centroids)
        clipboard_btn = QPushButton("Clipboard")
        clipboard_btn.setFixedWidth(100)
        clipboard_btn.clicked.connect(self._copy_centroids_to_clipboard)

        # 補間モード（表示時の補間法）と自動更新/手動再計算の UI 部品を先に作成
        from qt_compat.QtWidgets import QComboBox, QCheckBox
        self.interp_mode = 'auto'  # 'auto' | 'nearest' | 'linear'
        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(['Auto', 'Nearest', 'Smooth'])
        self.cmb_interp.setCurrentIndex(0)
        def _on_interp_changed(idx):
            try:
                # nested ternary: 'auto' if idx==0, 'nearest' if idx==1, otherwise 'linear'
                self.interp_mode = 'auto' if idx == 0 else ('nearest' if idx == 1 else 'linear')
                self.schedule_update(force=True)
            except Exception:
                pass
        self.cmb_interp.currentIndexChanged.connect(_on_interp_changed)

        self.auto_update_mode = True
        self.chk_auto_update = QCheckBox("Auto Update")
        self.chk_auto_update.setChecked(True)
        self.chk_auto_update.stateChanged.connect(lambda s: self._on_toggle_auto_update(bool(s)))
        self.btn_recalc = QPushButton("Recalculate Centroids")
        self.btn_recalc.clicked.connect(self._on_manual_recalc)

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
        self.show_boundaries = True
        self.btn_toggle_edges = QPushButton(STR.BUTTON_TOGGLE_BOUNDARIES)
        self.btn_toggle_edges.setCheckable(True)
        self.btn_toggle_edges.setChecked(True)
        self.btn_toggle_edges.toggled.connect(self._on_toggle_boundaries)

        # 左右反転モード切替（Auto / Normal / Flip）
        self.flip_mode = 'auto'  # 'auto' | 'normal' | 'flip'
        self.btn_flip_mode = QPushButton(f"{STR.FLIP_PREFIX}{STR.FLIP_LABELS.get('auto','Auto')}")
        self.btn_flip_mode.setCheckable(False)
        try:
            self.btn_flip_mode.setToolTip(STR.FLIP_TOOLTIP)
            self.btn_flip_mode.setAccessibleName("flip-mode-toggle")
        except Exception:
            pass
        self.btn_flip_mode.clicked.connect(self._on_cycle_flip_mode)

        # 画像領域レイアウト：上にボタン群（画像左上に Open、中央上に Export/自動系、右上に Flip/境界）
        img_layout = QVBoxLayout()
        img_header = QHBoxLayout()
        # 左上に Open ボタン
        try:
            img_header.addWidget(open_btn, 0, Qt.AlignLeft)
        except Exception:
            pass
        # 中央上には補間/自動更新/手動再計算をまとめる（Export は table_between の上へ移動）
        try:
            center_controls = QHBoxLayout()
            # 補間/自動更新/手動再計算は中央上へ移動
            try:
                center_controls.addWidget(self.cmb_interp)
            except Exception:
                pass
            try:
                center_controls.addWidget(self.chk_auto_update)
            except Exception:
                pass
            try:
                center_controls.addWidget(self.btn_recalc)
            except Exception:
                pass
            img_header.addLayout(center_controls)
        except Exception:
            pass
        img_header.addStretch(1)
        # 右上に Overlay スライダー（Flipの左）と Flip、境界トグルを配置
        try:
            # small overlay control placed at right-top next to Flip
            overlay_ctrl = QWidget()
            try:
                ol_layout = QHBoxLayout(overlay_ctrl)
                ol_layout.setContentsMargins(0, 0, 0, 0)
                ol_layout.setSpacing(4)
                lbl_ol = QLabel(self.display_labels.get('overlay_ratio', STR.NAME_OVERLAY_RATIO))
                try:
                    lbl_ol.setFont(QFont('Segoe UI', 11))
                except Exception:
                    pass
                ol_layout.addWidget(lbl_ol)
                self.slider_overlay = QSlider(Qt.Horizontal)
                self.slider_overlay.setMinimum(0)
                self.slider_overlay.setMaximum(100)
                self.slider_overlay.setSingleStep(1)
                self.slider_overlay.setValue(self.overlay_mix)
                try:
                    self.slider_overlay.setFixedWidth(140)
                except Exception:
                    pass
                self.slider_overlay.valueChanged.connect(self._on_overlay_ratio_changed)
                ol_layout.addWidget(self.slider_overlay)
            except Exception:
                pass
            try:
                img_header.addWidget(overlay_ctrl, 0, Qt.AlignRight)
            except Exception:
                img_header.addWidget(overlay_ctrl)
        except Exception:
            pass
        # 右上に Flip と境界トグルを配置
        try:
            img_header.addWidget(self.btn_flip_mode, 0, Qt.AlignRight)
        except Exception:
            pass
        try:
            img_header.addWidget(self.btn_toggle_edges, 0, Qt.AlignRight)
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
            sliders_layout.setSpacing(18)
            sliders_layout.setContentsMargins(6, 6, 6, 6)
        except Exception:
            pass

        # NOTE: overlay slider moved to image header (right-top). See img_header insertion below.

        # Helper to build a single-row control with label, slider, and numeric box (+/-)
        def _build_control_row(name, edit_widget, slider_widget, nudger_minus, nudger_plus):
            try:
                row = QHBoxLayout()
                lbl = QLabel(name)
                try:
                    lbl.setFont(ctrl_font)
                except Exception:
                    pass
                try:
                    # 固定幅にして、すぐ隣に数値ボックスが来るようにする（ラベルと数値の間に可変スペースを入れない）
                    lbl.setFixedWidth(140)
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
                bhl.addSpacing(6)            # gap between minus and number
                bhl.addWidget(edit_widget)
                bhl.addSpacing(6)            # gap between number and plus (same as above)
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

        # PosterLevel row
        try:
            r = _build_control_row(self.display_labels.get('poster_level', STR.NAME_POSTERLEVEL), self.edit_levels, self.slider_levels, self._nudge_levels, self._nudge_levels)
            if r is not None:
                sliders_layout.addLayout(r)
        except Exception:
            pass

        # Min Area row
        try:
            r = _build_control_row(self.display_labels.get('min_area', STR.NAME_MIN_AREA), self.edit_min_area, self.slider_min_area, self._nudge_min_area, self._nudge_min_area)
            if r is not None:
                sliders_layout.addLayout(r)
        except Exception:
            pass

        # Trim row
        try:
            r = _build_control_row(self.display_labels.get('trim', STR.NAME_TRIM), self.edit_trim, self.slider_trim, self._nudge_trim, self._nudge_trim)
            if r is not None:
                sliders_layout.addLayout(r)
        except Exception:
            pass

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
        # Flipボタンを Clear の右隣に配置
        actions_row.addWidget(self.btn_flip_mode)
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
                    target_w = min(500, pix.width())
                    self._left_top_pix = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
                    self.left_top_image.setPixmap(self._left_top_pix)
                    try:
                        self.left_top_image.setFixedWidth(500)
                    except Exception:
                        pass
                except Exception:
                    self._left_top_pix = pix
                    self.left_top_image.setPixmap(pix)
            else:
                self._left_top_pix = None
                self.left_top_image.setText("Px2XY")
                try:
                    self.left_top_image.setFixedWidth(500)
                except Exception:
                    pass
        except Exception:
            self.left_top_image = QLabel("Px2XY")
            self._left_top_pix = None
            try:
                self.left_top_image.setFixedWidth(500)
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
            self.table_between.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.table_between.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
        left_col.addLayout(sliders_layout, 0)
        # 左カラムの表の上に Add/Update/Clear ボタンを配置
        try:
            left_controls = QHBoxLayout()
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
            left_col.addLayout(left_controls, 0)
        except Exception:
            pass
        left_col.addWidget(self.table_ref_view, 1)
        # Wrap left column layout in a QWidget and cap its maximum width so it doesn't grow too wide
        left_container = QWidget()
        left_container.setLayout(left_col)
        try:
            # 固定幅にして左カラムを確実に400pxにする
            left_container.setFixedWidth(500)
        except Exception:
            try:
                left_container.setMaximumWidth(500)
            except Exception:
                pass
        main_row.addWidget(left_container, 0)
        # Center area: place the transposed bottom table between left and image
        # Create a center column layout so we can place Export button above the table_between
        try:
            center_col = QVBoxLayout()
            try:
                center_col.addWidget(export_btn, 0, Qt.AlignTop)
            except Exception:
                pass
            try:
                center_col.addWidget(clipboard_btn, 0, Qt.AlignTop)
            except Exception:
                pass
            center_col.addWidget(self.table_between, 1)
            # Wrap the center column in a QWidget so we can control the column width
            self.center_container = QWidget()
            self.center_container.setLayout(center_col)
            try:
                self.center_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
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

        self.open_image()

    # オーバーレイ比率スライダー変更ハンドラ
    def _on_overlay_ratio_changed(self, v):
        self.overlay_mix = int(v)
        self.schedule_update(force=True)

    # 境界線表示トグルハンドラ
    def _on_toggle_boundaries(self, checked):
        self.show_boundaries = bool(checked)
        self.schedule_update(force=True)

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
        return dict(
            levels=self.levels_value,
            min_area=self.slider_min_area.value(),
            trim_px=self.slider_trim.value(),
        )

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
                cache_trim = self._cache.get("trim_px")
                cache_poster = self._cache.get("poster")
                cache_centroids = self._cache.get("centroids")

                need_poster_recalc = (
                    cache_poster is None
                    or cache_levels != params["levels"]
                    or cache_img_id != id(self.proc_img)
                )

                # 自動モードでは通常通り重い処理を行う
                if self.auto_update_mode:
                    if need_poster_recalc:
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
                    else:
                        # reuse cached poster
                        poster = cache_poster
                        # If only min_area/trim changed, recompute centroids from cached poster
                        if (cache_poster is not None) and (
                            cache_min_area != params.get("min_area") or cache_trim != params.get("trim_px")
                        ):
                            try:
                                centroids = self.centroid_processor.get_centroids(params, poster=poster)
                                # update cached params and centroids (keep poster and img_id/levels)
                                self._cache.update({
                                    "min_area": params["min_area"],
                                    "trim_px": params["trim_px"],
                                    "centroids": centroids,
                                })
                            except Exception:
                                centroids = cache_centroids
                        else:
                            centroids = cache_centroids
                else:
                    # 手動モード: 可能ならキャッシュを使い、重い poster 再生成は行わない
                    if cache_poster is not None and cache_img_id == id(self.proc_img):
                        poster = cache_poster
                        # Use centroid_processor to recompute centroids from cached poster with current params
                        try:
                            centroids = self.centroid_processor.get_centroids(params, poster=poster)
                        except Exception:
                            # fallback to cached centroids if recompute fails
                            if cache_centroids is not None:
                                centroids = cache_centroids
                    else:
                        # キャッシュが無ければフォールバックで軽めに計算（呼び出し元でエラーは吸収）
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
                # 表示用にポスター画像をフル解像度へ拡大
                scale = 1.0 / self.scale_proc_to_full
                if scale != 1.0:
                    new_w = self.img_full.shape[1]
                    new_h = self.img_full.shape[0]
                    poster_full = cv2.resize(poster, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    poster_full = poster.copy()
                # 乗算オーバーレイ: base=poster_full, overlay=img_full の multiply を比率で補間
                mix = max(0.0, min(1.0, (self.overlay_mix or 0) / 100.0))
                mult = cv2.multiply(poster_full, self.img_full, scale=1.0/255.0)
                overlay_full = cv2.addWeighted(poster_full, 1.0 - mix, mult, mix, 0)
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
                            # poster_full is already at full resolution
                            poster_fe = poster_full.copy()
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
                        # 隣接画素の色が異なる箇所を検出（横/縦）
                        diff_h = np.any(edge_src[:, 1:, :] != edge_src[:, :-1, :], axis=2)
                        diff_v = np.any(edge_src[1:, :, :] != edge_src[:-1, :, :], axis=2)
                        edge_mask = np.zeros((h, w), dtype=np.uint8)
                        edge_mask[:, 1:][diff_h] = 255
                        edge_mask[1:, :][diff_v] = 255
                        # 黒枠は不要 → スムージング（ガウシアン）で柔らかい白線へ
                        # 小さなぼかしで1px相当の境界を滑らかにする
                        blurred = cv2.GaussianBlur(edge_mask, (3, 3), 0.8)
                        alpha = (blurred.astype(np.float32) / 255.0).reshape(h, w, 1)
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
        # Build a pixmap for the current overlay (or proc_img fallback) and set it directly on the label.
        if self._last_overlay_full is None:
            # Fallback to proc_img if available
            if self.proc_img is not None:
                try:
                    pm, (off_x, off_y), (new_w, new_h) = build_zoomed_canvas(
                        self.proc_img,
                        self.proc_zoom,
                        self.view_padding,
                        self.centroids,
                        self.selected_index,
                        self.ref_points,
                        self.scale_proc_to_full,
                        colors=None,
                        interp_mode=self.interp_mode,
                    )
                    if pm is not None:
                        try:
                            self.img_label_proc.setPixmap(pm)
                            self.img_label_proc.resize(pm.width(), pm.height())
                        except Exception:
                            pass
                        # Compute display_scale based on actual drawn pixels to keep coordinate mapping correct
                        try:
                            pad = int(self.view_padding)
                            full_w = int(self._img_base_size[0]) if self._img_base_size is not None else max(1, new_w)
                            drawn_w = max(1, pm.width() - 2 * pad)
                            self._display_scale = float(drawn_w) / float(full_w)
                            # physical offset (in label coordinates) is pad
                            self._display_offset = (pad, pad)
                        except Exception:
                            self._display_offset = (off_x, off_y)
                            self._display_scale = float(self.proc_zoom)
                        self._display_img_size = (new_w, new_h)
                        self._display_pm_base = pm
                        return
                except Exception:
                    pass
            self.img_label_proc.clear()
            return

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
                self._display_scale = float(self.proc_zoom)
            self._display_offset = (off_x, off_y)
        if pm is None:
            self.img_label_proc.clear()
            return
        self._display_img_size = (new_w, new_h)
        self._display_pm_base = pm
        # update statusbar with interp info
        try:
            actual_interp = 'nearest' if self.interp_mode == 'nearest' else ('linear' if self.interp_mode == 'linear' else ('nearest' if (self.proc_zoom > 1.5) else 'linear'))
            msg = f"補間モード={self.interp_mode} (描画補間={actual_interp})"
            if getattr(self, '_large_file_hint', False):
                msg += " ｜ 軽負荷モード有効"
            self.ui_footer.showMessage(msg)
        except Exception:
            pass

        # Directly set pixmap and resize label to match pixmap size (no virtual canvas)
        try:
            self.img_label_proc.setPixmap(pm)
            try:
                self.img_label_proc.resize(pm.width(), pm.height())
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
            self.ref_selected_index = int(curRow)
        except Exception:
            pass

    def _on_add_ref_point(self):
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
            self.table_ref.setCurrentCell(0, target)
            self.table_ref.selectColumn(target)
        finally:
            self.table_ref.blockSignals(False)
        # ピックモード開始（Add）
        self._start_pick_mode('add', ref_index=target)

    def _on_update_xy(self):
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
        self.flip_mode = nxt
        try:
            label = 'Auto' if nxt == 'auto' else ('Normal' if nxt == 'normal' else 'Flip')
            self.btn_flip_mode.setText(f"Flip:{label}")
        except Exception:
            pass
        # 再描画・テーブル更新
        try:
            self._safe_populate_tables(self.table_ref, self.table, self.ref_points, self.ref_obs, self.centroids, self.selected_index, self.ref_selected_index, flip_mode=self.flip_mode, visible_ref_cols=self.visible_ref_cols)
            self._apply_proc_zoom()
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
        outname = f"{STR.EXPORT_FILENAME_PREFIX}{dt_str}.txt"
        try:
            with open(outname, "w", encoding="utf-8") as f:
                f.write(STR.EXPORT_HEADER)
                for g, x, y in centroids:
                    f.write(f"{g},{int(round(x))},{int(round(y))}\n")
            from qt_compat.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", f"重心座標を {outname} に保存しました。")
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
            idx = int(curRow)
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
            if row in (2, 3, 4):
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

    def _end_pick_mode(self):
        self.pick_mode = None
        self.pick_ref_index = None
        # 通常は手のカーソル
        self.img_label_proc.setCursor(QCursor(Qt.OpenHandCursor))
        # ルーペは存在しない

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

                        # canonical: table_ref (row 0/1, col idx)
                        t = getattr(self, 'table_ref', None)
                        if t is not None:
                            try:
                                if t.columnCount() <= int(idx):
                                    t.setColumnCount(int(idx) + 1)
                                    try:
                                        t.setHorizontalHeaderLabels([str(i + 1) for i in range(t.columnCount())])
                                    except Exception:
                                        pass
                                if t.rowCount() >= 2:
                                    itx = t.item(0, int(idx))
                                    if itx is None:
                                        itx = QTableWidgetItem("")
                                        t.setItem(0, int(idx), itx)
                                    ity = t.item(1, int(idx))
                                    if ity is None:
                                        ity = QTableWidgetItem("")
                                        t.setItem(1, int(idx), ity)
                                    itx.setText(xi)
                                    ity.setText(yi)
                                    try:
                                        itx.setFlags(itx.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                        ity.setFlags(ity.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                        # transposed: table_ref_view (row idx, col 0/1)
                        rv = getattr(self, 'table_ref_view', None)
                        if rv is not None:
                            try:
                                if rv.rowCount() <= int(idx):
                                    rv.setRowCount(int(idx) + 1)
                                    try:
                                        rv.setVerticalHeaderLabels([str(i + 1) for i in range(rv.rowCount())])
                                    except Exception:
                                        pass
                                # Ensure at least X/Y columns exist
                                if rv.columnCount() < 2:
                                    try:
                                        rv.setColumnCount(max(2, len(STR.TABLE_LEFT_ROW_LABELS)))
                                    except Exception:
                                        rv.setColumnCount(2)
                                    try:
                                        rv.setHorizontalHeaderLabels(list(STR.TABLE_LEFT_ROW_LABELS))
                                    except Exception:
                                        pass
                                if rv.columnCount() >= 2:
                                    vix = rv.item(int(idx), 0)
                                    if vix is None:
                                        vix = QTableWidgetItem("")
                                        rv.setItem(int(idx), 0, vix)
                                    viy = rv.item(int(idx), 1)
                                    if viy is None:
                                        viy = QTableWidgetItem("")
                                        rv.setItem(int(idx), 1, viy)
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
                    if self.pick_mode == 'add':
                        self._end_pick_mode()

    def _on_ref_item_changed(self, item):
        # 左テーブル（Ref）の Obs.* 行（2,3,4行目）入力を半角へ正規化し、内部配列に反映
        row = item.row()
        col = item.column()
        if row not in (2, 3, 4):
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
        key = 'x' if row == 2 else ('y' if row == 3 else 'z')
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
            # transposed: view[r,c] corresponds to source[c,r]
            src_r = c
            src_c = r
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
                        if src_r in (2, 3, 4) and 0 <= src_c < len(self.ref_obs):
                            key = 'x' if src_r == 2 else ('y' if src_r == 3 else 'z')
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
            # Copy each cell from view into the source table (transposed mapping)
            rv = self.table_ref_view
            if rv is None:
                return
            rows = rv.rowCount()
            cols = rv.columnCount()
            for r in range(rows):
                for c in range(cols):
                    try:
                        it = rv.item(r, c)
                        txt = it.text() if it is not None else ""
                        src_r = c
                        src_c = r
                        if 0 <= src_r < self.table_ref.rowCount() and 0 <= src_c < self.table_ref.columnCount():
                            src_item = self.table_ref.item(src_r, src_c)
                            if src_item is None:
                                src_item = QTableWidgetItem(txt)
                                self.table_ref.setItem(src_r, src_c, src_item)
                            else:
                                src_item.setText(txt)
                    except Exception:
                        pass
        finally:
            try:
                self.table_ref.blockSignals(False)
            except Exception:
                pass
        # Now update internal ref_obs from table_ref's Obs rows (2,3,4)
        try:
            for col in range(self.table_ref.columnCount()):
                if col >= len(self.ref_obs):
                    continue
                for row, key in ((2, 'x'), (3, 'y'), (4, 'z')):
                    try:
                        it = self.table_ref.item(row, col)
                        val = it.text() if it is not None else ""
                        self.ref_obs[col][key] = val
                    except Exception:
                        pass
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
            app = QApplication.instance()
            if app is None:
                return
            tbl = self.table
            if tbl is None or tbl.columnCount() == 0:
                return
            lines = []
            # header
            lines.append("No\tCalcX\tCalcY\tCalcZ")
            for c in range(tbl.columnCount()):
                try:
                    # Calc.* are at rows 2,3,4 per tables.populate_tables
                    itx = tbl.item(2, c)
                    ity = tbl.item(3, c)
                    itz = tbl.item(4, c)
                    sx = itx.text() if itx is not None else ""
                    sy = ity.text() if ity is not None else ""
                    sz = itz.text() if itz is not None else ""
                    lines.append(f"{c+1}\t{sx}\t{sy}\t{sz}")
                except Exception:
                    lines.append(f"{c+1}\t\t\t")
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
        except Exception:
            pass

    def _refresh_transposed_views(self):
        # Create/update transposed copies of `self.table_ref` and `self.table`.
        try:
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
                            # - RefX/RefY (typically row 0/1)
                            # - Residual rows
                            update_src_rows = (0, 1, 5, 6, 7, 8)
                            cur = None
                            try:
                                cur = rv.currentItem()
                            except Exception:
                                cur = None
                            cur_r = cur.row() if cur is not None else -1
                            cur_c = cur.column() if cur is not None else -1

                            max_view_rows = rv.rowCount()
                            max_view_cols = rv.columnCount()

                            for view_r in range(max_view_rows):
                                src_c = view_r  # source column == view row
                                if not (0 <= src_c < src.columnCount()):
                                    continue
                                for src_r in update_src_rows:
                                    view_c = src_r  # source row == view column
                                    if not (0 <= view_c < max_view_cols):
                                        continue
                                    # Avoid touching the actively edited cell
                                    if view_r == cur_r and view_c == cur_c:
                                        continue
                                    try:
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

            # update left ref view (skip if currently editing)
            if not editing_left:
                try:
                    new_ref_view = make_transposed(self.table_ref)
                    # replace contents of self.table_ref_view
                    try:
                        self.table_ref_view.blockSignals(True)
                        self.table_ref_view.setRowCount(new_ref_view.rowCount())
                        self.table_ref_view.setColumnCount(new_ref_view.columnCount())
                        # copy headers
                        try:
                            self.table_ref_view.setHorizontalHeaderLabels([new_ref_view.horizontalHeaderItem(i).text() if new_ref_view.horizontalHeaderItem(i) is not None else "" for i in range(new_ref_view.columnCount())])
                        except Exception:
                            pass
                        try:
                            self.table_ref_view.setVerticalHeaderLabels([new_ref_view.verticalHeaderItem(i).text() if new_ref_view.verticalHeaderItem(i) is not None else "" for i in range(new_ref_view.rowCount())])
                        except Exception:
                            pass
                        for r in range(new_ref_view.rowCount()):
                            for c in range(new_ref_view.columnCount()):
                                try:
                                    item = new_ref_view.item(r, c)
                                    if item is not None:
                                        it = QTableWidgetItem(item.text())
                                        try:
                                            # preserve editability/flags where possible so Obs.* remain editable
                                            it.setFlags(item.flags())
                                        except Exception:
                                            pass
                                        try:
                                            # Ensure residual rows (original source rows 5..8) are not editable
                                            # In this transposed loop, `c` is the source row index.
                                            non_edit_rows = (5, 6, 7, 8)
                                            if c in non_edit_rows:
                                                try:
                                                    from qt_compat.QtCore import Qt as _Qt
                                                    it.setFlags(it.flags() & ~getattr(_Qt, 'ItemIsEditable', 0))
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                        try:
                                            it.setTextAlignment(item.textAlignment())
                                        except Exception:
                                            pass
                                        self.table_ref_view.setItem(r, c, it)
                                    else:
                                        self.table_ref_view.setItem(r, c, QTableWidgetItem(""))
                                except Exception:
                                    pass
                    finally:
                        try:
                            self.table_ref_view.blockSignals(False)
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    self._pending_ref_view_refresh = False
                except Exception:
                    pass

            # update bottom/transposed table_between
            try:
                new_mid = make_transposed(self.table)
                self.table_between.blockSignals(True)
                try:
                    self.table_between.setRowCount(new_mid.rowCount())
                    self.table_between.setColumnCount(new_mid.columnCount())
                    try:
                        self.table_between.setHorizontalHeaderLabels([new_mid.horizontalHeaderItem(i).text() if new_mid.horizontalHeaderItem(i) is not None else "" for i in range(new_mid.columnCount())])
                    except Exception:
                        pass
                    try:
                        self.table_between.setVerticalHeaderLabels([new_mid.verticalHeaderItem(i).text() if new_mid.verticalHeaderItem(i) is not None else "" for i in range(new_mid.rowCount())])
                    except Exception:
                        pass
                    for r in range(new_mid.rowCount()):
                        for c in range(new_mid.columnCount()):
                            try:
                                item = new_mid.item(r, c)
                                if item is not None:
                                    it = QTableWidgetItem(item.text())
                                    try:
                                        it.setFlags(item.flags())
                                    except Exception:
                                        pass
                                    try:
                                        it.setTextAlignment(item.textAlignment())
                                    except Exception:
                                        pass
                                    self.table_between.setItem(r, c, it)
                                else:
                                    self.table_between.setItem(r, c, QTableWidgetItem(""))
                            except Exception:
                                pass
                finally:
                    try:
                        self.table_between.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass
            # After updating transposed views, ensure fixed pixel widths are applied
            try:
                # schedule immediately so layout has applied sizes
                QTimer.singleShot(0, self._shrink_visible_columns)
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
                            if ref_tbl is not None and ref_tbl.columnCount() > 0:
                                hdr2.setDefaultSectionSize(max(8, int(ref_tbl.columnWidth(0))))
                            else:
                                hdr2.setDefaultSectionSize(max(8, 40))
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _adjust_center_column_widths(self, fixed_px: int = 150):
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
                if hasattr(self, 'table_between') and 0 <= idx < self.table_between.rowCount():
                    try:
                        self.table_between.blockSignals(True)
                        # choose column 0 for current cell; selection behavior is rows
                        self.table_between.setCurrentCell(idx, 0)
                        self.table_between.selectRow(idx)
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
            self._adjust_center_column_widths(fixed_px=150)
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
                else:
                    poster_full = poster.copy()
                # multiply overlay and blend
                mix = max(0.0, min(1.0, (self.overlay_mix or 0) / 100.0))
                mult = cv2.multiply(poster_full, self.img_full, scale=1.0/255.0)
                overlay_full = cv2.addWeighted(poster_full, 1.0 - mix, mult, mix, 0)
                # draw boundaries if enabled
                try:
                    if self.show_boundaries:
                        # Build poster_for_edges at full resolution and apply trim in full-pixel units
                        try:
                            trim_px_full = int(params.get('trim_px', 0) or 0)
                        except Exception:
                            trim_px_full = 0
                        try:
                            poster_fe = poster_full.copy()
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
                        blurred = cv2.GaussianBlur(edge_mask, (3, 3), 0.8)
                        alpha = (blurred.astype(np.float32) / 255.0).reshape(h, w, 1)
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