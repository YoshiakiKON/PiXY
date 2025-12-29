from qt_compat.QtCore import Qt, QEvent, QPoint, QTimer, QObject
from qt_compat.QtGui import QCursor
from collections import deque
from time import monotonic


def _evt_point(event):
    """Return a QPoint for event position compatible with Qt5/Qt6 events.

    Qt6's QWheelEvent/QMouseEvent may provide position() returning QPointF;
    Qt5 used pos() returning QPoint. Normalize to QPoint.
    """
    try:
        if hasattr(event, 'position'):
            p = event.position()
            # QPointF -> QPoint
            try:
                return QPoint(int(round(p.x())), int(round(p.y())))
            except Exception:
                return QPoint(int(p.x()), int(p.y()))
        elif hasattr(event, 'pos'):
            return event.pos()
    except Exception:
        pass
    return QPoint(0, 0)


class ImageViewController(QObject):
    """Handles mouse/wheel interactions for the zoomable image view.

    Depends on a Ui-like object providing:
      - img_label_proc (QLabel)
      - proc_scroll (QScrollArea)
      - proc_zoom: float
      - _apply_proc_zoom(): redraws label pixmap according to proc_zoom and updates display geometry
      - _display_to_full(QPoint) -> (x_full, y_full) or None
      - _full_to_display(x_full, y_full) -> (x_label, y_label) or None
      - _viewport_pos_to_label_pos(QPoint), _label_pos_to_viewport_pos(QPoint)
      - _draw_crosshair(QPoint)
      - _handle_image_click(QPoint)
      - _set_scroll(sx, sy)
      - pick_mode: None / 'add' / 'update'
      - _display_pm_base (QPixmap) for clearing crosshair overlay
    """

    def __init__(self, ui):
        super().__init__(ui)
        self.ui = ui
        # drag state
        self._mouse_pressed = False
        self._dragging = False
        self._drag_start_vp = None
        self._drag_start_scroll = (0, 0)
        self._drag_recent = deque(maxlen=8)  # (t, QPoint)
        # hover/select state
        self._hover_point_idx = None
        self._press_on_point_idx = None
        self._lock_to_point_select = False
        # kinetic
        self._kinetic_timer = QTimer(self)
        self._kinetic_timer.setInterval(16)
        self._kinetic_timer.timeout.connect(self._on_kinetic_tick)
        self._kinetic_vx = 0.0
        self._kinetic_vy = 0.0
        self._kinetic_last_t = 0.0

        # install event filters
        ui.proc_scroll.viewport().installEventFilter(self)
        ui.proc_scroll.viewport().setMouseTracking(True)
        ui.img_label_proc.installEventFilter(self)
        ui.img_label_proc.setMouseTracking(True)

    # Qt expects QObject-style eventFilter, but we don't subclass QObject; Qt accepts any PyObject with eventFilter
    def eventFilter(self, obj, event):
        # Wrap entire handler in try/except to prevent uncaught exceptions
        # from propagating out of the event loop and crashing the app.
        try:
            is_proc = (obj is self.ui.img_label_proc) or (obj is self.ui.proc_scroll.viewport())
            if not is_proc:
                return False
            et = event.type()

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos_vp = _evt_point(event) if obj is self.ui.proc_scroll.viewport() else self.ui._label_pos_to_viewport_pos(_evt_point(event))
                pos_label = _evt_point(event) if obj is self.ui.img_label_proc else self.ui._viewport_pos_to_label_pos(_evt_point(event))
                # 近傍の点を判定（通常モード時）
                self._hover_point_idx = self._nearest_point_idx(pos_label)
                self._press_on_point_idx = self._hover_point_idx
                self._lock_to_point_select = self._press_on_point_idx is not None and self.ui.pick_mode is None
                self._mouse_pressed = True
                self._dragging = False
                self._drag_start_vp = QPoint(pos_vp)
                self._drag_start_scroll = (
                    self.ui.proc_scroll.horizontalScrollBar().value(),
                    self.ui.proc_scroll.verticalScrollBar().value(),
                )
                self._drag_recent.clear()
                self._drag_recent.append((monotonic(), QPoint(pos_vp)))
                self._stop_kinetic()
                return True
            elif et == QEvent.MouseMove:
                pos_vp = _evt_point(event) if obj is self.ui.proc_scroll.viewport() else self.ui._label_pos_to_viewport_pos(_evt_point(event))
                pos_label = _evt_point(event) if obj is self.ui.img_label_proc else self.ui._viewport_pos_to_label_pos(_evt_point(event))
                # 近傍の点があればカーソルを矢印に、それ以外は手のひら（ピックモード中は十字）
                if self.ui.pick_mode in ('add', 'update'):
                    # ピックモードはカーソル固定（Ui側で設定）
                    pass
                else:
                    self._hover_point_idx = self._nearest_point_idx(pos_label)
                    try:
                        if self._hover_point_idx is not None:
                            self.ui.img_label_proc.setCursor(QCursor(Qt.ArrowCursor))
                        else:
                            # ドラッグ中以外は手のひら
                            self.ui.img_label_proc.setCursor(QCursor(Qt.OpenHandCursor))
                    except Exception:
                        pass
                if self._mouse_pressed and self._drag_start_vp is not None:
                    dx = pos_vp.x() - self._drag_start_vp.x()
                    dy = pos_vp.y() - self._drag_start_vp.y()
                    if self._lock_to_point_select:
                        # 点選択意図時はドラッグ開始させない
                        pass
                    elif not self._dragging and (abs(dx) > 3 or abs(dy) > 3):
                        self._dragging = True
                    if self._dragging:
                        sx0, sy0 = self._drag_start_scroll
                        self.ui._set_scroll(sx0 - dx, sy0 - dy)
                        self._drag_recent.append((monotonic(), QPoint(pos_vp)))
                        return True
                # draw crosshair in pick modes when not dragging
                if self.ui.pick_mode in ('add', 'update') and not self._dragging:
                    self.ui._draw_crosshair(pos_label)
            elif et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                pos_label = _evt_point(event) if obj is self.ui.img_label_proc else self.ui._viewport_pos_to_label_pos(_evt_point(event))
                vx = vy = 0.0
                if self._dragging and len(self._drag_recent) >= 2:
                    t2, p2 = self._drag_recent[-1]
                    t_cut = t2 - 0.12
                    t1, p1 = self._drag_recent[0]
                    for (tt, pp) in reversed(self._drag_recent):
                        t1, p1 = (tt, pp)
                        if tt <= t_cut:
                            break
                    dt = max(1e-3, t2 - t1)
                    vx = -(p2.x() - p1.x()) / dt
                    vy = -(p2.y() - p1.y()) / dt
                was_drag = self._dragging
                self._mouse_pressed = False
                self._dragging = False
                self._drag_start_vp = None
                self._drag_recent.clear()
                if not was_drag:
                    # 点選択が意図されていた場合はその点を選択、それ以外は既存のクリック処理
                    if self._press_on_point_idx is not None and self.ui.pick_mode is None:
                        idx = self._press_on_point_idx
                        self._press_on_point_idx = None
                        self._lock_to_point_select = False
                        # 範囲チェック
                        if 0 <= idx < len(getattr(self.ui, 'centroids', [])):
                            try:
                                if self.ui.selected_index != idx:
                                    self.ui.selected_index = idx
                                    # テーブル/表示を更新
                                    self.ui.schedule_update(force=True)
                                    # immediately sync visible table selection to reflect change
                                    try:
                                        self.ui._sync_table_selection()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        return True
                    else:
                        self.ui._handle_image_click(pos_label)
                        if getattr(self.ui, '_display_pm_base', None) is not None:
                            self.ui.img_label_proc.setPixmap(self.ui._display_pm_base)
                        return True
                speed = (vx*vx + vy*vy) ** 0.5
                if speed > 200:
                    self._start_kinetic(vx, vy)
                return True
            elif et == QEvent.Wheel:
                pos_vp = _evt_point(event) if obj is self.ui.proc_scroll.viewport() else self.ui._label_pos_to_viewport_pos(_evt_point(event))
                pos_label_before = self.ui._viewport_pos_to_label_pos(pos_vp)
                xf_yf = self.ui._display_to_full(pos_label_before)
                delta = event.angleDelta().y() / 120.0
                step = 0.1
                mods = event.modifiers() if hasattr(event, 'modifiers') else Qt.NoModifier
                if mods & Qt.ControlModifier:
                    step = 0.05
                new_zoom = self.ui.proc_zoom * (1.0 + step * delta)
                # Allow much larger zoom; rendering will downsample for display safety
                new_zoom = max(0.01, min(1024.0, new_zoom))
                if abs(new_zoom - self.ui.proc_zoom) > 1e-6:
                    self.ui.proc_zoom = new_zoom
                    self.ui._apply_proc_zoom()
                    if xf_yf is not None:
                        x_full, y_full = xf_yf
                        lx, ly = self.ui._full_to_display(x_full, y_full)
                        sx = lx - pos_vp.x()
                        sy = ly - pos_vp.y()
                        self.ui._set_scroll(sx, sy)
                    # ピックモード中は十字線を再描画
                    if self.ui.pick_mode in ('add', 'update'):
                        pos_label = self.ui._viewport_pos_to_label_pos(pos_vp)
                        self.ui._draw_crosshair(pos_label)
                return True
            elif et == QEvent.Resize:
                # ラベル/ビューポートのサイズ変更時に、ピックモードなら十字線を現在のカーソル位置で再描画
                if self.ui.pick_mode in ('add', 'update'):
                    try:
                        global_pt = QCursor.pos()
                        vp = self.ui.proc_scroll.viewport()
                        pos_vp = vp.mapFromGlobal(global_pt)
                        pos_label = self.ui._viewport_pos_to_label_pos(pos_vp)
                        self.ui._draw_crosshair(pos_label)
                    except Exception:
                        pass
                return False
            # If no branch handled the event, indicate we did not consume it.
            return False
        except KeyboardInterrupt:
            # User pressed Ctrl+C. Quit the QApplication cleanly to avoid
            # Qt printing a Python override error. Return False to indicate
            # the event was not consumed and allow shutdown.
            try:
                from qt_compat.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    try:
                        app.quit()
                    except Exception:
                        pass
            except Exception:
                pass
            return False
        except Exception:
            # Log other exceptions and avoid letting them crash the Qt event loop.
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            return False

    def _nearest_point_idx(self, pos_label):
        """Return index of nearest centroid within display-pixel radius; else None."""
        try:
            if not getattr(self.ui, 'centroids', None):
                return None
            # 近傍判定半径（表示ピクセル）
            radius = float(getattr(self.ui, 'select_radius_display', 10.0) or 10.0)
            r2 = radius * radius
            best_i = None
            best_d2 = None
            for i, (_g, xp, yp) in enumerate(self.ui.centroids):
                x_full = xp * getattr(self.ui, 'scale_proc_to_full', 1.0)
                y_full = yp * getattr(self.ui, 'scale_proc_to_full', 1.0)
                dxy = self.ui._full_to_display(x_full, y_full)
                if dxy is None:
                    continue
                dx = float(pos_label.x()) - float(dxy[0])
                dy = float(pos_label.y()) - float(dxy[1])
                d2 = dx*dx + dy*dy
                if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_i = i
            return best_i
        except Exception:
            return None

    def _start_kinetic(self, vx, vy):
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
        t = monotonic()
        dt = max(0.0, t - self._kinetic_last_t)
        self._kinetic_last_t = t
        if dt <= 0.0:
            return
        hsb = self.ui.proc_scroll.horizontalScrollBar()
        vsb = self.ui.proc_scroll.verticalScrollBar()
        sx = hsb.value() + self._kinetic_vx * dt
        sy = vsb.value() + self._kinetic_vy * dt
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
        self.ui._set_scroll(sx, sy)
        decay = 0.92
        self._kinetic_vx *= decay
        self._kinetic_vy *= decay
        if hit_edge_x:
            self._kinetic_vx *= 0.3
        if hit_edge_y:
            self._kinetic_vy *= 0.3
        if abs(self._kinetic_vx) < 5 and abs(self._kinetic_vy) < 5:
            self._stop_kinetic()
