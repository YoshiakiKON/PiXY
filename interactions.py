from qt_compat.QtCore import Qt, QEvent, QPoint, QTimer, QObject
from qt_compat.QtGui import QCursor
from collections import deque
from time import monotonic, perf_counter
import os


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
        self._hover_ref_idx = None
        self._press_on_point_idx = None
        self._press_on_ref_idx = None
        self._lock_to_point_select = False
        # kinetic
        self._kinetic_timer = QTimer(self)
        self._kinetic_timer.setInterval(16)
        self._kinetic_timer.timeout.connect(self._on_kinetic_tick)
        self._kinetic_vx = 0.0
        self._kinetic_vy = 0.0
        self._kinetic_last_t = 0.0

        # wheel zoom: coalesce frequent wheel events into ~1 redraw per frame
        self._wheel_zoom_timer = QTimer(self)
        self._wheel_zoom_timer.setSingleShot(True)
        self._wheel_zoom_timer.setInterval(16)
        self._wheel_zoom_timer.timeout.connect(self._on_wheel_zoom_tick)
        self._wheel_zoom_pending = False
        self._wheel_zoom_target = None
        self._wheel_zoom_anchor_full = None
        self._wheel_zoom_anchor_vp = None
        self._wheel_zoom_pick_mode = None
        self._wheel_zoom_last_event_t = None
        self._wheel_fast_render_applied = False
        self._wheel_fast_max_pixels = 4096 * 4096
        self._wheel_settle_timer = QTimer(self)
        self._wheel_settle_timer.setSingleShot(True)
        self._wheel_settle_timer.setInterval(140)
        self._wheel_settle_timer.timeout.connect(self._on_wheel_zoom_settled)

        # Optional performance logging
        self._wheel_profile = bool(str(os.environ.get('PIXY_WHEEL_PROFILE', '')).strip())
        self._wheel_ev_count = 0
        self._wheel_apply_count = 0
        self._wheel_apply_ms = deque(maxlen=120)
        self._wheel_latency_ms = deque(maxlen=120)
        self._wheel_last_report_t = perf_counter()

        # install event filters
        ui.proc_scroll.viewport().installEventFilter(self)
        ui.proc_scroll.viewport().setMouseTracking(True)
        ui.img_label_proc.installEventFilter(self)
        ui.img_label_proc.setMouseTracking(True)

        # Some platforms/configs can dispatch wheel events to the focused widget
        # even when the cursor is over a different widget (e.g., sliders change
        # while the cursor is over the image). Install an application-wide filter
        # so wheel-zoom works reliably when the cursor is over the image viewport.
        try:
            from qt_compat.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
        except Exception:
            pass

    # Qt expects QObject-style eventFilter, but we don't subclass QObject; Qt accepts any PyObject with eventFilter
    def eventFilter(self, obj, event):
        # Wrap entire handler in try/except to prevent uncaught exceptions
        # from propagating out of the event loop and crashing the app.
        try:
            et = event.type()

            # Determine whether the cursor is currently over the image viewport.
            # This allows us to handle wheel events even if Qt sent the event to a
            # different focused widget.
            cursor_over_proc = False
            pos_vp_from_cursor = None
            try:
                vp = self.ui.proc_scroll.viewport()
                gp = QCursor.pos()
                pv = vp.mapFromGlobal(gp)
                if vp.rect().contains(pv):
                    cursor_over_proc = True
                    pos_vp_from_cursor = QPoint(pv)
            except Exception:
                cursor_over_proc = False
                pos_vp_from_cursor = None

            is_proc_obj = (obj is self.ui.img_label_proc) or (obj is self.ui.proc_scroll.viewport())
            if not is_proc_obj and not (et == QEvent.Wheel and cursor_over_proc):
                return False

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # If wheel-zoom updates are pending, commit them before hit-testing.
                # This prevents one-step-late selection and post-click viewport jumps.
                try:
                    self._flush_pending_wheel_zoom_on_click()
                except Exception:
                    pass
                pos_vp = _evt_point(event) if obj is self.ui.proc_scroll.viewport() else self.ui._label_pos_to_viewport_pos(_evt_point(event))
                pos_label = _evt_point(event) if obj is self.ui.img_label_proc else self.ui._viewport_pos_to_label_pos(_evt_point(event))
                # 近傍の点を判定（通常モード時）: centroid / ref のうち近い方を選択
                hit_kind, hit_idx = self._nearest_hit(pos_label)
                self._hover_point_idx = hit_idx if hit_kind == 'centroid' else None
                self._hover_ref_idx = hit_idx if hit_kind == 'ref' else None
                self._press_on_point_idx = self._hover_point_idx
                self._press_on_ref_idx = self._hover_ref_idx
                self._lock_to_point_select = (hit_idx is not None) and self.ui.pick_mode is None
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
                try:
                    self.ui._update_cursor_info_overlay(pos_label)
                except Exception:
                    pass
                # 近傍の点があればカーソルを矢印に、それ以外は手のひら（ピックモード中は十字）
                if self.ui.pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                    # ピックモードはカーソル固定（Ui側で設定）
                    pass
                else:
                    hit_kind, hit_idx = self._nearest_hit(pos_label)
                    self._hover_point_idx = hit_idx if hit_kind == 'centroid' else None
                    self._hover_ref_idx = hit_idx if hit_kind == 'ref' else None
                    try:
                        if self._hover_point_idx is not None or self._hover_ref_idx is not None:
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
                if self.ui.pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update') and not self._dragging:
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
                    if self.ui.pick_mode is None:
                        # Prefer ref selection if a ref was hit
                        if self._press_on_ref_idx is not None:
                            ridx = self._press_on_ref_idx
                            self._press_on_ref_idx = None
                            self._press_on_point_idx = None
                            self._lock_to_point_select = False
                            try:
                                if 0 <= int(ridx) < len(getattr(self.ui, 'ref_points', []) or []):
                                    self.ui.ref_selected_index = int(ridx)
                                    try:
                                        self.ui._sync_ref_selection()
                                    except Exception:
                                        # fallback: redraw at least
                                        try:
                                            self.ui._apply_proc_zoom()
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            return True

                        # Otherwise, centroid selection
                        if self._press_on_point_idx is not None:
                            idx = self._press_on_point_idx
                            self._press_on_point_idx = None
                            self._lock_to_point_select = False
                            # 範囲チェック
                            if 0 <= idx < len(getattr(self.ui, 'centroids', [])):
                                try:
                                    if self.ui.selected_index != idx:
                                        self.ui.selected_index = idx
                                        # 画像クリック時はベース画像を固定し、
                                        # 選択マーカーのみ差分描画する。
                                        try:
                                            self.ui._refresh_selected_overlay_only()
                                        except Exception:
                                            self.ui.schedule_update(force=True, recompute_centroids=False)
                                    # Always sync visible table selection/scroll even when index is unchanged.
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
            elif et == QEvent.Leave:
                try:
                    self.ui._update_cursor_info_overlay(None)
                except Exception:
                    pass
                try:
                    # End add-fiducial mode on Leave after first add.
                    if self.ui.pick_mode == 'add' and getattr(self.ui, '_ref_add_has_added', False):
                        try:
                            if hasattr(self.ui, '_log_info'):
                                self.ui._log_info("AddRef: mode end on Leave after first add")
                        except Exception:
                            pass
                        self.ui._end_pick_mode()

                    # End target_add mode when cursor leaves the image, but only after
                    # the first point has been placed (before that, mode persists until cancel).
                    if self.ui.pick_mode == 'target_add' and getattr(self.ui, '_target_add_has_added', False):
                        try:
                            if hasattr(self.ui, '_log_info'):
                                self.ui._log_info("AddTarget: mode end on Leave after first add")
                        except Exception:
                            pass
                        self.ui._end_pick_mode()
                except Exception:
                    pass
                return False
            elif et == QEvent.Wheel:
                # Center-anchored zoom:
                # keep viewport center fixed while changing zoom to avoid jumpy behavior
                # near limits and make zoom in/out predictable.
                try:
                    vp0 = self.ui.proc_scroll.viewport()
                    pos_vp = QPoint(int(vp0.width() // 2), int(vp0.height() // 2))
                except Exception:
                    if pos_vp_from_cursor is not None:
                        pos_vp = QPoint(pos_vp_from_cursor)
                    else:
                        pos_vp = _evt_point(event) if obj is self.ui.proc_scroll.viewport() else self.ui._label_pos_to_viewport_pos(_evt_point(event))

                pos_label_before = self.ui._viewport_pos_to_label_pos(pos_vp)
                xf_yf = self.ui._display_to_full(pos_label_before)
                if xf_yf is None:
                    try:
                        # Fallback: use full-image center when current center maps outside image area.
                        img_sz = getattr(self.ui, '_img_base_size', None)
                        if img_sz is not None and len(img_sz) >= 2:
                            xf_yf = (float(img_sz[0]) / 2.0, float(img_sz[1]) / 2.0)
                    except Exception:
                        xf_yf = None

                if self._wheel_profile:
                    try:
                        self._wheel_ev_count += 1
                    except Exception:
                        pass

                # Prefer angleDelta, but fall back to pixelDelta for trackpads.
                dy = 0
                try:
                    ad = event.angleDelta()
                    ax = int(getattr(ad, 'x', lambda: 0)())
                    ay = int(getattr(ad, 'y', lambda: 0)())
                    dy = ay if abs(ay) >= abs(ax) else ax
                except Exception:
                    dy = 0
                if dy == 0:
                    try:
                        pd = event.pixelDelta()
                        px = int(getattr(pd, 'x', lambda: 0)())
                        py = int(getattr(pd, 'y', lambda: 0)())
                        dy = py if abs(py) >= abs(px) else px
                    except Exception:
                        dy = 0
                if dy == 0:
                    return False

                delta = float(dy) / 120.0
                # Exponential scaling feels more map-like than linear scaling.
                base = 1.2
                mods = event.modifiers() if hasattr(event, 'modifiers') else Qt.NoModifier
                if mods & Qt.ControlModifier:
                    base = 1.1
                try:
                    factor = float(base) ** float(delta)
                except Exception:
                    factor = 1.0

                base_zoom = self.ui.proc_zoom
                try:
                    if self._wheel_zoom_target is not None and (self._wheel_zoom_timer.isActive() or self._wheel_zoom_pending):
                        base_zoom = float(self._wheel_zoom_target)
                except Exception:
                    base_zoom = self.ui.proc_zoom

                new_zoom = float(base_zoom) * float(factor)
                # Max zoom endpoint: roughly target visible full-image pixels in viewport
                # (longer viewport side). Falls back to legacy cap when unavailable.
                max_zoom = 1024.0
                try:
                    target_px = float(getattr(self.ui, 'max_zoom_target_visible_px', 300) or 300)
                    if target_px > 0:
                        vp = self.ui.proc_scroll.viewport()
                        vw = max(1.0, float(vp.width()))
                        vh = max(1.0, float(vp.height()))
                        z_target = max(vw, vh) / target_px
                        max_zoom = min(max_zoom, max(0.01, float(z_target)))
                except Exception:
                    pass
                new_zoom = max(0.01, min(float(max_zoom), new_zoom))

                if abs(new_zoom - float(base_zoom)) > 1e-9:
                    # Coalesce redraws to reduce stutter during continuous wheel input.
                    self._wheel_zoom_target = float(new_zoom)
                    self._wheel_zoom_anchor_full = xf_yf
                    self._wheel_zoom_anchor_vp = QPoint(pos_vp)
                    self._wheel_zoom_pick_mode = self.ui.pick_mode
                    self._wheel_zoom_pending = True
                    try:
                        # During active wheel input, lower display pixel cap for responsiveness.
                        self.ui._max_render_pixels_override = int(self._wheel_fast_max_pixels)
                    except Exception:
                        pass
                    try:
                        self._wheel_settle_timer.start()
                    except Exception:
                        pass
                    if self._wheel_profile:
                        try:
                            self._wheel_zoom_last_event_t = perf_counter()
                        except Exception:
                            self._wheel_zoom_last_event_t = None
                    if not self._wheel_zoom_timer.isActive():
                        self._wheel_zoom_timer.start()
                return True
            elif et == QEvent.Resize:
                # ラベル/ビューポートのサイズ変更時に、ピックモードなら十字線を現在のカーソル位置で再描画
                try:
                    self.ui._reposition_viewport_overlays()
                except Exception:
                    pass
                try:
                    global_pt = QCursor.pos()
                    vp = self.ui.proc_scroll.viewport()
                    pos_vp = vp.mapFromGlobal(global_pt)
                    pos_label = self.ui._viewport_pos_to_label_pos(pos_vp)
                    self.ui._update_cursor_info_overlay(pos_label)
                except Exception:
                    pass
                if self.ui.pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
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

    def _flush_pending_wheel_zoom_on_click(self):
        """Finalize wheel zoom state before point-selection click handling.

        - Apply pending zoom tick now so hit-testing uses current mapping.
        - Cancel delayed settle pass to avoid post-click viewport jump.
        """
        try:
            if bool(getattr(self, '_wheel_zoom_pending', False)):
                try:
                    self._on_wheel_zoom_tick()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if getattr(self, '_wheel_zoom_timer', None) is not None and self._wheel_zoom_timer.isActive():
                self._wheel_zoom_timer.stop()
        except Exception:
            pass

        try:
            if getattr(self, '_wheel_settle_timer', None) is not None and self._wheel_settle_timer.isActive():
                self._wheel_settle_timer.stop()
        except Exception:
            pass

        # Do not allow deferred fast->full pass to reposition viewport after click.
        try:
            self.ui._max_render_pixels_override = None
        except Exception:
            pass
        try:
            self._wheel_fast_render_applied = False
        except Exception:
            pass
        except Exception:
            # Log other exceptions and avoid letting them crash the Qt event loop.
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            return False

    def _nearest_centroid_hit(self, pos_label):
        """Return (idx, d2) of nearest centroid within radius; else (None, None)."""
        try:
            if not getattr(self.ui, 'centroids', None):
                return None, None
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
            return best_i, best_d2
        except Exception:
            return None, None

    def _nearest_ref_hit(self, pos_label):
        """Return (idx, d2) of nearest ref point within radius; else (None, None)."""
        try:
            pts = getattr(self.ui, 'ref_points', None) or []
            if not pts:
                return None, None
            radius = float(getattr(self.ui, 'select_radius_display', 10.0) or 10.0)
            r2 = radius * radius
            spf = float(getattr(self.ui, 'scale_proc_to_full', 1.0) or 1.0)
            best_i = None
            best_d2 = None
            for i, pt in enumerate(pts):
                if not pt:
                    continue
                try:
                    x_full = float(pt[0]) * spf
                    y_full = float(pt[1]) * spf
                except Exception:
                    continue
                dxy = self.ui._full_to_display(x_full, y_full)
                if dxy is None:
                    continue
                dx = float(pos_label.x()) - float(dxy[0])
                dy = float(pos_label.y()) - float(dxy[1])
                d2 = dx*dx + dy*dy
                if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_i = i
            return best_i, best_d2
        except Exception:
            return None, None

    def _nearest_hit(self, pos_label):
        """Return ('ref'|'centroid'|None, idx|None) for nearest selectable marker."""
        try:
            ci, cd2 = self._nearest_centroid_hit(pos_label)
            ri, rd2 = self._nearest_ref_hit(pos_label)
            if ci is None and ri is None:
                return None, None
            if ci is None:
                return 'ref', ri
            if ri is None:
                return 'centroid', ci
            # both hit: choose closer
            try:
                if rd2 is not None and cd2 is not None and float(rd2) <= float(cd2):
                    return 'ref', ri
            except Exception:
                pass
            return 'centroid', ci
        except Exception:
            return None, None

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

    def _on_wheel_zoom_tick(self):
        """Apply pending wheel zoom changes (debounced)."""
        try:
            if not getattr(self, '_wheel_zoom_pending', False):
                return
            if getattr(self, '_wheel_zoom_target', None) is None:
                self._wheel_zoom_pending = False
                return

            target_zoom = float(self._wheel_zoom_target)
            anchor_full = getattr(self, '_wheel_zoom_anchor_full', None)
            anchor_vp = getattr(self, '_wheel_zoom_anchor_vp', None)
            pick_mode = getattr(self, '_wheel_zoom_pick_mode', None)
            evt_t = getattr(self, '_wheel_zoom_last_event_t', None)
            self._wheel_zoom_pending = False

            def _recenter_to_anchor():
                try:
                    if anchor_full is None:
                        return
                    try:
                        vp_now = self.ui.proc_scroll.viewport()
                        anchor_vp_live = QPoint(int(vp_now.width() // 2), int(vp_now.height() // 2))
                    except Exception:
                        anchor_vp_live = anchor_vp
                    if anchor_vp_live is None:
                        return
                    x_full, y_full = anchor_full
                    dxy = self.ui._full_to_display(x_full, y_full)
                    if dxy is None:
                        return
                    lx, ly = dxy
                    sx = float(lx) - float(anchor_vp_live.x())
                    sy = float(ly) - float(anchor_vp_live.y())
                    self.ui._set_scroll(sx, sy)
                except Exception:
                    pass

            t0 = perf_counter() if self._wheel_profile else None
            if abs(float(getattr(self.ui, 'proc_zoom', 1.0)) - target_zoom) > 1e-6:
                self.ui.proc_zoom = target_zoom
                try:
                    self._wheel_fast_render_applied = bool(getattr(self.ui, '_max_render_pixels_override', None) is not None)
                except Exception:
                    self._wheel_fast_render_applied = False
                self.ui._apply_proc_zoom()

                # Keep anchor at viewport center.
                try:
                    # Immediate correction
                    _recenter_to_anchor()
                    # Deferred correction after layout/scroll range settles
                    QTimer.singleShot(0, _recenter_to_anchor)
                except Exception:
                    pass

                # ピックモード中は十字線を再描画
                try:
                    if pick_mode in ('add', 'update', 'target_add', 'target_update', 'center_uv_update'):
                        try:
                            vp_now = self.ui.proc_scroll.viewport()
                            anchor_vp_live = QPoint(int(vp_now.width() // 2), int(vp_now.height() // 2))
                        except Exception:
                            anchor_vp_live = anchor_vp
                        if anchor_vp_live is None:
                            raise ValueError('anchor viewport unavailable')
                        pos_label = self.ui._viewport_pos_to_label_pos(anchor_vp_live)
                        self.ui._draw_crosshair(pos_label)
                except Exception:
                    pass

            if self._wheel_profile and t0 is not None:
                t1 = perf_counter()
                try:
                    self._wheel_apply_count += 1
                    self._wheel_apply_ms.append((t1 - t0) * 1000.0)
                    if evt_t is not None:
                        self._wheel_latency_ms.append((t0 - float(evt_t)) * 1000.0)
                except Exception:
                    pass

                # Report roughly once per second
                try:
                    now = t1
                    if (now - float(self._wheel_last_report_t)) >= 1.0:
                        self._wheel_last_report_t = now
                        ev = int(getattr(self, '_wheel_ev_count', 0) or 0)
                        ap = int(getattr(self, '_wheel_apply_count', 0) or 0)
                        co = max(0, ev - ap)
                        am = list(getattr(self, '_wheel_apply_ms', []) or [])
                        lm = list(getattr(self, '_wheel_latency_ms', []) or [])
                        avg_apply = (sum(am) / len(am)) if am else 0.0
                        max_apply = (max(am)) if am else 0.0
                        avg_lat = (sum(lm) / len(lm)) if lm else 0.0
                        max_lat = (max(lm)) if lm else 0.0
                        print(f"[PIXY wheel] events={ev} applies={ap} coalesced={co} apply_ms(avg/max)={avg_apply:.1f}/{max_apply:.1f} latency_ms(avg/max)={avg_lat:.1f}/{max_lat:.1f}")
                except Exception:
                    pass

            # If more wheel events arrived while rendering, schedule another tick.
            if getattr(self, '_wheel_zoom_pending', False):
                try:
                    self._wheel_zoom_timer.start()
                except Exception:
                    pass
        except Exception:
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

    def _on_wheel_zoom_settled(self):
        """Restore full-quality render after wheel input pause."""
        try:
            if getattr(self, '_wheel_zoom_pending', False):
                return
            try:
                self.ui._max_render_pixels_override = None
            except Exception:
                pass

            # If we used fast rendering while wheeling, refresh once at full quality
            # -- but only when the full-quality cap is actually higher than what
            # the fast path already delivered.  This avoids an expensive no-op
            # re-render when the image is small or the zoom level is low.
            if bool(getattr(self, '_wheel_fast_render_applied', False)):
                def _estimate_draw_size(max_pixels_cap):
                    try:
                        source_img = self.ui._last_overlay_full if self.ui._last_overlay_full is not None else self.ui.proc_img
                        if source_img is None:
                            return None
                        h, w = source_img.shape[:2]
                        z = max(0.001, float(getattr(self.ui, 'proc_zoom', 1.0)))
                        cap = int(max_pixels_cap)
                        desired_pixels = float(w) * float(h) * (z * z)
                        if desired_pixels > float(cap):
                            scale_down = (float(cap) / float(desired_pixels)) ** 0.5
                            draw_w = max(1, int(round(float(w) * z * scale_down)))
                            draw_h = max(1, int(round(float(h) * z * scale_down)))
                        else:
                            draw_w = max(1, int(round(float(w) * z)))
                            draw_h = max(1, int(round(float(h) * z)))
                        return (draw_w, draw_h)
                    except Exception:
                        return None

                need_redraw = True
                try:
                    full_cap = self.ui._get_render_max_pixels()
                    fast_cap = int(getattr(self, '_wheel_fast_max_pixels', 0) or 0)
                    if fast_cap > 0 and fast_cap >= full_cap:
                        need_redraw = False          # fast render was already at full quality
                    else:
                        fast_size = _estimate_draw_size(fast_cap) if fast_cap > 0 else None
                        full_size = _estimate_draw_size(full_cap)
                        # If the full-quality redraw would change the actual canvas size,
                        # it can look like a zoom jump after continuous wheel input.
                        # Keep the current geometry stable in that case.
                        if fast_size is not None and full_size is not None and fast_size != full_size:
                            need_redraw = False
                except Exception:
                    pass
                if need_redraw:
                    try:
                        self.ui._apply_proc_zoom()
                        # Recenter again after the full-quality redraw; the canvas size can
                        # change slightly compared with the fast path on very large images.
                        try:
                            anchor_full = getattr(self, '_wheel_zoom_anchor_full', None)
                            anchor_vp = getattr(self, '_wheel_zoom_anchor_vp', None)
                            if anchor_full is not None:
                                try:
                                    vp_now = self.ui.proc_scroll.viewport()
                                    anchor_vp_live = QPoint(int(vp_now.width() // 2), int(vp_now.height() // 2))
                                except Exception:
                                    anchor_vp_live = anchor_vp
                                if anchor_vp_live is not None:
                                    x_full, y_full = anchor_full
                                    dxy = self.ui._full_to_display(x_full, y_full)
                                    if dxy is not None:
                                        lx, ly = dxy
                                        sx = float(lx) - float(anchor_vp_live.x())
                                        sy = float(ly) - float(anchor_vp_live.y())
                                        self.ui._set_scroll(sx, sy)
                        except Exception:
                            pass
                    except Exception:
                        pass
            self._wheel_fast_render_applied = False
        except Exception:
            pass
