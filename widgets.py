from qt_compat.QtWidgets import QSlider, QStyle, QStyledItemDelegate, QLineEdit, QAbstractItemDelegate
from qt_compat.QtCore import Qt

# Pylance対策: Qt列挙を定数に退避
QT_LEFT_BUTTON = getattr(Qt, "LeftButton", 0)


class ClickableSlider(QSlider):
    """QSlider 拡張: クリックでジャンプ、ホイール感度スケーリング対応。

    _wheel_scale を変更するとホイールの1ノッチ当たりのステップ倍率を調整できます。
    例: 1/3 にしたい場合は 1.0/3.0 を設定。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wheel_accum = 0.0
        self._wheel_scale = 1.0
        self._wheel_wrap = False
        # Optional custom tick overlay (disabled by default).
        self._use_custom_ticks = False

    def mousePressEvent(self, event):
        if event.button() == QT_LEFT_BUTTON:
            minv, maxv = self.minimum(), self.maximum()
            tick = self.tickInterval() if self.tickInterval() > 0 else 1
            pos = event.pos().x()
            val = QStyle.sliderValueFromPosition(minv, maxv, pos, self.width())
            val = round((val - minv) / tick) * tick + minv
            val = max(minv, min(maxv, val))
            self.setValue(val)
            event.accept()
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        delta_y = event.angleDelta().y() / 120.0
        scale = getattr(self, "_wheel_scale", 1.0)
        scaled_steps = delta_y * scale
        self._wheel_accum += scaled_steps
        if abs(self._wheel_accum) >= 1.0:
            step_int = int(self._wheel_accum)
            self._wheel_accum -= step_int
            new_val = self.value() + step_int * self.singleStep()
            mn = self.minimum()
            mx = self.maximum()
            if bool(getattr(self, "_wheel_wrap", False)) and mx > mn:
                span = int(mx - mn + 1)
                try:
                    new_val = int(mn + ((int(new_val) - int(mn)) % span))
                except Exception:
                    new_val = max(mn, min(mx, int(new_val)))
            else:
                new_val = max(mn, min(mx, new_val))
            if new_val != self.value():
                self.setValue(new_val)
        event.accept()

    def paintEvent(self, event):
        # Draw default slider first. Optional custom tick overlay can be enabled per instance.
        try:
            super().paintEvent(event)
        except Exception:
            try:
                QSlider.paintEvent(self, event)
            except Exception:
                pass

        if not bool(getattr(self, '_use_custom_ticks', False)):
            return

        try:
            from qt_compat.QtGui import QPainter, QPen, QColor
            from qt_compat.QtWidgets import QStyleOptionSlider, QStyle

            painter = QPainter(self)
            tick_color = QColor(68, 68, 68)

            mn = int(self.minimum())
            mx = int(self.maximum())
            if mx <= mn:
                painter.end()
                return

            # Use style metrics to find the groove area for accurate tick placement
            opt = QStyleOptionSlider()
            try:
                opt.initFrom(self)
            except Exception:
                pass
            opt.orientation = self.orientation()
            opt.minimum = mn
            opt.maximum = mx
            opt.sliderPosition = int(self.value())

            groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

            usable_left = groove.left()
            usable_right = groove.right()
            usable_w = max(1, usable_right - usable_left)

            # Reduce tick mark lengths to roughly one-third of the previous size
            major_len = 7
            minor_len = 5

            # Draw ticks fully inside the widget: prefer below groove, otherwise above.
            h = max(1, int(self.height()))
            # give slightly more padding below the groove to allow long ticks
            space_below = int(h - 1 - (groove.bottom() + 6))
            if space_below >= major_len:
                y_base = int(groove.bottom() + 6)
                direction = 1
            else:
                y_base = int(groove.top() - 6)
                direction = -1

            # Draw only major ticks every 90 degrees (-180, -90, 0, 90, 180).
            step = 90
            for v in range(mn, mx + 1, step):
                x = int(usable_left + QStyle.sliderPositionFromValue(mn, mx, v, groove.width()))
                # Optionally skip drawing the tick exactly under the handle to avoid overlap
                try:
                    curv = int(self.value())
                except Exception:
                    curv = None

                if curv is not None and v == curv:
                    continue

                length = major_len
                penw = 1

                pen = QPen(tick_color, penw)
                painter.setPen(pen)
                y2 = y_base + direction * int(length)
                y1c = max(0, min(h - 1, y_base))
                y2c = max(0, min(h - 1, y2))
                painter.drawLine(x, y1c, x, y2c)

            painter.end()
        except Exception:
            pass


class RefTableDelegate(QStyledItemDelegate):
    """左テーブル（Ref）の編集ナビゲーション用デリゲート。

    要件:
    - 2行目(X)、3行目(Y)は編集不可（テーブル側でフラグ設定済みを前提）
        - 行定義変更後:
            2行目(Obs. X) -> 3行目(Obs. Y) -> 4行目(Obs. Z) と進み、
            4行目の次は隣列の 2行目(Obs. X) へ移動。
    """

    def __init__(self, table):
        super().__init__(table)
        self.table = table

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        try:
            # Enterキーでの遷移を編集ウィジェットに紐づける
            if isinstance(editor, QLineEdit):
                r, c = index.row(), index.column()

                def on_return():
                    try:
                        try:
                            # Commit without manually emitting commitData (can cause warnings if editor association changes)
                            self.table.closeEditor(editor, QAbstractItemDelegate.SubmitModelCache)
                        except Exception:
                            pass
                        # 全列で 2(Obs.X) -> 3(Obs.Y) -> 4(Obs.Z) と進み、
                        # 4 の次は 隣の列の 2(Obs.X) へ移動。
                        # Note: canonical ref table has 2 pseudo-header rows.
                        # Editable Stage rows are 4(Stage X),5(Stage Y),6(Stage Z).
                        if r == 4:
                            self.table.setCurrentCell(5, c)
                            item = self.table.item(5, c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                        if r == 5:
                            self.table.setCurrentCell(6, c)
                            item = self.table.item(6, c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                        if r == 6:
                            next_c = c + 1 if (c + 1) < self.table.columnCount() else c
                            self.table.setCurrentCell(4, next_c)
                            item = self.table.item(4, next_c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                    except Exception:
                        pass

                editor.returnPressed.connect(on_return)
        except Exception:
            pass
        return editor
