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
            new_val = max(self.minimum(), min(self.maximum(), new_val))
            if new_val != self.value():
                self.setValue(new_val)
        event.accept()


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
                        if r == 2:
                            self.table.setCurrentCell(3, c)
                            item = self.table.item(3, c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                        if r == 3:
                            self.table.setCurrentCell(4, c)
                            item = self.table.item(4, c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                        if r == 4:
                            next_c = c + 1 if (c + 1) < self.table.columnCount() else c
                            self.table.setCurrentCell(2, next_c)
                            item = self.table.item(2, next_c)
                            if item is not None and (item.flags() & Qt.ItemIsEditable):
                                self.table.editItem(item)
                            return
                    except Exception:
                        pass

                editor.returnPressed.connect(on_return)
        except Exception:
            pass
        return editor
