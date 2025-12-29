"""
テーブル関連のユーティリティ関数とデータ投入関数を提供するモジュール。

このモジュールは、参照点テーブルと重心データテーブルの構築、更新、
およびレイアウト調整を行う関数を定義する。
"""

from qt_compat.QtCore import Qt, QTimer
from qt_compat.QtWidgets import QTableWidgetItem, QAbstractItemView, QHeaderView
from qt_compat.QtGui import QFont, QColor
import numpy as np
from Util import (
    fit_affine_2d_to_3d,
    apply_affine_2d_to_3d,
    max_decimal_places,
    round_to_decimals,
)
import Strings as STR


def _fit_similarity_2d(P, Q):
    """Fit similarity transform Q ~= s * R * P + t.

    P, Q: (n,2)
    Returns (s, R, t) where R is (2,2), t is (2,).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        raise ValueError("P and Q must be (n,2)")
    if P.shape[0] != Q.shape[0] or P.shape[0] < 2:
        raise ValueError("Need at least 2 point pairs")

    muP = np.mean(P, axis=0)
    muQ = np.mean(Q, axis=0)
    X = P - muP
    Y = Q - muQ
    # covariance
    C = (X.T @ Y) / float(P.shape[0])
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    # enforce proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    varP = float(np.mean(np.sum(X * X, axis=1)))
    if not np.isfinite(varP) or varP <= 0:
        raise ValueError("Degenerate configuration")
    s = float(np.sum(S) / varP)
    t = muQ - (s * (R @ muP))
    return s, R, t


def _apply_similarity_2d(s, R, t, P):
    P = np.asarray(P, dtype=float)
    return (s * (P @ R.T)) + np.asarray(t, dtype=float)


def _fit_plane_z(uv, z):
    """Fit z ~= a*u + b*v + c by least squares."""
    uv = np.asarray(uv, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != z.shape[0] or uv.shape[0] < 3:
        raise ValueError("Need at least 3 points")
    A = np.c_[uv[:, 0], uv[:, 1], np.ones((uv.shape[0], 1), dtype=float)]
    coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coef  # (3,)


def _apply_plane_z(coef, uv):
    uv = np.asarray(uv, dtype=float)
    A = np.c_[uv[:, 0], uv[:, 1], np.ones((uv.shape[0], 1), dtype=float)]
    return (A @ np.asarray(coef, dtype=float)).reshape(-1)

# Pylance対策: Qt列挙をgetattr経由で整数として取得
ALIGN_CENTER = getattr(Qt, "AlignCenter", 0)
ALIGN_RIGHT = getattr(Qt, "AlignRight", 0)
ALIGN_VCENTER = getattr(Qt, "AlignVCenter", 0)
ITEM_EDITABLE = getattr(Qt, "ItemIsEditable", 0)


# 参照点テーブルの列幅を固定し、全体幅を調整
def fix_ref_table_width(table_ref):
    try:
        cols = table_ref.columnCount()
        if cols <= 0:
            return
        # 各列の幅を一定値に揃えて、初期から狙いの列数が確実に見えるようにする
        const_w = 16  # px per column (fixed-width operation)
        min_visible_cols = 5  # Ensure left table is wide enough to show at least 5 columns
        total = 0
        # Apply widths to actual existing columns
        for c in range(cols):
            table_ref.setColumnWidth(c, const_w)
            total += const_w
        # If there are fewer actual columns than our min target, account for extra space
        if cols < min_visible_cols:
            total += (min_visible_cols - cols) * const_w
        # 未表示時は width() が 0 になりがちなので sizeHint を利用
        vh = table_ref.verticalHeader()
        vh_w_now = vh.width() if vh.isVisible() else 0
        vh_w_hint = vh.sizeHint().width() if hasattr(vh, 'sizeHint') else 0
        vh_w = max(vh_w_now, vh_w_hint)
        frame = table_ref.frameWidth() * 2
        margin = 2
        total_w = total + vh_w + frame + margin
        table_ref.setFixedWidth(total_w)
    except Exception:
        pass


# 両テーブルの行高さを内容に合わせて調整
def fix_tables_height(table_ref, table):
    try:
        # If the main window enforces a fixed table height (e.g. Ui sets
        # `FIXED_TABLE_HEIGHT`), respect that and do not recompute/override
        # heights here. This prevents layout callbacks from expanding the
        # bottom table beyond the intended fixed size.
        try:
            win = table.window()
            if win is not None and hasattr(win, 'FIXED_TABLE_HEIGHT') and getattr(win, 'FIXED_TABLE_HEIGHT'):
                h = getattr(win, 'FIXED_TABLE_HEIGHT')
                try:
                    table_ref.setFixedHeight(h)
                except Exception:
                    pass
                try:
                    table.setFixedHeight(h)
                except Exception:
                    pass
                return
        except Exception:
            pass
        for t in (table_ref, table):
            t.resizeRowsToContents()
        header_h = max(table_ref.horizontalHeader().height(), table.horizontalHeader().height())
        rows_h = max(sum(table_ref.rowHeight(r) for r in range(table_ref.rowCount())),
                     sum(table.rowHeight(r) for r in range(table.rowCount())))
        frame = max(table_ref.frameWidth(), table.frameWidth()) * 2
        hsb = table.horizontalScrollBar()
        need_hsb = (hsb.maximum() > 0) or hsb.isVisible()
        hsb_h = hsb.sizeHint().height() if need_hsb else 0
        margin = 2
        total_h = header_h + rows_h + frame + hsb_h + margin
        table_ref.setFixedHeight(total_h)
        table.setFixedHeight(total_h)
    except Exception:
        pass


# 両テーブルにデータを投入し、レイアウトを調整
def populate_tables(table_ref, table, ref_points, ref_obs, centroids, selected_index, ref_selected_index, flip_mode='auto', visible_ref_cols=None):
    table.blockSignals(True)
    table_ref.blockSignals(True)
    try:
        # 左テーブル（Ref）: 右表と下揃えにし、残差行（Res.*）を追加
        row_labels_ref = STR.TABLE_LEFT_ROW_LABELS
        table_ref.clear()
        table_ref.setRowCount(len(row_labels_ref))
        total_cols = 10 if visible_ref_cols is None else max(1, min(10, int(visible_ref_cols)))
        table_ref.setColumnCount(total_cols)
        table_ref.setVerticalHeaderLabels(row_labels_ref)
        try:
            vh = table_ref.verticalHeader()
            func = getattr(vh, "setDefaultAlignment", None)
            if callable(func):
                func(ALIGN_CENTER)
        except Exception:
            pass
        # 列見出しは太字の数字
        table_ref.setHorizontalHeaderLabels([str(i + 1) for i in range(total_cols)])
        # 左テーブルの水平ヘッダーはデフォルトの外観を維持（スタイルシートは適用しない）
        for c in range(total_cols):
            pt = ref_points[c] if 0 <= c < len(ref_points) else None
            x_item = QTableWidgetItem("" if pt is None else str(int(round(pt[0]))))
            y_item = QTableWidgetItem("" if pt is None else str(int(round(pt[1]))))
            x_item.setTextAlignment(ALIGN_CENTER)
            y_item.setTextAlignment(ALIGN_CENTER)
            # X(0), Y(1) は入力不可
            try:
                x_item.setFlags(x_item.flags() & ~ITEM_EDITABLE)
                y_item.setFlags(y_item.flags() & ~ITEM_EDITABLE)
            except Exception:
                pass
            table_ref.setItem(0, c, x_item)
            table_ref.setItem(1, c, y_item)
            # Obs. X/Y/Z は編集可（2,3,4行目）
            obs = ref_obs[c] if 0 <= c < len(ref_obs) else {"x": "", "y": "", "z": ""}
            ox = QTableWidgetItem(obs.get("x", ""))
            oy = QTableWidgetItem(obs.get("y", ""))
            oz = QTableWidgetItem(obs.get("z", ""))
            for it in (ox, oy, oz):
                it.setTextAlignment(ALIGN_CENTER)
                # 3〜5行目（Obs.*）は薄い灰色背景 + 太字
                try:
                    it.setBackground(QColor(245, 245, 245))
                    f = it.font(); f.setBold(True); it.setFont(f)
                except Exception:
                    pass
            table_ref.setItem(2, c, ox)
            table_ref.setItem(3, c, oy)
            table_ref.setItem(4, c, oz)
            # 残差セル初期化（編集不可）
            rx = QTableWidgetItem("")
            ry = QTableWidgetItem("")
            rz = QTableWidgetItem("")
            rr = QTableWidgetItem("")
            for it in (rx, ry, rz, rr):
                it.setTextAlignment(ALIGN_CENTER)
                try:
                    it.setFlags(it.flags() & ~ITEM_EDITABLE)
                except Exception:
                    pass
            table_ref.setItem(5, c, rx)
            table_ref.setItem(6, c, ry)
            table_ref.setItem(7, c, rz)
            table_ref.setItem(8, c, rr)
        table_ref.resizeColumnsToContents()
        fix_ref_table_width(table_ref)

        # 右テーブル（重心リスト）: Lv 行は不要
        row_labels = STR.TABLE_RIGHT_ROW_LABELS
        table.clear()
        if not centroids:
            table.setRowCount(len(row_labels))
            table.setColumnCount(0)
            table.setVerticalHeaderLabels(row_labels)
            try:
                vh = table.verticalHeader()
                func = getattr(vh, "setDefaultAlignment", None)
                if callable(func):
                        func(ALIGN_CENTER)
            except Exception:
                pass
            return
        n = len(centroids)
        table.setRowCount(len(row_labels))
        table.setColumnCount(n)
        table.setVerticalHeaderLabels(row_labels)
        try:
            vh = table.verticalHeader()
            func = getattr(vh, "setDefaultAlignment", None)
            if callable(func):
                    func(ALIGN_CENTER)
        except Exception:
            pass
        table.setHorizontalHeaderLabels([str(i + 1) for i in range(n)])
        # 生のXYとLvを先に埋める
        for c, (g, x, y) in enumerate(centroids):
            item_x = QTableWidgetItem(str(int(round(x))))
            item_y = QTableWidgetItem(str(int(round(y))))
            for it in (item_x, item_y):
                it.setTextAlignment(ALIGN_CENTER)
            table.setItem(0, c, item_x)
            table.setItem(1, c, item_y)
        # Calc.* を計算（回転角度+拡大縮小率ベース: 2D similarity for X/Y + plane for Z）
        # 参照点の収集
        ref_uv = []
        ref_xyz = []
        obs_x_vals = []
        obs_y_vals = []
        obs_z_vals = []
        used_cols = []
        for c in range(total_cols):
            pt = ref_points[c] if 0 <= c < len(ref_points) else None
            obs = ref_obs[c] if 0 <= c < len(ref_obs) else None
            if pt is None or not obs:
                continue
            try:
                u, v = float(pt[0]), float(pt[1])
                X = float(obs.get("x", "")) if str(obs.get("x", "")).strip() != "" else None
                Y = float(obs.get("y", "")) if str(obs.get("y", "")).strip() != "" else None
                Z = float(obs.get("z", "")) if str(obs.get("z", "")).strip() != "" else None
                if X is None or Y is None or Z is None:
                    continue
                ref_uv.append((u, v))
                ref_xyz.append((X, Y, Z))
                used_cols.append(c)
                obs_x_vals.append(obs.get("x", ""))
                obs_y_vals.append(obs.get("y", ""))
                obs_z_vals.append(obs.get("z", ""))
            except Exception:
                continue
        model = None
        flipped = False
        if len(ref_uv) >= 3:
            try:
                P0 = np.array(ref_uv, dtype=float)
                T = np.array(ref_xyz, dtype=float)
                Txy = T[:, 0:2]
                Tz = T[:, 2]

                def _fit_for(P):
                    # Fit XY similarity + Z plane
                    s, R, t = _fit_similarity_2d(P, Txy)
                    coef_z = _fit_plane_z(P, Tz)
                    pred_xy = _apply_similarity_2d(s, R, t, P)
                    pred_z = _apply_plane_z(coef_z, P)
                    pred = np.c_[pred_xy, pred_z]
                    err = T - pred
                    rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if err.size else float('inf')
                    return (s, R, t, coef_z), rms

                mode = str(flip_mode).lower()
                if mode == 'flip':
                    flipped = True
                    P = P0.copy(); P[:, 0] *= -1.0
                    params, _rms = _fit_for(P)
                elif mode == 'normal':
                    flipped = False
                    params, _rms = _fit_for(P0)
                else:
                    # auto: compare rms for non-flip vs flip
                    params0, rms0 = _fit_for(P0)
                    P1 = P0.copy(); P1[:, 0] *= -1.0
                    params1, rms1 = _fit_for(P1)
                    if rms1 < rms0:
                        flipped = True
                        params = params1
                    else:
                        flipped = False
                        params = params0

                s, R, t, coef_z = params
                model = {
                    "s": s,
                    "R": R,
                    "t": t,
                    "coef_z": coef_z,
                }
            except Exception:
                model = None
        # 出力の丸め桁を推定（入力の小数桁から決定）
        dp_x = max_decimal_places(obs_x_vals) if obs_x_vals else 0
        dp_y = max_decimal_places(obs_y_vals) if obs_y_vals else 0
        dp_z = max_decimal_places(obs_z_vals) if obs_z_vals else 0
        # 変換適用
        if model is not None:
            try:
                pts = []
                for _, x, y in centroids:
                    u, v = float(x), float(y)
                    u2 = -u if flipped else u
                    pts.append((u2, v))
                pts = np.asarray(pts, dtype=float)
                pred_xy = _apply_similarity_2d(model["s"], model["R"], model["t"], pts)
                pred_z = _apply_plane_z(model["coef_z"], pts)
                pred = np.c_[pred_xy, pred_z]  # (n,3)
                # 丸め
                pred_x = round_to_decimals(pred[:, 0], dp_x)
                pred_y = round_to_decimals(pred[:, 1], dp_y)
                pred_z = round_to_decimals(pred[:, 2], dp_z)
                for c in range(n):
                    cx = QTableWidgetItem(str(pred_x[c]).rstrip('0').rstrip('.') if dp_x > 0 else str(int(round(pred_x[c]))))
                    cy = QTableWidgetItem(str(pred_y[c]).rstrip('0').rstrip('.') if dp_y > 0 else str(int(round(pred_y[c]))))
                    cz = QTableWidgetItem(str(pred_z[c]).rstrip('0').rstrip('.') if dp_z > 0 else str(int(round(pred_z[c]))))
                    for it in (cx, cy, cz):
                        it.setTextAlignment(ALIGN_CENTER)
                        # 右表の3〜5行目（Calc.*）は太字
                        try:
                            f = it.font(); f.setBold(True); it.setFont(f)
                        except Exception:
                            pass
                        it.setFlags(it.flags() & ~ITEM_EDITABLE)
                    table.setItem(2, c, cx)
                    table.setItem(3, c, cy)
                    table.setItem(4, c, cz)

                # 参照点の残差を計算して左テーブルへ表示
                if used_cols:
                    # 参照点の予測（予測時にのみフリップを反映）
                    pts_ref = []
                    for (u, v) in ref_uv:
                        u2 = -u if flipped else u
                        pts_ref.append((u2, v))
                    pts_ref = np.asarray(pts_ref, dtype=float)
                    pred_xy_ref = _apply_similarity_2d(model["s"], model["R"], model["t"], pts_ref)
                    pred_z_ref = _apply_plane_z(model["coef_z"], pts_ref)
                    pred_ref = np.c_[pred_xy_ref, pred_z_ref]
                    ref_arr = np.asarray(ref_xyz, dtype=float)
                    res = ref_arr - pred_ref  # (m,3)
                    # 表示桁: 残差は「有効数字」ベース（有効数字2桁、最大小数4桁）
                    def _decimals_for_sig(arr, sig=2, cap=4):
                        try:
                            a = np.asarray(arr, dtype=float)
                            maxabs = float(np.nanmax(np.abs(a))) if a.size else 0.0
                            if not np.isfinite(maxabs) or maxabs == 0.0:
                                return 3  # 全て0相当なら小数3桁（目安）
                            import math
                            dec = sig - 1 - int(math.floor(math.log10(maxabs)))
                            return int(max(0, min(cap, dec)))
                        except Exception:
                            return 3

                    dp_rx = _decimals_for_sig(res[:, 0])
                    dp_ry = _decimals_for_sig(res[:, 1])
                    dp_rz = _decimals_for_sig(res[:, 2])
                    # 統一感のため、各列で丸めて表示
                    res_x = round_to_decimals(res[:, 0], dp_rx)
                    res_y = round_to_decimals(res[:, 1], dp_ry)
                    res_z = round_to_decimals(res[:, 2], dp_rz)
                    mag = np.sqrt(res[:, 0]**2 + res[:, 1]**2 + res[:, 2]**2)
                    dp_mag = _decimals_for_sig(mag)
                    mag_r = round_to_decimals(mag, dp_mag)
                    for i, col in enumerate(used_cols):
                        if not (0 <= col < total_cols):
                            continue
                        # 各セルへ書き込み
                        def _fmt(val, dp):
                            try:
                                if dp and dp > 0:
                                    s = ("%.*f" % (dp, float(val))).rstrip('0').rstrip('.')
                                    return s
                                else:
                                    return str(int(round(float(val))))
                            except Exception:
                                return ""
                        items = [
                            QTableWidgetItem(_fmt(res_x[i], dp_rx)),
                            QTableWidgetItem(_fmt(res_y[i], dp_ry)),
                            QTableWidgetItem(_fmt(res_z[i], dp_rz)),
                            QTableWidgetItem(_fmt(mag_r[i], dp_mag)),
                        ]
                        for r_offset, it in enumerate(items):
                            it.setTextAlignment(ALIGN_CENTER)
                            try:
                                it.setFlags(it.flags() & ~ITEM_EDITABLE)
                            except Exception:
                                pass
                            table_ref.setItem(5 + r_offset, col, it)
            except Exception:
                # 失敗時は空欄のまま
                pass
        table.resizeColumnsToContents()

        # 高さ調整（非同期でも再度）。左は上に余白を入れて下揃え
        fix_tables_height(table_ref, table)
        try:
            top_margin = table.rowHeight(0)
            table_ref.setViewportMargins(0, top_margin, 0, 0)
        except Exception:
            pass
        # 表示が落ち着いたタイミングで幅・高さを再調整（ヘッダー幅が確定してから反映）
        QTimer.singleShot(0, lambda: fix_ref_table_width(table_ref))
        QTimer.singleShot(0, lambda: fix_tables_height(table_ref, table))

        # 選択反映
        if selected_index is not None and 'n' in locals() and 0 <= selected_index < n:
            table.setCurrentCell(0, selected_index)
            table.selectColumn(selected_index)
        if 0 <= ref_selected_index < table_ref.columnCount():
            table_ref.setCurrentCell(0, ref_selected_index)
            table_ref.selectColumn(ref_selected_index)
    finally:
        table.blockSignals(False)
        table_ref.blockSignals(False)
