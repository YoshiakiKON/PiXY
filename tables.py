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
        # Keep the left table width stable regardless of current visible columns
        # so ref-point additions don't shift the overall layout.
        min_visible_cols = 10
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
def populate_tables(
    table_ref,
    table,
    ref_points,
    ref_obs,
    centroids,
    selected_index,
    ref_selected_index,
    flip_mode='auto',
    visible_ref_cols=None,
    image_base_size=None,
    scale_proc_to_full=1.0,
    excluded_ref_indices=None,
):
    table.blockSignals(True)
    table_ref.blockSignals(True)
    try:
        # 左テーブル（Ref）: 右表と下揃えにし、残差行（Res.*）を追加
        row_labels_ref = STR.TABLE_LEFT_ROW_LABELS
        # Don't use clear() as it removes pseudo-headers; instead adjust row count
        # table_ref.clear()
        table_ref.setRowCount(len(row_labels_ref) + 2)  # +2 for pseudo-header rows
        total_cols = 10 if visible_ref_cols is None else max(1, min(10, int(visible_ref_cols)))
        table_ref.setColumnCount(total_cols)
        # Set vertical header labels starting from row 2 (skip pseudo-header rows 0-1)
        for i, label in enumerate(row_labels_ref):
            try:
                table_ref.setVerticalHeaderItem(i + 2, QTableWidgetItem(label))
            except Exception:
                pass
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
        # Data starts from row 2 to preserve pseudo-header rows 0-1
        DATA_ROW_OFFSET = 2
        try:
            _spf = float(scale_proc_to_full) if scale_proc_to_full is not None else 1.0
        except Exception:
            _spf = 1.0
        try:
            _h_full = int(image_base_size[1]) if image_base_size is not None else None
        except Exception:
            _h_full = None

        def _fmt_uv_from_proc(xp, yp):
            # Display convention: origin at bottom-left pixel of FULL image.
            # u = x_full
            # v = (h_full - 1) - y_full
            try:
                x_full = float(xp) * _spf
                y_full = float(yp) * _spf
                u = int(round(x_full))
                if _h_full is not None and _h_full > 0:
                    v = int(round((_h_full - 1) - y_full))
                else:
                    # fallback: at least make +v point upward
                    v = int(round(-y_full))
                return str(u), str(v)
            except Exception:
                return "", ""

        for c in range(total_cols):
            pt = ref_points[c] if 0 <= c < len(ref_points) else None
            if pt is None:
                su, sv = "", ""
            else:
                su, sv = _fmt_uv_from_proc(pt[0], pt[1])
            x_item = QTableWidgetItem(su)
            y_item = QTableWidgetItem(sv)
            x_item.setTextAlignment(ALIGN_CENTER)
            y_item.setTextAlignment(ALIGN_CENTER)
            # X(0), Y(1) は入力不可
            try:
                x_item.setFlags(x_item.flags() & ~ITEM_EDITABLE)
                y_item.setFlags(y_item.flags() & ~ITEM_EDITABLE)
            except Exception:
                pass
            table_ref.setItem(DATA_ROW_OFFSET + 0, c, x_item)
            table_ref.setItem(DATA_ROW_OFFSET + 1, c, y_item)
            # Stage X/Y/Z は編集可（2,3,4行目）
            obs = ref_obs[c] if 0 <= c < len(ref_obs) else {"x": "", "y": "", "z": ""}
            ox = QTableWidgetItem(obs.get("x", ""))
            oy = QTableWidgetItem(obs.get("y", ""))
            oz = QTableWidgetItem(obs.get("z", ""))
            for it in (ox, oy, oz):
                it.setTextAlignment(ALIGN_CENTER)
                # 3〜5行目（Stage.*）は薄い灰色背景 + 太字
                try:
                    it.setBackground(QColor(245, 245, 245))
                    f = it.font(); f.setBold(True); it.setFont(f)
                except Exception:
                    pass
            table_ref.setItem(DATA_ROW_OFFSET + 2, c, ox)
            table_ref.setItem(DATA_ROW_OFFSET + 3, c, oy)
            table_ref.setItem(DATA_ROW_OFFSET + 4, c, oz)
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
            table_ref.setItem(DATA_ROW_OFFSET + 5, c, rx)
            table_ref.setItem(DATA_ROW_OFFSET + 6, c, ry)
            table_ref.setItem(DATA_ROW_OFFSET + 7, c, rz)
            table_ref.setItem(DATA_ROW_OFFSET + 8, c, rr)
        table_ref.resizeColumnsToContents()
        fix_ref_table_width(table_ref)

        # 右テーブル（重心リスト）: Lv 行は不要
        row_labels = STR.TABLE_RIGHT_ROW_LABELS
        # Don't use clear() as it removes pseudo-headers
        # table.clear()
        if not centroids:
            table.setRowCount(len(row_labels) + 2)  # +2 for pseudo-header rows
            table.setColumnCount(0)
            # Set vertical header labels starting from row 2
            for i, label in enumerate(row_labels):
                try:
                    table.setVerticalHeaderItem(i + 2, QTableWidgetItem(label))
                except Exception:
                    pass
            try:
                vh = table.verticalHeader()
                func = getattr(vh, "setDefaultAlignment", None)
                if callable(func):
                        func(ALIGN_CENTER)
            except Exception:
                pass
            return
        n = len(centroids)
        table.setRowCount(len(row_labels) + 2)  # +2 for pseudo-header rows
        table.setColumnCount(n)
        # Set vertical header labels starting from row 2
        for i, label in enumerate(row_labels):
            try:
                table.setVerticalHeaderItem(i + 2, QTableWidgetItem(label))
            except Exception:
                pass
        try:
            vh = table.verticalHeader()
            func = getattr(vh, "setDefaultAlignment", None)
            if callable(func):
                    func(ALIGN_CENTER)
        except Exception:
            pass
        table.setHorizontalHeaderLabels([str(i + 1) for i in range(n)])
        # 生のXYとLvを先に埋める（Data starts from row 2）
        DATA_ROW_OFFSET = 2
        for c, (g, x, y) in enumerate(centroids):
            su, sv = _fmt_uv_from_proc(x, y)
            item_x = QTableWidgetItem(su)
            item_y = QTableWidgetItem(sv)
            for it in (item_x, item_y):
                it.setTextAlignment(ALIGN_CENTER)
            table.setItem(DATA_ROW_OFFSET + 0, c, item_x)
            table.setItem(DATA_ROW_OFFSET + 1, c, item_y)
        # Calc.* を計算（回転角度+拡大縮小率ベース）
        # - XY: 2D similarity (need ≥2 point pairs with X/Y)
        # - Z : plane fit (need ≥3 point pairs with X/Y/Z)
        ref_uv_xy = []
        ref_xy = []
        used_cols_xy = []
        obs_x_vals_xy = []
        obs_y_vals_xy = []

        ref_uv_xyz = []
        ref_xyz = []
        used_cols_xyz = []
        obs_x_vals_xyz = []
        obs_y_vals_xyz = []
        obs_z_vals_xyz = []

        for c in range(total_cols):
            pt = ref_points[c] if 0 <= c < len(ref_points) else None
            obs = ref_obs[c] if 0 <= c < len(ref_obs) else None
            if pt is None or not obs:
                continue
            # Skip excluded ref points from transformation calculation
            _excl_set = set(excluded_ref_indices or []) if excluded_ref_indices else set()
            if c in _excl_set:
                continue
            try:
                u, v = float(pt[0]), float(pt[1])
            except Exception:
                continue

            try:
                Xs = obs.get("x", "")
                Ys = obs.get("y", "")
                Zs = obs.get("z", "")
                X = float(Xs) if str(Xs).strip() != "" else None
                Y = float(Ys) if str(Ys).strip() != "" else None
                Z = float(Zs) if str(Zs).strip() != "" else None
            except Exception:
                X = Y = Z = None

            if X is not None and Y is not None:
                ref_uv_xy.append((u, v))
                ref_xy.append((X, Y))
                used_cols_xy.append(c)
                obs_x_vals_xy.append(Xs)
                obs_y_vals_xy.append(Ys)
                if Z is not None:
                    ref_uv_xyz.append((u, v))
                    ref_xyz.append((X, Y, Z))
                    used_cols_xyz.append(c)
                    obs_x_vals_xyz.append(Xs)
                    obs_y_vals_xyz.append(Ys)
                    obs_z_vals_xyz.append(Zs)

        model_xy = None
        model_xyz = None
        flipped_xy = False
        flipped_xyz = False

        def _fit_similarity_with_flip(P0, Q, mode_str: str):
            """Fit similarity with optional u-axis flip selection.
            Returns (s,R,t,flipped,rms).
            """
            mode_local = str(mode_str or 'auto').lower()

            def _fit_one(P):
                s, R, t = _fit_similarity_2d(P, Q)
                pred = _apply_similarity_2d(s, R, t, P)
                err = np.asarray(Q, dtype=float) - np.asarray(pred, dtype=float)
                rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if err.size else float('inf')
                return (s, R, t), rms

            if mode_local == 'flip':
                P = np.asarray(P0, dtype=float).copy()
                P[:, 0] *= -1.0
                (s, R, t), rms = _fit_one(P)
                return s, R, t, True, rms
            if mode_local == 'normal':
                (s, R, t), rms = _fit_one(P0)
                return s, R, t, False, rms

            # auto
            (s0, R0, t0), rms0 = _fit_one(P0)
            P1 = np.asarray(P0, dtype=float).copy()
            P1[:, 0] *= -1.0
            (s1, R1, t1), rms1 = _fit_one(P1)
            if rms1 < rms0:
                return s1, R1, t1, True, rms1
            return s0, R0, t0, False, rms0

        # XY-only model (≥2)
        if len(ref_uv_xy) >= 2:
            try:
                P0 = np.asarray(ref_uv_xy, dtype=float)
                Q = np.asarray(ref_xy, dtype=float)
                s, R, t, flipped_xy, _rms = _fit_similarity_with_flip(P0, Q, flip_mode)
                model_xy = {"s": s, "R": R, "t": t}
            except Exception:
                model_xy = None

        # Full XYZ model (≥3)
        if len(ref_uv_xyz) >= 3:
            try:
                P0 = np.asarray(ref_uv_xyz, dtype=float)
                T = np.asarray(ref_xyz, dtype=float)
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
                    flipped_xyz = True
                    P = P0.copy(); P[:, 0] *= -1.0
                    params, _rms = _fit_for(P)
                elif mode == 'normal':
                    flipped_xyz = False
                    params, _rms = _fit_for(P0)
                else:
                    # auto: compare rms for non-flip vs flip
                    params0, rms0 = _fit_for(P0)
                    P1 = P0.copy(); P1[:, 0] *= -1.0
                    params1, rms1 = _fit_for(P1)
                    if rms1 < rms0:
                        flipped_xyz = True
                        params = params1
                    else:
                        flipped_xyz = False
                        params = params0

                s, R, t, coef_z = params
                model_xyz = {"s": s, "R": R, "t": t, "coef_z": coef_z}
            except Exception:
                model_xyz = None

        # 出力の丸め桁を推定（入力の小数桁から決定）
        dp_x = max_decimal_places(obs_x_vals_xy) if obs_x_vals_xy else 0
        dp_y = max_decimal_places(obs_y_vals_xy) if obs_y_vals_xy else 0
        dp_z = max_decimal_places(obs_z_vals_xyz) if obs_z_vals_xyz else 0

        # 変換適用（優先: XYZモデル、なければXYモデル）
        model = model_xyz if model_xyz is not None else model_xy
        flipped = flipped_xyz if model_xyz is not None else flipped_xy
        if model is not None:
            try:
                pts = []
                for _, x, y in centroids:
                    u, v = float(x), float(y)
                    u2 = -u if flipped else u
                    pts.append((u2, v))
                pts = np.asarray(pts, dtype=float)
                pred_xy = _apply_similarity_2d(model["s"], model["R"], model["t"], pts)

                # Z is only available for the full XYZ model
                has_z = (model_xyz is not None and model_xyz.get("coef_z", None) is not None and dp_z is not None)
                if has_z:
                    pred_z = _apply_plane_z(model_xyz["coef_z"], pts)
                    pred = np.c_[pred_xy, pred_z]  # (n,3)
                else:
                    pred = np.c_[pred_xy, np.full((pred_xy.shape[0],), np.nan, dtype=float)]  # (n,3)

                # 丸め
                pred_x = round_to_decimals(pred[:, 0], dp_x)
                pred_y = round_to_decimals(pred[:, 1], dp_y)
                pred_z = round_to_decimals(pred[:, 2], dp_z) if has_z else None

                for c in range(n):
                    cx = QTableWidgetItem(str(pred_x[c]).rstrip('0').rstrip('.') if dp_x > 0 else str(int(round(pred_x[c]))))
                    cy = QTableWidgetItem(str(pred_y[c]).rstrip('0').rstrip('.') if dp_y > 0 else str(int(round(pred_y[c]))))
                    cz_text = ""
                    if has_z and pred_z is not None and np.isfinite(pred_z[c]):
                        cz_text = str(pred_z[c]).rstrip('0').rstrip('.') if dp_z > 0 else str(int(round(pred_z[c])))
                    cz = QTableWidgetItem(cz_text)
                    for it in (cx, cy, cz):
                        it.setTextAlignment(ALIGN_CENTER)
                        # 右表の3〜5行目（Calc.*）は太字
                        try:
                            f = it.font(); f.setBold(True); it.setFont(f)
                        except Exception:
                            pass
                        it.setFlags(it.flags() & ~ITEM_EDITABLE)
                    table.setItem(DATA_ROW_OFFSET + 2, c, cx)
                    table.setItem(DATA_ROW_OFFSET + 3, c, cy)
                    table.setItem(DATA_ROW_OFFSET + 4, c, cz)

                # 参照点の残差を計算して左テーブルへ表示
                if model_xyz is not None and used_cols_xyz:
                    # XYZ residuals (3D)
                    pts_ref = []
                    for (u, v) in ref_uv_xyz:
                        u2 = -u if flipped_xyz else u
                        pts_ref.append((u2, v))
                    pts_ref = np.asarray(pts_ref, dtype=float)
                    pred_xy_ref = _apply_similarity_2d(model_xyz["s"], model_xyz["R"], model_xyz["t"], pts_ref)
                    pred_z_ref = _apply_plane_z(model_xyz["coef_z"], pts_ref)
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
                    for i, col in enumerate(used_cols_xyz):
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
                            # Residual rows are offset by DATA_ROW_OFFSET (pseudo-header rows)
                            try:
                                table_ref.setItem(DATA_ROW_OFFSET + 5 + r_offset, col, it)
                            except Exception:
                                table_ref.setItem(5 + r_offset, col, it)

                elif model_xy is not None and used_cols_xy:
                    # XY-only residuals (2D). Leave Z residual blank.
                    pts_ref = []
                    for (u, v) in ref_uv_xy:
                        u2 = -u if flipped_xy else u
                        pts_ref.append((u2, v))
                    pts_ref = np.asarray(pts_ref, dtype=float)
                    pred_xy_ref = _apply_similarity_2d(model_xy["s"], model_xy["R"], model_xy["t"], pts_ref)
                    ref_arr2 = np.asarray(ref_xy, dtype=float)
                    res2 = ref_arr2 - pred_xy_ref  # (m,2)

                    def _decimals_for_sig(arr, sig=2, cap=4):
                        try:
                            a = np.asarray(arr, dtype=float)
                            maxabs = float(np.nanmax(np.abs(a))) if a.size else 0.0
                            if not np.isfinite(maxabs) or maxabs == 0.0:
                                return 3
                            import math
                            dec = sig - 1 - int(math.floor(math.log10(maxabs)))
                            return int(max(0, min(cap, dec)))
                        except Exception:
                            return 3

                    dp_rx = _decimals_for_sig(res2[:, 0])
                    dp_ry = _decimals_for_sig(res2[:, 1])
                    res_x = round_to_decimals(res2[:, 0], dp_rx)
                    res_y = round_to_decimals(res2[:, 1], dp_ry)
                    mag = np.sqrt(res2[:, 0]**2 + res2[:, 1]**2)
                    dp_mag = _decimals_for_sig(mag)
                    mag_r = round_to_decimals(mag, dp_mag)

                    def _fmt(val, dp):
                        try:
                            if dp and dp > 0:
                                s = ("%.*f" % (dp, float(val))).rstrip('0').rstrip('.')
                                return s
                            else:
                                return str(int(round(float(val))))
                        except Exception:
                            return ""

                    for i, col in enumerate(used_cols_xy):
                        if not (0 <= col < total_cols):
                            continue
                        items = [
                            QTableWidgetItem(_fmt(res_x[i], dp_rx)),
                            QTableWidgetItem(_fmt(res_y[i], dp_ry)),
                            QTableWidgetItem(""),
                            QTableWidgetItem(_fmt(mag_r[i], dp_mag)),
                        ]
                        for r_offset, it in enumerate(items):
                            it.setTextAlignment(ALIGN_CENTER)
                            try:
                                it.setFlags(it.flags() & ~ITEM_EDITABLE)
                            except Exception:
                                pass
                            try:
                                table_ref.setItem(DATA_ROW_OFFSET + 5 + r_offset, col, it)
                            except Exception:
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
