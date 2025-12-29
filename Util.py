"""
ユーティリティ関数を提供するモジュール。

画像変換、アフィン変換推定、ポスタライズなどの
汎用的な処理関数を定義する。
"""

import cv2
import numpy as np
from qt_compat.QtGui import QPixmap, QImage


def cvimg_to_qpixmap(img_bgr):
    """
    OpenCV BGR画像をQPixmapに変換。

    Args:
        img_bgr: BGR形式のNumPy配列

    Returns:
        QPixmapオブジェクト
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def kmeans_posterize(img_bgr, levels=2):
    """
    K-meansクラスタリングによるポスタライズ処理。

    Args:
        img_bgr: 入力画像 (BGR)
        levels: 色数 (クラスタ数)

    Returns:
        ポスタライズされた画像
    """
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    K = max(1, int(levels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 再現性のために固定シードを設定（同一入力・同一Kで結果が安定）
    try:
        cv2.setRNGSeed(12345)
    except Exception:
        pass
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    poster = res.reshape(img_bgr.shape)
    return poster


# ===== 2D -> 3D Affine estimation (least squares with simple robust option) =====

def _design_matrix(points_2d):
    """Build Nx3 design matrix [u, v, 1] from Nx2 points."""
    P = np.asarray(points_2d, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_2d must be (N,2)")
    N = P.shape[0]
    ones = np.ones((N, 1), dtype=float)
    return np.hstack([P, ones])  # (N,3)


def _fit_affine_core(P, T, weights=None):
    """Solve P(=Nx3) * a = T(=Nx3) for 3 outputs independently.
    Returns A(3x3) s.t. pred = P @ A.T
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        Pw = P * w
        Tw = T * w
    else:
        Pw, Tw = P, T
    A = np.zeros((3, 3), dtype=float)
    for j in range(3):
        a, _, _, _ = np.linalg.lstsq(Pw, Tw[:, j], rcond=None)
        A[j, :] = a
    return A


def _residuals(A, P, T):
    """
    アフィン変換の残差を計算。

    Args:
        A: アフィン行列 (3x3)
        P: 設計行列 (Nx3)
        T: ターゲット点 (Nx3)

    Returns:
        (誤差ベクトル, 予測値)
    """
    pred = P @ A.T  # (N,3)
    res = T - pred
    err = np.linalg.norm(res, axis=1)
    return err, pred


def fit_affine_2d_to_3d(points_2d, points_3d, robust=True, try_lr_flip=True, max_iters=5):
    """
    2D→3Dアフィン変換を最小二乗法で推定。

    Args:
        points_2d: 2D点リスト (N,2)
        points_3d: 3D点リスト (N,3)
        robust: 外れ値除去を行うか
        try_lr_flip: u軸反転モデルも試すか
        max_iters: ロバスト推定の最大反復数

    Returns:
        (A, info) A: アフィン行列 (3x3), info: 情報辞書
    """
    P = _design_matrix(points_2d)
    T = np.asarray(points_3d, dtype=float)
    if T.ndim != 2 or T.shape[1] != 3 or P.shape[0] != T.shape[0]:
        raise ValueError("points_2d (N,2) and points_3d (N,3) with same N required")

    def _fit_with_option(P_base):
        mask = np.ones((P_base.shape[0],), dtype=bool)
        A = _fit_affine_core(P_base, T)
        for _ in range(max_iters if robust else 1):
            err, _ = _residuals(A, P_base, T)
            med = np.median(err)
            mad = np.median(np.abs(err - med))
            thr = med + 2.5 * (1.4826 * mad + 1e-9)
            new_mask = err <= thr
            if new_mask.sum() < 3:  # keep at least 3
                new_mask = err.argsort()[:3]
                tmp = np.zeros_like(mask); tmp[new_mask] = True
                new_mask = tmp
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            A = _fit_affine_core(P_base[mask], T[mask])
        err, _ = _residuals(A, P_base, T)
        rms = float(np.sqrt(np.mean((err[mask])**2))) if mask.any() else float('inf')
        return A, mask, rms

    # normal model
    A1, mask1, rms1 = _fit_with_option(P)

    # left-right flip on u-axis
    if try_lr_flip:
        Pflip = P.copy()
        Pflip[:, 0] *= -1.0  # u -> -u
        A2, mask2, rms2 = _fit_with_option(Pflip)
        if rms2 < rms1:
            return A2, { 'flipped': True, 'inliers': mask2, 'rms': rms2 }

    return A1, { 'flipped': False, 'inliers': mask1, 'rms': rms1 }


def apply_affine_2d_to_3d(A, points_2d):
    """
    2D点にアフィン変換を適用して3D点を得る。

    Args:
        A: アフィン行列 (3x3)
        points_2d: 2D点リスト

    Returns:
        変換後の3D点リスト
    """
    P = _design_matrix(points_2d)
    pred = P @ A.T
    return pred  # (N,3)


def max_decimal_places(values):
    """
    数値文字列リストから最大小数桁数を推定。

    Args:
        values: 数値または文字列のリスト

    Returns:
        最大小数桁数
    """
    max_dp = 0
    for v in values:
        try:
            s = str(v)
            if 'e' in s or 'E' in s:
                # scientific notation: rough fallback to 3
                max_dp = max(max_dp, 3)
            elif '.' in s:
                dp = len(s.split('.')[-1])
                max_dp = max(max_dp, dp)
        except Exception:
            pass
    return max_dp


def round_to_decimals(arr, dp):
    """
    配列を指定小数桁数で丸める。

    Args:
        arr: 数値配列
        dp: 小数桁数

    Returns:
        丸められた配列
    """
    if dp is None:
        return arr
    try:
        return np.round(arr, int(dp))
    except Exception:
        return arr