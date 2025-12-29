"""
重心計算処理を行うモジュール。

画像からポスタライズ処理と連結成分解析により、
各色の領域の重心を計算する。
"""

import cv2
import numpy as np
from Util import kmeans_posterize
try:
    from Config import DEBUG
except Exception:
    DEBUG = False
import time


class CentroidProcessor:
    """
    重心計算プロセッサクラス。

    処理用画像とフル解像度画像のスケール情報を保持し、
    パラメータに基づいて重心を計算する。
    """

    def __init__(self, proc_img, scale_proc_to_full, img_full):
        """
        初期化。

        Args:
            proc_img: 処理用画像 (リサイズ済み)
            scale_proc_to_full: 処理用からフル解像度へのスケール倍率
            img_full: フル解像度画像
        """
        self.proc_img = proc_img
        self.scale_proc_to_full = scale_proc_to_full
        self.img_full = img_full

    def get_centroids(self, params, poster=None):
        """
        重心を計算する。

        Args:
            params: 処理パラメータ (levels, min_area, trim_px)
            poster: ポスタライズ画像 (Noneなら内部生成)

        Returns:
            重心リスト [[group_no, cx, cy], ...]
        """
        # posterが渡されなければここで生成（後方互換）
        start_t = time.time()
        if DEBUG:
            print(f"[DEBUG][CentroidProcessor] get_centroids start levels={params.get('levels')} min_area={params.get('min_area')} trim={params.get('trim_px')}")
        if poster is None:
            poster = kmeans_posterize(self.proc_img, params["levels"])
        min_area = params["min_area"]
        # `params['trim_px']` is provided in full-image pixels (UI-visible units).
        # Convert to processing-image (proc_img) pixels for morphological operations
        # because `poster` and masks are at proc resolution.
        trim_px_full = int(params.get("trim_px", 0) or 0)
        try:
            trim_px_proc = int(round(float(trim_px_full) / max(1.0, float(self.scale_proc_to_full))))
        except Exception:
            trim_px_proc = int(trim_px_full)
        unique_colors = np.unique(poster.reshape(-1, 3), axis=0)
        results = []
        for group_no, color in enumerate(unique_colors, 1):
            if DEBUG and group_no % 5 == 0:
                print(f"[DEBUG][CentroidProcessor] processing color group {group_no}/{len(unique_colors)}")
            mask = cv2.inRange(poster, color, color)
            # トリム（収縮）: UIで指定されたフル画像ピクセル単位を proc 解像度へ変換した
            # `trim_px_proc` を iterations に使って形態学的収縮を行う。
            if trim_px_proc > 0:
                k = int(trim_px_proc)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=k)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                cx, cy = centroids[lab]
                results.append([group_no, cx, cy])
        if DEBUG:
            print(f"[DEBUG][CentroidProcessor] get_centroids done: found {len(results)} centroids in {time.time()-start_t:.2f}s")
        return results