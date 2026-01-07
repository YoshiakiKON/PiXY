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

    def _split_by_neck_separation(self, comp_mask, neck_separation):
        """
        Detect and split particles by neck constriction using morphological operations.
        
        Args:
            comp_mask: Binary mask of the component (0-255)
            neck_separation: Threshold for neck detection (0-10, 0=no splitting)
        
        Returns:
            List of binary masks for split components
        """
        if neck_separation <= 0 or comp_mask is None or comp_mask.sum() == 0:
            return [comp_mask]
        
        try:
            # Normalize neck_separation (0-10) to erosion strength
            # Higher value = more aggressive erosion to break necks
            erosion_strength = int(neck_separation)
            if erosion_strength <= 0:
                return [comp_mask]
            
            # Apply erosion to thin out the component, revealing connection points
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(comp_mask, kernel, iterations=erosion_strength)
            
            # Find connected components in eroded mask (these are the "cores")
            num_cores, core_labels = cv2.connectedComponents(eroded, connectivity=4)[:2]
            
            if DEBUG:
                print(f"[DEBUG] _split_by_neck_separation: neck_sep={erosion_strength}, num_cores={num_cores}")
            
            # num_cores includes background (0), so we need at least 3 (bg + 2 actual cores)
            if num_cores < 3:
                # 0 is background, only 1 or fewer actual cores -> no split needed
                return [comp_mask]
            
            # Use multiple cores as seeds for splitting original component
            # by marker propagation
            markers = np.zeros(comp_mask.shape, dtype=np.int32)
            for core_id in range(1, num_cores):
                markers[core_labels == core_id] = core_id
            
            # Ensure background is marked
            markers[comp_mask == 0] = 0
            
            # For each unmarked pixel in comp_mask, find nearest marked pixel
            unmarked = (comp_mask > 0) & (markers == 0)
            if unmarked.any():
                # Dilate markers iteratively until all pixels are assigned
                for iteration in range(20):  # increased iterations
                    if not unmarked.any():
                        break
                    new_markers = markers.copy()
                    for i in range(new_markers.shape[0]):
                        for j in range(new_markers.shape[1]):
                            if unmarked[i, j]:
                                # Check neighbors
                                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < new_markers.shape[0] and 0 <= nj < new_markers.shape[1]:
                                        if new_markers[ni, nj] > 0:
                                            new_markers[i, j] = new_markers[ni, nj]
                                            break
                    markers = new_markers
                    unmarked = (comp_mask > 0) & (markers == 0)
            
            # Extract split masks
            split_masks = []
            for core_id in range(1, num_cores):
                split_mask = ((markers == core_id) & (comp_mask > 0)).astype(np.uint8) * 255
                if split_mask.sum() > 0:
                    split_masks.append(split_mask)
            
            if DEBUG and len(split_masks) > 1:
                print(f"[DEBUG] _split_by_neck_separation: split into {len(split_masks)} masks")
            
            if split_masks:
                return split_masks
            else:
                return [comp_mask]
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] _split_by_neck_separation failed: {e}")
            return [comp_mask]

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
        max_area = params.get("max_area", None)
        neck_separation = int(params.get("neck_separation", 0) or 0)
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
        # For histogram: store component areas BEFORE applying min/max filters.
        self.last_component_areas = []
        # For boundary display: mask AFTER applying min/max filters (and trim).
        self.last_boundary_mask = np.zeros(poster.shape[:2], dtype=np.uint8)

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
            
            # Simple connected components analysis (4-connectivity)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])

                # Optional neck separation: detect and split pinched particles
                comp_mask = (labels == lab).astype(np.uint8) * 255
                split_masks = self._split_by_neck_separation(comp_mask, neck_separation)

                # If no split occurred (or single piece), use original area
                if len(split_masks) <= 1:
                    if area > 0:
                        self.last_component_areas.append(area)
                    if area < min_area:
                        continue
                    if max_area is not None:
                        try:
                            if area > int(max_area):
                                continue
                        except Exception:
                            pass
                    cx, cy = centroids[lab]
                    results.append([group_no, cx, cy])
                    try:
                        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(self.last_boundary_mask, contours, -1, 255, 1)
                    except Exception:
                        pass
                    continue

                # Process each split component (only split areas counted)
                for split_mask in split_masks:
                    # Re-calculate centroid for this component
                    split_num_labels, split_labels, split_stats, split_centroids = cv2.connectedComponentsWithStats(split_mask, connectivity=4)
                    # Add all non-background components from this split
                    for split_lab in range(1, int(split_num_labels)):
                        split_area = int(split_stats[split_lab, cv2.CC_STAT_AREA])
                        if split_area > 0:
                            self.last_component_areas.append(split_area)
                        if split_area < min_area:
                            continue
                        if max_area is not None:
                            try:
                                if split_area > int(max_area):
                                    continue
                            except Exception:
                                pass
                        cx, cy = split_centroids[split_lab]
                        results.append([group_no, cx, cy])
                        try:
                            comp_split = (split_labels == split_lab).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(comp_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                cv2.drawContours(self.last_boundary_mask, contours, -1, 255, 1)
                        except Exception as e:
                            if DEBUG:
                                print(f"[DEBUG] Failed to draw contours for split mask: {e}")
                            pass
        if DEBUG:
            print(f"[DEBUG][CentroidProcessor] get_centroids done: found {len(results)} centroids in {time.time()-start_t:.2f}s")
        return results