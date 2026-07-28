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


class CalculationCancelled(Exception):
    """Raised when centroid calculation is cancelled by user request."""


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
        self.last_rim_points_proc = []

    def _rim_point_from_contours(self, contours, cx, cy, inward_px_proc):
        """Compute rim point: move inward from farthest contour point from centroid."""
        try:
            if not contours:
                return None
            pts = []
            for cnt in contours:
                try:
                    arr = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)
                    if arr.size > 0:
                        pts.append(arr)
                except Exception:
                    continue
            if not pts:
                return None
            p = np.vstack(pts)
            cxy = np.array([float(cx), float(cy)], dtype=np.float32)
            d = p - cxy
            d2 = np.sum(d * d, axis=1)
            if d2.size <= 0:
                return None
            idx = int(np.argmax(d2))
            tip = p[idx]
            vec = tip - cxy
            dist = float(np.linalg.norm(vec))
            if not np.isfinite(dist) or dist <= 1e-9:
                return float(tip[0]), float(tip[1])
            inward = max(0.0, float(inward_px_proc or 0.0))
            t = max(0.0, min(1.0, (dist - inward) / dist))
            rp = cxy + (vec * t)
            return float(rp[0]), float(rp[1])
        except Exception:
            return None

    def _split_by_neck_separation(self, comp_mask, neck_separation):
        """
        Detect and split particles by neck constriction using morphological operations.
        Optimized with cv2.dilate for fast marker propagation.
        
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
            # by nearest-seed assignment with OpenCV distance transform labels.
            # This avoids iterative per-seed dilations in Python and is much faster.
            seed_img = np.full(comp_mask.shape, 255, dtype=np.uint8)
            seed_img[core_labels > 0] = 0
            _, seed_labels = cv2.distanceTransformWithLabels(
                seed_img,
                cv2.DIST_L2,
                3,
                labelType=cv2.DIST_LABEL_CCOMP,
            )
            markers = seed_labels.astype(np.int32, copy=False)
            markers[comp_mask == 0] = 0
            
            # Extract split masks
            split_masks = []
            used_labels = np.unique(markers[comp_mask > 0]) if np.any(comp_mask > 0) else []
            for core_id in used_labels:
                if int(core_id) <= 0:
                    continue
                split_mask = ((markers == int(core_id)) & (comp_mask > 0)).astype(np.uint8) * 255
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

    def get_centroids(
        self,
        params,
        poster=None,
        *,
        collect_timings=False,
        emit_timing=True,
        compute_boundary_mask=True,
        stop_requested=None,
        stop_check_interval_sec=1.0,
    ):
        """
        重心を計算する。

        Args:
            params: 処理パラメータ (levels, min_area, trim_px)
            poster: ポスタライズ画像 (Noneなら内部生成)

        Returns:
            重心リスト [[group_no, cx, cy], ...]
        """
        # posterが渡されなければここで生成（後方互換）
        now = time.perf_counter
        t0 = now()
        poster_time = 0.0
        timings = None
        if collect_timings:
            timings = {
                "poster_time": 0.0,
                "unique_colors_time": 0.0,
                "mask_time": 0.0,
                "cc_time": 0.0,
                "comp_mask_time": 0.0,
                "split_time": 0.0,
                "split_cc_time": 0.0,
                "centroid_time": 0.0,
                "boundary_time": 0.0,
                "total_time": 0.0,
                "levels": params.get("levels"),
                "min_area": params.get("min_area"),
                "max_area": params.get("max_area"),
                "trim_px_full": params.get("trim_px"),
                "neck_separation": params.get("neck_separation"),
                "groups": 0,
                "components": 0,
                "split_components": 0,
                "centroids": 0,
            }
        check_interval = max(0.1, float(stop_check_interval_sec or 1.0))
        next_stop_check_t = time.monotonic() + check_interval

        def _check_stop_if_needed():
            nonlocal next_stop_check_t
            if stop_requested is None:
                return
            now_mono = time.monotonic()
            if now_mono < next_stop_check_t:
                return
            next_stop_check_t = now_mono + check_interval
            should_stop = False
            try:
                should_stop = bool(stop_requested())
            except TypeError:
                should_stop = bool(stop_requested)
            except Exception:
                should_stop = False
            if should_stop:
                raise CalculationCancelled("Centroid calculation cancelled by user.")

        if DEBUG:
            print(f"[DEBUG][CentroidProcessor] get_centroids start levels={params.get('levels')} min_area={params.get('min_area')} trim={params.get('trim_px')}")
        if poster is None:
            t_p_start = now()
            poster = kmeans_posterize(self.proc_img, params["levels"])
            poster_time = now() - t_p_start
        if timings is not None:
            timings["poster_time"] = float(poster_time)
        min_area = params["min_area"]
        max_area = params.get("max_area", None)
        neck_separation = int(params.get("neck_separation", 0) or 0)
        # Shape complexity filter strength (0-10).
        # 0 means "no filtering", 10 means "strongest filtering".
        try:
            shape_complexity = int(params.get("shape_complexity", 3) if params is not None else 3)
        except Exception:
            shape_complexity = 3

        def _passes_shape_complexity(binary_mask_255: np.ndarray) -> bool:
            """Return True if the component shape is acceptable.

            We use a compactness metric based on perimeter and area:
              ratio = P^2 / (4*pi*A)
            ratio == 1 for a circle and increases for elongated/irregular shapes.

            The UI parameter is 0-10; 0 disables this filter.
            """
            if shape_complexity <= 0:
                return True
            try:
                if binary_mask_255 is None or binary_mask_255.sum() == 0:
                    return False
                contours, _ = cv2.findContours(binary_mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return True
                # Use the largest contour by area for stability
                cnt = max(contours, key=cv2.contourArea)
                area_c = float(cv2.contourArea(cnt))
                if area_c <= 0.0:
                    return False
                perim = float(cv2.arcLength(cnt, True))
                ratio = (perim * perim) / (4.0 * float(np.pi) * area_c)
                # Map slider (1..10) to stricter threshold (5.5..1.0)
                thr = 6.0 - 0.5 * float(shape_complexity)
                return bool(ratio <= thr)
            except Exception:
                # Best-effort: if metric fails, don't drop detections
                return True
        # `params['trim_px']` is provided in full-image pixels (UI-visible units).
        # Convert to processing-image (proc_img) pixels for morphological operations
        # because `poster` and masks are at proc resolution.
        trim_px_full = int(params.get("trim_px", 0) or 0)
        try:
            # Use ceil so trim=1 in full-pixel units has a visible effect after downscaling.
            trim_px_proc = int(np.ceil(float(trim_px_full) / max(1.0, float(self.scale_proc_to_full))))
        except Exception:
            trim_px_proc = int(trim_px_full)
        rim_offset_full = int(params.get("rim_offset_px", 3) or 0)
        try:
            rim_offset_proc = float(rim_offset_full) / max(1.0, float(self.scale_proc_to_full))
        except Exception:
            rim_offset_proc = float(rim_offset_full)
        t_uc0 = now() if timings is not None else None
        unique_colors = np.unique(poster.reshape(-1, 3), axis=0)
        if timings is not None:
            timings["unique_colors_time"] += float(now() - t_uc0)
            timings["groups"] = int(len(unique_colors))
        results = []
        rim_results = []
        # For histogram: store component areas BEFORE applying min/max filters.
        self.last_component_areas = []
        # For boundary display: mask AFTER applying min/max filters (and trim).
        self.last_boundary_mask = np.zeros(poster.shape[:2], dtype=np.uint8)

        for group_no, color in enumerate(unique_colors, 1):
            _check_stop_if_needed()
            if DEBUG and group_no % 5 == 0:
                print(f"[DEBUG][CentroidProcessor] processing color group {group_no}/{len(unique_colors)}")
            t_mask0 = now() if timings is not None else None
            mask = cv2.inRange(poster, color, color)
            # トリム（収縮）: UIで指定されたフル画像ピクセル単位を proc 解像度へ変換した
            # `trim_px_proc` を iterations に使って形態学的収縮を行う。
            if trim_px_proc > 0:
                k = int(trim_px_proc)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=k)
            if timings is not None:
                timings["mask_time"] += float(now() - t_mask0)
            
            # Simple connected components analysis (4-connectivity)
            t_cc0 = now() if timings is not None else None
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            if timings is not None:
                timings["cc_time"] += float(now() - t_cc0)
                timings["components"] += int(max(0, int(num_labels) - 1))
            for lab in range(1, num_labels):
                _check_stop_if_needed()
                area = int(stats[lab, cv2.CC_STAT_AREA])
                left = int(stats[lab, cv2.CC_STAT_LEFT])
                top = int(stats[lab, cv2.CC_STAT_TOP])
                width = int(stats[lab, cv2.CC_STAT_WIDTH])
                height = int(stats[lab, cv2.CC_STAT_HEIGHT])

                # Components smaller than min_area can never produce a valid split
                # result, so skip the costly neck-separation path entirely.
                if area > 0 and area < min_area:
                    self.last_component_areas.append(area)
                    continue

                # Optional neck separation: detect and split pinched particles
                t_cm0 = now() if timings is not None else None
                comp_labels = labels[top:top + height, left:left + width]
                comp_mask = (comp_labels == lab).astype(np.uint8) * 255
                if timings is not None:
                    timings["comp_mask_time"] += float(now() - t_cm0)
                t_split0 = now() if timings is not None else None
                split_masks = self._split_by_neck_separation(comp_mask, neck_separation)
                if timings is not None:
                    timings["split_time"] += float(now() - t_split0)

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
                    # Optional shape filter (run after size filter)
                    if not _passes_shape_complexity(comp_mask):
                        continue
                    t_cent0 = now() if timings is not None else None
                    cx, cy = centroids[lab]
                    results.append([group_no, cx, cy])
                    rim_pt = None
                    try:
                        contours_r, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours_r:
                            contour_offset = np.array([[[left, top]]], dtype=np.int32)
                            shifted_contours_r = [cnt.astype(np.int32, copy=False) + contour_offset for cnt in contours_r]
                            rim_pt = self._rim_point_from_contours(shifted_contours_r, float(cx), float(cy), rim_offset_proc)
                    except Exception:
                        rim_pt = None
                    rim_results.append(rim_pt)
                    if timings is not None:
                        timings["centroid_time"] += float(now() - t_cent0)
                    if compute_boundary_mask:
                        t_b0 = now() if timings is not None else None
                        try:
                            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                contour_offset = np.array([[[left, top]]], dtype=np.int32)
                                shifted_contours = [cnt.astype(np.int32, copy=False) + contour_offset for cnt in contours]
                                cv2.drawContours(self.last_boundary_mask, shifted_contours, -1, 255, 1)
                        except Exception:
                            pass
                        if timings is not None and t_b0 is not None:
                            timings["boundary_time"] += float(now() - t_b0)
                    continue

                # Process each split component (only split areas counted)
                if timings is not None:
                    timings["split_components"] += int(len(split_masks))
                for split_mask in split_masks:
                    _check_stop_if_needed()
                    # Re-calculate centroid for this component
                    t_scc0 = now() if timings is not None else None
                    split_num_labels, split_labels, split_stats, split_centroids = cv2.connectedComponentsWithStats(split_mask, connectivity=4)
                    if timings is not None:
                        timings["split_cc_time"] += float(now() - t_scc0)
                    # Add all non-background components from this split
                    for split_lab in range(1, int(split_num_labels)):
                        _check_stop_if_needed()
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
                        # Optional shape filter for split components
                        if shape_complexity > 0:
                            try:
                                comp_split = (split_labels == split_lab).astype(np.uint8) * 255
                            except Exception:
                                comp_split = None
                            if comp_split is not None and (not _passes_shape_complexity(comp_split)):
                                continue
                        t_cent0 = now() if timings is not None else None
                        cx, cy = split_centroids[split_lab]
                        cx += left
                        cy += top
                        results.append([group_no, cx, cy])
                        rim_pt = None
                        try:
                            comp_split = (split_labels == split_lab).astype(np.uint8) * 255
                            contours_r, _ = cv2.findContours(comp_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours_r:
                                contour_offset = np.array([[[left, top]]], dtype=np.int32)
                                shifted_contours_r = [cnt.astype(np.int32, copy=False) + contour_offset for cnt in contours_r]
                                rim_pt = self._rim_point_from_contours(shifted_contours_r, float(cx), float(cy), rim_offset_proc)
                        except Exception:
                            rim_pt = None
                        rim_results.append(rim_pt)
                        if timings is not None:
                            timings["centroid_time"] += float(now() - t_cent0)
                        if compute_boundary_mask:
                            t_b0 = now() if timings is not None else None
                            try:
                                comp_split = (split_labels == split_lab).astype(np.uint8) * 255
                                contours, _ = cv2.findContours(comp_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    contour_offset = np.array([[[left, top]]], dtype=np.int32)
                                    shifted_contours = [cnt.astype(np.int32, copy=False) + contour_offset for cnt in contours]
                                    cv2.drawContours(self.last_boundary_mask, shifted_contours, -1, 255, 1)
                            except Exception as e:
                                if DEBUG:
                                    print(f"[DEBUG] Failed to draw contours for split mask: {e}")
                                pass
                            if timings is not None and t_b0 is not None:
                                timings["boundary_time"] += float(now() - t_b0)
        total_time = now() - t0

        # Save last timings (even if not collecting detailed breakdown)
        if timings is not None:
            timings["centroids"] = int(len(results))
            timings["total_time"] = float(total_time)
            # Derived buckets for easy comparison
            detection_time = (
                float(timings.get("unique_colors_time", 0.0))
                + float(timings.get("mask_time", 0.0))
                + float(timings.get("cc_time", 0.0))
                + float(timings.get("comp_mask_time", 0.0))
                + float(timings.get("split_time", 0.0))
                + float(timings.get("split_cc_time", 0.0))
            )
            timings["particle_detection_time"] = detection_time
            timings["centroid_calc_time"] = float(timings.get("centroid_time", 0.0))
            timings["boundary_time"] = float(timings.get("boundary_time", 0.0))
            timings["poster_time"] = float(poster_time)
            self.last_timings = timings
        else:
            # Keep attribute available for callers, but don't overwrite with partials.
            try:
                self.last_timings = {
                    "poster_time": float(poster_time),
                    "total_time": float(total_time),
                    "centroids": int(len(results)),
                }
            except Exception:
                pass

        # Always emit a timing summary line so user can measure performance
        if emit_timing:
            try:
                print(f"[TIMING][CentroidProcessor] levels={params.get('levels')} min_area={params.get('min_area')} centroids={len(results)} poster_time={poster_time:.3f}s total_time={total_time:.3f}s")
            except Exception:
                try:
                    print(f"[TIMING][CentroidProcessor] centroids={len(results)} total_time={total_time:.3f}s")
                except Exception:
                    pass
        if DEBUG:
            print(f"[DEBUG][CentroidProcessor] get_centroids done: found {len(results)} centroids in {total_time:.2f}s")
        try:
            self.last_rim_points_proc = list(rim_results)
        except Exception:
            self.last_rim_points_proc = []
        return results