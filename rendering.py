from qt_compat.QtCore import QPoint
from qt_compat.QtGui import QPixmap, QPainter, QPen, QColor
from Util import cvimg_to_qpixmap
import cv2


def build_zoomed_canvas(overlay_full_img, proc_zoom, view_padding,
                        centroids, selected_index, ref_points, scale_proc_to_full,
                        colors=None, interp_mode='auto'):
    """
    入力: 
      - overlay_full_img: numpy (H,W,3) BGR
      - proc_zoom: float
      - view_padding: int
      - centroids: list[(g, x_proc, y_proc)]
      - selected_index: Optional[int]
      - ref_points: list[Optional[(x_proc,y_proc)]]
      - scale_proc_to_full: float
      - colors: dict オプション
    出力:
      - pm: QPixmap キャンバス（余白＋画像＋マーカー済み）
      - display_offset: (off_x, off_y)
      - display_img_size: (new_w, new_h)
    """
    if overlay_full_img is None:
        return None, (0, 0), (0, 0)
    h, w = overlay_full_img.shape[:2]
    # Allow large proc_zoom values; compute desired pixel count using floats to avoid huge ints
    z = max(0.001, float(proc_zoom))
    # estimate desired pixel count (float) and decide whether to downsample for display
    # Safety cap for displayed pixel count.
    # NOTE: If the image is large, zoom can become "visually capped" because we downsample once
    # desired pixel count exceeds this. Raising this increases the maximum effective magnification.
    # Display safety cap for rendered pixel count.
    # Higher cap => higher effective zoom but heavier CPU/RAM.
    # 6144^2 is a compromise: noticeably higher than the original 4096^2 without the large slowdown of 8192^2.
    MAX_PIXELS = 6144 * 6144
    desired_pixels = float(w) * float(h) * (z * z)
    # decide interpolation method
    # interp_mode: 'auto'|'nearest'|'linear'
    use_nearest = False
    if interp_mode == 'nearest':
        use_nearest = True
    elif interp_mode == 'linear':
        use_nearest = False
    else:
        # auto: if zoom is large (enlarging), prefer nearest to preserve pixel blocks
        use_nearest = (z > 1.5)

    if desired_pixels > MAX_PIXELS:
        scale_down = (MAX_PIXELS / float(desired_pixels)) ** 0.5
        # compute draw size directly from w * z * scale_down to avoid constructing huge intermediate sizes
        draw_w = max(1, int(round(float(w) * z * scale_down)))
        draw_h = max(1, int(round(float(h) * z * scale_down)))
        interp = cv2.INTER_LINEAR if not use_nearest else cv2.INTER_NEAREST
        img_resized = cv2.resize(overlay_full_img, (draw_w, draw_h), interpolation=interp)
        downsampled = True
        # ds_factor maps drawn pixels to the logical zoomed size (w*z)
        ds_factor = float(draw_w) / (float(w) * z)
    else:
        draw_w = max(1, int(round(float(w) * z)))
        draw_h = max(1, int(round(float(h) * z)))
        interp = cv2.INTER_NEAREST if use_nearest else cv2.INTER_LINEAR
        img_resized = cv2.resize(overlay_full_img, (draw_w, draw_h), interpolation=interp)
        downsampled = False
        ds_factor = 1.0

    pad = int(view_padding)
    canvas_w = img_resized.shape[1] + 2 * pad
    canvas_h = img_resized.shape[0] + 2 * pad
    pm = QPixmap(canvas_w, canvas_h)
    pm.fill(QColor(30, 30, 30))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # draw resized pixmap; if downsampled we will still compute coordinates scaled to the displayed size
    qpix = cvimg_to_qpixmap(img_resized)
    painter.drawPixmap(pad, pad, qpix)

    # Compute offsets in logical (label) coordinates. If the image was downsampled,
    # a physical pad of 'pad' pixels corresponds to pad * (1/ds_factor) logical pixels.
    # For consistent mapping between the QPixmap contents and the UI coordinate
    # conversions, return the physical pixel offset of the drawn image inside
    # the pixmap. The label/UI expects the pad value (in pixmap coordinates)
    # so use pad directly here. Marker coordinates should be computed using
    # the display scale (draw_w / original_width).
    off_x = int(round(pad))
    off_y = int(round(pad))
    # display_scale maps full-image pixels -> displayed (physical) pixels
    display_scale = float(draw_w) / float(w) if w != 0 else 1.0
    # マーカー色/半径
    cfg = {
        'pen_width': 2,
        'centroid_fill': QColor(64, 64, 64),
        'centroid_radius': 4,
        'selected_fill': QColor(0, 102, 255),
        'selected_radius': 6,
        'ref_fill': QColor(255, 0, 0),
        'ref_radius': 6,
    }
    if colors:
        cfg.update(colors)

    # 1) 通常重心
    if centroids:
        for idx, (_, xp, yp) in enumerate(centroids):
            xf = xp * scale_proc_to_full
            yf = yp * scale_proc_to_full
            # use display_scale for mapping full-image coords to physical pixels
            xd = int(round(xf * display_scale)) + off_x
            yd = int(round(yf * display_scale)) + off_y
            if selected_index is not None and idx == selected_index:
                continue
            painter.setPen(QPen(QColor(255, 255, 255), cfg['pen_width']))
            painter.setBrush(cfg['centroid_fill'])
            painter.drawEllipse(QPoint(xd, yd), cfg['centroid_radius'], cfg['centroid_radius'])

    # 2) Ref
    for pt in (ref_points or []):
        if not pt:
            continue
        x_proc, y_proc = pt
        xf = x_proc * scale_proc_to_full
        yf = y_proc * scale_proc_to_full
        xd = int(round(xf * display_scale)) + off_x
        yd = int(round(yf * display_scale)) + off_y
        painter.setPen(QPen(QColor(255, 255, 255), cfg['pen_width']))
        painter.setBrush(cfg['ref_fill'])
        painter.drawEllipse(QPoint(xd, yd), cfg['ref_radius'], cfg['ref_radius'])

    # 3) 選択
    if centroids and selected_index is not None and 0 <= selected_index < len(centroids):
        _, xp, yp = centroids[selected_index]
        xf = xp * scale_proc_to_full
        yf = yp * scale_proc_to_full
        xd = int(round(xf * display_scale)) + off_x
        yd = int(round(yf * display_scale)) + off_y
        painter.setPen(QPen(QColor(255, 255, 255), cfg['pen_width']))
        painter.setBrush(cfg['selected_fill'])
        painter.drawEllipse(QPoint(xd, yd), cfg['selected_radius'], cfg['selected_radius'])

    painter.end()
    # return actual drawn image physical size (img_resized) and logical display size (width,height)
    # physical drawn size (in pixels inside the pixmap)
    draw_w = img_resized.shape[1]
    draw_h = img_resized.shape[0]
    return pm, (off_x, off_y), (draw_w, draw_h)


def draw_crosshair(base_pixmap, display_offset, display_img_size, pos_label,
                   outline_color=QColor(0, 0, 0), outline_width=4,
                   line_color=QColor(255, 255, 255), line_width=2):
    """白＋黒縁取りの十字線をベースPixmapに描いて返す。"""
    if base_pixmap is None:
        return None
    pad_x, pad_y = display_offset
    w, h = display_img_size
    if w <= 0 or h <= 0:
        return base_pixmap
    x = min(max(pos_label.x(), pad_x), pad_x + w - 1)
    y = min(max(pos_label.y(), pad_y), pad_y + h - 1)
    pm2 = QPixmap(base_pixmap)
    painter = QPainter(pm2)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(outline_color, outline_width))
    painter.drawLine(pad_x, y, pad_x + w - 1, y)
    painter.drawLine(x, pad_y, x, pad_y + h - 1)
    painter.setPen(QPen(line_color, line_width))
    painter.drawLine(pad_x, y, pad_x + w - 1, y)
    painter.drawLine(x, pad_y, x, pad_y + h - 1)
    painter.end()
    return pm2
