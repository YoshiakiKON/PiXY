"""
Simple benchmark for CentroidProcessor.get_centroids.
Usage:
    py bench_centroids.py [image_path] [runs]
If image_path is omitted, looks for DemoBSE.png, DemoBMP.bmp, or last_image_path.txt.
"""
import sys
import os
import time
import numpy as np
import cv2
from Config import PROC_TARGET_WIDTH, load_last_image_path
from CalcCentroid import CentroidProcessor


def load_image_try(paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img, p
            except Exception:
                pass
    return None, None


def build_proc_img(img_full, target_width=PROC_TARGET_WIDTH):
    h, w = img_full.shape[:2]
    if w <= target_width:
        return img_full.copy(), 1.0
    scale = float(target_width) / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    proc = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return proc, (float(w) / float(new_w))


def main():
    args = sys.argv[1:]
    img_arg = args[0] if len(args) >= 1 else None
    runs = int(args[1]) if len(args) >= 2 else 5

    project_dir = os.path.dirname(__file__)
    candidates = []
    if img_arg:
        candidates.append(img_arg)
    candidates.extend([
        os.path.join(project_dir, 'DemoBSE.png'),
        os.path.join(project_dir, 'DemoBMP.bmp'),
        load_last_image_path()
    ])

    img_full, used = load_image_try(candidates)
    if img_full is None:
        print('No image found for benchmark. Provide a path or place DemoBSE.png in project root.')
        return

    proc_img, scale = build_proc_img(img_full)
    # scale_proc_to_full should be full / proc
    scale_proc_to_full = float(img_full.shape[1]) / float(proc_img.shape[1])
    print(f'Using image: {used} full={img_full.shape[1]}x{img_full.shape[0]} proc={proc_img.shape[1]}x{proc_img.shape[0]} scale_proc_to_full={scale_proc_to_full:.3f}')

    cp = CentroidProcessor(proc_img, scale_proc_to_full, img_full)
    params = {'levels': 6, 'min_area': 5, 'trim_px': 0}

    times = []
    for i in range(runs):
        t0 = time.monotonic()
        res = cp.get_centroids(params)
        dt = time.monotonic() - t0
        times.append(dt)
        print(f'Run {i+1}/{runs}: centroids={len(res)} time={dt:.3f}s')

    print(f'Average: {sum(times)/len(times):.3f}s over {runs} runs')


if __name__ == '__main__':
    main()
