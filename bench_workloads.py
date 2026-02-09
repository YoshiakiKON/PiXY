"""bench_workloads.py

K-Meansポスタリゼーション / 粒子検出 / 重心計算 の計算負荷(時間)を比較するベンチ。

PiXY の既存実装をそのまま呼び出し、同一画像・同一パラメータで以下を計測します。
  1) K-Meansポスタリゼーション (Util.kmeans_posterize)
  2) 粒子検出 (マスク生成 + connected components + neck separation + split CC)
  3) 重心計算 (結果リストへの重心追加)

Usage:
  py bench_workloads.py [image_path] [runs]

Options (environment-agnostic, simplest CLI):
  - 画像パス省略時は DemoBSE.png / DemoBMP.bmp / last_image_path.txt を探索

Note:
  - OpenCVのkmeansは乱数を使うため、Util.kmeans_posterize側で固定シード設定済み。
  - 計時は time.perf_counter を使用。
"""

import os
import sys
import time
import statistics
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from Config import PROC_TARGET_WIDTH, load_last_image_path
from Util import kmeans_posterize
from CalcCentroid import CentroidProcessor


def _load_image_try(paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for p in paths:
        if p and os.path.exists(p):
            try:
                # Windowsパス/日本語パス対応: fromfile + imdecode
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img, p
            except Exception:
                pass
    return None, None


def _build_proc_img(img_full: np.ndarray, target_width: int = PROC_TARGET_WIDTH) -> Tuple[np.ndarray, float]:
    """アプリと同じ思想: 表示/処理のため横幅を抑える。返すscaleは full/proc。"""
    h, w = img_full.shape[:2]
    if w <= int(target_width):
        return img_full.copy(), 1.0
    scale = float(target_width) / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    proc = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_proc_to_full = float(w) / float(new_w)
    return proc, scale_proc_to_full


def _fmt_ms(x: float) -> str:
    return f"{x * 1000.0:.2f} ms"


def _summ(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "stdev": float("nan"), "min": float("nan"), "max": float("nan")}
    if len(vals) == 1:
        return {"mean": vals[0], "stdev": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }


def _pop_flag(args: List[str], name: str) -> bool:
    try:
        i = args.index(name)
    except ValueError:
        return False
    args.pop(i)
    return True


def _pop_opt(args: List[str], name: str, cast_fn, default):
    try:
        i = args.index(name)
    except ValueError:
        return default
    if i + 1 >= len(args):
        return default
    try:
        v = cast_fn(args[i + 1])
    except Exception:
        v = default
    # remove option + value
    args.pop(i + 1)
    args.pop(i)
    return v


def _get_breakdown_keys() -> List[str]:
    # Store in a consistent order for printing and summaries.
    return [
        "unique_colors_time",
        "mask_time",
        "cc_time",
        "comp_mask_time",
        "split_time",
        "split_cc_time",
        "centroid_time",
        "boundary_time",
    ]


def main() -> int:
    args = sys.argv[1:]

    # Minimal flags (keep parsing intentionally simple)
    no_boundary = _pop_flag(args, "--no-boundary")
    levels = _pop_opt(args, "--levels", int, 6)
    min_area = _pop_opt(args, "--min-area", int, 5)
    trim_px = _pop_opt(args, "--trim", int, 0)
    neck_sep = _pop_opt(args, "--neck", int, 0)
    proc_width = _pop_opt(args, "--proc-width", int, PROC_TARGET_WIDTH)
    runs = _pop_opt(args, "--runs", int, 10)

    img_arg = args[0] if len(args) >= 1 else None
    if len(args) >= 2:
        # Backward-compatible positional runs
        try:
            runs = int(args[1])
        except Exception:
            pass

    # ベンチ条件
    params = {
        "levels": int(levels),
        "min_area": int(min_area),
        "max_area": None,
        "trim_px": int(trim_px),
        "neck_separation": int(neck_sep),
    }

    project_dir = os.path.dirname(__file__)
    candidates: List[str] = []
    if img_arg:
        candidates.append(img_arg)
    candidates.extend(
        [
            os.path.join(project_dir, "DemoBSE.png"),
            os.path.join(project_dir, "DemoBMP.bmp"),
            load_last_image_path(),
        ]
    )

    img_full, used = _load_image_try(candidates)
    if img_full is None:
        print("No image found for benchmark. Provide a path or place DemoBSE.png in project root.")
        return 2

    proc_img, scale_proc_to_full = _build_proc_img(img_full, target_width=int(proc_width))

    print(
        "Using image:\n"
        f"  path: {used}\n"
        f"  full: {img_full.shape[1]}x{img_full.shape[0]}\n"
        f"  proc: {proc_img.shape[1]}x{proc_img.shape[0]} (PROC_TARGET_WIDTH={PROC_TARGET_WIDTH})\n"
        f"  scale_proc_to_full: {scale_proc_to_full:.3f}\n"
        f"  params: levels={params['levels']} min_area={params['min_area']} trim_px={params['trim_px']} neck_sep={params['neck_separation']}\n"
        f"  boundary_mask: {'OFF' if no_boundary else 'ON'}\n"
        f"  runs: {runs}"
    )

    cp = CentroidProcessor(proc_img, scale_proc_to_full, img_full)

    # Warmup（初回の内部キャッシュ/メモリ確保を避ける）
    try:
        poster_w = kmeans_posterize(proc_img, params["levels"])
        cp.get_centroids(params, poster=poster_w, collect_timings=False, emit_timing=False)
    except Exception:
        pass

    t_kmeans: List[float] = []
    t_detect: List[float] = []
    t_cent: List[float] = []
    t_boundary: List[float] = []
    t_total_centroids: List[float] = []
    breakdown: Dict[str, List[float]] = {k: [] for k in _get_breakdown_keys()}

    for i in range(runs):
        # 1) K-Means posterization
        t0 = time.perf_counter()
        poster = kmeans_posterize(proc_img, params["levels"])
        dt_k = time.perf_counter() - t0
        t_kmeans.append(dt_k)

        # 2) 粒子検出 + 3) 重心計算（内訳タイミングを CalcCentroid から取得）
        t1 = time.perf_counter()
        res = cp.get_centroids(
            params,
            poster=poster,
            collect_timings=True,
            emit_timing=False,
            compute_boundary_mask=(not no_boundary),
        )
        dt_all = time.perf_counter() - t1
        t_total_centroids.append(dt_all)

        info = getattr(cp, "last_timings", {}) or {}
        t_detect.append(float(info.get("particle_detection_time", 0.0)))
        t_cent.append(float(info.get("centroid_calc_time", 0.0)))
        t_boundary.append(float(info.get("boundary_time", 0.0)))
        for k in breakdown.keys():
            breakdown[k].append(float(info.get(k, 0.0)))

        # Detailed breakdown line (dominant contributors)
        cm = float(info.get("comp_mask_time", 0.0))
        cc = float(info.get("cc_time", 0.0))
        mk = float(info.get("mask_time", 0.0))
        sp = float(info.get("split_time", 0.0))

        print(
            f"Run {i+1:2d}/{runs}: "
            f"kmeans={_fmt_ms(dt_k)} | "
            f"detect={_fmt_ms(t_detect[-1])} (comp_mask={_fmt_ms(cm)}, cc={_fmt_ms(cc)}, mask={_fmt_ms(mk)}, split={_fmt_ms(sp)}) | "
            f"centroid={_fmt_ms(t_cent[-1])} | "
            f"boundary={_fmt_ms(t_boundary[-1])} | "
            f"get_centroids_total={_fmt_ms(dt_all)} | "
            f"centroids={len(res)}"
        )

    sk = _summ(t_kmeans)
    sd = _summ(t_detect)
    sc = _summ(t_cent)
    sb = _summ(t_boundary)
    st = _summ(t_total_centroids)

    print("\n=== Summary (mean ± stdev) ===")
    print(f"K-Means posterize: {_fmt_ms(sk['mean'])} ± {_fmt_ms(sk['stdev'])}")
    print(f"Particle detect:   {_fmt_ms(sd['mean'])} ± {_fmt_ms(sd['stdev'])}")
    print(f"Centroid calc:     {_fmt_ms(sc['mean'])} ± {_fmt_ms(sc['stdev'])}")
    print(f"Boundary extract:  {_fmt_ms(sb['mean'])} ± {_fmt_ms(sb['stdev'])}")
    print(f"get_centroids():   {_fmt_ms(st['mean'])} ± {_fmt_ms(st['stdev'])}")

    # Breakdown summary
    print("\n=== Breakdown inside get_centroids() (mean ± stdev) ===")
    for k in _get_breakdown_keys():
        s = _summ(breakdown.get(k, []))
        print(f"{k:18s}: {_fmt_ms(s['mean'])} ± {_fmt_ms(s['stdev'])}")

    denom = sk["mean"] + sd["mean"] + sc["mean"]
    if denom > 0:
        print("\n=== Share (kmeans + detect + centroid) ===")
        print(f"K-Means:  {100.0 * sk['mean'] / denom:.1f}%")
        print(f"Detect:   {100.0 * sd['mean'] / denom:.1f}%")
        print(f"Centroid: {100.0 * sc['mean'] / denom:.1f}%")

    # Share inside get_centroids
    denom2 = sd["mean"] + sc["mean"] + (0.0 if no_boundary else sb["mean"])
    if denom2 > 0:
        print("\n=== Share inside get_centroids() (detect + centroid + boundary) ===")
        print(f"Detect:   {100.0 * sd['mean'] / denom2:.1f}%")
        print(f"Centroid: {100.0 * sc['mean'] / denom2:.1f}%")
        if not no_boundary:
            print(f"Boundary: {100.0 * sb['mean'] / denom2:.1f}%")

    print("\nNote: Boundary extract is extra work for UI overlay; use --no-boundary for pure detection/centroid cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
