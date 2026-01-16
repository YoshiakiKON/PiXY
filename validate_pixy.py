#!/usr/bin/env python3
"""
validate_pixy.py - Batch validation script for PiXY centroid detection

Process all local BSE images, compute statistics, and generate validation report.
"""

import os
import cv2
import numpy as np
import time
import json
from pathlib import Path

# Assuming CalcCentroid is available
try:
    from CalcCentroid import CentroidProcessor
    from Util import kmeans_posterize
except ImportError:
    print("Error: CalcCentroid module not found. Ensure running from project root.")
    exit(1)

# ========== CONFIGURATION ==========
IMAGE_DIR = "bse_samples"  # Change to your local BSE image folder
SCALE_MICROMETERS_PER_PIXEL = 3.33  # From 1800px = 6mm = 6000μm
OUTPUT_REPORT = "validation_report.json"

# PiXY standard parameters for BSE images
DEFAULT_PARAMS = {
    "levels": 4,          # Number of color groups (K-means clusters)
    "min_area": 30,       # Minimum grain area (pixels)
    "max_area": 5000,     # Maximum grain area (pixels)
    "trim_px": 0,         # Boundary erosion
    "neck_separation": 0  # Neck splitting strength
}

# ========== MAIN VALIDATION ROUTINE ==========

def process_single_image(img_path, params, scale_um_per_px):
    """
    Process a single BSE image and extract centroids + statistics.
    
    Returns:
        dict with statistics
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Failed to load: {img_path}")
        return None
    
    h_orig, w_orig = img.shape[:2]
    
    # Resize for processing (standard 512×512 for speed)
    proc_size = 512
    scale_proc_to_full = max(h_orig, w_orig) / proc_size
    proc_img = cv2.resize(img, (proc_size, proc_size))
    
    # Measure processing time
    t_start = time.time()
    
    # Initialize processor
    processor = CentroidProcessor(
        proc_img=proc_img,
        scale_proc_to_full=scale_proc_to_full,
        img_full=img
    )
    
    # Get centroids
    centroids = processor.get_centroids(params, poster=None)
    
    t_elapsed = time.time() - t_start
    
    # Compute statistics
    if len(centroids) > 0:
        # Centroids: [group_no, cx, cy]
        areas_px = processor.last_component_areas if hasattr(processor, 'last_component_areas') else []
        
        if areas_px:
            areas_um2 = [a * (scale_um_per_px ** 2) for a in areas_px]
            grain_sizes_um = [np.sqrt(4 * a / np.pi) for a in areas_um2]  # Approximate diameter
            
            stats = {
                "particle_count": len(centroids),
                "grain_size_mean_um": float(np.mean(grain_sizes_um)),
                "grain_size_std_um": float(np.std(grain_sizes_um)),
                "grain_size_min_um": float(np.min(grain_sizes_um)),
                "grain_size_max_um": float(np.max(grain_sizes_um)),
                "areas_um2": [float(a) for a in areas_um2[:10]],  # First 10 for reference
            }
        else:
            stats = {
                "particle_count": len(centroids),
                "grain_size_mean_um": None,
                "grain_size_std_um": None,
            }
    else:
        stats = {
            "particle_count": 0,
            "grain_size_mean_um": None,
        }
    
    stats["processing_time_sec"] = round(t_elapsed, 3)
    stats["image_size_px"] = [w_orig, h_orig]
    
    return stats


def validate_all_images(image_dir, params, scale_um_per_px):
    """
    Process all images in directory.
    
    Returns:
        List of result dicts
    """
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' not found.")
        return []
    
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.tiff', '.tif', '.png', '.bmp', '.jpg', '.jpeg'))
    ])
    
    if not image_files:
        print(f"Warning: No image files found in '{image_dir}'")
        return []
    
    print(f"\n🔬 Processing {len(image_files)} BSE images...")
    print(f"   Scale: {scale_um_per_px:.2f} μm/pixel")
    print(f"   Params: levels={params['levels']}, min_area={params['min_area']}\n")
    
    results = []
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        print(f"  [{i:2d}/{len(image_files)}] {img_file}...", end=" ", flush=True)
        
        stat = process_single_image(img_path, params, scale_um_per_px)
        if stat:
            stat["filename"] = img_file
            results.append(stat)
            print(f"✓ {stat['particle_count']} particles, {stat['processing_time_sec']:.2f}s")
        else:
            print("✗")
    
    return results


def compute_aggregate_statistics(results):
    """
    Compute mean and standard deviation across all images.
    """
    if not results:
        return {}
    
    particle_counts = [r['particle_count'] for r in results]
    times = [r['processing_time_sec'] for r in results]
    grain_sizes = [r['grain_size_mean_um'] for r in results if r['grain_size_mean_um'] is not None]
    
    agg = {
        "total_images": len(results),
        "particle_count_mean": float(np.mean(particle_counts)),
        "particle_count_std": float(np.std(particle_counts)),
        "particle_count_range": [int(np.min(particle_counts)), int(np.max(particle_counts))],
        "processing_time_mean_sec": float(np.mean(times)),
        "processing_time_std_sec": float(np.std(times)),
        "grain_size_mean_um": float(np.mean(grain_sizes)) if grain_sizes else None,
        "grain_size_std_um": float(np.std(grain_sizes)) if grain_sizes else None,
    }
    
    return agg


def print_report(results, aggregate):
    """
    Print formatted validation report.
    """
    print("\n" + "=" * 70)
    print("📊 VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\n✓ Processed: {aggregate['total_images']} images")
    print(f"\n📈 Particle Detection Statistics:")
    print(f"   Count per image:  {aggregate['particle_count_mean']:.1f} ± {aggregate['particle_count_std']:.1f}")
    print(f"   Range:            {aggregate['particle_count_range'][0]} - {aggregate['particle_count_range'][1]}")
    
    if aggregate['grain_size_mean_um'] is not None:
        print(f"\n🔹 Grain Size Statistics:")
        print(f"   Mean diameter:    {aggregate['grain_size_mean_um']:.2f} ± {aggregate['grain_size_std_um']:.2f} μm")
    
    print(f"\n⏱️  Processing Performance:")
    print(f"   Time per image:   {aggregate['processing_time_mean_sec']:.3f} ± {aggregate['processing_time_std_sec']:.3f} sec")
    print(f"   Total time:       {aggregate['processing_time_mean_sec'] * aggregate['total_images']:.1f} sec")
    
    print("\n" + "=" * 70)
    print(f"📁 Detailed results saved to: {OUTPUT_REPORT}")
    print("=" * 70 + "\n")


# ========== MAIN ==========

if __name__ == "__main__":
    print("\n🚀 PiXY Validation Script")
    print(f"   Image directory: {IMAGE_DIR}")
    
    # Process all images
    results = validate_all_images(IMAGE_DIR, DEFAULT_PARAMS, SCALE_MICROMETERS_PER_PIXEL)
    
    if results:
        # Compute aggregate stats
        agg = compute_aggregate_statistics(results)
        
        # Save detailed report
        report = {
            "metadata": {
                "script": "validate_pixy.py",
                "image_dir": IMAGE_DIR,
                "scale_um_per_px": SCALE_MICROMETERS_PER_PIXEL,
                "pixy_params": DEFAULT_PARAMS,
            },
            "per_image": results,
            "aggregate": agg,
        }
        
        with open(OUTPUT_REPORT, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print_report(results, agg)
        
        # Print markdown table for paper
        print("\n📋 Markdown table for JOSS paper:\n")
        print("| Image | Particle Count | Mean Grain Size (μm) | Processing Time (s) |")
        print("|-------|----------------|----------------------|---------------------|")
        for r in results:
            grain_str = f"{r['grain_size_mean_um']:.2f}" if r['grain_size_mean_um'] else "N/A"
            print(f"| {r['filename']} | {r['particle_count']} | {grain_str} | {r['processing_time_sec']:.3f} |")
        
        print(f"\n**Aggregate**: {agg['particle_count_mean']:.1f} ± {agg['particle_count_std']:.1f} particles/image")
        print(f"**Mean Processing Time**: {agg['processing_time_mean_sec']:.3f} ± {agg['processing_time_std_sec']:.3f} sec")
    else:
        print("\n✗ No valid results generated.")
