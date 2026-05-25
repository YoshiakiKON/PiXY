# PiXY v1.3.3 — Release Notes (2026-05-26)

## Overview

PiXY is a desktop GUI application for extracting target centroids from microscopy images and converting image coordinates (u, v) to instrument stage coordinates (X, Y, Z) using fiducial-based alignment.

## What is new in v1.3.3

### Performance
- Major speedup for Neck Separation during centroid detection.
- Split processing now runs on per-component ROI masks instead of full-frame masks.
- Marker propagation was replaced with OpenCV distance-transform label assignment.
- Components smaller than `min_area` skip neck split work early.

### UI behavior and stability
- Reduced apparent center jump when adding Target/Fiducial points at high zoom.
- Improved consistency of add-point continuation behavior.

## Packaging

- Windows single-file executable is generated with `build_exe.ps1`.
- Output binary: `dist/PiXY.exe`

## Notes

- Citation metadata and project version were updated to `1.3.3`.
- See `CHANGELOG.md` for details.
