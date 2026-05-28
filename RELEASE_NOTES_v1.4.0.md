# PiXY v1.4.0 — Release Notes (2026-05-28)

## Overview

PiXY is a desktop GUI application for extracting target centroids from microscopy images and converting image coordinates (u, v) to instrument stage coordinates (X, Y, Z) using fiducial-based alignment.

## Highlights in v1.4.0

### Workflow redesign (Centroid Extraction mode)
- Added a Start/Finish Centroid Extraction workflow.
- Left workflow tabs are now hidden in regular operation; the Start/Finish button controls mode switching.
- Default view shows On-line Alignment contents.
- Entering extraction mode opens the Off-line targeting surface and enables extraction-specific controls.

### Display behavior consistency
- In normal mode, display is fixed to:
  - Boundary: OFF
  - Display Mode: Original
- In extraction mode, Boundary and Display Mode are restored from saved extraction preferences.
- Extraction display preferences are saved and reused across app launches.

### Left panel and group-list refinements
- Removed the Auto-detect title row for a cleaner parameter area.
- Simplified group cards:
  - removed Add List button,
  - group header is now an action button (`Add GroupN`),
  - retained Show/Hide toggle for per-group visibility.
- Multiple spacing and sizing fixes were applied to improve visual stability.

### Interaction and stability fixes
- Fixed center-table selection behavior during extraction mode so it does not select unrelated points on the image.
- Fixed Update u,v cursor/crosshair behavior by consistently handling `center_uv_update` as a pick mode.

## Packaging

- Windows single-file executable is built with `build_exe.ps1`.
- Output binary for this release: `dist/PiXY_ver140.exe`

## Notes

- Project version is now `1.4.0`.
- See `CHANGELOG.md` for detailed changes.
