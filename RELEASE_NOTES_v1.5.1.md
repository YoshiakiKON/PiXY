# PiXY v1.5.1 — Release Notes (2026-08-10)

## Overview

PiXY is a desktop GUI tool for image-based centroid extraction and pixel-to-stage coordinate conversion.

**Recommended screen size**: 1200×900 or larger.

## What's Included

- Standalone Windows executable: `PiXY_ver151.exe`
- Source code (Python)

## What's New in v1.5.1

This release is a focused quality-and-correctness update based on v1.5.0.

### Overlay rendering architecture (problem A / problem B separation)

The image overlay is now built in three independent layers:

1. **Base image** — the background image with rotation and grid.
2. **All-points overlay** — all registered Target Points and Fiducial Points drawn at normal size, rebuilt when the list changes.
3. **Selection highlight** — the currently selected point drawn at the foreground at larger size, rebuilt only when the selection changes.

Previously these layers were mixed, which caused two bugs:

- **Problem A (add lag)**: Adding a point to the list did not show it immediately, because the base layer did not include newly added points.
- **Problem B (residual highlight)**: Changing the selection sometimes left the previous selection marker visible.

Both are now fixed by the layer separation.

### Manual Rim target: point appears immediately after add

When adding a Target Point with the **Rim** position selected, the point now appears on the image immediately after clicking. Previously, manual Rim targets had no separate rim coordinate, causing the snapshot to return `None` and the row to be skipped.

### Online — Target Points XYZ computed from stage transform

In Online Alignment mode, the Target Points table now shows X, Y, Z columns computed from the fitted stage transform. The values are updated automatically whenever the stage transform is recalculated (fiducial edits, zoom, etc.).

### Fiducial Show/Hide triggers Target XYZ recompute

Toggling a Fiducial point between visible and hidden in the Online table now immediately recomputes the stage transform and updates the Target Points XYZ.

### Fiducial exclusion applied to stage transform calculation

Fiducial points that are toggled to hidden are now correctly excluded from the stage transform fitting (previously they were included regardless of their visibility state).

### New Project resets more parameters

New Project now resets Trim, Neck Separation, Shape Complexity sliders, Image Rotate, and Flip to their default values, in addition to clearing point lists.

### Fiducial point labels on the image overlay

Fiducial points are now labeled **Fid. 1**, **Fid. 2**, … on the image overlay instead of bare numbers.

## Notes for Upgraders

If you are coming from v1.5.0:
- No data-format changes; existing `.pixy` project files load without modification.
- The EXE is renamed to `PiXY_ver151.exe`.

## Known Issues / Limitations

- The bundled EXE is large (Qt, NumPy, OpenCV, and related dependencies).
- Very large images may require additional memory.

## Documentation

See the v1.5.0 documentation set for the full manual.

## Citation

If you use PiXY in published work, please cite the software.
- See `CITATION.cff` for the recommended citation metadata.
- DOI: https://doi.org/10.5281/zenodo.18174474
