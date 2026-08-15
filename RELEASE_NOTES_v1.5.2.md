# PiXY v1.5.2 — Release Notes (2026-08-16)

## Overview

PiXY is a desktop GUI tool for image-based centroid extraction and pixel-to-stage coordinate conversion.

**Recommended screen size**: 1200×900 or larger.

## What's Included

- Standalone Windows executable: `PiXY_ver152.exe`
- Source code (Python)

## What's New in v1.5.2

This release is a UI polish and code cleanup update based on v1.5.1.

### Parameter naming: "Grain Size Threshold" → "Particle Size Range (pix)"

The particle area filter parameter has been renamed throughout the UI for clarity:

- Histogram title: **"Particle Size Range (pix)"** (was "Grain Size Threshold (pix)")
- Slider label: **"Particle Size Range (pix)"** (was "Minimum Grain Area (pix)")

Functionality is unchanged; only the display names were updated.

### Parameter naming: "Number of Groups (K)" label consistent

The `Number of Groups (K)` label is now shown consistently in the centroid extraction panel. The internal `display_labels` entry was also updated to match.

### Number of Groups maximum reduced to 10

The slider maximum for **Number of Groups** has been reduced from 20 to 10. Values above 10 are rarely useful for typical microscope images and can cause slow computation.

### Dead code removal: poster_level / slider_levels

The deprecated `poster_level` slider and its associated widgets, signal handlers, and methods (`_wire_levels`, `_on_levels_slider_changed`, `_on_levels_edit_finished`, `_nudge_levels`) have been removed. The active `Number of Groups` control (`slider_num_groups`) is the sole parameter for K-means cluster count.

## Notes for Upgraders

If you are coming from v1.5.1:
- No data-format changes; existing `.pixy` project files load without modification.
- The EXE is renamed to `PiXY_ver152.exe`.
- UI parameter names changed (cosmetic only; no effect on saved projects or exports).

## Known Issues / Limitations

- The bundled EXE is large (Qt, NumPy, OpenCV, and related dependencies).
- Very large images may require additional memory.

## Documentation

See the v1.5.1 documentation set for the full manual. Updated v1.5.2 manuals are available as draft files (`InstructionManual_EN_v1.5.2.md`, `InstructionManual_JP_v1.5.2.md`).

## Citation

If you use PiXY in published work, please cite the software.
- See `CITATION.cff` for the recommended citation metadata.
- DOI: https://doi.org/10.5281/zenodo.18174474
