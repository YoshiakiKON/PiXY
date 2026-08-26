# PiXY Operation Manual (English)

---

## Cover Information

- Software: PiXY — Pixel-to-Stage XY Coordinate Converter
- Author: Yoshiaki KON
- Repository: https://github.com/YoshiakiKON/PiXY
- Zenodo DOI: 10.5281/zenodo.18174474
- License: MIT
- Last updated: 2026-08-16 (v1.5.2 spec)

---

## Version Notes (v1.5.2)

**Changes from v1.5.1:**
- UI parameter naming: "Grain Size Threshold" → "Particle Size Range (pix)" for clarity
- Unified control mode for Boundary Offset, Neck Separation, Shape Complexity (optional; can switch to individual mode)
- Configuration file-based parameter management (`pixy_settings.ini`)

**Notes for upgraders from v1.5.1:**
- No data-format changes; existing `.pixy` project files load without modification.
- UI terminology updated for consistency; functionality unchanged.

---

## Overview

- Purpose: Pair candidate points (centroids) in microscope images with instrument fiducials and convert image coordinates to stage coordinates to reduce instrument time and improve reproducibility.
- Intended users: Micro-area analysis operators, sample analysis staff, instrument administrators
- Main features: Image loading, particle detection (K-means + connected components), fiducial input, affine transform estimation, residual visualization, CSV export / clipboard copy
- Benefits: Earlier targeting preparation, less manual work at the instrument, and statistically analyzing more points

---

## Getting Started (Installation & Launch)

- System requirements
  - OS: Windows recommended (distributed EXE available). Linux/Mac may work when running from source.
  - Python: 3.8 or later (when running from source)
  - Disk and memory: Depends on image resolution (large images require more memory)
- Dependencies: See `requirements.txt` (PySide6, OpenCV, NumPy, etc.)
- Installation (from source)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Main.py
```

- Running the distributed EXE: After downloading, right-click → [Properties] → [Unblock] (if needed), then run.
- Note: When running from source, enable the virtual environment and match the versions in `requirements.txt`.

---

## Quick Start (Minimum Steps)

1. Run `PiXY_ver152.exe`.
2. Click `New Project` and select an image file to load.
3. Click `START Centroid Extraction`.
4. In extraction mode, adjust the left-side detection parameters (`Number of Groups`, `Boundary Offset`, `Neck Separation`, `Shape Complexity`, and `Particle Size Range` histogram) and confirm that centroids are detected.
5. (Optional but recommended) Enter a manual group name in the text field below each `Add GroupN` button.
6. Add candidate points from each group using `Add GroupN` buttons (left cards) so they appear in the center list.
7. If `Number of Groups` is changed and you recompute, manual group names are transferred to the nearest-color group automatically. When multiple names map to one group, the source group with more points is prioritized.
8. Click `Finish Centroid Extraction` to return to on-line alignment.
9. Click `Add Fiducial Point`, click fiducials on the image, and enter the corresponding stage coordinates in the table. Enter 3 or more points and check residuals. If residuals are large overall, add more points. If one point has a much larger residual, re-check that point or exclude it.
10. If everything looks good, export coordinates via `Export XYZ` or `Clipboard` and pass them to the instrument.

---

## UI Overview

- Main buttons
  - Upper left (project operations)
    - `New Project`: Start a new project and select an image file to load.
    - `Load Project`: Load a saved `.pixy` project (image, parameters, results).
    - `Save Project`: Save the current state as a `.pixy` project.

  - Middle left (fiducial operations)
    - `Add Fiducial Point`: Enter fiducial registration mode; click fiducial points on the image to add them to the table.
    - `Update XY`: Update the stage coordinates (`Stage X/Y/Z`) for the selected fiducial row.
    - `Clear` (fiducials): Delete the selected fiducial row in the table, or clear its values.

  - Lower left (particle detection parameter settings)
    - `START Centroid Extraction` / `Finish Centroid Extraction`: Enter/exit extraction mode.
    - Detection parameters are operated in `Advanced` mode only.
    - `Recalculation Trigger` (`Auto` / `Manual`): Control when particle detection is recomputed. `Auto` recomputes on every parameter change; `Manual` recomputes only when you click `ReCalculate`.
    - **Unified Control Mode** (new in v1.5.2): A checkbox allows switching between:
      - **Unified Control ON**: Adjust a single "Aggressiveness" slider (0-10) to scale all three parameters proportionally.
      - **Unified Control OFF**: Adjust individual parameters separately (fine-tuning mode).
    - `Number of Groups (K)` — Number of K-means clusters. Suggested 3–10 (try based on the number of colors; avoid over-segmentation).
    - `Boundary Offset` — Offset to avoid incomplete regions near the image boundary (e.g., exclude edges).
    - `Neck Separation` — Strength for separating touching particles (larger = stronger separation).
    - `Shape Complexity` — Tuning parameter to suppress/split regions with complex shapes.
    - `Particle Size Range (pix)` — Histogram-based min/max area selection (replaces "Grain Size Threshold").  **← RENAMED**
    - Group cards (bottom left)
      - `Add GroupN`: Add detected points of group N to the center list.
      - `Group name` input (under each `Add GroupN`): Assign a manual display name for that group.
      - `Show` / `Hide`: Per-group visibility toggle.
      - `Add ALL Group to List`: Add all groups to the center list.
      - When `Number of Groups` or posterization level changes, custom group visibility (`Show`/`Hide`) is reset and all groups are shown.
      - If `Number of Groups` changes and centroids are recomputed, manual group names are inherited by nearest group color; collisions are resolved by preferring the source group with more points.

  - Center (candidate table)
    - `Export XYZ`: Convert candidate points to stage coordinates (X,Y,Z) using the estimated transform and save as CSV.
    - `Clipboard`: Copy output data to the clipboard in a paste-ready tab-separated format (TSV) for instrument software.
    - `Add Target`: Manually add a target point on the image (use when you want to add a point not found by automatic extraction).
    - `Update u, v`: Update `u, v` (image coordinates) of the selected target row to the currently clicked position.
    - `Clear` (targets): Clear the target selection or any temporary designation.

    Note (`Add Target` / `Update u, v` with group visibility)
    - `Update u, v` (Update Target) is currently a beta feature. The internal behavior and UI may change in future versions.
    - Points added via `Add Target` are treated as manual targets and always belong to `Group 0`.

---

## Parameter Tuning Guide

### Centroid Detection Parameters (Quick Reference)

| Parameter | Typical Range | Recommended Default | Effect |
|-----------|---|---|---|
| `Number of Groups (K)` | 2–20 | 4–5 | Higher = more color groups segmented. Too high causes over-splitting. |
| `Boundary Offset` | 0–10 px | 0–2 | Exclude edges of the image; prevents incomplete regions. |
| `Neck Separation` | 0–10 | 0–5 | Higher = stronger separation of touching particles; slower. |
| `Shape Complexity` | 0–10 | 3–5 | Higher = retain fine details; lower = smooth contours. |
| `Particle Size Range (pix)` | 10–5000 px | 20–100 px | Filters small noise (below min) and large artifacts (above max). **← RENAMED** |

### Tuning Workflow

1. First adjust `K` and `Particle Size Range` (see recommended ranges above).
2. Confirm that desired particles are detected and unwanted noise is removed.
3. Only if many touching particles remain, tune `Neck Separation` etc. (Advanced parameters can be computationally expensive).
4. For manual recalculation, use `Manual` trigger mode to avoid lag while adjusting.

### Unified Control Mode (new in v1.5.2)

- Check "Unified Control" to enable a single "Aggressiveness" slider (0-10).
- This scales `Boundary Offset`, `Neck Separation`, and `Shape Complexity` together.
- Useful for rapid parameter sweeps; switch to individual mode for fine-tuning.
- Preset values are defined in `pixy_settings.ini` and can be customized.

---

## Fiducial Point Registration (Online Mode)

- Click `Add Fiducial Point` to enter registration mode.
- Click on each fiducial marker in the image.
- For each fiducial, enter its stage coordinates (X, Y, Z) in the table.
- Register at least 3 fiducials (3 non-collinear points define an affine transform).
- Review residuals (differences between predicted and observed positions):
  - If residuals are uniformly small, the transform is good.
  - If one point has a much larger residual than others, re-check that fiducial or exclude it.
- After confirming the transform, export target point coordinates via `Export XYZ`.

---

## Troubleshooting

### Centroids Not Detected or Noisy

1. Check `Particle Size Range` — adjust min/max to remove noise and retain particles.
2. Increase `K` to segment more color groups if particles are mixed with background.
3. Check image quality (contrast, sharpness) — PiXY performs best on high-contrast images.

### Centroids Detected but Fragments Incorrectly

1. Increase `Neck Separation` to separate touching particles.
2. Increase `Shape Complexity` to retain finer details (if particles have complex shapes).
3. Check `Boundary Offset` — ensure edges are not cutting off partial regions.

### Large Residuals After Fiducial Registration

1. Re-check stage coordinates; typos are a common source.
2. Add more fiducials (3–5 minimum) for robust estimation.
3. Check image quality and zoom — ensure fiducials are clearly visible.
4. Exclude outlier fiducials if one point has much larger residual.

---

## File Formats

### Project File (.pixy)

A `.pixy` file is a JSON-based project archive containing:
- Image data (encoded in base64)
- Detected target centroids and their metadata
- Fiducial points and their stage coordinates
- Parameter settings (zoom, rotation, extraction parameters)
- Extraction mode (Basic / Advanced)

### Export Format (CSV)

When you click `Export XYZ`, the output is a tab-separated CSV with columns:
```
Target_ID  Pixel_X  Pixel_Y  Stage_X  Stage_Y  Stage_Z
```

---

## Configuration File (pixy_settings.ini)

*New in v1.5.2*

`pixy_settings.ini` allows customization of:
- **Aggressiveness presets** (unified control mode): Each level 0-10 is mapped to a specific combination of `Boundary Offset`, `Neck Separation`, `Shape Complexity`.
- **UI defaults**: Default aggressiveness level, unified control enable/disable.

Example:
```ini
[Aggressiveness]
preset_0  = 0, 0, 3
preset_1  = 1, 0, 3
...
preset_5  = 5, 3, 5
...

[UI]
default_aggressiveness = 5
unified_control_default = True
```

Edit this file and restart PiXY for changes to take effect.

---

## Advanced Topics

### Batch Processing

PiXY is designed for interactive image analysis, not batch processing. For bulk extraction:
1. Prepare a list of image files.
2. For each, use the GUI interactively or write a custom Python script using the underlying modules (e.g., `CalcCentroid.py`).

### Custom Color Spaces

Currently, PiXY uses RGB color space by default. Support for other color spaces (e.g., HSV, Lab) may be added in future versions.

### Online Mode Alignment Algorithm

PiXY fits an **affine transformation** (2D rotation, scale, translation, and optional Z-plane equation) to map image coordinates (u, v) to stage coordinates (X, Y, Z). The fit uses least-squares minimization of residuals.

---

## Tips & Tricks

- **Keyboard shortcuts**: Hold Ctrl while using the scroll wheel to zoom in/out on the image.
- **Quick copy**: Use `Clipboard` button to copy tab-separated coordinates; paste directly into spreadsheet or instrument software.
- **Group naming**: Assign meaningful names (e.g., "matrix", "inclusion") to groups for easier documentation.
- **Saving projects**: Save early and often. A `.pixy` file preserves all your work and parameters.

---

## Citation

If you use PiXY in published work, please cite:

```
@software{KON_PiXY,
  author = {Yoshiaki KON},
  title = {PiXY: Pixel-to-Stage XY Coordinate Converter},
  url = {https://github.com/YoshiakiKON/PiXY},
  doi = {10.5281/zenodo.18174474},
  version = {1.5.2},
  year = {2026},
}
```

---

## License

PiXY is distributed under the MIT License. See `LICENSE` file for details.

---

**Manual Version**: 1.5.2  
**Last Updated**: 2026-08-16  
**Status**: Release

### TODO for v1.5.2 Release

- [ ] Update version number in this manual (currently "1.5.2 Draft")
- [ ] Update "Last updated" date to actual release date
- [ ] Update `.exe` filename references (if applicable)
- [ ] Test all UI screenshots/descriptions match actual UI
- [ ] Convert to HTML via Pandoc and verify formatting
- [ ] Commit to git and tag as `v1.5.2`
