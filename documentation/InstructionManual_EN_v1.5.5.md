# PiXY Operation Manual (English)

---

## Cover Information

- Software: PiXY — Pixel-to-Stage XY Coordinate Converter
- Author: Yoshiaki KON
- Repository: https://github.com/YoshiakiKON/PiXY
- Zenodo DOI: 10.5281/zenodo.18174474
- License: MIT
- Last updated: 2026-08-26 (v1.5.5)

---

## Version Notes (v1.5.5)

**Changes from v1.5.4:**
- Exported image markers now follow the same overlay payload used by the live UI, preventing stale centroid sets from being written into exported images.
- Marker color and label sizing were tuned to better match the on-screen appearance and reduce overlapping text on saved images.
- Online export remains in Image coordinate space, which preserves the original pixel positions used in the loaded image.

**Notes for upgraders from v1.5.4:**
- No project data format changes are required; existing `.pixy` project files remain compatible.
- The visible point set and the export output are now better aligned.

---

## Overview

- Purpose: Associate target points in microscopy images with instrument fiducials and convert image coordinates to stage coordinates to reduce instrument time and improve reproducibility.
- Intended users: Micro-area analysis operators, sample analysis staff, instrument administrators
- Main features: Image loading, particle detection (K-means + connected components), fiducial input, transform estimation, residual visualization, CSV export / clipboard copy
- Benefits: Earlier targeting preparation, less manual work at the instrument, and more consistent exported image records

---

## Getting Started (Installation & Launch)

- System requirements
  - OS: Windows recommended (distributed EXE available). Linux/Mac may work when running from source.
  - Python: 3.8 or later (when running from source)
  - Disk and memory: depends on image resolution (large images require more memory)
- Dependencies: See `requirements.txt` (PySide6, OpenCV, NumPy, etc.)
- Installation (from source)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Main.py
```

- Running the distributed EXE: After downloading, run the application. If Windows blocks the file, right-click and choose `Properties` → `Unblock`.
- Note: When running from source, enable the virtual environment and match the versions in `requirements.txt`.

---

## Quick Start (Minimum Steps)

1. Run `PiXY_ver155.exe`.
2. Click `New Project` and select an image file to load.
3. Click `START Centroid Extraction`.
4. In extraction mode, adjust the left-side detection parameters and confirm that centroids are detected.
5. Add candidate points from each group using `Add GroupN` buttons so they appear in the center list.
6. Finish centroid extraction to return to online alignment.
7. Click `Add Fiducial Point`, click fiducials on the image, and enter the corresponding stage coordinates in the table.
8. Enter 3 or more points and check residuals.
9. If everything looks good, export coordinates via `Export XYZ` or `Clipboard`.

---

## UI Overview

- Main buttons
  - Upper left (project operations)
    - `New Project`: Start a new project and select an image file to load.
    - `Load Project`: Load a saved `.pixy` project.
    - `Save Project`: Save the current project state as a `.pixy` file.

  - Middle left (fiducial operations)
    - `Add Fiducial Point`: Enter fiducial registration mode and click fiducial points on the image.
    - `Update XY`: Update the stage coordinates for the selected fiducial row.
    - `Clear`: Delete or clear fiducial rows.

  - Lower left (particle detection parameter settings)
    - `START Centroid Extraction` / `Finish Centroid Extraction`: Enter/exit extraction mode.
    - Detection parameters are operated in `Advanced` mode.
    - `Recalculation Trigger` (`Auto` / `Manual`): Controls when particle detection is recomputed.
    - `Number of Groups (K)`: K-means cluster count.
    - `Boundary Offset`: Excludes image-edge artifacts.
    - `Neck Separation`: Separates touching particles.
    - `Shape Complexity`: Filters irregular or complex shapes.
    - `Particle Size Range (pix)`: Filter by area range.

  - Center (candidate table)
    - `Export XYZ`: Export target point stage coordinates as CSV.
    - `Clipboard`: Copy output data to the clipboard as tab-separated text.
    - `Add Target`: Manually add a target point on the image.
    - `Update u, v`: Adjust the selected point position.
    - `Clear`: Clear the target selection or temporary designation.

---

## Coordinate System

PiXY supports two coordinate modes in the user interface:

- `Image` coordinates: pixel coordinates in the loaded image space.
- `Stage` coordinates: instrument stage coordinates after transformation estimation.

For image export, PiXY writes marker locations in the original image coordinate frame. This is deliberate: the exported image is meant to match the loaded image and preserve the visual positions of the target points seen by the user.

---

## Fiducial Point Registration (Online Mode)

- Click `Add Fiducial Point` to enter registration mode.
- Click on each fiducial marker in the image.
- Enter each stage coordinate (X, Y, Z) in the table.
- Register at least 3 fiducials.
- Review residuals and exclude outliers if needed.
- After confirming the transform, export target point coordinates via `Export XYZ`.

---

## File Formats

### Project File (.pixy)

A `.pixy` file is a JSON-based project archive containing:
- Image data
- Detected target centroids and metadata
- Fiducial points and stage coordinates
- Parameter settings
- Extraction mode information

### Export Format (CSV)

When you click `Export XYZ`, the output is a CSV with columns such as:

```
Target_ID  Pixel_X  Pixel_Y  Stage_X  Stage_Y  Stage_Z
```

---

## Configuration File (pixy_settings.ini)

`pixy_settings.ini` allows customization of:
- Aggressiveness presets used in unified control mode
- UI defaults for extraction behavior

---

## Troubleshooting

### Centroids not detected or too noisy

1. Check the particle size range.
2. Adjust `K` or segmentation parameters.
3. Improve contrast and image quality.

### Large residuals after fiducial registration

1. Re-check stage coordinates.
2. Add more fiducials.
3. Exclude an outlier fiducial if one point has a much larger residual.

---

## Documentation

- `InstructionManual_EN_v1.5.5.md`
- `InstructionManual_JP_v1.5.5.md`
- `RELEASE_NOTES_v1.5.5.md`

---

## Version

**Current release: v1.5.5**
