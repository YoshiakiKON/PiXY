# PiXY v1.2.0 — Release Notes (2026-01-29)

## Overview

PiXY is a desktop GUI tool for image-based centroid extraction and pixel→stage coordinate conversion.

Typical use cases:
- Detect particle/grain-like regions in microscope/SEM images and extract their centroids.
- Convert centroid pixel coordinates to stage coordinates using user-defined reference points.
- Export results for downstream measurement workflows (e.g., stage navigation, automated acquisition).

**Recommended screen size**: 1200×900 or larger (the UI is more comfortable at this size).

## Overview figure

![PiXY overview workflow](documentation/images/workflow_v2.svg)

## What’s Included

- Standalone Windows executable: `PiXY_ver120.exe`
- Source code (Python)
- Documentation and submission materials (JOSS draft, citation metadata)

## Installation & Quick Start

### Option A: Standalone Windows executable
1. Download `PiXY_ver120.exe`.
2. Run it (no installer).
3. Open an image via **Open Image**.
4. (Optional but recommended) Add reference points via **Add Ref** (≥3 non-collinear points) to enable pixel→stage conversion.
5. Adjust detection parameters (e.g., number of groups, area thresholds) and run grain identification.
6. Export results (CSV / clipboard as supported by the UI).

### Option B: Run from source (Windows PowerShell)
```powershell
cd C:\Python\Px2XY
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Main.py
```

## What’s New in v1.2.0

This release focuses on stability, predictable behavior in manual workflows, and packaging improvements for Windows.

### User-facing improvements
- **Manual/Auto recalculation workflow**: Centroid recomputation can be gated by a clear **Auto/Manual** trigger, allowing manual runs when you want to avoid expensive recomputation on every parameter tweak.
- **More responsive manual recompute**: Clicking **ReCalculate** immediately provides UI feedback (busy/disabled state) before the heavy processing begins.
- **More predictable startup image selection**: On startup, PiXY prefers the last opened image when available.

### Packaging / Windows EXE improvements
- **Better frozen-path handling**: Asset lookup is robust under PyInstaller builds (works reliably when run from a bundled EXE).
- **Build script robustness**: The build script tolerates missing optional assets and still produces the EXE.

### Defaults and guardrails
- **Area-threshold defaults**: Conservative default min/max grain area thresholds to reduce surprising detections for first-time users.
- **Histogram initial selection clamping**: Prevents invalid initial ranges when the histogram content is small.

## Notes for Upgraders

If you are coming from older builds:
- The recalculation trigger UI may be in a different location than early v1.1.x builds; it is grouped under grain identification controls.
- For reproducible results, keep your area thresholds and segmentation parameters consistent when comparing runs.

## Known Issues / Limitations

- The bundled EXE size is relatively large due to GUI and scientific dependencies (Qt + NumPy + OpenCV).
- If you use unusual file paths or restricted folders, Windows permissions may prevent writing runtime state files; prefer a user-writable folder.

## Documentation

- `documentation/QuickManual_EN.md` — Quick start
- `documentation/Manual_EN.md` — Full manual

## UI screenshot (optional)

![PiXY main UI](documentation/images/fig_ui.png)

## Citation

If you use PiXY in published work, please cite the software.
- See `CITATION.cff` for the recommended citation metadata.

---

Build info
- Version: 1.2.0
- Release date: 2026-01-29
