# PiXY v1.2.3 — Release Notes (2026-02-10)

## Overview

PiXY is a desktop GUI tool for image-based centroid extraction and pixel→stage coordinate conversion.

Typical use cases:
- Detect particle/grain-like regions in microscope/SEM images and extract their centroids.
- Convert centroid pixel coordinates to stage coordinates using user-defined fiducial points.
- Export results for downstream measurement workflows (e.g., stage navigation, automated acquisition).

**Recommended screen size**: 1200×900 or larger (the UI is more comfortable at this size).

## What’s Included

- Standalone Windows executable: `PiXY.exe` (built locally)
- Source code (Python)
- Documentation and submission materials (JOSS draft, citation metadata)

## What’s New in v1.2.3

This release improves UI responsiveness and visual stability when users adjust parameters in Manual mode.

### User-facing improvements
- Performance/UI: in Manual mode, slider changes no longer trigger heavy recomputation and avoid full-frame poster resize/composition.
- UI: keep the last rendered posterized overlay/boundaries visible during Manual parameter tweaks (no flicker/disappearance).

### Packaging / Build notes
- The single-file Windows EXE was created locally and is available in `dist/` (see `dist/PiXY.exe`).

## Download / Run

1. Download the standalone EXE from GitHub Releases (if published) or use the local `dist` output.
2. If Windows shows a security prompt, use the file properties “Unblock” option (if present), then run.

Notes:

- The standalone EXE is the easiest option for users who do not want to install Python.
- If you prefer running from source (or need to tweak code), see `requirements.txt` and the README files in this repository.

## Build (developers)

PowerShell (example):

```powershell
.\build_exe.ps1 -Clean -Name PiXY
# or, to produce a versioned filename:
.\build_exe.ps1 -Clean -Name PiXY_ver123
```

Output examples:
- `dist/PiXY.exe` (default)
- `dist/PiXY_ver123.exe` (when `-Name PiXY_ver123` specified)

--

Zenodo DOI (fixed across versions): https://doi.org/10.5281/zenodo.18174474
