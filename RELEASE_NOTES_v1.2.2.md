# PiXY v1.2.2 — Release Notes (2026-02-05)

- Standalone Windows executable: `PiXY_ver122.exe`
- Source: Python code in this repository
- Zenodo DOI (fixed across versions): https://doi.org/10.5281/zenodo.18174474

---

## Overview

PiXY is a lightweight GUI tool for **converting microscopy image pixel coordinates into physical stage coordinates** for microanalysis instruments.

In a typical workflow, PiXY helps you:

- Detect particles/targets on an image and extract their centroids as $(u, v)$
- Register the image to the instrument stage using user-measured fiducial points
- Export the converted stage coordinates $(X, Y, Z)$ as CSV or via clipboard for instrument software

PiXY is intended for workflows such as LA-ICP-MS, SEM-EDS, EPMA, and SIMS where pre-acquired images are used to plan many measurement points efficiently.

---

## Key Features

- **Target detection**: K-means clustering + connected-component analysis for centroid extraction (or import externally pre-processed images).
- **Coordinate transformation**: Least-squares estimation of a 2D→3D affine mapping from fiducial points, with residual inspection.
- **Practical operations**: Manual rotate/flip controls to match instrument orientation.
- **Output**: CSV export and clipboard transfer for instrument input.

---

## What’s New in v1.2.2

- **UI tweak**: SegmentControl button widths adjusted for clearer labeling (e.g., Normal/Flip).
- **Docs**: Minor proofreading and formatting updates in the JOSS draft.
- **JOSS submission materials**: `paper.md` and a GitHub Actions workflow to generate a draft PDF (`.github/workflows/draft-pdf.yml`).

---

## Download / Run

1. Download `PiXY_ver122.exe` from GitHub Releases.
2. If Windows shows a security prompt, use the file properties “Unblock” option (if present), then run.

Notes:

- The standalone EXE is the easiest option for users who do not want to install Python.
- If you prefer running from source (or need to tweak code), see `requirements.txt` and the README files in this repository.

---

## Build (developers)

PowerShell:

```powershell
.\build_exe.ps1 -Clean -Name PiXY_ver122
```

Output:
- `dist/PiXY_ver122.exe`
