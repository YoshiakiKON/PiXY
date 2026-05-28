# Manual — PiXY (English)

Detailed reference describing current v1.4 workflow, key controls, and files.

## Contents
- Overview
- Installation
- Startup options
- Operation flow (v1.4)
- Main UI
- Fiducial points
- Output files (CSV, centroids_*.txt)
- Configuration (`Config.py`)
- Logs & debugging

## Overview
- PiXY detects candidate points (centroids) in microscopy images and converts image coordinates (`u`, `v`) to stage coordinates (`X`, `Y`, `Z`) using fiducial points.

## Installation
1. Prepare Python 3.10+
2. Create venv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Startup options
- `--auto`: automatically process last image.
- `--auto-exit`: auto process and exit.

## Operation flow (v1.4)
1. Open image via `New Project`.
2. Click `START Centroid Extraction`.
3. Tune extraction parameters on the left panel (`Number of Groups`, `Boundary Offset`, `Neck Separation`, `Shape Complexity`, `Grain Size Threshold` histogram).
4. Use `Add GroupN` to move detected points to the center list.
5. Click `Finish Centroid Extraction`.
6. Add and refine fiducial points using `Add Fiducial Point` / `Update u, v`, then export (`Export XYZ` or clipboard).

## Main UI
- Left panel
	- `START/Finish Centroid Extraction`: enter/exit extraction mode.
	- `Recalculation Trigger` (`Auto` / `Manual`).
	- Group cards with `Add GroupN` and `Show/Hide`.
- Center panel
	- Candidate table and controls (`Export XYZ`, `Clipboard`, `Add Target`, `Update u, v`, `Clear`).
- Right panel
	- Image display and orientation controls.
	- `Boundary` and `Display Mode` (`Original`/`Posterized`) are visible only in extraction mode.
	- In normal mode, display is fixed to `Original` + `Boundary OFF`.

## Fiducial points
1. Click `Add Fiducial Point` to enter fiducial mode.
2. Click known points on the image to add observations.
3. Edit image coordinates (`u`, `v`) and stage coordinates (`Stage X`, `Stage Y`, `Stage Z`) in the table and re-fit.

## Output files
- `centroids_YYYYMMDD_HHMMSS.txt`: detected centroid list.
- Export CSV for transformed coordinates.

## Configuration
- Edit `Config.py` for parameters (display, thresholds, etc.).

## Logs
- Check `debug_px2xy.log` for debug output.

## Screenshots
- Place screenshots in `documentation/images/` as referenced in the Quick Manual.
