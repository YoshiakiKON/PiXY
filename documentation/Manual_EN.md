# Manual — PiXY (English)

Detailed reference describing features, configuration, logs, and file formats.

## Contents
- Overview
- Installation
- Startup options
- UI components
- Handling reference points
- Output files (CSV, centroids_*.txt)
- Advanced configuration (`Config.py`)
- Logs & debugging

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

## Main UI
- Main window contains: image view, centroid list, reference table, toolbar.
- Reference table shows image coordinates (`u`, `v`) and stage coordinates (`Stage X`, `Stage Y`, `Stage Z`), transformed coordinates, and residuals.

## Reference points
1. Click `Add Ref` to enter reference mode.
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
