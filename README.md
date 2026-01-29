# PiXY

PiXY is a GUI tool that detects particle centroids in images and converts pixel coordinates to real-world (stage) coordinates using user-defined reference points.

## Key features

- Detect centroids from segmented image regions (posterization/clustering).
- Add and edit reference points to estimate an affine/similarity transform.
- Interactive, transposed reference table view for quick editing and verification.
- Export coordinates and reference data (CSV supported).

## Requirements

- Python 3.10+ (adjust as needed for your environment).
- See `requirements.txt` for Python package dependencies.
- Recommended window size: 1200×900 or larger.

## Documentation

User manuals and quick-start guides (English and Japanese) are in `documentation/`:

- `documentation/QuickManual_EN.md` — Quick Manual
- `documentation/Manual_EN.md` — Manual
- `documentation/SCREENSHOT_GUIDE.md` — Screenshot capture guide

Screenshots for the quick manuals should be placed under `documentation/images/`.

## Install (development)

PowerShell example:

```powershell
cd C:\Python\PiXY
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
python Main.py
```

You can optionally pass an image path:

```powershell
python Main.py path\to\image.jpg
```

## Basic usage

1. Open an image with `Open Image`.
2. Use `Add Ref` to enter reference-point mode, then click image points to add reference observations.
3. Edit reference `u`/`v` (image coordinates) and `Stage X`/`Stage Y` (and `Stage Z` if applicable) in the reference table to refine the transform.
4. Export results as CSV or copy to clipboard.

## Citation

If you publish results generated with this software, please cite this repository (see `CITATION.cff`).

## License

MIT License — see `LICENSE`.

