# PiXY: Pixel to stage-XY Coordinate Converter
Yoshiaki KON (Geological Survey of JAPAN, National Institute of Advanced Industrial Science and Technology)

PiXY is a desktop GUI tool that detects centroids of particle-like regions in images and converts pixel coordinates to real-world stage coordinates using user-defined fiducial points (naturally occurring specimen features such as scratches or particle tips; not pre-made markers).

[![Quick Manual preview](README.png)](README.png)


## Features

- Detect centroids from segmented image regions (posterization/clustering).
- Add and edit fiducial points to estimate an affine/similarity transform.
- Interactive, transposed fiducial table view for quick editing and verification.
- Export coordinates and fiducial data (CSV supported).

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

## Installation (development)

PowerShell example:

```powershell
cd C:\Python\Px2XY
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root (Python source):

```powershell
python Main.py
```

Or run the provided Windows executable: `PiXY_ver121.exe`.

## Basic usage

- Open an image with `Open Image`.
- Use `Add Fiducial Point` to enter fiducial-point mode, then click image points to add fiducial observations.
- Edit fiducial `u`/`v` (image coordinates) and `Stage X`/`Stage Y` (and `Stage Z` if applicable) in the fiducial table to refine the transform.
- The left transposed table shows residuals and transformed coordinates for quick inspection.

## Development & Contribution

- Report bugs and feature requests via GitHub Issues.
- Pull requests are welcome — please include tests or reproduction steps when possible.

## Citation

If you publish results generated with this software, please cite this repository.

- Zenodo DOI: https://doi.org/10.5281/zenodo.18174474
- See `CITATION.cff` for the recommended citation metadata.

## License

See the `LICENSE` file in the repository for license terms (e.g. MIT).

---

If you'd like, I can also:

- Commit this change and push a branch/PR.
- Produce a shorter `README-short.md` for display on PyPI/GitHub release pages.
- Draft a `paper.md` for JOSS submission using the repository metadata and release DOI (once available).

