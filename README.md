# PiXY — Pixel to Stage Coordinate Converter

PiXY is a small desktop application that detects particle centroids in microscopy images and converts image (pixel) coordinates to instrument stage coordinates using user-provided fiducial pairs. It is designed to speed offline target selection and prepare instrument-ready stage coordinates for microanalysis workflows.

Key features
- Detect particle centroids from segmented/processed images using a lightweight K-means + connected-component pipeline.
- Interactive GUI (PySide6) for placing and editing fiducial points and inspecting residuals.
- Estimate a 2D→3D affine transform from fiducial pairs and export instrument-ready coordinates (CSV / clipboard).
- Manual rotation and flip controls to help match image orientation during fiducial identification.

Quick links
- Repository: https://github.com/YoshiakiKON/PiXY
- Zenodo archive DOI: 10.5281/zenodo.18174474

Requirements
- Python 3.8+
- PySide6, NumPy, OpenCV (see `requirements.txt`)

Installation (development)
1. Clone the repository:

```bash
git clone https://github.com/YoshiakiKON/PiXY.git
cd PiXY
```

2. Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run
-----
- From source:

```bash
python Main.py
```

- A Windows executable is also provided in the `dist/` or `build/` folders for convenience.

Basic usage
- Open an image with the `Open Image` / `Export Image` controls.
- Add fiducial points using `Add Fiducial Point` and enter corresponding stage coordinates in the table.
- Inspect residuals, adjust fiducials if necessary, and export coordinates via CSV or clipboard.

Developer notes
- The GUI is implemented in `Ui.py`. Centroid extraction is in `CalcCentroid.py`. Rendering helpers are in `rendering.py` and utilities in `Util.py`.
- The `SegmentControl` helper lives inside `Ui.py` and implements compact segmented buttons used throughout the UI.

Versioning & citation
- Current release: 1.2.5
- Please cite the software when used in published research (see `CITATION.cff`).

License
- MIT — see `LICENSE`.

Contributing
- Issues and pull requests are welcome. For larger changes, open an issue first to discuss the proposed design.

Contact
- Yoshiaki KON — Geological Survey of Japan (GSJ), AIST

---

If you want, I can also:
- Update `README.md` with screenshots or a short animated GIF.
- Add a concise `README-short.md` for PyPI.
- Create a `CHANGELOG.md` entry for the `1.2.5` release.
