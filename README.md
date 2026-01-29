# PiXY: Pixel to stage-XY Coordinate Converter
Yoshiaki KON (Geological Survey of JAPAN, National Institute of Advanced Industrial Science and Technology)

PiXY is a desktop GUI tool that detects centroids of particle-like regions in images and converts pixel coordinates to real-world stage coordinates using user-defined reference points.

[![Quick Manual preview](README.png)](QuickManual.pdf)



## Features

- Detect centroids from posterized/segmented image regions (K-means + connected components).
- Add and edit reference points to estimate affine (or similarity) transforms between image and stage coordinates.
- Interactive reference table and overlay preview for quick inspection and parameter tuning.
- Export detected coordinates and reference data (CSV supported).

## Requirements

- Python 3.10 or newer (for running from source). See `requirements.txt` for dependencies.

## Installation (development)

PowerShell example:

```powershell
cd C:\Python\Px2XY
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root (Python source):

```powershell
python Main.py
```

Or run the provided Windows executable: `PiXY_ver118.exe`.

## Basic Usage

- Open an image with `Open Image`.
- Click `Add Ref` and add at least 3 (non-collinear) reference points: click image locations and enter corresponding stage `X,Y,Z`.
- Adjust detection parameters ("Number of Groups", area thresholds) and run detection.
- Export stage coordinates (CSV) or copy to clipboard for instrument import.

## Quick Manual

Click the preview image above to open the one-page Quick Manual PDF (`QuickManual.pdf`).

## Documentation

- `documentation/QuickManual_EN.md` — Quick start (English)
- `documentation/Manual_EN.md` — Detailed manual (English)

Screenshots for manuals are in `documentation/images/`.

## Citation

If you publish work that uses this software, please cite the repository. See `CITATION.cff` for citation metadata.

## License

This software is released under the MIT License (see `LICENSE`).

## Contributing

- Report issues on GitHub Issues.
- Pull requests are welcome; include reproduction steps or tests where possible.

## Acknowledgements

Some development assistance used AI-assisted tools; all outputs were reviewed and validated by the human authors. See `NOTICE.md` for details.

---

If you want additional edits (different preview image, shorter README for PyPI, or a README localized to Japanese), tell me which and I will update it.

