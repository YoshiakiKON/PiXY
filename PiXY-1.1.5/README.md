# PiXY

PiXY は画像中の粒子の重心を検出し、参照点（RefPoint）を使ってピクセル座標を実世界座標に変換するための GUI ツールです。

## 特徴
- 画像から重心（centroids）を抽出
- 参照点を追加・編集してアフィン / 類似変換を推定
# PiXY

PiXY is a desktop GUI tool that detects centroids of particle-like regions in images and converts pixel coordinates to real-world coordinates using user-defined reference points.

## Features

- Detect centroids from segmented image regions.
- Add and edit reference points to estimate affine or similarity transforms between image and stage coordinates.
- Interactive table view for quick inspection and editing of reference observations and residuals.
- Export coordinates and reference data (CSV supported).

## Requirements

- Python 3.10 or newer.
- See [requirements.txt](requirements.txt) for Python package dependencies.

## Installation

Create and activate a virtual environment, then install dependencies. Example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
python Main.py
```

Optionally provide an image path:

```powershell
python Main.py path\to\image.jpg
```

## Basic Usage

- Open an image using the `Open Image` action.
- Use the `Add Ref` tool to add reference points by clicking image locations and entering the corresponding real-world coordinates.
- Edit reference `u`/`v` (image coordinates) and `Stage X`/`Stage Y` in the reference table to refine the transform.
- Inspect residuals to evaluate transform quality and export results to CSV.

## Documentation

User manuals and quick-start guides are available in the `documentation/` folder:

- `documentation/QuickManual_EN.md` — Quick start (English)
- `documentation/Manual_EN.md` — Detailed manual (English)

Screenshots for the quick manual are stored under `documentation/images/`.

## Citation

If you publish work that uses this software, please cite this repository. See [CITATION.cff](CITATION.cff) for citation metadata.

## License

This project is released under the terms described in the `LICENSE` file.

## Contributing

- Report bugs and feature requests via GitHub Issues.
- Pull requests are welcome; include tests or reproduction steps when possible.

## Notes for Distributors

- For PyPI or packaged distributions, provide a short README (README-short.md) and ensure `requirements.txt` lists runtime dependencies only.

## Acknowledgements

Some development assistance used AI-assisted tools; all outputs were reviewed and validated by the human authors. See [NOTICE.md](NOTICE.md) for additional acknowledgements.

---

If you want, I can commit this update, create a branch/PR, and produce a shorter README for PyPI.


---
