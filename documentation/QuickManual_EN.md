# Quick Manual — PiXY (English)

Short, step-by-step guide with screenshots for common tasks.

## Overview
- Purpose: Detect particle centroids in images and convert pixel → real-world coordinates using fiducial points.

## Requirements
- Python 3.10+
- See `requirements.txt` for dependencies

## Run
```powershell
python Main.py
```

## Common tasks (see screenshots)

1) Open image
- Use the menu or `Open Image` to load an image.

![Open Image](images/quick_en_1.png)

2) Detect centroids
- Click `Detect` (or equivalent) to display centroids.

![Detect Centroids](images/quick_en_2.png)

3) Add fiducial points
- Click `Add Fiducial Point`, then click image points to add observations.

![Add Ref](images/quick_en_3.png)

4) Transform & export
- Edit image coordinates (`u`/`v`) and stage coordinates (`Stage X`/`Stage Y`) in the reference table to refine the fit, then export to CSV.

![Export](images/quick_en_4.png)

## Quick Troubleshooting
- Image won't load: try different formats (ppm, bmp, jpg).
- Fiducial mismatch: re-add points and check residuals.

---
Place screenshots under `documentation/images/`. See `SCREENSHOT_GUIDE.md` for capture instructions.
