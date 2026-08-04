# Quick Manual — PiXY (English, v1.4.4)

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
- Use `New Project` to load an image.

![Open Image](images/quick_en_1.png)

2) Start centroid extraction
- Click `START Centroid Extraction`.
- Tune extraction parameters on the left and verify detected centroids.
- `Shape Complexity` default is `3` in the current spec.
- If `Number of Groups` changes, previous per-group `Show/Hide` settings are reset and all groups become visible.
- In Core+Rim display mode, each core/rim pair from the same particle is connected by a white line.

![Detect Centroids](images/quick_en_2.png)

3) Add groups to center list
- (Optional) Enter a manual group name in the text field below each `Add GroupN` button.
- Use `Add GroupN` buttons in the left group cards to move required points to the center list.
- If `Number of Groups` changes after recomputation, group names are inherited by nearest group color. In collision cases, the name from the larger source group is prioritized.

![Detect Centroids](images/quick_en_2.png)

4) Finish extraction and add fiducials
- Click `Finish Centroid Extraction`, then click `Add Fiducial Point`.
- Click `Add Fiducial Point`, then click image points to add observations.

![Add Ref](images/quick_en_3.png)

5) Transform & export
- Edit image coordinates (`u`/`v`) and stage coordinates (`Stage X`/`Stage Y`/`Stage Z`) in the reference table to refine the fit, then export to CSV.

![Export](images/quick_en_4.png)

## Quick Troubleshooting
- Image won't load: try different formats (ppm, bmp, jpg).
- Fiducial mismatch: re-add points and check residuals.
- After `Replace Image`, extraction-mode K-means is recalculated automatically so left-side detections match the replaced image.

---
Place screenshots under `documentation/images/`. See `SCREENSHOT_GUIDE.md` for capture instructions.

