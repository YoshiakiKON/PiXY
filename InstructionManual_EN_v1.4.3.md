# PiXY Operation Manual (English)

---

## Cover Information

- Software: PiXY — Pixel-to-Stage XY Coordinate Converter
- Author: Yoshiaki KON
- Repository: https://github.com/YoshiakiKON/PiXY
- Zenodo DOI: 10.5281/zenodo.18174474
- License: MIT
- Last updated: 2026-07-30 (v1.4.3 spec)

---

## Overview

- Purpose: Pair candidate points (centroids) in microscope images with instrument fiducials and convert image coordinates to stage coordinates to reduce instrument time and improve reproducibility.
- Intended users: Micro-area analysis operators, sample analysis staff, instrument administrators
- Main features: Image loading, particle detection (K-means + connected components), fiducial input, affine transform estimation, residual visualization, CSV export / clipboard copy
- Benefits: Earlier targeting preparation, less manual work at the instrument, and statistically analyzing more points

---

## Getting Started (Installation & Launch)

- System requirements
  - OS: Windows recommended (distributed EXE available). Linux/Mac may work when running from source.
  - Python: 3.8 or later (when running from source)
  - Disk and memory: Depends on image resolution (large images require more memory)
- Dependencies: See `requirements.txt` (PySide6, OpenCV, NumPy, etc.)
- Installation (from source)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Main.py
```

- Running the distributed EXE: After downloading, right-click → [Properties] → [Unblock] (if needed), then run.
- Note: When running from source, enable the virtual environment and match the versions in `requirements.txt`.

---

## Quick Start (Minimum Steps)

1. Run `PiXY_ver143.exe`.
2. Click `New Project` and select an image file to load.
3. Click `START Centroid Extraction`.
4. In extraction mode, adjust the left-side detection parameters (`Number of Groups`, `Boundary Offset`, `Neck Separation`, `Shape Complexity`, and `Grain Size Threshold` histogram) and confirm that centroids are detected.
5. (Optional but recommended) Enter a manual group name in the text field below each `Add GroupN` button.
6. Add candidate points from each group using `Add GroupN` buttons (left cards) so they appear in the center list.
7. If `Number of Groups` is changed and you recompute, manual group names are transferred to the nearest-color group automatically. When multiple names map to one group, the source group with more points is prioritized.
8. Click `Finish Centroid Extraction` to return to on-line alignment.
9. Click `Add Fiducial Point`, click fiducials on the image, and enter the corresponding stage coordinates in the table. Enter 3 or more points and check residuals. If residuals are large overall, add more points. If one point has a much larger residual, re-check that point or exclude it.
10. If everything looks good, export coordinates via `Export XYZ` or `Clipboard` and pass them to the instrument.

---

## UI Overview

- Main buttons
  - Upper left (project operations)
    - `New Project`: Start a new project and select an image file to load.
    - `Load Project`: Load a saved `.pixy` project (image, parameters, results).
    - `Save Project`: Save the current state as a `.pixy` project.

  - Middle left (fiducial operations)
    - `Add Fiducial Point`: Enter fiducial registration mode; click fiducial points on the image to add them to the table.
    - `Update XY`: Update the stage coordinates (`Stage X/Y/Z`) for the selected fiducial row.
    - `Clear` (fiducials): Delete the selected fiducial row in the table, or clear its values.

  - Lower left (particle detection parameter settings)
    - `START Centroid Extraction` / `Finish Centroid Extraction`: Enter/exit extraction mode.
    - Detection parameters are operated in `Advanced` mode only.
    - `Recalculation Trigger` (`Auto` / `Manual`): Control when particle detection is recomputed. `Auto` recomputes on every parameter change; `Manual` recomputes only when you click `ReCalculate`.
    - `Number of Groups (K)` — Number of K-means clusters. Suggested 3–10 (try based on the number of colors; avoid over-segmentation).
    - `Boundary Offset` — Offset to avoid incomplete regions near the image boundary (e.g., exclude edges).
    - `Neck Separation` — Strength for separating touching particles (larger = stronger separation).
    - `Shape Complexity` — Tuning parameter to suppress/split regions with complex shapes.
    - `Grain Size Threshold` — Histogram-based min/max area selection.
    - Group cards (bottom left)
      - `Add GroupN`: Add detected points of group N to the center list.
      - `Group name` input (under each `Add GroupN`): Assign a manual display name for that group.
      - `Show` / `Hide`: Per-group visibility toggle.
      - `Add ALL Group to List`: Add all groups to the center list.
      - When `Number of Groups` or posterization level changes, custom group visibility (`Show`/`Hide`) is reset and all groups are shown.
      - If `Number of Groups` changes and centroids are recomputed, manual group names are inherited by nearest group color; collisions are resolved by preferring the source group with more points.

  - Center (candidate table)
    - `Export XYZ`: Convert candidate points to stage coordinates (X,Y,Z) using the estimated transform and save as CSV.
    - `Clipboard`: Copy output data to the clipboard in a paste-ready tab-separated format (TSV) for instrument software.
    - `Add Target`: Manually add a target point on the image (use when you want to add a point not found by automatic extraction).
    - `Update u, v`: Update `u, v` (image coordinates) of the selected target row to the currently clicked position.
    - `Clear` (targets): Clear the target selection or any temporary designation.

    Note (`Add Target` / `Update u, v` with group visibility)
    - `Update u, v` (Update Target) is currently a beta feature. The internal behavior and UI may change in future versions.
    - Points added via `Add Target` are treated as manual targets and always belong to `Group 0`.
    - Manual targets are appended to the end of the auto-detected `Group 0` block (not inserted as No.1).
      - As a result, adding a manual target can shift the candidate `No.` indices.
    - `Update u, v` behaves differently depending on what is selected.
      - If a manual target (=`Group 0`) is selected: it updates that manual target in place (the number of points does not increase).
      - If an auto-detected target (e.g., `Group >= 1`) is selected: it is treated as a replace operation:
        (1) the original point is hidden/excluded, and (2) a new manual target (`Group 0`) is added at the clicked position.
    - Even if `Group 0` is hidden by group `Show`/`Hide`, when you replace a non-`Group 0` point via `Update u, v`, the newly added point is forced-visible
      so you can still see the result of your update.

  - Right side (image display / appearance adjustments)
    - `Export Image`: Merge the current overlays (boundaries, centroid IDs, etc.) onto the original image and export as an image.
    - `Original` / `Posterized` and `Boundary` (`Show` / `Hide`) are shown only in Centroid Extraction mode.
      - Normal mode is fixed to `Original` + `Boundary OFF`.
    - `Coordinate` (`Image` / `Stage`): Switch the displayed coordinate system between image coordinates and stage coordinates.
      - When showing Image coordinates (u, v)
        - `Flip` (`Normal` / `Flip`): Toggle a left-right flip to match the instrument view.
        - `Image Rotate` slider: Manually rotate the image to help re-identify fiducials (linked with the angle display).
      - When showing Stage coordinates (X, Y, Z)
        - `Right` (`+X` / `-X`): Horizontal axis setting for the instrument coordinate system. Choose +X if the screen-right direction is positive.
        - `Top` (`+Y` / `-Y`): Vertical axis setting for the instrument coordinate system. Choose +Y if the screen-up direction is positive.

- Image interaction
  - `Left-click + drag`: Pan the image.
  - `Mouse wheel (over the image)`: Zoom in/out.

- Table columns
  - `u, v`: Image coordinates (pixels)
  - `Stage X, Stage Y, Stage Z`: Instrument coordinates (units depend on the instrument)
  - `Res. X/Y/Z`: Residual for each fiducial point
  - `Toggle switch`: Show/hide each point; hidden points are excluded from calculation and export
---

## Workflow Details

- Image (generate candidates)
  - Load an image, set the number of clusters with `Number of Groups (K)`, and perform color segmentation via K-means.
  - Compute centroids per region using connected-component analysis and generate the candidate list.
  - Recompute automatically or manually when parameters change (`Auto` / `Manual` mode).
- Fiducial input
  - Click `Add Fiducial Point` and register re-identifiable feature points by clicking on the image.
  - Find the same points on the instrument and enter `Stage X/Y/Z`.
  - We recommend 4 or more non-collinear fiducial points (5+ for stability).
- Transform estimation
  - Estimate a 2D→3D affine transform from fiducial pairs via least squares.
  - Also try a flipped model and adopt the model with smaller residuals.
  - Check each fiducial residual; re-identify or exclude outliers and re-estimate.
- Export
  - Convert candidates to stage coordinates and export via `Export XYZ` / `Clipboard`.
  - If the instrument has a specific import format, confirm and adjust the CSV column order.

---

## Feature Guide

- Particle detection (K-means + connected components)
  - Goal: Extract target regions (e.g., particles) from an image and obtain centroids (candidate points) for each region.
  - Key parameters
    - `Number of Groups (K)` — Number of K-means clusters. Suggested 3–10 (try based on the number of colors; avoid over-segmentation).
    - `Grain Size Threshold` — Minimum region area (px). Suggested 10–50 px (noise removal).
    - `Maximum Area` — Maximum region area (px). Exclude regions that are too large.
    - `Boundary Offset` — Offset to avoid incomplete regions near the image boundary (e.g., exclude edges).
    - `Neck Separation` — Strength for separating touching particles (larger = stronger separation).
    - `Shape Complexity` — Tuning parameter to suppress/split regions with complex shapes (default: `3`).
  - Tuning tips
    - First adjust `K` and `Grain Size Threshold`, then tune `Neck Separation` etc. only if many touching particles remain. Advanced parameters can be computationally expensive, so manual recalculation is recommended.

- Centroid filter order (current spec)
  1. Apply trim (`Boundary Offset`) to the binary mask.
  2. Label connected components.
  3. Early-reject components smaller than the minimum area threshold.
  4. Apply neck separation splitting to remaining components.
  5. Re-apply minimum/maximum area constraints to split components.
  6. Apply shape complexity filter.
  7. Compute/export centroid and boundary overlays for accepted components.
- Residual visualization
  - Check the distribution using the GUI residual table and histogram.
  - Detect outliers using RMS or a median + MAD-based threshold.
- Project reproducibility
  - Save parameters and results into `.pixy`.
  - In the current version, the internal random seed for K-means is fixed (not user-configurable).

---

## Files and Formats

- `.pixy` — JSON project file (embeds the input image as base64; includes processing parameters, extracted results, fiducials, and the estimated transform)
- `centroids_*.csv` — Output CSV (No, Group, Stage X, Stage Y, Stage Z)
- Supported input images: PNG/JPG/BMP, etc.

---

## Practical Example (Short Tutorial)

1. Example: Load a microscope image (e.g., BSE) and extract candidates with `K=5`, `Min area=20px`.
2. Select 5 fiducial points and obtain stage coordinates on the instrument.
3. Confirm that 50% of residuals fall within 3–4 μm for each of X, Y, and Z.
4. Export CSV and paste/import into the instrument software to run the experiment.

---

## License and How to Cite

- License: MIT
- Citation example: Y. KON, PiXY v1.4.2. Zenodo:10.5281/zenodo.18174474

---

## Change Log (Excerpt)

- v1.4.2 (2026-07-28): Fixed Replace Image re-detection in centroid extraction mode and added core/rim pair connector lines in overlays.
- v1.4.1 (2026-06-08): Improved middle-table width/scroll stability, synchronized center-row XYZ after extraction adds, improved large-number readability, and reset group visibility to all-visible when group count changes.
- v1.4.0 (2026-05-28): Added Start/Finish Centroid Extraction workflow, mode-dependent display controls, and updated left-panel group operations.

- v1.3.2 (2026-02-18): Windows EXE distribution, improved UI flip processing, added README
- v1.2.3: Internal bug fixes, improved fiducial residual display

---

## Appendix

- Glossary: `fiducial` (reference point), `centroid`, etc.
- References: Key papers/libraries cited for PiXY algorithms and dependencies
- Issue submission template: Attach reproduction steps, OS/Python version, and log output

---

## Appendix: Equations and Algorithm Details

This appendix describes the main equations omitted from the paper and the algorithm logic used by major GUI features in as much detail as possible.

### Posterization by K-means (cluster segmentation)

- Objective function (to minimize)
  $$J = \sum_{k=1}^{K} \sum_{x\in S_k} \|x - \mu_k\|^2$$
  Here, $x$ is the pixel color vector (e.g., RGB), $S_k$ is the set of pixels assigned to cluster $k$, and $\mu_k$ is the cluster center (centroid).

- Iterative procedure (standard Lloyd's algorithm)
  1. Set initial centers $\{\mu_k\}$ (in this implementation, initialization uses a fixed internal RNG for reproducibility).
  2. Assignment step: Assign each pixel $x$ to the nearest center $\mu_k$ (Euclidean distance).
     $$\text{assign}(x) = \arg\min_k \|x - \mu_k\|^2$$
  3. Update step: Update each cluster center by the mean.
     $$\mu_k \leftarrow \frac{1}{|S_k|}\sum_{x\in S_k} x$$
  4. Convergence: Stop when assignments do not change, when center movement is below a threshold, or when max iterations is reached.

- Implementation notes
  - For reproducibility, initialization may use a fixed seed (e.g., `cv2.setRNGSeed(12345)`); the current version uses a fixed seed.
  - Selecting $K$ for the whole image depends on heuristics, so the user adjusts `Number of Groups (K)` in the GUI.
  - Posterization is visualized by replacing each cluster label with a representative color (e.g., the mean color of the cluster).

### Connected components and centroid computation

- Steps
  1. Binarization (select target clusters from the posterized result)
  2. Label connected components (4-neighborhood or 8-neighborhood)
  3. For each component, compute area (pixel count), perimeter, and moments
  4. Compute centroid $(u,v)$ from first moments
     $$u = \frac{M_{10}}{M_{00}},\quad v = \frac{M_{01}}{M_{00}}$$
    where $M_{pq}=\sum x^p y^q$ are region moments and $M_{00}$ is the area.

### Size filter (exclude by area)

- Logic
  - Measure area $A$ for each labeled region and discard regions below `min_area` or above `max_area`.

- Pseudocode

```
for region in labeled_regions:
    A = region.area
    if A < min_area or (max_area is not None and A > max_area):
        discard region
    else:
        keep region
```

### Neck separation (splitting touching regions)

- Goal: Separate clusters of touching particles into individual particles.
- Common approaches (implementation options)
  1. Distance transform + marker-based watershed
     - Compute the distance transform and extract local maxima as markers
     - Apply watershed using markers as initial labels to split touching regions
  2. Split based on convex hull / shape indices
     - Detect deviation from convex hull (lower solidity) or low circularity as split candidates
  3. Morphological erosion/dilation to detect thin neck parts and split there

- Implementation parameter
  - `Neck Separation` strength (small values = weak splitting; larger values = more aggressive splitting)

### Boundary offset

- To exclude incomplete particles near screen edges or cropped regions, trim a fixed number of pixels around the image boundary and exclude it from processing.

### Shape complexity (Shape Complexity / Compactness)

- Example metrics
  - Circularity from perimeter $P$ and area $A$
    $$C = \frac{4\pi A}{P^2}$$
    Values close to 1 indicate a circle (simple); smaller values indicate more complex shapes (indentations, elongated regions).
  - Solidity (region area / convex hull area) is also useful as a complexity measure.

### Estimating the coordinate transform (2D image → 3D stage)

- Model (affine approximation)
  An affine map from image coordinates $(u,v)$ to stage coordinates $(X,Y,Z)$ can be written as
  $$
  \begin{bmatrix} X\\ Y\\ Z \end{bmatrix}
  =
  \begin{bmatrix}
  a_0 & a_1 & a_2\\
  b_0 & b_1 & b_2\\
  c_0 & c_1 & c_2
  \end{bmatrix}
  \begin{bmatrix} 1\\ u\\ v \end{bmatrix}
  $$
  where each column contains the constant term and coefficients for $u,v$.

- Least squares derivation
  - Suppose there are $N$ fiducials $i$, each with image coordinates $(u_i,v_i)$ and measured stage coordinates $(X_i,Y_i,Z_i)$.
  - Let $A\in\mathbb{R}^{N\times 3}$ with rows $A_i=[1\; u_i\; v_i]$ and $B\in\mathbb{R}^{N\times 3}$ with rows $B_i=[X_i\;Y_i\;Z_i]$.
    Solving linear regression for each output coordinate yields $P\in\mathbb{R}^{3\times3}$.
    Using the normal equation:
    $$P = (A^T A)^{-1} A^T B$$

- Residuals and evaluation
  - Compute residual matrix $R = B - A P$ and the norm $\|r_i\|$ of each point's residual vector.
  - Evaluate accuracy using RMS (root mean square) or per-axis RMS.

---

## Glossary (grouped as in the Japanese master)

Below are short definitions of key terms used in this manual.

【A-row】
- Affine transform: A linear transform plus translation for image coordinates; used as a 2D→3D approximation in this app.

【Ka-row】
- K-means: A representative non-hierarchical clustering method that partitions color space into $K$ clusters.
- Cluster / Group: A set of pixels grouped by K-means.

【Sa-row】
- Fiducial (reference point): A re-identifiable feature point on the image that pairs with a measured stage coordinate.
- Residual: The difference between the predicted coordinate (from the estimated transform) and the measured coordinate; used for accuracy evaluation.

【Ta-row】
- Trimming / Boundary Offset: Operation to exclude a fixed pixel margin around the image from processing.

【Na-row】
- Neck separation: An algorithm that detects thin neck regions between touching particles and separates them.
- Noise: Small false regions or misdetections; removed by size filtering via `min_area`.

【Ha-row】
- Posterize (Posterized): Replace pixels with representative cluster colors for simplified display.
- Shape complexity: A shape index computed from perimeter and area (e.g., circularity, solidity).

【Ma-row】
- Area: The pixel count of a labeled region; used for size filtering.
- Centroid: A representative point computed from first moments (image coordinates $u,v$).

【Ra-row】
- Connected component: A contiguous pixel set in a binary image; identified by labeling.

【Wa-row】
- Workflow: The basic steps of this software: ImageMode (candidate extraction) → StageMode (fiducial input / transform estimation) → Export (output).

---

File generated: 2026-02-18
