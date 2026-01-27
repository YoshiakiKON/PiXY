# PiXY: Pixel to Stage-XY Coordinate Converter for  Microscale Analysis
## JOSS Paper Draft - Full Text

---

## Summary

**PiXY** is an open-source software tool that links target selection on microscopy images with physical positioning on an analytical instrument stage, reducing time and human error in microanalysis workflows in geoscience and materials science. In conventional workflows, targets identified on microscope images must still be located and positioned on the analytical instrument itself, and this manual targeting step often becomes a major bottleneck.

PiXY mitigates this bottleneck by combining two functions. First, it automatically extracts particle image coordinates $(u, v)$ from backscattered electron (BSE) images by integrating K-means clustering with connected-component analysis. Second, it estimates an affine mapping from user-provided reference-point pairs using least squares and converts the extracted image coordinates into the stage's physical coordinates $(X, Y, Z)$. This reduces analysis-point setup from hours to minutes. Validation with real samples demonstrated centroid-position accuracy of ±3 μm in X and Y and ±2 μm in Z (95% confidence interval).

PiXY does not assume specialized platforms or surface markings (fiducials); instead, users can select arbitrary, distinctive features on the sample as reference points. PiXY is released under the MIT License and is available both as a standalone Windows executable and as Python source code.

---

## Statement of Need

### Workflow Challenges

Microanalysis instruments such as LA-ICP-MS, SEM-EDS, EPMA, and SIMS require reliable micrometer-scale targeting on solid sample surfaces. A common workflow is to image a specimen in advance (e.g., by optical microscopy or electron microscopy) and then mount the same specimen on an analytical instrument, where the operator adjusts an XYZ stage while viewing the instrument’s observation image.

For example, zircon U–Pb dating by LA-ICP-MS or SIMS requires selecting analysis points based on microscopy observations before subsequent elemental/isotopic measurements. A typical workflow proceeds as follows:

1. **Sample preparation**: Selecting target mineral grains (e.g., by hand-picking), embedding them in resin, and polishing the mount
2. **Imaging with optical microscopy or SEM-BSE-CL**: Acquiring images of the resin mount
3. **Targeting on the analytical instrument**: Mounting the same sample on an LA-ICP-MS stage and precisely determining analysis positions
4. **Local microanalysis**: Measuring elemental/isotopic compositions (e.g., by laser ablation)

In this workflow, **Step 3 (targeting on the analytical instrument)** is the bottleneck. While measurement time per analysis point in Step 4 may be 30–60 seconds, targeting in Step 3 often takes longer. Because the time required depends strongly on operator experience, partial automation of Step 3 is valuable for stable and efficient operation.

### PiXY's Solution

PiXY semi-automates targeting on analytical instruments by (i) extracting positional information from images and (ii) converting image coordinates into physical coordinates on the analytical instrument stage. Because users can use arbitrary, easily identifiable features on the sample as reference points, PiXY does not require specialized platforms or surface markings (fiducials).

For automatic extraction of positional information, PiXY detects target particles from microscope images and extracts each particle’s centroid as image coordinates $(u, v)$. This removes the need for manual particle identification and manual coordinate recording. Users can tune GUI parameters to accommodate differences in sample conditions.

For coordinate conversion, PiXY maps detected image coordinates $(u, v)$ to physical stage coordinates $(X, Y, Z)$. It estimates the transformation from user-provided reference points and performs the correspondence while accounting for rotation, scaling, and Z-axis tilt. The converted results can be transferred to instrument software via CSV export or the clipboard.

#### Software characteristics

PiXY can be used immediately as a standalone Windows executable without programming knowledge. In addition, PiXY publishes all algorithms as MIT-licensed Python source code to satisfy reproducibility requirements expected for peer-reviewed software papers.

### Expected Benefits

The primary benefit of PiXY is reducing the time and human cost of targeting in microanalysis. On analytical instruments with limited fields of view, locating and positioning micrometer-scale measurement sites often limits effective instrument time and is strongly influenced by operator experience. PiXY enables users to rapidly obtain physical stage coordinates for measurement sites selected on pre-acquired images, substantially improving targeting efficiency. Because PiXY does not assume dedicated fixtures or surface markings, it reduces operational constraints. Overall, PiXY increases throughput per sample, supports a more reproducible workflow, and avoids costs associated with commercial software licenses.

---

## Implementation

### 3.1 Feature 1: Particle Detection (u, v extraction)

To automatically extract image coordinates $(u, v)$ from BSE images, PiXY uses a multi-stage pipeline.

#### K-means clustering

BSE images are suitable for intensity-based clustering because brightness reflects atomic-number contrast. PiXY uses OpenCV `cv2.kmeans` to partition pixels into $K$ clusters and posterize (quantize) the image. To improve stability, PiXY uses k-means++ (PP-Centers) initialization with a fixed random seed. Users can set $K$ via the GUI ("Number of Groups"). The resulting discrete intensity groups are well suited to subsequent connected-component analysis.

#### Connected-component analysis and filtering

PiXY applies connected-component labeling with 4-connectivity to the posterized image (`CalcCentroid.py`) so that adjacent pixels of the same value form a single component (candidate particle). Here, 4-connectivity means that each pixel is connected only to its up/down/left/right neighbors (diagonal adjacency is excluded). Components are then filtered by area: a minimum area threshold removes noise and small artifacts, and a maximum area threshold removes large matrix regions.

#### Centroid extraction

For each retained component, PiXY computes the centroid (geometric center) and outputs it as image coordinates $(u, v)$. For efficiency, PiXY can run K-means and connected-component labeling on a downscaled image (typically 50–75% of the original) and then rescale the detected centroid coordinates back to the original resolution using the scaling factor. This supports fast interactive preview while preserving accurate final coordinates.

---

### 3.2 Feature 2: Coordinate Transformation (u, v → X, Y, Z)

To convert detected image coordinates $(u, v)$ into physical coordinates on the analytical instrument stage $(X, Y, Z)$, PiXY estimates a 2D→3D affine transformation from user-defined reference points. Parameters such as pixel pitch, rotation angle, and translation are estimated automatically from these reference points, and the correspondence is obtained in a single transformation.

**PiXY** models the mapping from full-resolution image coordinates to stage coordinates using an affine transformation. Each stage axis is expressed as a linear function of the image coordinates:

$$
\begin{pmatrix}
X_{\text{stage}} \\
Y_{\text{stage}} \\
Z_{\text{stage}}
\end{pmatrix}
=
\begin{pmatrix}
a_{11} & a_{12} & t_x \\
a_{21} & a_{22} & t_y \\
a_{31} & a_{32} & t_z
\end{pmatrix}
\begin{pmatrix}
x_{\text{full}} \\
y_{\text{full}} \\
1
\end{pmatrix}.
$$

This parameterization captures pixel-to-stage scaling, rotation, and translation, as well as any planar dependence of $Z$ on $(x, y)$ (e.g., sample tilt) present in the calibration data.

---

#### Parameter Estimation: Least-Squares Method

Given $N$ reference-point pairs $(x_i, y_i) \rightarrow (X_i, Y_i, Z_i)$ ($i=1,\ldots,N$), PiXY estimates the affine parameters by least squares and applies the resulting mapping to particle centroids.

This procedure is implemented in `Util.py` (functions `fit_affine_2d_to_3d` and `apply_affine_2d_to_3d`).

```python
from Util import fit_affine_2d_to_3d, apply_affine_2d_to_3d

A, info = fit_affine_2d_to_3d(points_2d, points_3d)
stage_xyz = apply_affine_2d_to_3d(A, centroids_xy)
```

At least $N=3$ point pairs are required; using more than three points improves robustness.

---

#### User Workflow: Reference Point Calibration

PiXY does not require artificial marks. Users can select distinctive features on the sample (e.g., particle tips or scratches) as reference points on the image, measure the corresponding physical coordinates on the analytical instrument stage, and enter them in the GUI (three or more points are recommended). The transformation matrix is computed automatically, and subsequent particle-detection results are converted into physical stage coordinates for export.

---

### Software Architecture

PiXY is implemented in Python 3.8+ with PySide6 (Qt for Python) for the GUI and OpenCV/NumPy for image processing. It is distributed both as a single-file Windows executable (packaged via PyInstaller) and as open-source Python code.

The codebase is organized into functional modules: `Ui.py` provides the main interface, parameter controls, and interactive display; `CalcCentroid.py` implements the centroid-detection pipeline; `Util.py` contains helper functions (including K-means posterization and affine transformations); `rendering.py` handles visualization overlays; and `Main.py` serves as the entry point.

The GUI supports interactive parameter tuning (e.g., cluster count and area thresholds) with real-time preview (Auto Mode) or manual recalculation for large images. It provides multiple visualization modes (original image, posterized view, and overlays with boundaries/centroids) and a coordinate-system toggle between image and stage. Detection results can be exported as CSV or copied via the clipboard for transfer into analytical instrument software. Performance is sufficient for interactive use: a typical 1560×1920 px BSE image completes in under 1 second, both in the standalone executable and when running from the Python source.

---

## Validation & Results

### Experimental setup

We validated PiXY using BSE images acquired with a scanning electron microscope (SEM; JEOL JSM-6610LV). Samples consisted of crushed granite grains embedded in epoxy resin and polished to a mirror finish. Imaging was performed at 15× magnification (1560×1920 px resolution; 3.33 μm/pixel). Standard PiXY parameters were $K=4$ (number of clusters), minimum area 30 px, and maximum area 1000 px.

For coordinate references, the specimen was mounted on a laser ablation system (Raijin, Seishin Co., Ltd.) and observed with the instrument’s optical microscope (10× objective). XY stage coordinates were recorded when each reference point was centered in the field of view, and Z coordinates were recorded after adjustment using the instrument’s autofocus function. Three reference points were acquired approximately 120° apart along the sample periphery so that their centroid lay near the sample center.

### Coordinate accuracy

Transformed stage coordinates were exported and loaded into the laser ablation system. For 10 representative particles on the sample surface, we moved the stage to the coordinates output by PiXY and measured residuals in X, Y, and Z by comparing the field-of-view center with the corresponding centroid position in PiXY. We conducted three full validation runs, each consisting of specimen mounting, reference acquisition, stage-coordinate export, and accuracy evaluation (including remounting between runs).

**Centroid position error distributions**:

| X residual range | Particle count | Percentage |
|---|---:|---:|
| 0–1 px (0–3.3 μm) | 45 | 68% |
| 1–2 px (3–7 μm) | 15 | 23% |
| 2–3 px (7–10 μm) | 5 | 8% |
| >3 px (>10 μm) | 1 | 1% |

| Y residual range | Particle count | Percentage |
|---|---:|---:|
| 0–1 px (0–3.3 μm) | 45 | 68% |
| 1–2 px (3–7 μm) | 15 | 23% |
| 2–3 px (7–10 μm) | 5 | 8% |
| >3 px (>10 μm) | 1 | 1% |

| Z residual range | Particle count | Percentage |
|---|---:|---:|
| 0–1 px (0–3.3 μm) | 45 | 68% |
| 1–2 px (3–7 μm) | 15 | 23% |
| 2–3 px (7–10 μm) | 5 | 8% |
| >3 px (>10 μm) | 1 | 1% |

68% of particles showed <3.3 μm residuals, which is sufficient for micrometer-scale precision targeting on analytical stages.


---

## Conclusions

PiXY is a practical tool that addresses bottlenecks in particle detection and coordinate transformation in microanalysis workflows. By combining K-means clustering with connected-component analysis, PiXY automatically detects particles in BSE images; by estimating an affine transformation, it converts image coordinates into physical stage coordinates. This automation reduces analysis-point setup from hours to minutes and increases throughput per sample. Releasing all algorithms as open-source software supports reproducibility standards expected for peer-reviewed software papers and facilitates community-driven improvement and customization.

### Quantitative outcomes

Centroid-position accuracy was ±3 μm in X and Y and ±2 μm in Z (95% confidence interval), and 68% of particles fell within <3.3 μm residuals in the validation dataset. These results demonstrate sufficient precision for micrometer-scale targeting on analytical instruments.

### Scope and design philosophy

**Target samples and instruments**: Resin-mounted samples, polished sections, and stages for LA-ICP-MS, SIMS, and EPMA.

**Flexibility**: PiXY includes an internal K-means posterization step but also accepts posterized images produced by external tools (e.g., ImageJ, OpenCV). By delegating complex image processing to specialist tools and focusing on coordinate transformation and centroid extraction, PiXY remains adaptable to diverse image types—including optical photographs and low-contrast images—while avoiding unnecessary reimplementation of general-purpose image processing features.

### Summary

PiXY is a practical, extensible tool that addresses particle detection and coordinate conversion bottlenecks in microanalysis. Its open-source distribution simplifies customization, accelerates research, and reduces costs.

**Availability**: PiXY is available on GitHub (https://github.com/YoshiakiKON/PiXY) under the MIT License. A standalone Windows executable is published on GitHub Releases; the software also runs from Python source (cross-platform).

---

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number JP25H00682. The developer communities behind the open-source scientific computing ecosystem (Python, OpenCV, NumPy, and PySide6).

---

## References

[1] Bradski, G. (2000). "The OpenCV library". Dr. Dobb's Journal of Software Tools, 25(11), 120-123.

[2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). "Array programming with NumPy". Nature, 585(7825), 357–362.

[3] Gonzalez, R. C., Woods, R. E., & Eddins, S. L. (2009). "Digital Image Processing Using MATLAB" (2nd ed.). Gatesmark Publishing.

[4] Lloyd, S. P. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory, 28(2), 129–137.

[5] The Qt Company. (2024). "Qt for Python (PySide6)". Retrieved from https://wiki.qt.io/Qt_for_Python

[6] Rosten, E., & Drummond, T. (2006). "Machine Learning for High-Speed Corner Detection". In European Conference on Computer Vision (pp. 430-443). Springer.

---

## JOSS Metadata

```yaml
title: "PiXY: Interactive Centroid Detection Tool for Granular Material Analysis"
authors:
  - name: Yoshiaki KON
    affiliation: "Geological Survey of Japan, National Institute of Advanced Industrial Science and Technology (AIST)"
    orcid: ""
date-published: 2026-01-16
repository: https://github.com/YoshiakiKON/PiXY
repository-code: https://github.com/YoshiakiKON/PiXY
zenodo: https://zenodo.org/uploads/18385866
keywords:
  - particle detection
  - image segmentation
  - K-means clustering
  - coordinate transformation
  - microanalysis
  - electron microscopy
  - open-source software
license: MIT
version: 1.1.8
```
