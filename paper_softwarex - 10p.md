---
title: "PiXY: Pixel to stage-XY Coordinate Converter"
authors:
  - name: Yoshiaki KON
    affiliation: Geological Survey of Japan (GSJ), National Institute of Advanced Industrial Science and Technology (AIST)
date: 29 January 2026
bibliography: paper.bib
repository: https://github.com/YoshiakiKON/PiXY
archive_doi: 10.5281/zenodo.18174474
license: MIT
version: 1.4.2
---

Abstract

Accurate relocation of micrometer-scale targets is a major bottleneck in in-situ microanalysis. PiXY is open-source software that links target selection on microscopy images with physical positioning on an analytical-instrument stage, reducing time and human error in microanalysis workflows in geoscience and materials science. PiXY automatically extracts particle centroids using K-means–based color segmentation and connected-component analysis, and then estimates a least-squares affine transform from user-defined fiducial points to convert image coordinates into stage coordinates.

PiXY supports **offline targeting**, where users select targets on pre-acquired images and prepare stage coordinates before instrument time. This reduces the workload of targeting (relocation) on the instrument, shortens non-measurement stage operations, and improves instrument utilization. PiXY does not require dedicated fiducial markers or special pre-marking; instead, repeatably identifiable features on the sample can be used as fiducials.

In validation with real specimens ($N=100$, five fiducial points), residuals (targeting errors on the analytical instrument) were evaluated for positions specified on the image. Half of the residuals were within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% were within 10 μm (X), 16 μm (Y), and 21 μm (Z). These residuals support reliable offline targeting and practical coordinate preparation prior to instrument time.

Metadata

| Field | Value |
|---|---|
| Software title | PiXY: Pixel to stage-XY Coordinate Converter |
| Authors | Yoshiaki KON |
| Version | 1.3.3 |
| Repository URL | https://github.com/YoshiakiKON/PiXY |
| Archive DOI | 10.5281/zenodo.18174474 |
| License | MIT |
| Language | Python 3.8+ |
| Dependencies | PySide6, OpenCV, NumPy, PyInstaller |
| Release date | 15 February 2026 |

1. Introduction

Microanalysis instruments such as LA-ICP-MS, SIMS, EPMA, and SEM-EDS require micrometer-scale selection of analysis positions on solid sample surfaces. A common workflow is to image a sample in advance using optical or electron microscopy and then mount the same sample on an analytical instrument. On the instrument, operators adjust an XYZ stage while viewing the instrument’s observation image to decide measurement locations. In practice, this “positioning/targeting” step can take longer than the measurement per spot. Because the required time depends on operator experience, targeting frequently becomes a major bottleneck for instrument time.

To address this problem, high-accuracy registration methods using dedicated fiducial markers have been proposed [@sheriff2020autocrim]. However, such approaches may require pre-marking on the sample, which can be a barrier to adoption. Commercial tools also exist for specifying analysis positions on images and relocating them on the instrument; these tools can use image-based fiducial points for coordinate transformation.

Nevertheless, existing tools often require manual specification of each measurement point, which is inefficient when targeting must handle many candidate points obtained by automatic particle detection and centroid extraction. In addition, if fiducials are limited to only 2–3 points by design, it becomes difficult to use four or more fiducials and constrain the transform using least squares to improve robustness and accuracy. Therefore, an integrated workflow that can be iterated quickly—particle recognition → fiducial entry → coordinate transformation → instrument-ready export—without additional pre-processing is still not widely available.

In this work, we developed PiXY and provide an open-source GUI workflow that integrates particle detection (centroid extraction) from images through coordinate transformation and batch export of target coordinates. With PiXY, part of the targeting workload during instrument time can be shifted to **offline targeting** (selecting targets and preparing coordinates before instrument time), contributing to shortened instrument time and improved measurement throughput.


2. Software Description

PiXY links two steps in a single GUI workflow: (1) offline targeting on pre-acquired images, where targets are selected and particle centroids are extracted, and (2) fiducial-based coordinate transformation from image coordinates to analytical-instrument stage coordinates, followed by instrument-ready export. Figure 1 shows the GUI.

![Figure 1: Example of the PiXY GUI showing (i) particle recognition/centroid overlays, (ii) particle-recognition and centroid-extraction parameters, (iii) fiducial points and residuals, and (iv) exportable coordinate tables.](documentation/images/fig_ui.png)

2.1 Offline targeting

To efficiently set a large number of analysis spots, PiXY performs particle recognition and centroid extraction on pre-acquired microscope images (e.g., backscattered-electron images) to semi-automate targeting.

2.1.1 Particle recognition

PiXY prioritizes portability and interactive iteration (parameter tuning and immediate inspection) in routine laboratory workflows. Particle recognition results are overlaid on the image in the GUI so that users can inspect results immediately. To keep the GUI responsive even for large images, computation is performed on a “processing image”. When needed, the full image is downscaled (target width 640 px) for processing, and the resulting centroid coordinates are mapped back to full-resolution pixel coordinates for export.

For particle recognition, PiXY uses K-means clustering based on pixel colors. Each pixel is treated as a color vector $\mathbf{x}_i\in\mathbb{R}^3$ (BGR), and cluster centers $\{\boldsymbol{\mu}_k\}_{k=1}^{K}$ are estimated by minimizing
$$
\sum_i \min_{k}\lVert \mathbf{x}_i-\boldsymbol{\mu}_k\rVert^2
$$
[@macqueen1967]. Each pixel is replaced with its cluster center color to quantize the image into $K$ colors (posterization). PiXY uses the OpenCV implementation (K-means++ initialization) [@opencv] and fixes the random seed to ensure reproducibility for the same input and the same $K$. Next, PiXY creates a binary mask for each cluster (color) and extracts regions as 4-connected components.

The following parameters can be tuned interactively. In addition to selecting the number of clusters ($K$), they are intended to suppress noise and matrix regions, and to reduce small spurious regions caused by posterization boundaries and image edges.

**Number of Groups:**
Sets the number of K-means clusters $K$ (the number of representative colors after posterization). Increasing $K$ can improve particle/background separation in some cases, but may lead to over-segmentation. If $K$ is too small, particles and background can be mixed in the same cluster, leading to missed detections.

**Grain Size Threshold:**
For each extracted region (connected component), area $A$ (pixel count) is obtained from connected-component statistics (CC_STAT_AREA). Only regions satisfying $A_{min}\leq A \leq A_{max}$ are accepted (or only $A\geq A_{min}$ if no upper bound is set). $A_{min}$ and $A_{max}$ can be selected interactively with a histogram of region areas, and are used to exclude small noise regions and overly large regions caused by merged grains or background inclusion.

**Boundary Offset (Advanced):**
Applies morphological erosion $n$ times (where $n$ is specified in full-resolution pixels) to shrink the extracted mask and offset boundaries inward. Because processing is performed on the processing image, the actual erosion iterations are converted according to the downscaling factor. For a binary mask $M$, PiXY applies $M' = \mathrm{erode}(M; n)$ to suppress thin boundary features and small spurious regions derived from the image edge.

**Neck Separation (Advanced):**
If touching particles are extracted as a single region, PiXY attempts to split them by thinning the connection (neck) using erosion to create multiple cores and then assigning the original region to each core by fast marker propagation using dilation.

**Shape Complexity (Advanced):**
PiXY computes a compactness index $C=\frac{P^2}{4\pi A}$ from perimeter $P$ and area $A$, and accepts only regions with $C$ below a threshold. $C=1$ corresponds to a circle, and $C$ increases for elongated or highly irregular shapes. By decreasing the threshold, only near-circular regions remain. This helps exclude thin regions along scratches or edges and select compact, grain-like regions.

2.1.2 Centroid calculation

To place analysis spots on recognized particles, PiXY computes the centroid of each region. Centroids are obtained as $(x_{proc},y_{proc})$ from connected-component analysis on the processing image and are mapped to full-resolution pixel coordinates $(x_{full},y_{full})$ using the scale factor between processing and full-resolution images. PiXY then converts $(x_{full},y_{full})$ into GUI-displayed pixel coordinates $(u,v)$ (origin at the bottom-left corner).

In this paper, full-resolution pixel coordinates $(x_{full},y_{full})$ follow the standard image coordinate system (origin at the top-left; $x$ to the right; $y$ downward). In the GUI, PiXY displays $(u,v)$ with an origin at the bottom-left ($u$ to the right; $v$ upward). With full-image height $H_{full}$ (px), the conversion is
$$
u=x_{full},\quad v=(H_{full}-1)-y_{full}
$$
and the inverse is $x_{full}=u,\ y_{full}=(H_{full}-1)-v$.
Extracted spots are displayed in the GUI with a group ID (color class) and coordinate values, and are also drawn on the image with serial indices. The overlay image can be exported to streamline record keeping.

2.2 Fiducial-based coordinate transformation and export

To convert extracted spot coordinates to analytical-instrument coordinates, PiXY allows users to enter fiducial points interactively.

For coordinate conversion, PiXY estimates a 2D→3D affine approximation using least squares from fiducial pairs $(x_i,y_i)\rightarrow(X_i,Y_i,Z_i)$. While the transform can be estimated even with a small number of fiducials, increasing the number of fiducials constrains the least-squares estimate and can improve robustness and accuracy through residual evaluation. PiXY does not require dedicated fiducial markers; instead, distinctive, repeatably identifiable features on the sample surface (e.g., particle tips, scratches, edges) can be used as fiducials.

PiXY models the mapping with a 2D→3D affine transform:

$$
\begin{pmatrix}
X\\
Y\\
Z
\end{pmatrix}
=
\begin{pmatrix}
a_{11} & a_{12} & t_x\\
a_{21} & a_{22} & t_y\\
a_{31} & a_{32} & t_z
\end{pmatrix}
\begin{pmatrix}
x_{full}\\
y_{full}\\
1
\end{pmatrix}.
$$

Parameters are estimated using least squares from multiple fiducial pairs $(x_i,y_i)\rightarrow(X_i,Y_i,Z_i)$. Because fiducial entry on the instrument can be time-consuming, PiXY includes manual rotation/flip controls to help users match image orientation during fiducial identification, and it supports residual inspection before exporting instrument-ready coordinates (e.g., via CSV and clipboard).

Using the estimated parameters, PiXY converts centroid coordinates into stage coordinates and allows users to inspect residuals at fiducials in the GUI. Final target coordinates can be exported in instrument-friendly formats (CSV and clipboard). PiXY also provides image rotation/flip operations to support fiducial re-identification.

PiXY is implemented in Python 3.8+. The GUI is implemented with PySide6 [@pyside6]. Image processing uses OpenCV [@opencv] and NumPy [@numpy]. Code modules include `Ui.py` (GUI), `CalcCentroid.py` (centroid extraction), `Util.py` (helpers such as transform estimation), `rendering.py` (overlays), and `Main.py` (entry point). PiXY is distributed both as Python source and as a standalone Windows executable.

2.3 Instrument-ready coordinate export and project saving

Converted target coordinates can be exported in instrument-friendly formats (e.g., CSV save and clipboard copy). In addition, PiXY supports exporting centroid lists (e.g., `centroids_*.txt`) and overlay images (centroids and indices drawn on the original image) to assist record keeping.

PiXY can also save the full analysis state—input image, processing parameters, and computed results—as a project file (`.pixy`). The internal format is JSON, and the image is embedded using base64 encoding. This design supports traceability and helps reproduce identical outputs for the same input image under the same settings.

2.4 Installation and availability

PiXY source code is publicly available on GitHub, and a persistent archive is provided via Zenodo (DOI: 10.5281/zenodo.18174474). To run PiXY from Python, install dependencies and launch `Main.py`:

```bash
python -m pip install -r requirements.txt
python Main.py
```

PiXY is also distributed as a standalone Windows executable built with PyInstaller, so it can be used without a Python environment.

3. Illustrative examples

The following describes a typical workflow: users input a pre-acquired microscope image, estimate a coordinate transform from 3–5 fiducial points, and export instrument-ready stage coordinates.

3.1 Offline targeting

(1-1) Image acquisition:
Acquire an image of the sample surface using optical or electron microscopy.

(1-2) Load the image:
Launch PiXY on any Windows PC. Because a demo image is loaded at startup, select “New Project” and load the pre-acquired microscope image (e.g., a BSE image) into PiXY (Figure 2). After loading, particle recognition and centroid extraction are executed automatically, and particle regions and centroids are overlaid on the image.

(1-3) Centroid extraction (particle recognition):
Tune parameters such as Number of Groups ($K$) and Grain Size Threshold (area thresholds) to extract particle regions and centroids (Figure 2). Because results are overlaid on the image, users can immediately check for over-segmentation, missed detections, or noise. If needed, Boundary Offset, Neck Separation, and Shape Complexity can be used to suppress spurious regions and adjust splitting of touching particles.

(1-4) Save the data:
Select “Save Project” to save the processed image, processing parameters, and extracted centroids as a project file (`.pixy`, JSON format). In addition, selecting “Export Image” exports an overlay image with centroid indices drawn on the original image.

![Figure 2: Workflow in PiXY (image loading → centroid extraction → fiducial entry → residual inspection → coordinate export).](documentation/images/workflow_v2.svg)

3.2 Online targeting on the analytical instrument

(2-1) Load offline-targeting results:
Launch PiXY on a Windows PC. If possible, launching PiXY on the instrument control PC simplifies transferring coordinate data into the instrument control application. Select “Load Project” and load the project saved in (1-4). If working on the same PC as in (1-4), continue the workflow as is.

(2-2) Enter fiducial points:
Switch to fiducial-entry mode using `Add Fiducial Point` and click repeatably identifiable features on the image (e.g., particle tips, scratches, edges) to add fiducials (Figure 2). Then, on the analytical instrument, relocate the same fiducials and measure stage coordinates $(X,Y,Z)$, and enter them into the fiducial table in the GUI. PiXY also provides image rotation/flip operations to assist fiducial identification on the instrument.

(2-3) Inspect residuals and re-estimate:
After entry, PiXY estimates the coordinate transform using least squares from multiple fiducial pairs and displays residuals (errors) for each fiducial in the GUI. If a fiducial shows a large residual, re-identify it, add new fiducials, or exclude the outlier and re-estimate.

(2-4) Export coordinate data:
Export the converted target information (spot index, group ID, and stage coordinates $(X,Y,Z)$) in instrument-friendly formats (CSV save, clipboard copy). Paste the data into the coordinate input file used by the instrument (or load the CSV) to import coordinates.

4. Validation of coordinate transformation

We validated PiXY using BSE images acquired with a scanning electron microscope (JEOL JSM-6610LV) and a laser-ablation system (Raijin; Seishin Shoji). A representative dataset was acquired at 15× magnification (1560×1920 pixels; 3.33 μm/pixel) from a polished epoxy mount. For fiducials, stage coordinates were recorded on the analytical instrument while repeatedly relocating the same features. For each fiducial, the XY position was read when the feature was centered in the instrument view, and the Z coordinate was recorded after adjustment using the instrument’s autofocus (or equivalent focus-setting) function.

In this validation, the BSE image was loaded into PiXY, and centroids were extracted using default parameters ($K=5$, minimum area 20 px, maximum area 4000 px). Fiducials were selected from distinctive features near the mount rim so that they constrained the transform; two configurations were compared (three fiducials approximately 120° apart, and five fiducials approximately 70° apart around the rim). After estimating the transform, stage coordinates exported by PiXY were provided to the instrument and the stage was moved. The difference between the intended targets and the actual reached positions was evaluated as residuals. Measurement points were distributed over approximately 5000 μm in X and Y and 100–400 μm in Z.

Each configuration was evaluated in three runs with different sample orientations (rotation/tilt). In each run, 30–40 target points were measured, yielding $N=100$ residuals per configuration (Figure 3). With five fiducials, 50% of residuals were within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% were within 10 μm (X), 16 μm (Y), and 21 μm (Z). With three fiducials, 50% were within 11 μm, 12 μm, and 6 μm, and 90% were within 32 μm, 42 μm, and 20 μm, for X, Y, and Z, respectively. In general, increasing the number of fiducials strengthens the constraint in least-squares estimation and tends to reduce both bias and variability.

![Figure 3: Residual histograms for each axis (X, Y, Z) comparing three vs. five fiducial points.](documentation/images/fig_residual_hist.png)

5. Impact

The primary impact of PiXY is reducing time and human effort in targeting operations for in-situ microanalysis. On analytical instruments with limited fields of view, exploring and positioning micrometer-scale targets can easily become the bottleneck of machine time, and the time required depends on operator experience. With offline targeting in PiXY, stage coordinates for targets selected on pre-acquired images can be prepared rapidly, improving operational efficiency of analytical instruments.

Because particle recognition accuracy for centroid extraction depends on image simplicity and contrast, PiXY is designed to accept externally preprocessed images (e.g., segmentation or binarization). Future work includes incorporating segmentation modules and integrating with instrument-control APIs to further automate and streamline targeting.

6. Conclusions

PiXY integrates established image-processing and affine-registration methods into a practical workflow for offline targeting in in-situ microanalysis. By reducing manual relocation workload during instrument time and providing instrument-ready coordinate export, PiXY contributes to improved reproducibility and instrument utilization.

PiXY provides a single GUI workflow from centroid extraction (candidate generation) to fiducial entry, residual inspection, coordinate export, and project saving, enabling fast repetition of the procedure (re-identification and re-estimation). Validation with real specimens shows that residual distributions improve as the number of fiducial points increases, confirming sufficient accuracy for practical operation.

At the same time, because the success of particle recognition depends on image contrast and pre-processing, selecting an appropriate segmentation approach is important depending on the application and instrument. Future work includes integrating segmentation functions and instrument-control APIs to further automate and reduce labor in targeting.

PiXY is publicly available on GitHub and is persistently archived on Zenodo (DOI: 10.5281/zenodo.18174474). Reproducibility is supported by fixing the K-means random seed and by saving processing conditions and results in `.pixy` project files.

CRediT author statement

Yoshiaki KON: Conceptualization, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing – original draft, Writing – review and editing.

Declaration of competing interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Acknowledgements

This work was supported by JSPS KAKENHI (Grant Number: 25H00682). The author acknowledges the open-source communities behind Python, OpenCV, NumPy, and PySide6.

During development, the author used AI-assisted tools (GitHub Copilot, Google Gemini, xAI Grok, Anthropic Claude) for drafting suggestions and programming support. All generated outputs were reviewed and validated by the author, who takes full responsibility for the final content.

References

[See `paper.bib` for BibTeX entries; citations in-text use pandoc syntax and will be numbered on conversion.]
