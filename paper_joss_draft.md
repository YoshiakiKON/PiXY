---
title: 'PiXY: Pixel to stage-XY Coordinate Converter'
tags:
  - particle detection
  - image segmentation
  - K-means clustering
  - coordinate transformation
  - microanalysis
  - electron microscopy
  - open-source software
authors:
  - name: Yoshiaki KON
    orcid: 0000-0002-2826-6666
    affiliation: "1"
affiliations:
  - index: 1
    name: Geological Survey of Japan (GSJ), National Institute of Advanced Industrial Science and Technology (AIST)
    ror: ""
date: 29 January 2026
bibliography: paper.bib
repository: https://github.com/YoshiakiKON/PiXY
repository-code: https://github.com/YoshiakiKON/PiXY
zenodo: https://doi.org/10.5281/zenodo.18417046
license: MIT
version: 1.2.0
---


# PiXY: Pixel to stage-XY Coordinate Converter
## JOSS Paper Draft - English Version

---

## Summary

**PiXY** is open-source software that links target selection on microscopy images with physical positioning on an analytical instrument stage, reducing time in microanalysis workflows in geoscience and materials science. Conventionally, targets identified on microscope images must still be manually relocated on the analytical instrument, and this additional targeting step often becomes a major bottleneck.

PiXY addresses this bottleneck through two functions. First, it automatically extracts particle image coordinates $(u, v)$ by combining K-means color clustering with connected-component analysis. Second, it estimates an affine mapping from user-defined reference points using a least-squares method and converts the extracted image coordinates into physical stage coordinates $(X, Y, Z)$. We evaluated the positional accuracy using real specimens and calculated the residuals relative to the full scale. The residual distribution (mean ± SD) was 0.1 ± 0.1 %FS in X, Y, and 4 ± 4 %FS in Z (N = 100).

In particular, when using five reference points, the residual histograms show that approximately 60% of the measurement points fall within 5 μm on each axis (X, Y, and Z; Figure 2).

PiXY does not assume specialized platforms or surface markings and can use arbitrary, distinctive features on the specimen as reference points. PiXY is released under the MIT License and is available both as a standalone Windows executable and as Python source code.

![Figure 1: PiXY GUI screenshot showing centroid overlays, reference points/residuals, and exportable coordinate tables.](documentation/images/fig_ui.png)

---

## Statement of Need

### Workflow challenges

Microanalysis instruments such as LA-ICP-MS, SEM-EDS, EPMA, and SIMS require reliable micrometer-scale targeting on solid sample surfaces. A common workflow is to image a sample in advance (e.g., by optical microscopy or electron microscopy) and then mount the same sample on an analytical instrument, where an operator adjusts an XYZ stage while viewing the instrument’s observation image.

For example, zircon U–Pb dating by LA-ICP-MS requires selecting analysis points based on microscopy observations before subsequent elemental and isotopic measurements (e.g., [@Iizuka:2006]). A typical workflow proceeds as follows:

1. **Sample preparation**: Select target mineral grains (e.g., by hand-picking), embed them in resin, and polish the mount.
2. **Imaging**: Acquire optical or SEM–BSE/CL images of the resin mount.
3. **Targeting**: Mount the sample on the analytical instrument and confirm the analysis spots.
4. **Microanalysis**: Measure elemental or isotopic compositions at the spots.

In this workflow, Step 3 (targeting on the analytical instrument) is often the bottleneck for efficient use of instrument time. For instance, while Step 4 may require 30–60 seconds per analysis point, targeting frequently takes longer. Because targeting time depends strongly on operator experience, semi-automation of Step 3 is valuable for stable and efficient operation.

Although individual tools exist for parts of this process (image analysis, coordinate conversion, and spreadsheet-based instrument input), there remains a gap in integrated GUIs that consistently support the full operational chain: identifying measurement targets, performing reference-point-based coordinate transformation, and exporting results in instrument-ready formats. PiXY targets this gap by integrating established image-processing and coordinate-transformation methods into a single workflow, thereby simplifying procedures and reducing targeting time.

---

## State of the Field

Particle detection and coordinate extraction on images can be performed using general-purpose image-analysis software (e.g., ImageJ/FIJI) or image-processing libraries such as OpenCV [@opencv]. In recent years, machine-learning-based segmentation has advanced rapidly, and segmentation foundation models have also been applied in domain-specific contexts (e.g., Segment Anything in Medical Images) [@ma2024sami].

In practical microanalysis workflows, however, a common bottleneck lies in converting detected “microscopy image coordinates” into “physical stage coordinates” and providing outputs in formats usable by instrument software (e.g., CSV or clipboard transfer). Fiducial-marker-based registration methods are widely used to align microscopy images with stage coordinates (e.g., [@sheriff2020autocrim]), but they require prior marking and may be difficult to introduce depending on specimen constraints, instrument configuration, and operational policies.

PiXY integrates particle detection (via an internal lightweight method or by importing pre-processed images from external workflows), reference-point-based coordinate transformation, residual inspection, and instrument-ready output into a single GUI, enabling rapid iteration during targeting.

---

## Software Design

PiXY is designed to (1) automatically extract particle locations from microscopy images and (2) convert image coordinates into physical stage coordinates, thereby semi-automating targeting on analytical instruments. Because any distinctive points on a specimen can be used as reference points, PiXY does not require specialized platforms or surface marking. The GUI allows users to tune detection parameters and immediately inspect detection and transformation results. Converted coordinates can be exported as CSV or transferred via the clipboard for input into instrument software.

The contribution of PiXY lies less in proposing new algorithms than in integrating established image-processing and coordinate-transformation techniques into a workflow that is usable in routine laboratory operations. Specifically, PiXY emphasizes usability by providing immediate preview of detection results, reference point entry, residual inspection for coordinate transformation, and export (CSV/clipboard) within a single GUI, thereby reducing rework and human error.

### Design trade-offs

For particle detection, PiXY adopts a reproducible and portable approach by combining K-means clustering with connected-component analysis, leveraging the intensity contrast typically available in electron microscopy images. Segmentation quality, however, depends strongly on image modality and specimen conditions, and both classical and machine-learning-based methods evolve rapidly [@ma2024sami]. Rather than competing directly across the full segmentation landscape, PiXY is designed to accept inputs pre-processed by external specialized software or workflows (e.g., binarization, posterization, clustering, or ML-based segmentation) and to place those results into the same coordinate-transformation and export pipeline.
Accordingly, PiXY tolerates multiple pre-processing routes, enabling users to choose segmentation methods appropriate to their instruments, domains, and image quality.
In addition, because reference-point acquisition is often the time-limiting step in practice, PiXY adopts an affine approximation (2D→3D) that can be estimated stably from a small number of points, rather than using overly complex nonlinear models.
PiXY is distributed both as Python source code and as a standalone executable so that users without a programming environment can adopt it.

### Particle detection algorithm (u, v extraction)

To automatically extract particle image coordinates $(u, v)$ from electron microscopy images, PiXY applies the following pipeline:

1. The image is clustered by K-means to facilitate separation of particles and background (the number of clusters is configurable in the GUI) [@lloyd1982].
2. Candidate particle regions are extracted by connected-component analysis, and noise or matrix regions are excluded using area thresholds.
3. The centroid of each remaining region is computed and output as $(u, v)$.

For performance, PiXY processes a downscaled image during preview, and then rescales detected coordinates to the original resolution.

---

### Coordinate transformation algorithm (u, v → X, Y, Z)

To convert detected **image coordinates ($u, v$)** into **physical stage coordinates ($X, Y, Z$)**, PiXY estimates a **2D→3D affine transformation** from user-defined reference points. Pixel pitch, rotation angle, translation, and other parameters are estimated automatically from these reference points, and the correspondence is obtained in a single transformation.

#### Affine transformation estimation

PiXY models the mapping from image coordinates to physical stage coordinates using an affine approximation. Each stage axis is expressed as a linear combination of the image coordinates $(x, y, 1)$, allowing planar dependence of $Z$ on image position to be handled within the same framework.

Here, $(x_{\text{full}}, y_{\text{full}})$ denotes full-resolution image coordinates (corresponding to $(u, v)$ used elsewhere in this manuscript).

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

This model captures pixel-to-physical scaling, rotation, translation, and cases where $Z$ depends on $(x, y)$ (approximable as a plane). Given multiple reference-point pairs $(x_i, y_i) \rightarrow (X_i, Y_i, Z_i)$, parameters are estimated by least squares.

#### Reference point setting and workflow

PiXY does not require artificial marks; distinctive points on the specimen (e.g., particle tips or scratches) can be used as reference points. Users select reference points on the image, measure the corresponding stage coordinates on the analytical instrument, and enter them in the GUI (three or more points are recommended).

In practice, the most time-consuming operation is locating these reference points on the instrument stage to acquire coordinates. To support efficient operation, PiXY includes manual rotation and flip functions. Once reference points are entered, scaling/rotation/translation for XY and tilt parameters for Z are computed automatically, and subsequent particle-detection results are converted into stage coordinates.

---

### Software architecture

PiXY is implemented in Python 3.8+ and uses PySide6 (Qt for Python) [@pyside6] for the GUI and OpenCV [@opencv] and NumPy [@numpy] for image processing. It also provides a single-file Windows executable packaged with PyInstaller.

The software is modularized by functionality: `Ui.py` provides the main interface and display; `CalcCentroid.py` implements centroid extraction; `Util.py` provides helper routines such as K-means and affine transformation; `rendering.py` handles visualization overlays; and `Main.py` serves as the entry point.

The GUI enables interactive tuning of parameters (e.g., cluster count and area thresholds). Results can be exported as CSV or copied to the clipboard for input into instrument software.

---

## Validation

### Experimental setup

Validation used BSE images acquired by a scanning electron microscope (JEOL JSM-6610LV). The specimen consisted of crushed granite grains embedded in 6-mm-diameter epoxy resin and polished to a mirror finish. Imaging conditions were 15× magnification, 1560×1920 px resolution, and a scale factor of 3.33 μm/pixel. After imaging, the specimen was mounted on a laser ablation system (Raijin, Seishin Co., Ltd.). The system uses an optical microscope (10× objective) and a micro-step motorized stage to position the specimen. XY coordinates were recorded when each reference point was centered in the field of view, and Z coordinates were recorded after adjustment using the system’s autofocus function.

### Evaluation of coordinate accuracy and trueness

The BSE image was loaded into PiXY, and particle centroids were detected using standard parameters: K = 5 (number of clusters), minimum area 20 px, and maximum area 4000 px. Reference points for coordinate transformation were acquired under two conditions: (i) three points approximately 120° apart and (ii) five points approximately 70° apart, both placed near the specimen periphery so that their centroid lay near the specimen center.

After coordinate transformation, the stage coordinates calculated by PiXY were exported and loaded into the laser ablation system for targeting verification. The residuals—defined as the differences between the intended target positions and the actual reached positions—were obtained by comparing the centroid position displayed in PiXY with the position reached after moving the stage to the PiXY‑derived coordinates. Validation was performed using three sets of measurements, including one with remounting, with measurement points spanning approximately 5000 μm in X and Y and 100–400 μm in Z.

Residual statistics were computed from N = 100 measurement points. For reporting, residuals were also normalized by the corresponding measurement span (full scale), i.e., ~5000 μm for X/Y and 100–400 μm for Z, and expressed as %FS. Across all points, the normalized residual distribution (mean ± SD) was 0.1 ± 0.1 %FS in X and Y, and 4 ± 4 %FS in Z.
Results by reference-point count are shown below.

![Figure 2: Residual histograms for each axis (X, Y, Z) comparing three vs. five reference points. With five reference points, the first bin (0–5 μm) contains ~60% of the residuals for all axes, indicating that most targets are reached within 5 μm.](documentation/images/fig_residual_hist.png)

| Number of reference points | X residual (mean ± SD) [μm] | Y residual (mean ± SD) [μm] | Z residual (mean ± SD) [μm] |
|---:|---:|---:|---:|
| 3 | 15 ± 14 | 16 ± 17 | 9 ± 7 |
| 5 | 4 ± 5 | 6 ± 7 | 8 ± 8 |

Consistent with these statistics, the residual distributions are strongly concentrated near zero for the five-point condition: about 60% of points are within 5 μm in X, Y, and Z (Figure 2).

In general, increasing the number of reference points strengthens constraints in least-squares estimation, often reducing both the mean residual (bias; related to trueness) and the standard deviation (scatter; related to precision). In this dataset, the five-point condition improved X and Y residuals, whereas Z residuals were comparable (see the table above).

---

## Research Impact Statement

The primary impact of PiXY is reducing the time and human cost of targeting in microanalysis. Targeting—searching micrometer-scale measurement locations in the limited field of view on an analytical instrument and determining measurement positions—often limits effective instrument time, and the required time depends strongly on operator skill. With PiXY, physical stage coordinates for measurement sites selected on pre-acquired images can be obtained quickly, substantially improving targeting efficiency. In addition, because PiXY does not assume dedicated fixtures or surface marking, it reduces operational constraints. Overall, PiXY improves throughput per specimen, supports a more reproducible workflow, and avoids costs associated with commercial software licenses.

---

## AI Usage Disclosure

During development, the authors used AI-assisted pair-programming tools (GitHub Copilot, Google Gemini, xAI Grok, and Anthropic Claude). All generated code and text were reviewed and validated by the authors, who retain full responsibility for the contents. AI tools were used mainly to generate candidate code and alternative wording; final decisions were made by the authors based on specifications, execution results, and visual inspection.

---

## Acknowledgments

This work was supported by JSPS KAKENHI (Grant Number: 25H00682). We also thank the developer communities behind the open-source scientific computing ecosystem, including Python, OpenCV, NumPy, and PySide6.

---

## References

