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
zenodo: https://doi.org/10.5281/zenodo.18174474
license: MIT
version: 1.3.3
---


# PiXY: Pixel to stage-XY Coordinate Converter
## JOSS Paper Draft - English Version

---

## Summary

**PiXY** is open-source software that links target selection on microscopy images with physical positioning on an analytical instrument stage, reducing time in microanalysis workflows in geoscience and materials science. Conventionally, targets identified on microscope images must still be manually relocated on the analytical instrument, and this additional targeting step often becomes a major bottleneck.

PiXY addresses this bottleneck through two functions. First, it automatically extracts particle image coordinates $(u, v)$ by combining K-means color clustering with connected-component analysis. Second, it estimates an affine mapping from user-defined fiducial points using a least-squares method and converts the extracted image coordinates into physical stage coordinates $(X, Y, Z)$.

We evaluated the positional accuracy using real specimens and calculated the residuals. In particular, when using five fiducial points (N = 100), the residual distributions are strongly concentrated near zero: 50% of points fall within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% fall within 10 μm (X), 16 μm (Y), and 21 μm (Z).

PiXY does not assume specialized platforms or surface markings. In this work, the “fiducial points” are not pre-made fiducial markers; instead, we use naturally occurring distinctive specimen features (e.g., particle tips or scratches) as fiducials. PiXY is released under the MIT License and is available both as a standalone Windows executable and as Python source code.

---

## Statement of Need

### Workflow challenges

Microanalysis instruments such as LA-ICP-MS, SEM-EDS, EPMA, and SIMS require reliable micrometer-scale targeting on solid sample surfaces. A common workflow is to image a sample in advance (e.g., by optical microscopy or electron microscopy) and then mount the same sample on an analytical instrument, where an operator adjusts an XYZ stage while viewing the instrument’s observation image.

For example, zircon U–Pb dating by LA-ICP-MS requires selecting analysis points based on microscopy observations before subsequent elemental and isotopic measurements (e.g., [@Iizuka:2006]). A typical workflow proceeds as follows:

1. **Sample preparation**: Select target mineral grains (e.g., by hand-picking), embed them in resin, and polish the mount.
2. **Imaging**: Acquire optical or SEM–BSE/CL images of the resin mount.
3. **Targeting**: Mount the sample on the analytical instrument and confirm the analysis spots.
4. **Microanalysis**: Measure elemental or isotopic compositions at the spots.

In this workflow, Step 3 (targeting on the analytical instrument) is often the bottleneck for efficient use of instrument time. For instance, while Step 4 may require 30–60 seconds per analysis point, targeting frequently takes longer. Because targeting time depends strongly on operator experience, semi-automating Step 3 is valuable for stable and efficient operation.

Although individual tools exist for parts of this process (image analysis, coordinate conversion, and spreadsheet-based instrument input), there remains a gap in integrated GUIs (Figure 1) that consistently support the full operational chain: identifying measurement targets, performing fiducial-point-based coordinate transformation, and exporting results in instrument-ready formats. PiXY targets this gap by integrating established image-processing and coordinate-transformation methods into a single workflow, thereby simplifying procedures and reducing targeting time.

![Figure 1: PiXY GUI screenshot showing centroid overlays, fiducial points/residuals, and exportable coordinate tables.](documentation/images/fig_ui.png)

---

## State of the Field

Particle detection and coordinate extraction on images can be performed using general-purpose image-analysis software (e.g., ImageJ/FIJI) or image-processing libraries such as OpenCV [@opencv]. In practical microanalysis workflows, however, a common bottleneck lies in converting detected “microscopy image coordinates” into “physical stage coordinates” and providing outputs in formats usable by instrument software (e.g., CSV or clipboard transfer). Fiducial-marker-based registration methods are widely used to align microscopy images with stage coordinates (e.g., [@sheriff2020autocrim]), but they require prior marking and may be difficult to introduce depending on specimen constraints, instrument configuration, and operational policies.

PiXY integrates particle detection (via an internal lightweight method or by importing pre-processed images from external workflows), fiducial-point-based coordinate transformation, residual inspection, and instrument-ready output into a single GUI, enabling rapid iteration during targeting.

---

## Software Design

PiXY is designed to (1) automatically extract particle locations from microscopy images and (2) convert image coordinates into physical stage coordinates, thereby semi-automating targeting on analytical instruments. Because any distinctive points on a specimen can be used as fiducial points, PiXY does not require specialized platforms or surface marking. The GUI allows users to tune detection parameters and immediately inspect detection and transformation results. Converted coordinates can be exported as CSV or transferred via the clipboard for input into instrument software.

The contribution of PiXY lies less in proposing new algorithms than in integrating established image-processing and coordinate-transformation techniques into a workflow that is usable in routine laboratory operations. Specifically, PiXY emphasizes usability by providing immediate preview of detection results, fiducial point entry, residual inspection for coordinate transformation, and export (CSV/clipboard) within a single GUI, thereby reducing rework and human error.

### Design trade-offs

For particle detection, PiXY adopts a reproducible and portable approach by combining K-means clustering with connected-component analysis, leveraging the intensity contrast typically available in electron microscopy images. Segmentation quality, however, depends strongly on image modality and specimen conditions, and both classical and machine-learning-based methods evolve rapidly [@ma2024sami]. Rather than competing directly across the full segmentation landscape, PiXY is designed to accept inputs pre-processed by external specialized software or workflows (e.g., binarization, posterization, clustering, or ML-based segmentation) and to place those results into the same coordinate-transformation and export pipeline.

Accordingly, PiXY tolerates multiple pre-processing routes, enabling users to choose segmentation methods appropriate to their instruments, domains, and image quality.

In addition, because fiducial-point acquisition is often the time-limiting step in practice, PiXY adopts an affine approximation (2D→3D) that can be estimated stably from a small number of points, rather than using overly complex nonlinear models.

PiXY is distributed both as Python source code and as a standalone executable so that users without a programming environment can adopt it.

### Particle detection algorithm (u, v extraction)

To automatically extract particle image coordinates $(u, v)$ from electron microscopy images, PiXY applies the following pipeline:

1. The image is clustered by K-means to facilitate separation of particles and background (the number of clusters is configurable in the GUI) [@lloyd1982].
2. Candidate particle regions are extracted by connected-component analysis, and noise or matrix regions are excluded using area thresholds.
3. The centroid of each remaining region is computed and output as $(u, v)$.

For performance, PiXY processes a downscaled image during preview, and then rescales detected coordinates to the original resolution.

---

### Coordinate transformation algorithm (u, v → X, Y, Z)

To convert detected **image coordinates ($u, v$)** into **physical stage coordinates ($X, Y, Z$)**, PiXY estimates a **2D→3D affine transformation** from user-defined fiducial points. Pixel pitch, rotation angle, translation, and other parameters are estimated automatically from these fiducial points, and the correspondence is obtained in a single transformation.

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

This model captures pixel-to-physical scaling, rotation, translation, and cases where $Z$ depends on $(x, y)$ (approximable as a plane). Given multiple fiducial-point pairs $(x_i, y_i) \rightarrow (X_i, Y_i, Z_i)$, parameters are estimated by least squares.

#### Fiducial point setting and workflow

PiXY does not require artificial marks. Instead, naturally occurring distinctive points on the specimen (e.g., particle tips or scratches) can be used as fiducial points. Users select fiducial points on the image, measure the corresponding stage coordinates on the analytical instrument, and enter them in the GUI (three or more points are recommended).

In practice, the most time-consuming operation is locating these fiducial points on the instrument stage to acquire coordinates. To support efficient operation, PiXY includes manual rotation and flip functions. Once fiducial points are entered, scaling/rotation/translation for XY and tilt parameters for Z are computed automatically, and subsequent particle-detection results are converted into stage coordinates.

---

### Software architecture

PiXY is implemented in Python 3.8+ and uses PySide6 (Qt for Python) [@pyside6] for the GUI and OpenCV [@opencv] and NumPy [@numpy] for image processing. It also provides a single-file Windows executable packaged with PyInstaller.

The software is modularized by functionality: `Ui.py` provides the main interface and display; `CalcCentroid.py` implements centroid extraction; `Util.py` provides helper routines such as K-means and affine transformation; `rendering.py` handles visualization overlays; and `Main.py` serves as the entry point.

The GUI enables interactive tuning of parameters (e.g., cluster count and area thresholds). Results can be exported as CSV or copied to the clipboard for input into instrument software.

---

## Validation

### Experimental setup

Validation used BSE images acquired by a scanning electron microscope (JEOL JSM-6610LV). The specimen consisted of crushed granite grains embedded in 6-mm-diameter epoxy resin and polished to a mirror finish. Imaging conditions were 15× magnification, 1560×1920 px resolution, and a scale factor of 3.33 μm/pixel. After imaging, the specimen was mounted on a laser ablation system (Raijin, Seishin Co., Ltd.). The system uses an optical microscope (10× objective) and a micro-step motorized stage to position the specimen. XY coordinates were recorded when each fiducial point was centered in the field of view, and Z coordinates were recorded after adjustment using the system’s autofocus function.

### Evaluation of coordinate accuracy

The BSE image was loaded into PiXY, and particle centroids were detected using standard parameters: K = 5 (number of clusters), minimum area 20 px, and maximum area 4000 px. Fiducial points for coordinate transformation were acquired under two conditions: (i) three points approximately 120° apart and (ii) five points approximately 70° apart, both placed near the specimen periphery.

Using these fiducial points, PiXY estimated the coordinate transformation between the SEM image and the laser‑ablation stage. After computing the transformation, the stage coordinates generated by PiXY were exported and loaded into the laser‑ablation system for targeting verification. Residuals—defined as the differences between the intended target positions and the actual stage‑reached positions—were quantified by comparing PiXY‑displayed centroid coordinates with the positions reached after moving the stage to the PiXY‑derived coordinates. Measurement points spanned approximately 5000 μm in X and Y, and 100–400 μm in Z.

Each configuration was evaluated across three runs with different specimen orientations (rotation and tilt). Each run included 30–40 target points, yielding N = 100 residual measurements per configuration (Figure 2). The absolute residuals reveal clear differences in practical targeting precision between the two configurations. With five fiducial points, 90% of residuals fall within 10, 16, and 21 μm in X, Y, and Z, respectively, and 50% fall within 3, 3, and 4 μm. In contrast, with three fiducial points, 90% of residuals fall within 32, 42, and 20 μm, and 50% fall within 11, 12, and 6 μm for X, Y, and Z.

![Figure 2: Residual histograms for each axis (X, Y, Z) comparing three vs. five fiducial points.](documentation/images/fig_residual_hist.png)

Increasing the number of fiducial points generally improves the stability of the least‑squares coordinate transformation, reducing both systematic bias and random scatter. Consistent with this expectation, the five‑point configuration yields residuals that are more tightly clustered around zero across all axes, with particularly pronounced improvements in X and Y.

---

## Research Impact Statement

The primary impact of PiXY is the reduction of time and human effort required for targeting in microanalysis. Because locating micrometer‑scale measurement positions must be performed within the limited field of view of an analytical instrument, targeting often limits effective instrument use, and the time required differs markedly with operator expertise.
PiXY enables rapid retrieval of physical stage coordinates from selected measurement sites on pre‑acquired images, substantially improving targeting efficiency. Moreover, because PiXY does not rely on dedicated fixtures or surface markings, it imposes fewer operational constraints. Overall, PiXY increases throughput per specimen, supports a more reproducible workflow, and reduces costs associated with commercial software licenses.

---

## AI Usage Disclosure

During development, the authors used AI-assisted pair‑programming tools (GitHub Copilot, Google Gemini, xAI Grok, and Anthropic Claude). All AI‑generated code and text were reviewed and validated by the authors, who take full responsibility for the final content. AI tools were used primarily to generate candidate code and alternative phrasings; final decisions were made by the authors based on specifications, execution results, and visual inspection.

---

## Acknowledgments

This work was supported by JSPS KAKENHI (Grant Number: 25H00682). The authors also acknowledge the open-source scientific computing communities behind Python, OpenCV, NumPy, and PySide6.

---

## References

