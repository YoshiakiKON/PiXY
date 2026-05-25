---
title: "PiXY: Pixel to stage-XY Coordinate Converter"
authors:
  - name: Yoshiaki KON
    affiliation: Geological Survey of Japan (GSJ), National Institute of Advanced Industrial Science and Technology (AIST)
date: 29 January 2026
repository: https://github.com/YoshiakiKON/PiXY
archive_doi: 10.5281/zenodo.18174474
license: MIT
version: 1.3.2
---

Abstract

PiXY is an open-source software that connects “target selection on microscopy images” with “positioning on analytical instruments” in microanalysis workflows in geoscience and materials science, reducing manual time and human errors. PiXY combines K-means color clustering and connected-component analysis to extract particle image coordinates (u, v), and estimates an affine transform by least squares from user-acquired fiducial points to convert image coordinates into physical instrument stage coordinates (X, Y, Z).

This software supports offline targeting: users select targets on pre-acquired images and prepare stage coordinates before instrument time. This reduces the workload of on-instrument targeting (relocation), shortens instrument operation time beyond measurements, and improves instrument utilization. PiXY does not require dedicated fiducial markers or special sample preparation; instead, it can use repeatably identifiable features on the sample as fiducial points.

In validation with real specimens (N = 100, five fiducials), we evaluated the reaching error (residual) on the analytical instrument for positions specified on the image. As a result, 50% of residuals were within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% were within 10 μm (X), 16 μm (Y), and 21 μm (Z), confirming practical accuracy.

Keywords

- microscopy
- microanalysis
- registration
- affine
- centroid
- targeting

Highlights

- Links microscopy targets to instrument stage coordinates
- Extracts centroids and estimates affine transforms from multiple fiducials
- Enables offline targeting to reduce on-instrument workload
- Validated on real specimens (N=100) with practical residuals

Metadata

| Field | Value |
|---|---|
| Software title | PiXY: Pixel to stage-XY Coordinate Converter |
| Authors | Yoshiaki KON |
| Version | 1.3.2 |
| Repository URL | https://github.com/YoshiakiKON/PiXY |
| Archive DOI | https://doi.org/10.5281/zenodo.18174474 |
| License | MIT |
| Language | Python 3.8+ |
| Dependencies | PySide6, OpenCV, NumPy, PyInstaller |
| Release date | 15 February 2026 |


1. Motivation and significance

Microanalysis instruments such as LA-ICP-MS, SIMS, EPMA, and SEM-EDS require micrometer-scale selection of analysis positions on solid sample surfaces. In a common workflow, the sample surface is imaged in advance using optical or electron microscopy, and then the same sample is mounted on an analytical instrument. On the instrument, operators adjust an XYZ stage while viewing the instrument’s observation image to determine measurement positions. In practice, this “positioning/targeting” step can take longer than the measurement per spot, and the required time often depends on operator experience; therefore, targeting tends to become a major bottleneck of instrument time.

To address this issue, high-accuracy registration methods using dedicated fiducial markers have been proposed [1], but they may require pre-marking on the sample and can be a barrier to adoption. Commercial tools also exist that allow users to specify analysis positions on images and relocate them on the instrument; such tools may use image-based fiducial points for coordinate transformation.

However, many existing tools still require manual specification of measurement points, which is inefficient when targeting must handle many candidate points obtained by automatic particle detection and centroid extraction. In addition, if a tool restricts fiducials to only 2–3 points, it becomes difficult to increase fiducials to four or more and constrain the transform using least squares to improve robustness and accuracy. Therefore, an integrated GUI workflow that connects known methods and executes the full process consistently is still lacking.

Here, individual elements such as particle recognition and centroid extraction (image processing), and coordinate transformation based on fiducial points (least-squares estimation of an affine transform), are widely available as established algorithms. In practical targeting, however, a unified GUI is needed to support the full sequence: generating many candidates on an image, iteratively updating the transform while entering fiducials and inspecting residuals, and finally exporting an instrument-ready coordinate list. PiXY integrates this sequence into a single application, enabling trial-and-error and coordinate preparation to be completed before instrument time.

In this work, we developed PiXY and provide an open-source GUI workflow from particle detection (centroid extraction) to coordinate transformation and batch export of target coordinates. With PiXY, part of the targeting workload on the analytical instrument can be shifted to offline targeting (selecting targets and preparing coordinates before instrument time), contributing to shortened instrument time and improved measurement throughput.


2. Software Description

2.1 Software architecture

PiXY is a desktop GUI application that integrates, as a single workflow, (1) generating candidates (particle regions and centroids) from pre-acquired images and (2) converting image coordinates into analytical-instrument stage coordinates based on fiducial points and exporting instrument-ready coordinate tables (Figure 1). Internally, PiXY consists of GUI components (image display and overlay rendering; table editing for candidates and fiducials), image processing (color segmentation by K-means clustering [2], connected-component analysis, centroid calculation), transform estimation (least-squares estimation from fiducial pairs and residual evaluation), and input/output (image loading; CSV/clipboard export; project save/load). To keep the GUI responsive, computationally intensive processing is performed on a downscaled “processing image”, and results are converted back to full-resolution pixel coordinates for output. The project file format (.pixy) stores processing conditions and results, enabling reruns under identical conditions for the same input.

The project file format (.pixy) stores processing conditions and intermediate results, enabling reruns under identical conditions for the same input image. PiXY is implemented in Python 3.8+. It uses PySide6 (Qt for Python) [3] for the GUI and OpenCV [4] and NumPy [5] for image processing.

Figure 1: Example of the PiXY GUI showing (i) particle recognition and centroid overlays, (ii) particle-recognition and centroid-extraction parameters, (iii) fiducial points and residuals, and (iv) exportable coordinate tables.

![Figure 1: Example of the PiXY GUI.](documentation/images/fig_ui.png)

2.2 Software functionalities

PiXY supports a consistent workflow from offline candidate generation to instrument-ready coordinate list creation. Users load microscope/SEM images and extract particle regions based on simple color segmentation by K-means and connected-component analysis, obtaining centroid coordinates as candidates. Results are overlaid as boundaries and centroids on the GUI image, allowing interactive adjustment of major parameters and immediate inspection for over-segmentation, missed detections, and noise inclusion. Candidates can be managed as a table, and exporting an indexed overlay image and centroid lists helps create measurement records.

Next, users specify repeatably identifiable features on the sample as fiducial points on the image and enter corresponding stage coordinates measured on the analytical instrument into the table. PiXY estimates a 2D-to-3D affine approximation by least squares from multiple fiducial pairs and visualizes residuals (errors) for each fiducial, supporting outlier detection, re-identification, and re-estimation by adding fiducials. Based on the estimated transform, PiXY converts candidate centroids into stage coordinates and exports them in convenient formats (CSV save, clipboard copy).

To assist fiducial re-identification, PiXY provides image rotation/flip operations. The analysis state can be saved and loaded as a project file (.pixy), allowing integrated management of the image, processing conditions, extraction results, fiducials, and transform results. Source code is publicly available on GitHub and is persistently archived on Zenodo (DOI: 10.5281/zenodo.18174474). PiXY can be run from Python by installing dependencies listed in requirements.txt and launching Main.py. In addition, it is distributed for Windows as a standalone executable built with PyInstaller.


3. Illustrative examples

This section describes the main functions of PiXY as a continuous procedure from candidate generation (centroid extraction) through fiducial entry, residual inspection, and export of coordinates for instrument loading.

3.1 Offline targeting

3.1.1 Image acquisition: Acquire an image of the sample surface using optical or electron microscopy.

3.1.2 Load the image: Launch PiXY on any Windows PC. Because a demo image is loaded at startup, select “New Project” and load a pre-acquired microscope image (e.g., a BSE image) into PiXY (Figure 2). After loading, particle recognition and centroid extraction are executed automatically, and particle regions and centroids are overlaid on the sample image.

3.1.3 Centroid extraction (particle recognition): Tune parameters such as Number of Groups (K) and Grain Size Threshold (area thresholds) to extract particle regions and centroids (Figure 2). Because results are overlaid on the image, users can immediately check for over-segmentation, missed detections, or noise. If needed, Boundary Offset, Neck Separation, and Shape Complexity can be used to suppress spurious regions and adjust splitting of touching particles.

3.1.4 Save the data: Select “Save Project” to save the processed image, processing parameters, and extracted centroids as a project file (.pixy, JSON format). In addition, selecting “Export Image” exports an overlay image with centroid indices drawn on the original image.

3.2 Online targeting on the analytical instrument

3.2.1 Load offline-targeting results: Launch PiXY on a Windows PC. If possible, running PiXY on the instrument control PC simplifies transferring coordinate data into the instrument control application. Select “Load Project” and load the project saved in 3.1.4. If working on the same PC as in 3.1.4, continue the workflow as is.

3.2.2 Enter fiducial points: Switch to fiducial-entry mode using “Add Fiducial Point” and click repeatably identifiable features on the image (e.g., particle tips, scratches, edges) to add fiducials (Figure 2). Then, on the analytical instrument, relocate the same fiducials and measure stage coordinates (X, Y, Z), and enter them into the fiducial table in the GUI. PiXY also provides image rotation/flip operations to assist fiducial identification on the instrument.

3.2.3 Inspect residuals and re-estimate: After entry, PiXY estimates the coordinate transform using least squares from multiple fiducial pairs and displays residuals (errors) for each fiducial in the GUI. If a fiducial shows a large residual, re-identify it, add new fiducials, or exclude the outlier and re-estimate.

3.2.4 Export coordinate data: Export the converted target information (spot index, group index, and stage coordinates (X, Y, Z)) in instrument-friendly formats (CSV save, clipboard copy). Paste the data into the coordinate input file used by the instrument (or load the CSV) to import coordinates.

Figure 2: Workflow in PiXY (image loading → centroid extraction → fiducial entry → residual inspection → coordinate export).

![Figure 2: Workflow in PiXY.](documentation/images/workflow_v2.svg)

3.3 Validation of coordinate transformation

We validated the accuracy of coordinate transformation in PiXY using real specimens and microanalysis instruments. For validation, we used BSE images acquired with a scanning electron microscope (JEOL JSM-6610LV) and a laser-ablation system (Raijin; Seishin Shoji). A representative dataset was acquired at 15× magnification (1560×1920 pixels; 3.33 μm/pixel) from a polished epoxy mount. For fiducials, stage coordinates were recorded on the instrument while repeatedly relocating the same features. For each fiducial, the XY position was read when the feature was centered in the instrument view, and the Z coordinate was recorded after adjustment using the instrument’s autofocus (or equivalent focus-setting) function.

In the evaluation, the BSE image was loaded into PiXY, and centroids were extracted using standard parameters (K = 5, minimum area 20 px, maximum area 4000 px). Fiducials were selected from distinctive features near the mount rim so that they sufficiently constrained the transform; two configurations were compared: three fiducials (approximately 120° spacing) and five fiducials (approximately 70° spacing). After estimating the transform, stage coordinates exported by PiXY were provided to the instrument and the stage was moved. The difference between the intended target positions and the actual reached positions was evaluated as residuals. Measurement points were distributed over approximately 5000 μm in X and Y and 100–400 μm in Z.

Each configuration was evaluated in three runs with different sample orientations (rotation/tilt). In each run, 30–40 target points were measured, yielding N = 100 residuals per configuration (Figure 3). With five fiducials, 50% of residuals were within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% were within 10 μm (X), 16 μm (Y), and 21 μm (Z). With three fiducials, 50% were within 11 μm, 12 μm, and 6 μm, and 90% were within 32 μm, 42 μm, and 20 μm, for X, Y, and Z, respectively. In general, increasing the number of fiducials strengthens the constraint in least-squares estimation and tends to reduce both bias and variability.

Figure 3: Residual histograms for each axis (X, Y, Z) comparing three vs. five fiducial points.

![Figure 3: Residual histograms for each axis.](documentation/images/fig_residual_hist.png)


4. Impact

The primary impact of PiXY is shifting the targeting step in microanalysis from “manual work during instrument time” to “offline work before instrument time”, thereby reducing the burden of positioning and re-identification that can strongly limit machine time. On analytical instruments with limited fields of view, exploring and positioning micrometer-scale targets can be time-consuming, depends on operator experience, and is prone to human errors especially for many targets. PiXY integrates candidate generation (centroid extraction) on pre-acquired images with fiducial-based coordinate transformation, residual inspection, and coordinate export as a single GUI workflow, supporting reduction of instrument operation time beyond measurements.

(1) New research questions enabled: PiXY makes it easier to prepare many measurement-point candidates systematically before instrument loading, supporting a shift from “representative measurements at a small number of points” to “statistical evaluation based on many points” in microanalysis. Examples include dense point selection for particle populations and phase boundaries, sampling designs based on particle size/shape/composition correlations, and evaluation of how fiducial placement and the number of fiducials affect residuals (optimization of fiducial operation). Compared with automatic registration studies using dedicated markers (e.g., autoCRIM [1]), PiXY assumes operation without dedicated markers and focuses on workflow optimization under practical constraints.

(2) How existing work improves: In many microanalysis workflows, relocation of measurement points becomes a bottleneck and can constrain the number of measured points and reduce instrument efficiency. PiXY (i) automatically generates candidates on the image, (ii) allows using four or more fiducials to constrain least-squares estimation, and (iii) supports iterative re-identification and re-estimation while inspecting residuals, concentrating trial-and-error into offline work before instrument time. In the validation (3.3), with five fiducials, 50% of residuals were within 3 μm (X), 3 μm (Y), and 4 μm (Z), and 90% were within 10 μm (X), 16 μm (Y), and 21 μm (Z). These measurements show that increasing the number of fiducials contributes to improving residual distributions.

In an operational example by the author, setting up approximately 200 measurement points may take more than one hour with conventional manual work (searching and relocation on the instrument), whereas using PiXY with offline targeting reduced preparation time to approximately 20 minutes in a practical case. This can lower the barrier to higher-throughput measurements (e.g., >10^3 points).

(3) Changes in day-to-day practice: PiXY manages candidates and fiducials as tables and allows users to correct misidentifications or outliers while inspecting overlays and residuals. Because analysis state can be saved/loaded as a project file (.pixy), it is easier to hand off tasks among personnel and to re-analyze and re-export later (e.g., adding targets or adding fiducials and re-estimating). Fixing the random seed for K-means and saving processing conditions further supports reproducibility of procedures and outputs for the same input image.

(4) Adoption and dissemination: PiXY is publicly available on GitHub and is persistently archived on Zenodo (DOI: 10.5281/zenodo.18174474). Because the public release is recent, it is difficult to provide quantitative metrics such as citation counts at the time of writing. However, software can be cited by DOI with version specification for reproducible use. PiXY is distributed both as Python source and as a standalone Windows executable, enabling use on instrument control PCs without a Python environment. Download and usage statistics can be tracked via GitHub/Zenodo and can be used in future adoption evaluation.

(5) Commercial use and spin-offs: Because PiXY does not depend on manufacturer-specific instrument-control APIs or dedicated markers, it can be used in commercial environments (contract analysis, quality control, research support) as a tool to assist targeting steps that consume instrument time. At present, no spin-off companies or specific commercial productization based on PiXY are known.

Finally, because particle-recognition accuracy for centroid extraction depends on image simplicity and contrast, PiXY is designed to accept externally preprocessed images (e.g., segmentation or binarization) as inputs. Future work includes integrating segmentation functions and further automation through integration with instrument-control APIs.


5. Conclusions

PiXY integrates candidate generation (centroid extraction) and fiducial-based coordinate transformation into a single GUI workflow for practical offline targeting in microanalysis, contributing to reducing manual targeting burden during instrument time.

Validation with real specimens shows that increasing the number of fiducial points improves residual distributions, confirming that PiXY can achieve accuracy sufficient for practical operation. Future work includes integrating preprocessing/segmentation and further automation via instrument-control APIs to reduce labor in targeting.


CRediT author statement

Yoshiaki KON: Conceptualization, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing – original draft, Writing – review and editing.

Declaration of competing interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Acknowledgements

This work was supported by JSPS KAKENHI (Grant Number: 25H00682). The author acknowledges the open-source communities behind Python, OpenCV, NumPy, and PySide6.

During development, the author used AI-assisted tools (GitHub Copilot, Google Gemini, xAI Grok, Anthropic Claude) for manuscript editing suggestions and programming support. All generated outputs were reviewed and validated by the author, who takes full responsibility for the final content.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used GitHub Copilot, Google Gemini, xAI Grok, and Anthropic Claude in order to assist with drafting and programming tasks. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

References

[1] Sheriff J, Fletcher IW, Cumpson PJ. Computer-readable image markers for automated registration in correlative microscopy (autoCRIM) [preprint]. arXiv:2011.14949. 2020. https://doi.org/10.48550/arXiv.2011.14949.

[2] MacQueen JB. Some methods for classification and analysis of multivariate observations. In: Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics. Berkeley, CA: University of California Press; 1967, p. 281-97.

[3] The Qt Company. Qt for Python (PySide6) [software]. 2024. https://wiki.qt.io/Qt_for_Python; [accessed 29 January 2026].

[4] Bradski G. The OpenCV library. Dr Dobb's Journal of Software Tools. 2000.

[5] Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, et al. Array programming with NumPy. Nature. 2020;585(7825):357-62. https://doi.org/10.1038/s41586-020-2649-2.
