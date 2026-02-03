# PiXY: Pixel to stage-XY Coordinate Converter for Granular Material Analysis
## JOSS Paper Draft - Table of Contents & Outline

---

## 📑 Paper Structure (Target: 4-5 pages)

### 1. **Summary** (0.3 pages)
- Brief overview of PiXY and its purpose
- Key contribution to image analysis workflow

### 2. **Statement of Need** (0.7 pages)
- Challenge in automated grain/particle detection and coordinate conversion
- Existing limitations:
  - **Time bottleneck in microanalysis**: Positioning is the critical bottleneck in analytical instruments; efficient target positioning is essential for throughput improvement
  - **FIJI (ImageJ)** particle analysis exists but requires expertise and difficult parameter adjustment
  - **Python clustering libraries** are available but lack GUI + BSE-specific optimization
  - **Commercial software** (e.g., image-analysis suites) is expensive and proprietary
- **PiXY's unique contribution**: Given pre-acquired images of analysis targets, PiXY performs particle detection and coordinate transformation to instrument-ready coordinate systems in a single workflow
- **Broad applicability**: Compatible with any microanalytical instrument having XYZ coordinates (LA-ICP-MS, SIMS, EPMA, etc.)
- **Open-source advantage**: Transparent, modifiable, and freely accessible


### 3. **Implementation** (1.5 pages)

#### 3.1 **Particle Detection Algorithm** (0.7 pages)
- **K-means Color Clustering**:
  - Why K-means for BSE microscopy: contrast-based segmentation without manual thresholding
  - Implementation: `cv2.kmeans()` with PP-Centers initialization for stability
  - User-adjustable cluster count (2–20 groups)
  - Posterization output: discrete color regions corresponding to mineral phases

- **Connected Component Analysis**:
  - 4-connectivity labeling on posterized image
  - Each connected region = candidate particle
  
- **Filtering Pipeline**:
  - Min/Max area thresholds: remove noise and excessively large artifacts
  - Boundary offset (erosion): refine particle edges
  
- **Neck Separation** (optional):
  - Morphological erosion detects constriction points between touching particles
  - Fast marker propagation using `cv2.dilate` splits necked regions
  - Enables accurate counting of partially overlapping grains

- **Centroid Extraction**:
  - Centroid coordinates computed for each filtered component
  - Sub-pixel accuracy via moment calculations

#### 3.2 **Coordinate Transformation** (0.8 pages)
- **Coordinate Systems**:
  - **Image coordinates**: Pixel-based (origin at top-left)
  - **Stage coordinates**: Physical instrument reference frame (origin user-defined)
  
- **Transformation Pipeline**:
  1. Processing image (downscaled) → Full-resolution image scaling
  2. Full-resolution → Stage coordinates via affine transformation
  3. Affine parameters estimated from fiducial point pairs (user-defined)
  
- **Mathematical Framework**:
  - 2D affine transformation: $(x', y') = A \cdot (x, y) + b$
  - Least-squares estimation from ≥3 fiducial point pairs
  - Handles rotation, translation, scaling, and shear
  
- **Software Architecture**:
  - GUI framework: PySide6 (Qt for Python)
  - Real-time visualization with overlay
  - Parameter controls: Number of Groups, Min/Max Area, Boundary Offset, Neck Separation
  
- **Key Features**:
  - Interactive mode switching (Auto/Manual recalculation)
  - Image rotation and flipping
  - Multiple export formats (CSV, clipboard)
  - Performance: <1 second for typical 1560×1920 BSE images

### 4. **Validation & Results** (1.5 pages)

#### 4.1 **Experimental Setup**
- **Sample Material**: [Material type]
- **Imaging Method**: Backscattered Electron (BSE) microscopy
- **Image Characteristics**:
  - Resolution: [X×X pixels]
  - Contrast levels: [description]
  - Number of validation images: 10 representative samples
  
#### 4.2 **Quantitative Results**
- **Comparison Metrics**:
  - **Reproducibility/consistency** across images
  - **Positional accuracy**: Same sample tested 4 times with different orientations (RefPoint → coordinate transformation → position verification on analytical instrument)
  - **Deviation assessment**: Report degree of positional offset on analytical instrument stage
  - **Detection accuracy**: False positive/negative rates (if available)
  
- **Figures** (with captions):
  - **Figure 1**: Representative BSE image with detected particles overlay (3-4 samples)
  - **Figure 2**: Grain size distribution histogram (aggregated results)
  - **Figure 3**: GUI interface screenshot
  - **Figure 4**: Processing pipeline diagram (optional)

#### 4.3 **Computational Performance**
- Benchmarks on different image sizes
- CPU/memory requirements
- Standalone EXE availability

### 5. **Limitations & Future Work** (0.3 pages)
- Current limitations (if any)
- Potential extensions:
  - Additional detection methods (Edge detection, Watershed, ML-based)
  - GPU acceleration
  - Batch processing
  - 3D support

### 6. **Conclusions** (0.2 pages)
- Summary of capabilities
- Availability and open-source license
- Call for community feedback

### 7. **Acknowledgments** (0.1 pages)
- Funding, institutions, collaborators

### 8. **References** (0.5-1 page)
- OpenCV, NumPy, PySide6
- Related particle detection papers
- Similar commercial/academic tools

---

## 📊 Figure Placeholders

| Figure | Description | Notes |
|--------|-------------|-------|
| Fig 1  | Raw BSE image + detected particles overlay | Show 3-5 representative samples |
| Fig 2  | Grain size distribution (histogram) | Aggregate stats from 10 samples |
| Fig 3  | GUI screenshot | Show key controls and visualization |
| Fig 4  | Algorithm flow diagram (optional) | K-means → components → filtering |

---

## 💾 Data to Prepare

- [ ] 10 representative BSE images (raw + processed)
- [ ] Quantitative results:
  - Particle counts per image
  - Mean/min/max grain sizes
  - Processing times
  - Accuracy metrics (vs. manual counting if available)
- [ ] GUI screenshots showing:
  - Main window with image
  - Parameter controls
  - Overlay visualization
  - Results table

---

## ✍️ Writing Style for JOSS

- **Audience**: Software engineers, researchers in image analysis
- **Tone**: Professional but accessible
- **Format**: Markdown (JOSS uses GitHub-flavored Markdown)
- **Equations**: LaTeX inline format (e.g., $K$-means)
- **Citations**: BibTeX format

---

## 🎯 Next Steps

1. **Write Section 1-2** (Summary + Statement of Need)
2. **Complete Section 3** (Implementation details)
3. **Prepare figures** from your BSE validation images
4. **Compile quantitative results** (Section 4)
5. **Draft Section 5-8** (Conclusions, etc.)
6. **Submit to JOSS** with GitHub repo link + Zenodo link

---

このアウトラインでよろしいですか？それとも、特定のセクションを先に詳細に書きますか？
