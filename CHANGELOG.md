# Changelog

All notable changes to this project will be documented in this file.

## [1.3.3] - 2026-05-26
### Performance
- Neck separation path was significantly accelerated by processing component masks in ROI space and replacing iterative marker propagation with OpenCV distance-transform label assignment.
- Components smaller than `min_area` now skip neck-splitting work early to avoid unnecessary heavy processing.

### UI behavior
- Kept the active view center stable when adding targets/fiducial points under high zoom to reduce apparent jump after point insertion.
- Harmonized add-point continuation behavior between target and fiducial workflows.

## [Unreleased] - 2026-05-18
### UI redesign: 2-step workflow
- Left panel now shows two clearly labeled sections:
  - **Step 1: Off-line Targeting** (blue header) — project management and target point selection
  - **Step 2: On-line Alignment** (red header) — fiducial point input and coordinate export
- Target point selection is now **manual-first**: `Add Target` is the primary method; auto-detect (centroid extraction) is demoted to an auxiliary collapsible panel.
- Auto-detect panel (`▶ Auto-detect Targets (Auxiliary)`) is **collapsed by default**; click the label to expand.
- New projects loaded via `New Project` no longer auto-run centroid extraction on image open.
- `Strings.py`: added `STEP1_LABEL`, `STEP2_LABEL`, `SECTION_AUTO_DETECT`, `SECTION_AUTO_DETECT_HINT` constants.
- `Ui.py`: `_open_image_from_path` accepts `auto_detect` kwarg (default `False`); pass `True` to restore previous auto-run behaviour.
- `InstructionManual_JP.md`: Quick Start and Workflow sections rewritten to reflect 2-step design.

## [1.2.2] - 2026-02-05
- UI: adjust SegmentControl button widths for better readability (Normal/Flip).
- Docs: minor proofreading and formatting updates in the JOSS draft.

## [1.2.3] - 2026-02-10
- Performance/UI: in Manual mode, slider changes no longer trigger heavy recomputation and avoid full-frame poster resize/composition.
- UI: keep the last rendered posterized overlay/boundaries visible during Manual parameter tweaks (no flicker/disappearance).

## [1.2.1] - 2026-02-03
- Docs: unify terminology to "fiducial point(s)" (naturally occurring specimen features; not pre-made markers).
- Citation: use a fixed Zenodo DOI for all versions (10.5281/zenodo.18174474).
- Packaging: set Windows EXE icon via PyInstaller `--icon` (fixes missing icon in v1.2.0 EXE).

## [1.2.0] - 2026-01-29
- UI: restore v1.1.9-style manual recalculation trigger (Auto / ReCalculate) and improve responsiveness.
- Stability: guards around poster/boundary rendering and recomputation gating in manual mode.
- Packaging: prefer `sys._MEIPASS` for bundled assets (PyInstaller).
- Startup: prefer last opened image; fall back to bundled demo image.
- Defaults: add initial histogram selection caps for grain area.
- UI: show version in footer reliably (EXE + source).

## [1.1.8] - 2026-01-27
- Release: Stable release 1.1.8. Bumped project version and updated JOSS/metadata files.

## [1.1.6] - 2026-01-15
## [1.1.7] - 2026-01-20
- Fix: Guard poster initialization in `_update_image_actual()` to prevent `UnboundLocalError` when nudging group counts or opening images; overlay now falls back to original image when poster is unavailable; boundary rendering guarded.
- Performance: Separate posterization from centroid computation; posterization now updates immediately even in Manual mode (visual feedback), while heavy centroid recompute respects Manual/Auto mode setting.
- Docs: Align English/Japanese JOSS drafts to two-core-feature framing (Feature 1: u,v extraction; Feature 2: u,v→X,Y,Z transform).
- Impl: Rename Implementation headings to "Feature 1" and "Feature 2"; keep algorithm details intact.
- Release: Stable release 1.1.7.
- UI: Remove version from window title; show in footer instead (auto-update on startup)
- UI: Footer labels set to bold for better visibility
- UI: Footer text format changed to "PiXY — © Yoshiaki KON" (initial) and "PiXY (Ver. x.x.x) — © Yoshiaki KON" (after version parse)
- Feature: Improved footer layout with right-aligned credit label

## [1.1.5] - 2026-01-15
- Packaging: build `PiXY_ver115.exe` (single-file) and add multi-size `PiXY_icon.ico` for Windows.
- Packaging: remove PPM fallback assets from the repository and build inputs (use ICO/PNG instead) to reduce bundle size.
- Tooling: added `tools/convert_png_to_ppm.py` to regenerate PPM fallbacks from PNGs if needed.
- Asset: resized `app_icon` to 256×256 to avoid very large uncompressed PPM files.
- Version: bumped project version to 1.1.5 and pushed a git tag `v1.1.5`.
- Note: the built EXE was added to the repository; consider using Git LFS or GitHub Releases for large binaries going forward.

## [1.1.1] - 2026-01-08
- Performance optimization: replaced Python loop with cv2.dilate for fast marker propagation in neck separation
- 10-100x speedup in neck separation calculation

## [1.1.2] - 2026-01-13
- Fix: wheel zoom anchor and center drift on Image/Stage views (align top-left mapping)
- UI: moved Coordinate/Rotate/Flip controls to second header row; Stage status shows Magnification/Rotation/Shift X/Shift Y/Pitch/Roll
- Fix: Magnification displayed in proc (u,v) units to match on-screen conversion

## [1.1.3] - 2026-01-13
- Release: bump version to 1.1.3, packaging rebuild and minor fixes

## [1.1.4] - 2026-01-14
- Fix: Make Image `u,v` display use full-image coordinates (avoid proc/full confusion)
- Fix: Ensure rotation/transformed calls use Qt.TransformationMode and remove stale log confusion
- Dev: Improve INFO/ERROR logging so click/align events appear when DEBUG is off

## [1.1] - 2026-01-08
- Neck separation refinement: avoid double-counting areas when splitting components
- Improved area histogram accuracy: only split areas counted post-split
- Boundary contours drawn per split component with min/max filtering applied

## [0.1.0] - 2025-10-15
- Initial codebase prepared for JOSS submission

## [1.0.1] - 2026-01-01
- UI polish: renamed labels, fixed frozen headers, improved button layout
- Fixed cumulative button width bug and clipboard feedback
- Export now prompts save location; posterization column added
- Boundary rendering adjusted for visual parity at trim_px==0
- Misc. table/column width and header alignment tweaks
