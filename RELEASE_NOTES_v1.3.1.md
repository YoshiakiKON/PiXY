# PiXY v1.3.1 — Release Notes (2026-02-18)

## Software description

PiXY is a desktop GUI application that converts microscopy image pixel coordinates into analytical-instrument stage coordinates. The application integrates candidate generation (particle/centroid extraction via K-means and connected-component analysis), interactive fiducial entry for calibration, and affine transform estimation by least squares to convert image coordinates (u,v) into stage coordinates (X,Y,Z). Output formats include CSV and clipboard-friendly tables for direct import into instrument control software.

This release provides a minimal submission package (see below) that contains only the files required to reproduce and run the software for manuscript review.

## Screenshot

![Screenshot_XY](Screenshot_XY.png)

## Notes

- Change history is documented in `RELEASE_NOTES_v1.3.1.md` — no additional change log is included here.
- This `v1.3.1` release is intended as a minimal archive for paper submission: it contains the same codebase as `v1.3.1` but is packaged with only the files necessary for reproducible execution and review (source files, minimal assets required at runtime, and a README). Non-essential documentation and large auxiliary files (full image sets, intermediate build artifacts, and editorial material) are omitted from the minimal package.

If you want me to create the minimal archive now (tag `v1.3.1`, generate a zip of the selected files, and push the tag), confirm and I will proceed.

---
Generated: 2026-02-18
