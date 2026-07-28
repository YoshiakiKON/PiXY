# PiXY v1.4.2 — Release Notes (2026-07-28)

## Overview

PiXY is a desktop GUI application for extracting target centroids from microscopy images and converting image coordinates (u, v) to instrument stage coordinates (X, Y, Z) using fiducial-based alignment.

## Highlights in v1.4.2

### Replace Image behavior in centroid extraction mode
- Fixed `Replace Image` flow so centroid extraction now recomputes K-means/centroids immediately when extraction mode is active.
- This prevents stale left-side detection results after image replacement.
- Center list values are preserved while the detection-side list is refreshed for the new image.

### Core/Rim pairing visualization
- Added white connector lines between paired Core and Rim points that belong to the same particle.
- Pairing is resolved from overlay-local source/position mapping to remain consistent with current overlay source.

### Documentation and metadata refresh
- Project metadata bumped to `1.4.2` (`pyproject.toml`, `CITATION.cff`).
- Added `RELEASE_NOTES_v1.4.2.md`.
- Added versioned manuals for `v1.4.2` (English/Japanese, quick/manual variants).

## Packaging

- Project version: `1.4.2`
- Intended Windows EXE output name: `PiXY_ver142.exe`
- Added spec file for this release target: `PiXY_ver142.spec`

## Notes

- See `CHANGELOG.md` for per-change details.
- Zenodo DOI remains fixed across versions: https://doi.org/10.5281/zenodo.18174474
