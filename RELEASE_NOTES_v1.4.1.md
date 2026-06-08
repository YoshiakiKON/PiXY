# PiXY v1.4.1 — Release Notes (2026-06-08)

## Overview

PiXY is a desktop GUI application for extracting target centroids from microscopy images and converting image coordinates (u, v) to instrument stage coordinates (X, Y, Z) using fiducial-based alignment.

## Highlights in v1.4.1

### Middle column width and scrollbar stability
- Unified middle-column width ownership to `_adjust_center_column_widths()` to avoid conflicting fixed-width overwrites.
- Removed obsolete center extra-width constants and simplified width calculation to measured table/layout dimensions.
- Preserved scrollbar visibility behavior while preventing right-edge clipping in the middle table.

### Center-row XYZ consistency after centroid-based add
- Fixed a case where points added from centroid extraction could keep blank `X/Y/Z` in the middle table.
- Middle-table rows now synchronize `X/Y/Z` from the canonical right table whenever transposed views are refreshed.

### Numeric readability improvements
- Middle-table `X/Y/Z` display now uses integer formatting when `|value| >= 100`.
- For long numeric text that would otherwise be elided, font size is reduced only for affected `X/Y/Z` cells.

## Packaging

- Project version: `1.4.1`
- Intended Windows EXE output name: `PiXY_ver141.exe`

## Notes

- See `CHANGELOG.md` for detailed per-change history.
- Zenodo DOI remains fixed across versions: https://doi.org/10.5281/zenodo.18174474
