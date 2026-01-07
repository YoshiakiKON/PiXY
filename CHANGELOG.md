# Changelog

All notable changes to this project will be documented in this file.

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
