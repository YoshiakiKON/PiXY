# PiXY v1.5.5
![PiXY header](PiXY_Header.png)

**Release date:** 2026-08-26

## Overview

PiXY (Pixel to stage-XY Coordinate Converter) is an open-source graphical user interface (GUI) for offline targeting in microanalysis. It enables measurement positions to be selected on pre-acquired sample images before sample loading and subsequently converted into instrument stage coordinates during online alignment.

This release focuses on consistency between the on-screen marker display and the exported image output. The main goal is to ensure that exported images reflect the same visible target points and labels that the user sees in the UI, without drift caused by stale or differently filtered centroid sets.

## What changed in v1.5.5

### Export-image consistency fix

- Exported images now prefer the current overlay-render payload used by the UI rather than an older raw centroid list.
- Marker labels now use the same overlay label text and ordering when available.
- Point colors and sizes were adjusted to remain closer to the display appearance used by the live UI.
- Label text is kept compact enough to avoid oversized overlap on exported full-resolution images.

### User-visible behavior

- Online mode still exports in Image coordinate space, matching the original image pixels used for display overlays.
- The resulting image now better matches what the operator sees in the UI when evaluating extracted points.
- The export is more stable when filters, manual targets, or group visibility settings modify the visible set.

## Included files

- Standalone Windows executable: `PiXY_ver155.exe`
- Source code for running PiXY in a Python environment
- `InstructionManual_EN_v1.5.5.md`
- `InstructionManual_JP_v1.5.5.md`

**Recommended screen size:** 1200 x 900 pixels or larger.

## Citation

If you use PiXY in published research, please cite the software and the associated publication.

**DOI:** 10.5281/zenodo.18174474

See `CITATION.cff` for the recommended citation metadata.

## Platform

The pre-built executable is provided for Windows. Users can also run PiXY from the source code in a Python environment.

## Documentation

The manuals provide a step-by-step description of the offline targeting and online alignment workflow, including image-based target extraction, manual target selection, fiducial-point registration, and coordinate export.
