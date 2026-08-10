# PiXY Official Current Release Marker

Current official release
- Version: 1.5.1
- Release date: 2026-08-10
- Release notes: `RELEASE_NOTES_v1.5.1.md`
- Positioning: Quality-and-correctness update; fixes overlay rendering architecture, Manual Rim add lag, Online XYZ sync, and Fiducial exclusion.

Canonical files for v1.5.1
- Metadata:
  - `pyproject.toml`
  - `CITATION.cff`

Runtime verification checklist
1. Launch PiXY and check the footer text shows `Ver. 1.5.1`.
2. Confirm `pyproject.toml` contains `version = "1.5.1"`.
3. Confirm `CITATION.cff` contains `version: "1.5.1"`.

Notes
- Treat this file as the single source of truth when multiple versioned documents coexist.
- When next release is prepared, update this file first, then docs/metadata.

