# PiXY Official Current Release Marker

Current official release
- Version: 1.4.4
- Release date: 2026-08-04
- Release notes: `RELEASE_NOTES_v1.4.4.md`
- Positioning: Pre-release for the upcoming v1.5 line.

Canonical files for v1.4.4
- Metadata:
  - `pyproject.toml`
  - `CITATION.cff`
- Manuals:
  - `InstructionManual_EN_v1.4.4.md`
  - `InstructionManual_JP_v1.4.4.md`
  - `documentation/InstructionManual_EN_v1.4.4.html`
  - `documentation/InstructionManual_JP_v1.4.4.html`
- Quick manuals:
  - `documentation/QuickManual_EN_v1.4.4.md`
  - `documentation/QuickManual_JP_v1.4.4.md`

Runtime verification checklist
1. Launch PiXY and check the footer text shows `Ver. 1.4.4`.
2. Confirm `pyproject.toml` contains `version = "1.4.4"`.
3. Confirm `CITATION.cff` contains `version: "1.4.4"`.

Notes
- Treat this file as the single source of truth when multiple versioned documents coexist.
- When next release is prepared, update this file first, then docs/metadata.

