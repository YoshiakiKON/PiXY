# PiXY Official Current Release Marker

Current official release
- Version: 1.5.0
- Release date: 2026-08-05
- Release notes: `RELEASE_NOTES_v1.5.0.md`
- Positioning: Official stable release with restored online alignment overlay controls.

Canonical files for v1.5.0
- Metadata:
  - `pyproject.toml`
  - `CITATION.cff`
- Manuals:
  - `InstructionManual_EN_v1.5.0.md`
  - `InstructionManual_JP_v1.5.0.md`
  - `documentation/InstructionManual_EN_v1.5.0.html`
  - `documentation/InstructionManual_JP_v1.5.0.html`
- Quick manuals:
  - `documentation/QuickManual_EN_v1.5.0.md`
  - `documentation/QuickManual_JP_v1.5.0.md`

Runtime verification checklist
1. Launch PiXY and check the footer text shows `Ver. 1.5.0`.
2. Confirm `pyproject.toml` contains `version = "1.5.0"`.
3. Confirm `CITATION.cff` contains `version: "1.5.0"`.

Notes
- Treat this file as the single source of truth when multiple versioned documents coexist.
- When next release is prepared, update this file first, then docs/metadata.

