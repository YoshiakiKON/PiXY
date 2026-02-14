# PiXY v1.2.3 — Release Notes (2026-02-10)

This release focuses on UI/interaction responsiveness and visual stability during manual parameter adjustments.

## What’s New

- Performance/UI: in Manual mode, slider changes no longer trigger heavy recomputation and avoid full-frame poster resize/composition.
- UI: keep the last rendered posterized overlay/boundaries visible during Manual parameter tweaks (no flicker/disappearance).

## Notes

- Build: single-file Windows EXE can be produced via `build_exe.ps1` (uses PyInstaller). The build was created locally and is available in `dist/`.
- Version metadata and documentation were updated to `1.2.3`.

-- Yoshiaki KON
