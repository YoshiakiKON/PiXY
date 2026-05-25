# PiXY v1.3.2 — Release Notes (2026-02-18)

## Summary

This patch release fixes project save/load and UI asset issues and updates packaging so the Windows executable contains the required UI images.

## Notable changes

- Fix: Restore embedded images on project load — `.pixy` files that embed the input image (base64) are now restored first when loading; if embedded decoding fails the loader falls back to `image_path`.
- Fix: Ensure saved `.pixy` records the active image path (`self.image_path`) for better consistency.
- Fix: Improved Save Project logic to embed images (when available) and warn for large images.
- Fix: EXE packaging: include `PiXY_XY.png` in the PyInstaller `--add-data` assets and add runtime fallback to a generic `PiXY.png` if the mode-specific logo is missing.
- Build: Rebuilt Windows EXE (output: `dist/PiXY_ver124.exe`) including updated assets.
- Docs: Update `InstructionManual_JP.md` and manuscript metadata to v1.3.2.

## Notes for users

- Projects saved with this version will include the embedded input image where possible; copying only the `.pixy` file to another PC is sufficient to reproduce the session without the original image file.
- Older `.pixy` files saved by prior versions may still reference the original image path if the image was not embedded.

## Upcoming 1.3.2 packaging (paper-submission build)

Planned: a forthcoming release `v1.3.2` will contain the exact same code as `v1.3.2` but will be packaged as a minimal archive intended for manuscript submission. That distribution will include only the files required for reproducible execution and review (paper/manuscript-related auxiliary files will be excluded). The purpose is to provide a compact, submission-ready artifact while keeping the code identical to the release here.

If you want me to produce the `v1.3.2` minimal package now (create the archive and a dedicated tag), say so and I will prepare it.

---
Generated: 2026-02-18
