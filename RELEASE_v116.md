# PiXY v1.1.6 Release Notes

**Release Date**: January 15, 2026

## Overview
PiXY v1.1.6 is a maintenance and UI improvement release focused on enhancing footer visibility and removing unnecessary clutter from the window title bar.

## New Features
- **Footer credit display**: Moved version information from the window title bar to the footer for a cleaner interface.
- **Auto-updating version**: Footer displays version number automatically parsed from `pyproject.toml` on startup.
- **Bold footer labels**: Improved footer text visibility with bold font rendering.

## Changes

### UI Improvements
- **Window title**: Removed "(Ver. x.x.x)" from the window title bar — now displays only "PiXY".
- **Footer layout**: 
  - Left label for status messages (bold, white, 11px)
  - Right-aligned credit label (bold, white, 11px) showing "PiXY — © Yoshiaki KON" initially
  - After startup, footer updates to show "PiXY (Ver. 1.1.6) — © Yoshiaki KON" with the actual parsed version
- **Font styling**: Both footer labels now render in bold for improved readability.

### Code Changes
- Modified `Ui.py`:
  - Updated window title initialization to exclude version string
  - Added `self._app_version` attribute to store parsed version from `pyproject.toml`
  - Modified `Footer` class to include bold font styling for both labels
  - Added auto-update logic for footer credit label with version info
- Updated `pyproject.toml` version from 1.1.5 to 1.1.6

### Documentation
- Updated `CHANGELOG.md` with v1.1.6 entry

## Known Issues
None reported for this release.

## Build Information
- **Executable**: `PiXY_ver116.exe` (single-file Windows executable)
- **Python Version**: 3.8+
- **Dependencies**: numpy, opencv-python, PySide6

## Installation & Usage
1. Download `PiXY_ver116.exe`
2. Run directly (no installation required)
3. Provide an image file when prompted to extract centroids

## Feedback & Support
For issues or feature requests, please create an issue in the GitHub repository.

---
**Build Date**: 2026-01-15
**Maintainer**: Yoshiaki KON
