# PiXY v1.1.6 Release Notes

**Release Date**: January 16, 2026

## Overview
PiXY v1.1.6 is a maintenance and UI enhancement release focused on improving footer visibility, removing unnecessary window title clutter, and providing better visual feedback through dynamic version display.

## New Features
- **Dynamic Footer Version Display**: Version information is now automatically parsed from `pyproject.toml` and displayed in the footer at startup.
- **Cleaner Window Title**: Removed version string from the window title bar for a minimalist interface.
- **Bold Footer Labels**: Enhanced footer text visibility with bold font rendering for both status and credit labels.

## Improvements

### User Interface Enhancements
- **Window Title**: Now displays only "PiXY" without the version number for a cleaner appearance.
- **Footer Layout**:
  - **Left label**: Status messages displayed in bold, white, 11pt font
  - **Right-aligned label**: Application credit "PiXY — © Yoshiaki KON" (initial display)
  - **Dynamic update**: After startup version parsing, displays "PiXY (Ver. 1.1.6) — © Yoshiaki KON" with actual version
- **Typography**: Both footer labels render in bold for improved readability and emphasis.

### Code Changes
- Modified `Ui.py`:
  - Updated window title initialization to remove version string
  - Added `self._app_version` attribute to store parsed version from `pyproject.toml`
  - Enhanced `Footer` class with bold font styling for both status and credit labels
  - Implemented auto-update logic for footer credit label to display version information
  - Version parsing happens during application initialization and updates footer dynamically
- Updated `pyproject.toml`:
  - Version bumped from 1.1.5 to 1.1.6

### Documentation & Build
- Updated `CHANGELOG.md` with v1.1.6 entry
- Enhanced `build_exe.ps1` with corrected asset file references
- Generated `PiXY_ver116.exe` (single-file Windows executable, ~94 MB)

## Technical Details

### Version Display Logic
1. **Startup**: Application parses `pyproject.toml` to extract version
2. **Footer Init**: Credit label initializes with "PiXY — © Yoshiaki KON"
3. **Post-Parse**: If version successfully extracted, footer updates to "PiXY (Ver. X.X.X) — © Yoshiaki KON"

### Build Specifications
- **Executable**: `PiXY_ver116.exe` (single-file, no installation required)
- **Size**: ~94 MB
- **Python Version**: 3.8+
- **Key Dependencies**: numpy, opencv-python, PySide6
- **Target Platform**: Windows 11/10

## Known Issues
None reported for this release.

## Installation & Usage

### Option 1: Standalone Executable
1. Download `PiXY_ver116.exe` from the release
2. Run directly (no installation required)
3. Provide an image file when prompted to extract centroids

### Option 2: From Source
```bash
git clone <repository-url>
cd Px2XY
git checkout v1.1.6
python Main.py
```

## Contributors
- **Maintainer**: Yoshiaki KON
- **Version**: 1.1.6
- **Release**: January 16, 2026

## Feedback & Support
For issues, bug reports, or feature requests, please create an issue in the GitHub repository.

---

**Build Information**
- Commit: `c8c268b`
- Tag: `v1.1.6`
- Build Date: 2026-01-16
- Builder: PyInstaller 6.18.0
