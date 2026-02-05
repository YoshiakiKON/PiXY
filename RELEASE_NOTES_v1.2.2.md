# PiXY v1.2.2 — Release Notes (2026-02-05)

- Standalone Windows executable: `PiXY_ver122.exe`
- Source: Python code in this repository
- Zenodo DOI (fixed across versions): https://doi.org/10.5281/zenodo.18174474

---

## What’s New in v1.2.2

- **UI tweak**: SegmentControl button widths adjusted for clearer labeling (e.g., Normal/Flip).
- **Docs**: Minor proofreading and formatting updates in the JOSS draft.

---

## Download / Run

1. Download `PiXY_ver122.exe` from GitHub Releases.
2. If Windows shows a security prompt, use the file properties “Unblock” option (if present), then run.

---

## Build (developers)

PowerShell:

```powershell
.\build_exe.ps1 -Clean -Name PiXY_ver122
```

Output:
- `dist/PiXY_ver122.exe`
