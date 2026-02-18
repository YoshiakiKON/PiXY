import base64
from pathlib import Path
p = Path(__file__).resolve().parent
img = p.parent / 'Screenshot_XY.png'
if not img.exists():
    print('Missing image', img)
    raise SystemExit(1)
b = base64.b64encode(img.read_bytes()).decode()
md = f'''# PiXY v1.3.0 — Release Notes (2026-02-18)

## Software description

PiXY is a desktop GUI application that converts microscopy image pixel coordinates into analytical-instrument stage coordinates. The application integrates candidate generation (particle/centroid extraction via K-means and connected-component analysis), interactive fiducial entry for calibration, and affine transform estimation by least squares to convert image coordinates (u,v) into stage coordinates (X,Y,Z). Output formats include CSV and clipboard-friendly tables for direct import into instrument control software.

This release provides a minimal submission package (see below) that contains only the files required to reproduce and run the software for manuscript review.

## Screenshot (embedded)

![Screenshot_XY](data:image/png;base64,{b})

## Notes

- Change history is documented in `RELEASE_NOTES_v1.2.5.md` — no additional change log is included here.
- This `v1.3.0` release is intended as a minimal archive for paper submission: it contains the same codebase as `v1.2.5` but is packaged with only the files necessary for reproducible execution and review (source files, minimal assets required at runtime, and a README). Non-essential documentation and large auxiliary files (full image sets, intermediate build artifacts, and editorial material) are omitted from the minimal package.

If you want me to create the minimal archive now (tag `v1.3.0`, generate a zip of the selected files, and push the tag), confirm and I will proceed.

---
Generated: 2026-02-18
'''
out = p.parent / 'RELEASE_NOTES_v1.3.0.md'
out.write_text(md, encoding='utf-8')
print('WROTE', out)
