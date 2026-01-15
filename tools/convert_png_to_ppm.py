#!/usr/bin/env python3
"""Convert selected PNG files to binary PPM (P6) for bundling.

Usage:
  python tools/convert_png_to_ppm.py          # auto-detect and regenerate app_icon.ppm and splash.ppm
  python tools/convert_png_to_ppm.py --force  # always overwrite
  python tools/convert_png_to_ppm.py app_icon.ppm:source.png  # custom mapping

The script prefers existing PNG assets in the repository and falls back to skipping if not found.
"""
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    print('Pillow not found. Install with: pip install Pillow')
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAP = {
    'app_icon.ppm': ['PiXY_icon.png', 'PiXY_icon.png', 'PiXY.png', 'SplashIcon.png'],
    'splash.ppm': ['splash.png', 'PiXY_splash.png']
}


def find_source(candidates):
    for name in candidates:
        p = ROOT / name
        if p.exists():
            return p
    return None


def convert(src: Path, dest: Path, force=False):
    if dest.exists() and not force:
        print(f'Skipping {dest.name}: already exists (use --force to overwrite)')
        return False
    im = Image.open(src).convert('RGB')
    im.save(dest, format='PPM')
    print(f'Wrote {dest} from {src.name}')
    return True


def main():
    args = sys.argv[1:]
    force = '--force' in args
    pairs = {}

    # If user provided custom mappings like dest:source, parse them
    for a in args:
        if a == '--force':
            continue
        if ':' in a:
            dest, src = a.split(':', 1)
            pairs[dest] = Path(src)

    # Fill defaults
    for dest, candidates in DEFAULT_MAP.items():
        if dest in pairs:
            continue
        src = find_source(candidates)
        if src:
            pairs[dest] = src

    if not pairs:
        print('No source PNGs found for conversion. Ensure PNG assets exist in project root.')
        sys.exit(0)

    for dest, src in pairs.items():
        destp = ROOT / dest
        convert(src, destp, force=force)


if __name__ == '__main__':
    main()
