# -*- mode: python ; coding: utf-8 -*-

import os

_datas = [
    ('DemoBSE.png', '.'),
    ('pyproject.toml', '.'),
    ('last_image_path.txt', '.'),
    ('PiXY_splash.png', '.'),
    ('PiXY_icon.ico', '.'),
    ('PiXY_Pix.png', '.'),
    ('PiXY_XY.png', '.'),
    ('PiXY.png', '.'),
]

for _opt in (
    'startup_image_guard.json',
    'DemoBMP.bmp',
    'DemoBMP.png',
    'splash.ppm',
    'app_icon.ppm',
):
    if os.path.exists(_opt):
        _datas.append((_opt, '.'))


a = Analysis(
    ['Main.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PiXY_ver142',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Python\\PiXY\\PiXY_icon.ico'],
)
