Packaging to EXE (Windows)

This project is a PyQt5 application. To produce a single-file Windows executable, use PyInstaller.

Prerequisites
- Python 3.x installed and `py` launcher available in PATH (Windows)
- Recommended: create a virtualenv

Quick steps (PowerShell)

1. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

2. Install PyInstaller (if you don't have it):

```powershell
py -m pip install pyinstaller
```

3. Run the build script in the project root (this script invokes PyInstaller and bundles common image/data files):

```powershell
.\build_exe.ps1 -Name Px2XY
```

After build completes, the single EXE will be in `dist\Px2XY.exe`.

Notes and tips
- If your application loads external image files (bmp/png), include them using `--add-data` (the provided `build_exe.ps1` already includes some common ones). On Windows, separate source;dest by `;`.
- PyInstaller may need Qt plugins (platforms). If the EXE fails to start with Qt platform errors, try building without `--onefile` to inspect the bundled tree, or add the `--paths` option pointing to the PyQt5 plugins folder. Example troubleshooting commands are in PyInstaller docs.
- To debug build problems, run PyInstaller without `--windowed` to see console output.

If you want, I can:
- Run the PyInstaller build here and verify the produced EXE (needs time and may produce a large binary).
- Generate a `.spec` file customized to include additional data files or hidden imports (e.g. for OpenCV/PyQt hooks).
- Add an icon by supplying `--icon=app.ico` and placing `app.ico` in the project root.

Tell me whether you want me to run the build here, or if you prefer instructions only.