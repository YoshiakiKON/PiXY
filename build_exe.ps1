# PowerShell build script for creating a single-file GUI EXE using PyInstaller
# Usage (run in project root):
#   .\build_exe.ps1 -Clean -Name PiXY

param(
    [switch]$Clean,
    [string]$Name = "PiXY"
)

# Ensure working directory is script directory
Set-Location -Path $PSScriptRoot

# Install build-time dependencies
Write-Host "Installing PyInstaller (if missing)..."
py -m pip install --upgrade pyinstaller --user

if ($Clean) {
    Write-Host "Removing previous build/dist/spec files..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .\build
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .\dist
    Remove-Item -Force -ErrorAction SilentlyContinue .\$Name.spec
}

# Build command notes:
# - --onefile : single exe
# - --windowed: no console (GUI app)
# - --add-data: include image/data files; on Windows use ";" as separator
# Adjust the --add-data entries if you use other assets.

$addData = @(
    "DemoBMP.bmp;.",
    "last_image_path.txt;.",
    "PiXY_splash.png;.",
    "PiXY_icon.ico;.",
    "PiXY.png;.",
    "px2XY2.png;.",
    "px2XY.png;.",
    "app_icon.png;.",
    "splash.ppm;.",
    "app_icon.ppm;."
)

# Build add-data arguments without Join-String (PowerShell 5 compatible)
$addDataArgs = ($addData | ForEach-Object { "--add-data `"$($_)`"" }) -join ' '

# Exclude PyQt bindings to avoid mixed Qt stacks (we use PySide6 via qt_compat)
$exclude = @("PyQt5", "PyQt6")
$excludeArgs = ($exclude | ForEach-Object { "--exclude-module $($_)" }) -join ' '

$main = "Main.py"
$cmd = "py -m PyInstaller --noconfirm --onefile --windowed --name $Name $excludeArgs $addDataArgs $main"
Write-Host "Running: $cmd"
Invoke-Expression $cmd

Write-Host "Build finished. Check the .\dist\$Name.exe file."