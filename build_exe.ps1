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
    "OD3_BEC.bmp;.",
    "Otake-fresh-3_XBSE_STD_MINERAL_SAMPLE.bmp;.",
    "水玉-01.png;.",
    "last_image_path.txt;."
)

$addDataArgs = $addData | ForEach-Object { "--add-data `"$_`"" } | Join-String ' '

$main = "Main.py"
$cmd = "py -m PyInstaller --noconfirm --onefile --windowed --name $Name $addDataArgs $main"
Write-Host "Running: $cmd"
Invoke-Expression $cmd

Write-Host "Build finished. Check the .\dist\$Name.exe file."