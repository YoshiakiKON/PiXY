<#
Windows PowerShell helper to convert `paper.md` -> `paper_softwarex.docx` using Pandoc.

Prerequisites:
- Install Pandoc: https://pandoc.org/installing.html (Windows installer)
- (Optional) If you have SoftwareX Word template, set `$referenceDoc` to its path.

Usage:
.
.
  .\build_docx.ps1

#>

$ErrorActionPreference = 'Stop'

Write-Host "Starting DOCX build: paper.md -> paper_softwarex.docx"

# Locate pandoc
$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    Write-Error "pandoc not found. Please install Pandoc and ensure it's on PATH: https://pandoc.org/installing.html"
    exit 1
}

# Paths
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$md = Join-Path $root 'paper.md'
$out = Join-Path $root 'paper_softwarex.docx'
$bib = Join-Path $root 'paper.bib'
$respath = Join-Path $root 'documentation/images'

# Optional: reference docx (SoftwareX Word template). Leave empty to use default styling.
$referenceDoc = "" # e.g. 'SoftwareX_template.docx'

if (!(Test-Path $md)) { Write-Error "paper.md not found at $md"; exit 1 }

$args = @(
    $md,
    '-s',
    '-o', $out,
    '--resource-path=' + $respath,
    '--citeproc'
)

if (Test-Path $bib) { $args += @('--bibliography', $bib) }
if ($referenceDoc -ne "" -and (Test-Path $referenceDoc)) { $args += @('--reference-doc', $referenceDoc) }

Write-Host "Running: pandoc $($args -join ' ')"
& pandoc @args

if (Test-Path $out) {
    Write-Host "Generated: $out"
} else {
    Write-Error "DOCX generation failed. Check pandoc output above."
    exit 1
}
