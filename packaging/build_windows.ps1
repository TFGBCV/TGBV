<#
.SYNOPSIS
  One-shot local build of the portable GBV Monitor exe bundle (Windows).

.DESCRIPTION
  Creates .venv-build, installs app + build requirements, downloads the llama.cpp
  engine + models, installs the bundled Chromium, runs PyInstaller and assembles
  everything into dist/GBVMonitor/ plus a distributable zip.

  Run from the project root or anywhere:
      powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
  Smaller build (no GPU engine):
      powershell -File packaging\build_windows.ps1 -SkipVulkan

  ~7-8 GB of free disk space and a decent internet connection are required
  (the default chat+vision model is ~6 GB: Q4_K_M GGUF + F16 mmproj tower).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -ChatPrefix Qwen2.5-VL-7B-Instruct-Q4_K_M
  Smaller text-only build (no image screening): -NoVision
#>
[CmdletBinding()]
param(
  [string]$ChatRepo   = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
  [string]$ChatPrefix = "Qwen2.5-VL-7B-Instruct-Q4_K_M",
  [string]$EmbedRepo  = "nomic-ai/nomic-embed-text-v1.5-GGUF",
  [string]$EmbedFile  = "nomic-embed-text-v1.5.Q8_0.gguf",
  [switch]$NoVision,
  [switch]$SkipVulkan,
  [switch]$SkipModels,
  [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# resolve repo root (script lives in packaging/)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
Write-Host "[build] project root: $root" -ForegroundColor Cyan

$py = Join-Path $root ".venv-build\Scripts\python.exe"
if (!(Test-Path $py)) {
  Write-Host "[build] creating build venv ..." -ForegroundColor Cyan
  python -m venv .venv-build
  if ($LASTEXITCODE -ne 0) { throw "python not found/failed to create venv (need Python 3.10-3.13 64-bit)." }
}
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install -r requirements.txt -r packaging\requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }

# ----------------------------------------------------------------- assets
$backendArg = if ($SkipVulkan) { "cpu" } else { "cpu,vulkan" }
$assetArgs  = @("--dest", ".")
if ($SkipModels) { $assetArgs += "--skip-models" }
if ($NoVision)   { $assetArgs += "--no-vision" }
Write-Host "[build] fetching llama.cpp engine + models ..." -ForegroundColor Cyan
& $py packaging\fetch_assets.py --dest . --backends $backendArg `
     --chat-repo $ChatRepo --chat-prefix $ChatPrefix `
     --embed-repo $EmbedRepo --embed-file $EmbedFile `
     @assetArgs
if ($LASTEXITCODE -ne 0) { throw "fetch_assets failed." }

# ----------------------------------------------------------------- chromium
Write-Host "[build] installing the bundled Chromium (Playwright) ..." -ForegroundColor Cyan
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $root "bundle-browsers"
& $py -m playwright install chromium
if ($LASTEXITCODE -ne 0) { throw "playwright install failed." }

# ----------------------------------------------------------------- pyinstaller
Write-Host "[build] PyInstaller packaging (this takes a few minutes) ..." -ForegroundColor Cyan
& $py -m PyInstaller packaging\gbv_monitor.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

if (!(Test-Path "dist\GBVMonitor\GBVMonitor.exe")) { throw "GBVMonitor.exe was not produced -- inspect build warnings above." }

# ----------------------------------------------------------------- assemble dist
$dest = New-Item -ItemType Directory -Force -Path "dist\GBVMonitor"
New-Item -ItemType Directory -Force -Path "$root\dist\GBVMonitor\models"  | Out-Null
New-Item -ItemType Directory -Force -Path "$root\dist\GBVMonitor\output"  | Out-Null

Write-Host "[build] assembling dist\GBVMonitor ..." -ForegroundColor Cyan
& robocopy "$root\runtime"  "$root\dist\GBVMonitor\runtime"  /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy runtime failed ($LASTEXITCODE)." }
if (!$SkipModels) {
  & robocopy "$root\models" "$root\dist\GBVMonitor\models" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy models failed ($LASTEXITCODE)." }
}
& robocopy (Join-Path $root "bundle-browsers") (Join-Path $root "dist\GBVMonitor\ms-playwright") /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy browsers failed ($LASTEXITCODE)." }
Copy-Item "$root\packaging\DIST_README.txt" "$root\dist\GBVMonitor\README.txt" -Force

# ----------------------------------------------------------------- zip
if (!$SkipZip) {
  $zipPath = "$root\dist\GBVMonitor-win64.zip"
  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
  Write-Host "[build] zipping (may take a while for ~6 GB) ..." -ForegroundColor Cyan
  $7z = Get-Command 7z -ErrorAction SilentlyContinue
  if ($7z) {
    & 7z a -mx=1 -y $zipPath "$root\dist\GBVMonitor\*"
  } else {
    Compress-Archive -Path "$root\dist\GBVMonitor\*" -DestinationPath $zipPath
  }
  if ($LASTEXITCODE -ne 0 -and !$7z) { throw "zip failed." }
}

$sizeGB = [math]::Round((Get-ChildItem "$root\dist\GBVMonitor" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host ""
Write-Host "[build] DONE." -ForegroundColor Green
Write-Host "[build] distribution folder: $root\dist\GBVMonitor ($sizeGB GB)"
if (!$SkipZip) { Write-Host "[build] distributable zip: $root\dist\GBVMonitor-win64.zip" }
Write-Host "[build] smoke test: dist\GBVMonitor\GBVMonitor.exe  (console shows model loading, browser opens)"
Write-Host "[build] (headless CI): GBVMonitor.exe --no-browser --port 8502"
