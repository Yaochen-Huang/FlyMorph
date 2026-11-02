# Build FlyMorph.exe (PyInstaller, one-folder). Run inside a conda env that has
# PySide6 + onnxruntime + pyinstaller installed:
#   conda activate eyeapp
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$root  = Split-Path -Parent $here          # repository root
$model = Join-Path $root "models\eye_unet_v4.onnx"
if (-not (Test-Path $model)) { throw "model not found: $model" }

python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { python -m pip install pyinstaller }

Remove-Item -Recurse -Force build, dist, FlyMorph.spec -ErrorAction SilentlyContinue

python -m PyInstaller `
  --noconfirm --clean --windowed --uac-admin `
  --name FlyMorph `
  --paths "$here" `
  --add-data "$model;." `
  --hidden-import core `
  --hidden-import core.infer `
  --hidden-import core.tables `
  --hidden-import editor `
  --hidden-import onnxruntime `
  --hidden-import onnxruntime.capi._pybind_state `
  --collect-submodules onnxruntime `
  --exclude-module torch --exclude-module matplotlib --exclude-module tkinter `
  main.py

if (-not (Test-Path "dist\EyeArea\EyeArea.exe")) { throw "build failed" }
$size = (Get-ChildItem "dist\EyeArea" -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("OK -> dist\EyeArea\EyeArea.exe   ({0:N0} MB total)" -f $size)
