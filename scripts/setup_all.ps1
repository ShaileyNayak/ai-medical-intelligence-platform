# Setup script for Windows (PowerShell)
# Usage: .\scripts\setup_all.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$PY = "C:\Users\SHAILEY NAYAK\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $PY)) { $PY = "python" }

Write-Host "==> Copying env files"
Copy-Item "$ROOT\.env.example" "$ROOT\.env" -Force
Copy-Item "$ROOT\backend\.env.example" "$ROOT\backend\.env" -Force

Write-Host "==> Backend venv + deps"
& $PY -m venv "$ROOT\backend\.venv"
& "$ROOT\backend\.venv\Scripts\pip.exe" install -r "$ROOT\backend\requirements.txt"

Write-Host "==> Model venv + data + train"
& $PY -m venv "$ROOT\model\.venv"
& "$ROOT\model\.venv\Scripts\pip.exe" install -r "$ROOT\model\requirements.txt"
& "$ROOT\model\.venv\Scripts\python.exe" "$ROOT\model\prepare_demo_data.py"
& "$ROOT\model\.venv\Scripts\python.exe" "$ROOT\model\train.py" --data-dir "$ROOT\data" --epochs 3 --batch-size 8

Write-Host "==> Frontend npm install"
Push-Location "$ROOT\frontend"
npm install
Pop-Location

Write-Host "==> Done. Start backend: cd backend; .\.venv\Scripts\uvicorn.exe app.main:app --reload"
Write-Host "             Start UI:     cd frontend; npm run dev"
