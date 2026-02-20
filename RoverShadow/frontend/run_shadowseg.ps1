$ErrorActionPreference = 'Stop'
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is not available in PATH. Run `conda activate shadowseg` first.' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'npm is not available in PATH. Install Node.js LTS and retry.' }
if (-not $env:CONDA_DEFAULT_ENV) { throw 'Conda environment is not active. Run `conda activate shadowseg` first.' }
if ($env:CONDA_DEFAULT_ENV -ne 'shadowseg') { Write-Host "Active conda env: $env:CONDA_DEFAULT_ENV" }
Set-Location $PSScriptRoot
python -c "import fastapi, uvicorn, multipart" 2>$null
if ($LASTEXITCODE -ne 0) { python -m pip install -r ..\app\requirements.txt }
if (-not (Test-Path '.\node_modules')) { npm install }
npm run start
