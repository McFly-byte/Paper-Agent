$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Write-Host ".env not found. Copy .env.example to .env and fill in your API key before running the backend."
    exit 1
}

poetry run python main.py
