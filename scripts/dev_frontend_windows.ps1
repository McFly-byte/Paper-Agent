$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WebDir = Join-Path $ProjectRoot "web"
Set-Location $WebDir

if (-not (Test-Path "node_modules")) {
    Write-Host "node_modules not found. Run npm install in the web directory first."
    exit 1
}

npm run dev
