$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:MAMBA_ROOT_PREFIX = Join-Path $ProjectRoot ".mamba"
$Micromamba = Join-Path $ProjectRoot ".tools\Library\bin\micromamba.exe"
$EnvPath = Join-Path $ProjectRoot ".envs\isaac-serpent"

if (-not (Test-Path -LiteralPath $Micromamba)) {
    throw "micromamba not found: $Micromamba"
}

if (-not (Test-Path -LiteralPath $EnvPath)) {
    throw "environment not found: $EnvPath"
}

(& $Micromamba shell hook -s powershell) | Out-String | Invoke-Expression
micromamba activate $EnvPath
Write-Host "Activated isaac-serpent: $EnvPath"
