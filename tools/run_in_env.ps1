param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Command
)

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

if ($Command.Count -eq 0) {
    Write-Host "Usage: .\tools\run_in_env.ps1 <command> [args...]"
    Write-Host "Example: .\tools\run_in_env.ps1 python --version"
    exit 1
}

& $Micromamba run -p $EnvPath @Command
exit $LASTEXITCODE
