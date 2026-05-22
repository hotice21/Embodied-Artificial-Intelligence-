$ProjectRoot = Split-Path -Parent $PSScriptRoot

$GameRootCandidates = @(
    $env:ISAAC_GAME_ROOT,
    (Join-Path $ProjectRoot "The Binding of Isaac Rebirth Repentance"),
    (Join-Path (Split-Path -Parent $ProjectRoot) "The Binding of Isaac Rebirth Repentance")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

if ($GameRootCandidates.Count -eq 0) {
    throw "Game root not found. Set ISAAC_GAME_ROOT to the folder that contains isaac-ng.exe."
}

$GameRoot = $GameRootCandidates[0]
$ModsDir = Join-Path $GameRoot "mods"

$ZipCandidates = @(
    (Join-Path $ProjectRoot "game\mod\szx_chinese_console_3001774454.zip"),
    (Join-Path $ProjectRoot ".team_repo\game\mod\szx_chinese_console_3001774454.zip")
) | Where-Object { Test-Path -LiteralPath $_ }

if ($ZipCandidates.Count -eq 0) {
    throw "Missing teammate mod zip. Expected game\mod\szx_chinese_console_3001774454.zip."
}

$ZipPath = $ZipCandidates[0]
$InstallDir = Join-Path $ModsDir "szx_chinese_console_3001774454"
$BackupRoot = Join-Path $ProjectRoot "backups\mods"

New-Item -ItemType Directory -Force -Path $ModsDir | Out-Null
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

if (Test-Path -LiteralPath $InstallDir) {
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BackupDir = Join-Path $BackupRoot "szx_chinese_console_3001774454-$Stamp"
    Move-Item -LiteralPath $InstallDir -Destination $BackupDir
    Write-Host "Backed up old mod to $BackupDir"
}

Expand-Archive -LiteralPath $ZipPath -DestinationPath $ModsDir -Force
Write-Host "Installed teammate mod to $InstallDir"
