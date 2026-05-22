param(
    [Parameter(Mandatory = $true)]
    [string] $SaveSource
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = Resolve-Path -LiteralPath $SaveSource -ErrorAction Stop
$BackupRoot = Join-Path $ProjectRoot "backups\saves"
$CodexRoot = "C:\Users\Public\Documents\Steam\CODEX\250900"
$CodexRemote = Join-Path $CodexRoot "remote"
$DocumentsRoot = Join-Path $env:USERPROFILE "Documents\My Games\Binding of Isaac Repentance"

function Backup-Directory {
    param(
        [string] $Path,
        [string] $Name
    )

    if (Test-Path -LiteralPath $Path) {
        $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $Target = Join-Path $BackupRoot "$Name-$Stamp"
        Copy-Item -LiteralPath $Path -Destination $Target -Recurse -Force
        Write-Host "Backed up $Path to $Target"
    }
}

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
Backup-Directory -Path $CodexRoot -Name "codex-250900"
Backup-Directory -Path $DocumentsRoot -Name "documents-repentance"

if (Test-Path -LiteralPath $CodexRemote) {
    $Destination = $CodexRemote
} elseif (Test-Path -LiteralPath $CodexRoot) {
    $existingSave = Get-ChildItem -LiteralPath $CodexRoot -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "persistentgamedata|gamestate|options|rep_" } |
        Select-Object -First 1
    if ($existingSave) {
        $Destination = $CodexRoot
    } else {
        $Destination = $CodexRemote
    }
} else {
    $Destination = $CodexRemote
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Item = Get-Item -LiteralPath $SourcePath
if ($Item.PSIsContainer) {
    Copy-Item -LiteralPath (Join-Path $Item.FullName "*") -Destination $Destination -Recurse -Force
} else {
    Copy-Item -LiteralPath $Item.FullName -Destination $Destination -Force
}

Write-Host "Imported save from $SourcePath to $Destination"
Write-Host "Start the game once and verify the save slot before doing experiments."
