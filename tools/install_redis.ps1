$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DownloadsDir = Join-Path $ProjectRoot ".downloads"
$RedisDir = Join-Path $ProjectRoot ".tools\redis"
$AssetName = "Redis-x64-5.0.14.1.zip"
$Url = "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/$AssetName"
$ZipPath = Join-Path $DownloadsDir $AssetName

New-Item -ItemType Directory -Force -Path $DownloadsDir, $RedisDir | Out-Null

if (-not (Test-Path -LiteralPath $ZipPath)) {
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath
}

Remove-Item -LiteralPath $RedisDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $RedisDir | Out-Null
Expand-Archive -LiteralPath $ZipPath -DestinationPath $RedisDir -Force

$ConfigPath = Join-Path $RedisDir "redis.isaac.conf"
@"
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 0
loglevel notice
logfile redis.isaac.log
databases 16
save ""
appendonly no
dir ./
"@ | Set-Content -Path $ConfigPath -Encoding ASCII

Write-Host "Redis installed to $RedisDir"
Write-Host "Run .\tools\start_redis.cmd to start it."
