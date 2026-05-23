@echo off
set "PROJECT_ROOT=%~dp0.."
set "REDIS_DIR=%PROJECT_ROOT%\.tools\redis"
set "REDIS_SERVER=%REDIS_DIR%\redis-server.exe"
set "REDIS_CONF=%REDIS_DIR%\redis.isaac.conf"

if not exist "%REDIS_SERVER%" (
  echo redis-server.exe not found: %REDIS_SERVER%
  echo Run .\tools\install_redis.cmd first.
  exit /b 1
)

if not exist "%REDIS_CONF%" (
  > "%REDIS_CONF%" echo bind 127.0.0.1
  >> "%REDIS_CONF%" echo port 6379
  >> "%REDIS_CONF%" echo timeout 0
  >> "%REDIS_CONF%" echo tcp-keepalive 0
  >> "%REDIS_CONF%" echo loglevel notice
  >> "%REDIS_CONF%" echo logfile redis.isaac.log
  >> "%REDIS_CONF%" echo databases 16
  >> "%REDIS_CONF%" echo save ""
  >> "%REDIS_CONF%" echo appendonly no
  >> "%REDIS_CONF%" echo dir ./
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$existing = Get-NetTCPConnection -LocalPort 6379 -ErrorAction SilentlyContinue; if ($existing) { Write-Host 'Redis port 6379 already in use.'; exit 0 }; Start-Process -FilePath '%REDIS_SERVER%' -ArgumentList 'redis.isaac.conf' -WorkingDirectory '%REDIS_DIR%' -WindowStyle Hidden; Start-Sleep -Seconds 1; $conn = Get-NetTCPConnection -LocalPort 6379 -ErrorAction SilentlyContinue; if ($conn) { Write-Host 'Redis started on 127.0.0.1:6379'; exit 0 } else { Write-Host 'Redis did not start.'; exit 1 }"
