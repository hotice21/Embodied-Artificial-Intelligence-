@echo off
set "PROJECT_ROOT=%~dp0.."
set "REDIS_CLI=%PROJECT_ROOT%\.tools\redis\redis-cli.exe"

if exist "%REDIS_CLI%" (
  "%REDIS_CLI%" -h 127.0.0.1 -p 6379 shutdown
  exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$conn = Get-NetTCPConnection -LocalPort 6379 -ErrorAction SilentlyContinue; if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }"
