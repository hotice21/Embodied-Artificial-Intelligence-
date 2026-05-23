@echo off
set "PROJECT_ROOT=%~dp0.."
set "MAMBA_ROOT_PREFIX=%PROJECT_ROOT%\.mamba"
"%PROJECT_ROOT%\.tools\Library\bin\micromamba.exe" run -p "%PROJECT_ROOT%\.envs\isaac-serpent" serpent.exe %*
