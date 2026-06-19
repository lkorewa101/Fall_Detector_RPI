@echo off
setlocal

set "FW_DIR="
for /d %%D in ("%~dp0IWR6843ISK_C3CD_*") do (
  if exist "%%~fD\VERIFY_PACKAGE.ps1" set "FW_DIR=%%~fD"
)

if not defined FW_DIR (
  echo Firmware release folder was not found.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%FW_DIR%\VERIFY_PACKAGE.ps1"
if errorlevel 1 exit /b %ERRORLEVEL%

set "APP_DIR="
for /d %%D in ("%~dp0*") do (
  if exist "%%~fD\tools\smoke_model_load.py" set "APP_DIR=%%~fD"
)

if not defined APP_DIR (
  echo Desktop app folder was not found.
  exit /b 1
)

cd /d "%APP_DIR%"
set "LOCAL_PY_PKGS=%USERPROFILE%\py310_pkgs"
if exist "%LOCAL_PY_PKGS%" set "PYTHONPATH=%LOCAL_PY_PKGS%;%PYTHONPATH%"
python tools\smoke_model_load.py
exit /b %ERRORLEVEL%
