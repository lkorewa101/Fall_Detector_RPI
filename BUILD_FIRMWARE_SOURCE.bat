@echo off
setlocal

set "GMAKE=C:\ti\xdctools_3_50_08_24_core\gmake.exe"
set "FW_SRC="

for /d %%D in ("%~dp0IWR6843ISK_C3CD_*") do (
  if exist "%%~fD\people_tracking_fall\makefile" set "FW_SRC=%%~fD\people_tracking_fall"
)

if not defined FW_SRC (
  echo Firmware source folder not found.
  exit /b 1
)

if not exist "%GMAKE%" (
  echo TI gmake was not found:
  echo %GMAKE%
  echo Install TI build tools or edit BUILD_FIRMWARE_SOURCE.bat to point to gmake.exe.
  exit /b 1
)

cd /d "%FW_SRC%"
"%GMAKE%" all
exit /b %ERRORLEVEL%
