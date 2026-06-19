@echo off
cd /d "%~dp0"
set "LOCAL_PY_PKGS=%USERPROFILE%\py310_pkgs"
if exist "%LOCAL_PY_PKGS%" set "PYTHONPATH=%LOCAL_PY_PKGS%;%PYTHONPATH%"
python -c "import numpy, PyQt5, pyqtgraph, serial" >nul 2>nul
if errorlevel 1 (
    echo [desktop] Missing Python packages.
    echo [desktop] Install them with:
    echo   install_requirements.bat
    echo.
    pause
    exit /b 1
)
python main.py
pause
