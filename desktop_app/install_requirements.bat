@echo off
cd /d "%~dp0"
set "LOCAL_PY_PKGS=%USERPROFILE%\py310_pkgs"
if not exist "%LOCAL_PY_PKGS%" mkdir "%LOCAL_PY_PKGS%"
python -c "import sys; raise SystemExit(0 if sys.version_info < (3, 11) else 1)"
if not errorlevel 1 (
    python -m pip install --target "%LOCAL_PY_PKGS%" -r requirements-py310.txt
) else (
    python -m pip install --target "%LOCAL_PY_PKGS%" -r requirements.txt
)
pause
