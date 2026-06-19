@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

python launcher.py

pause
exit /b %ERRORLEVEL%
