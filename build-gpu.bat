@echo off
REM Build GPU installer using venv-gpu
REM Usage: build-gpu.bat

cd /d "%~dp0"
setlocal enabledelayedexpansion
set VENV_PYTHON="%cd%\venv-gpu\Scripts\python.exe"
echo Building GPU installer with: !VENV_PYTHON!
!VENV_PYTHON! build_installers.py gpu
pause
