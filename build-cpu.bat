@echo off
REM Build CPU installer using venv-cpu
REM Usage: build-cpu.bat

cd /d "%~dp0"
setlocal enabledelayedexpansion
set VENV_PYTHON="%cd%\venv-cpu\Scripts\python.exe"
echo Building CPU installer with: !VENV_PYTHON!
!VENV_PYTHON! build_installers.py cpu
pause
