@echo off
REM Build the Photo Search installer using venv
REM Usage: build.bat

cd /d "%~dp0"
setlocal enabledelayedexpansion
set VENV_PYTHON="%cd%\venv\Scripts\python.exe"
echo Building installer with: !VENV_PYTHON!
!VENV_PYTHON! build_installers.py
pause
