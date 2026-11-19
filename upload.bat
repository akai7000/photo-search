@echo off
REM Upload installers to CloudFlare R2 using venv
REM Usage: upload.bat [options]

cd /d "%~dp0"
setlocal enabledelayedexpansion
set VENV_PYTHON="%cd%\venv\Scripts\python.exe"
echo Uploading with: !VENV_PYTHON!
!VENV_PYTHON! upload_installers.py %*
pause
