@echo off
REM Double-click wrapper for run.ps1. Bypasses the default ExecutionPolicy so
REM the script can run without the user changing system settings.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
