@echo off
REM Ifa voice-mode launcher for Windows (PowerShell). Double-click this file.
REM Thin wrapper: sets IFA_MODE=voice then delegates to run-ps.bat.
cd /d "%~dp0"
set "IFA_MODE=voice"
call "%~dp0run-ps.bat"
