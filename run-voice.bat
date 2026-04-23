@echo off
REM Ifa voice-mode launcher for Windows. Double-click this file to start.
REM Thin wrapper: sets IFA_MODE=voice then delegates to run.bat.
cd /d "%~dp0"
set "IFA_MODE=voice"
call "%~dp0run.bat"
