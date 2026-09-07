@echo off
REM Ifa launcher for Windows. Double-click this file in Explorer to start.
REM Self-heals: creates venv, installs deps, starts Ollama, then pulls the
REM model configured by IFA_OLLAMA_MODEL in .env.
REM on first use, then runs python -m ifa.main.
REM
REM Structure: main logic is a `:main` subroutine. The outer script always
REM falls through to the trailing `pause`, so the window never closes on you
REM no matter which code path exited.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
cd /d "%~dp0"
title Ifa

call :main
set "_rc=%errorlevel%"

echo.
echo ===============================================
if %_rc% == 0 (
    echo   Ifa exited normally.
) else (
    echo   Ifa exited with code %_rc%.
)
echo   Press any key to close this window.
echo ===============================================
pause >nul
endlocal
exit /b %_rc%


:main
echo.
echo =====================================
echo   Ifa  -  Personal AI Assistant
echo =====================================
echo.

REM -------- 1. Python venv --------
if not exist "venv\Scripts\python.exe" (
    echo [setup] venv not found. Creating...
    where python >nul 2>&1
    if errorlevel 1 (
        echo [error] Python is not on PATH.
        echo         Install Python 3.11+ from https://www.python.org/downloads/
        echo         and make sure "Add python.exe to PATH" is checked during install.
        exit /b 1
    )
    python -m venv venv || exit /b 1
    echo [setup] Installing dependencies from ifa\requirements.txt ...
    "venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    "venv\Scripts\python.exe" -m pip install -r "ifa\requirements.txt" || exit /b 1
)

REM -------- 2. Ollama installed? --------
where ollama >nul 2>&1
if errorlevel 1 (
    echo [error] Ollama is not installed.
    echo         Install from https://ollama.com/download, then re-run this script.
    exit /b 1
)

REM -------- 3. Ollama configuration --------
REM Run Ifa's Ollama instance on a dedicated port so it does not
REM conflict with another Ollama instance using the default 11434.
set "OLLAMA_HOST=127.0.0.1:11435"

REM -------- 4. Ollama running? --------
curl -s -m 3 http://127.0.0.1:11435/api/tags >nul 2>&1
if errorlevel 1 (
    echo [setup] Ollama is not running on %OLLAMA_HOST%. Starting it in the background...
    start "Ollama - Ifa" /MIN cmd /c "set OLLAMA_HOST=127.0.0.1:11435 && ollama serve"

    set /a _tries=0
    :wait_ollama
    timeout /t 1 /nobreak >nul

    curl -s -m 2 http://127.0.0.1:11435/api/tags >nul 2>&1
    if not errorlevel 1 goto :ollama_ok

    set /a _tries+=1
    if !_tries! LSS 15 goto :wait_ollama

    echo [error] Ollama did not come up within 15 seconds.
    echo         Check the Ollama process/window.
    exit /b 1
)

:ollama_ok

REM -------- 5. Configured Ollama model pulled? --------
set "IFA_OLLAMA_MODEL="
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="IFA_OLLAMA_MODEL" set "IFA_OLLAMA_MODEL=%%B"
)

if not defined IFA_OLLAMA_MODEL (
    echo [error] IFA_OLLAMA_MODEL is not set in .env.
    exit /b 1
)

ollama list 2>nul | findstr /I /L /C:"%IFA_OLLAMA_MODEL%" >nul
if errorlevel 1 (
    echo [setup] %IFA_OLLAMA_MODEL% not found. Pulling now (one-time)...
    ollama pull "%IFA_OLLAMA_MODEL%" || exit /b 1
)

REM -------- 6. Launch Ifa --------
echo.
echo [launch] Starting Ifa. Type 'exit' to quit.
echo.
set "HF_HUB_OFFLINE=0"
"venv\Scripts\python.exe" -m ifa.main
exit /b %errorlevel%
