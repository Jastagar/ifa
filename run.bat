@echo off
REM Ifa launcher for Windows. Double-click this file in Explorer to start.
REM Self-heals: creates venv, installs deps, starts Ollama, pulls qwen2.5:7b-instruct
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

REM -------- 3. Ollama running? --------
curl -s -m 3 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [setup] Ollama is not running. Starting it in the background...
    start "Ollama" /MIN cmd /c "ollama serve"
    set /a _tries=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    curl -s -m 2 http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 goto :ollama_ok
    set /a _tries+=1
    if !_tries! LSS 15 goto :wait_ollama
    echo [error] Ollama did not come up within 15 seconds. Check the Ollama window.
    exit /b 1
)
:ollama_ok

REM -------- 4. qwen2.5:7b-instruct pulled? --------
ollama list 2>nul | findstr /I "qwen2.5:7b-instruct" >nul
if errorlevel 1 (
    echo [setup] qwen2.5:7b-instruct not found. Pulling now (one-time, ~5GB)...
    ollama pull qwen2.5:7b-instruct || exit /b 1
)

REM -------- 5. Voice-mode models pre-cached --------
REM Explicit pre-download so runtime can run with HF_HUB_OFFLINE=1.
REM openWakeWord (~1 MB) + faster-whisper small.en (~470 MB) — idempotent.
set "PYTHONPATH=."
"venv\Scripts\python.exe" -m scripts.setup_voice_models || exit /b 1

REM -------- 6. Launch Ifa --------
echo.
echo [launch] Starting Ifa. Type 'exit' to quit.
echo.
set "HF_HUB_OFFLINE=1"
"venv\Scripts\python.exe" -m ifa.main
exit /b %errorlevel%
