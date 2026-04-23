@echo off
REM Ifa launcher for Windows. Double-click this file to start the assistant.
REM Self-heals: creates venv if missing, installs requirements, starts Ollama,
REM pulls qwen2.5:7b-instruct if not present, then runs python -m ifa.main.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

REM Work relative to this script's location (double-click from Explorer sets
REM cwd to system32 otherwise).
cd /d "%~dp0"

title Ifa

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
        echo         and make sure "Add python.exe to PATH" is checked.
        goto :error
    )
    python -m venv venv
    if errorlevel 1 goto :error
    echo [setup] Installing dependencies from ifa\requirements.txt ...
    "venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    "venv\Scripts\python.exe" -m pip install -r "ifa\requirements.txt"
    if errorlevel 1 goto :error
)

REM -------- 2. Ollama installed? --------
where ollama >nul 2>&1
if errorlevel 1 (
    echo [error] Ollama is not installed.
    echo         Download and install from: https://ollama.com/download
    echo         Then re-run this script.
    goto :error
)

REM -------- 3. Ollama running? --------
curl -s -m 3 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [setup] Ollama is not running. Starting it in the background...
    REM Launch ollama serve in a detached window so this script keeps control.
    start "Ollama" /MIN cmd /c "ollama serve"
    REM Wait up to 15s for Ollama to become reachable.
    set /a _tries=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    curl -s -m 2 http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 goto :ollama_ok
    set /a _tries+=1
    if !_tries! LSS 15 goto :wait_ollama
    echo [error] Ollama did not come up within 15 seconds. Check the Ollama window.
    goto :error
)
:ollama_ok

REM -------- 4. qwen2.5:7b-instruct pulled? --------
ollama list 2>nul | findstr /I "qwen2.5:7b-instruct" >nul
if errorlevel 1 (
    echo [setup] qwen2.5:7b-instruct not found. Pulling now (one-time, ~5GB)...
    ollama pull qwen2.5:7b-instruct
    if errorlevel 1 goto :error
)

REM -------- 5. Launch Ifa --------
echo.
echo [launch] Starting Ifa. Type 'exit' to quit.
echo.
set PYTHONPATH=.
"venv\Scripts\python.exe" -m ifa.main
set _rc=%errorlevel%

echo.
if %_rc% neq 0 (
    echo Ifa exited with code %_rc%.
) else (
    echo Ifa exited normally.
)
pause
endlocal
exit /b %_rc%

:error
echo.
echo *** Setup failed. See the error above. Press any key to close. ***
pause >nul
endlocal
exit /b 1
