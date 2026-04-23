# Ifa launcher for Windows (PowerShell). Same self-healing flow as run.bat.
#
# How to launch:
#   - Right-click this file in Explorer and choose "Run with PowerShell", OR
#   - Double-click `run-ps.bat` (a thin wrapper around this script), OR
#   - From a PowerShell prompt: `.\run.ps1`
#
# If you hit an ExecutionPolicy error, launch via:
#   powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1

$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'Ifa'

# cd to this script's directory — right-click→Run-with-PowerShell sets $PWD to
# wherever you were; we always want to operate on the repo root.
Set-Location -LiteralPath $PSScriptRoot

$rc = 0

function Write-Header {
    Write-Host ''
    Write-Host '=====================================' -ForegroundColor Cyan
    Write-Host '  Ifa  -  Personal AI Assistant' -ForegroundColor Cyan
    Write-Host '=====================================' -ForegroundColor Cyan
    Write-Host ''
}

function Write-Step {
    param([string]$Message)
    Write-Host "[setup] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[error] $Message" -ForegroundColor Red
}

function Pause-Exit {
    param([int]$Code = 0)
    Write-Host ''
    Write-Host '===============================================' -ForegroundColor Cyan
    if ($Code -eq 0) {
        Write-Host '  Ifa exited normally.' -ForegroundColor Green
    } else {
        Write-Host "  Ifa exited with code $Code." -ForegroundColor Red
    }
    Write-Host '  Press Enter to close this window.' -ForegroundColor Cyan
    Write-Host '===============================================' -ForegroundColor Cyan
    [void]$Host.UI.ReadLine()
    exit $Code
}

function Test-OllamaUp {
    try {
        $null = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

try {
    Write-Header

    # -------- 1. Python venv --------
    if (-not (Test-Path 'venv\Scripts\python.exe')) {
        Write-Step 'venv not found. Creating...'
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            Write-Err 'Python is not on PATH.'
            Write-Host '        Install Python 3.11+ from https://www.python.org/downloads/'
            Write-Host '        and make sure "Add python.exe to PATH" is checked during install.'
            Pause-Exit 1
        }
        python -m venv venv
        if ($LASTEXITCODE -ne 0) { Pause-Exit 1 }

        Write-Step 'Installing dependencies from ifa\requirements.txt ...'
        & 'venv\Scripts\python.exe' -m pip install --upgrade pip | Out-Null
        & 'venv\Scripts\python.exe' -m pip install -r 'ifa\requirements.txt'
        if ($LASTEXITCODE -ne 0) { Pause-Exit 1 }
    }

    # -------- 2. Ollama installed? --------
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Err 'Ollama is not installed.'
        Write-Host '        Install from https://ollama.com/download, then re-run this script.'
        Pause-Exit 1
    }

    # -------- 3. Ollama running? --------
    if (-not (Test-OllamaUp)) {
        Write-Step 'Ollama is not running. Starting it in the background...'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Minimized | Out-Null

        $started = $false
        for ($i = 1; $i -le 15; $i++) {
            Start-Sleep -Seconds 1
            if (Test-OllamaUp) { $started = $true; break }
        }
        if (-not $started) {
            Write-Err 'Ollama did not come up within 15 seconds. Check the Ollama window.'
            Pause-Exit 1
        }
    }

    # -------- 4. qwen2.5:7b-instruct pulled? --------
    $modelList = & ollama list 2>$null
    if (-not ($modelList -match 'qwen2\.5:7b-instruct')) {
        Write-Step 'qwen2.5:7b-instruct not found. Pulling now (one-time, ~5GB)...'
        ollama pull qwen2.5:7b-instruct
        if ($LASTEXITCODE -ne 0) { Pause-Exit 1 }
    }

    # -------- 5. Launch Ifa --------
    Write-Host ''
    Write-Host "[launch] Starting Ifa. Type 'exit' to quit." -ForegroundColor Green
    Write-Host ''
    $env:PYTHONPATH = '.'
    & 'venv\Scripts\python.exe' -m ifa.main
    $rc = $LASTEXITCODE
}
catch {
    Write-Err ("Unexpected failure: " + $_.Exception.Message)
    $rc = 1
}

Pause-Exit $rc
