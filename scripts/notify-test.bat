@echo off
REM Sends fake phone notifications to a running Ifa (POST /activate).
REM Usage: scripts\notify-test.bat [single^|burst^|urgent^|low^|email^|all]
REM Reads IFA_API_PORT and IFA_API_TOKEN from .env unless they are already set.

setlocal ENABLEEXTENSIONS
cd /d "%~dp0.."

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /I "%%A"=="IFA_API_PORT" if not defined IFA_API_PORT set "IFA_API_PORT=%%B"
        if /I "%%A"=="IFA_API_TOKEN" if not defined IFA_API_TOKEN set "IFA_API_TOKEN=%%B"
    )
)
if not defined IFA_API_PORT set "IFA_API_PORT=8787"
set "URL=http://127.0.0.1:%IFA_API_PORT%"

set "SCENARIO=%~1"
if "%SCENARIO%"=="" set "SCENARIO=single"

curl -s -m 3 "%URL%/health" >nul 2>&1
if errorlevel 1 (
    echo Ifa is not listening on %URL%. Launch Ifa first.
    exit /b 1
)

if /I "%SCENARIO%"=="single" goto :single
if /I "%SCENARIO%"=="burst" goto :burst
if /I "%SCENARIO%"=="urgent" goto :urgent
if /I "%SCENARIO%"=="low" goto :low
if /I "%SCENARIO%"=="email" goto :email
if /I "%SCENARIO%"=="all" goto :all
echo Unknown scenario "%SCENARIO%". Use: single, burst, urgent, low, email, all
exit /b 1

:single
echo [single] One WhatsApp message. Expect: listen sound, then ask "What is it?"
call :send com.whatsapp WhatsApp Joe "Are you free tonight?"
exit /b 0

:burst
echo [burst] Four broken messages from one sender, 1s apart. Expect: grouped into one, announced as urgent.
call :send com.whatsapp WhatsApp Mom "Hi"
timeout /t 1 /nobreak >nul
call :send com.whatsapp WhatsApp Mom "How are you"
timeout /t 1 /nobreak >nul
call :send com.whatsapp WhatsApp Mom "Are you free? not anything like a"
timeout /t 1 /nobreak >nul
call :send com.whatsapp WhatsApp Mom "Emergency, call me now"
exit /b 0

:urgent
echo [urgent] One urgent SMS. Expect: "Sir, there is an urgent message for you."
call :send com.google.android.apps.messaging Messages Dad "Accident on the highway. I am okay but come to the hospital right now"
exit /b 0

:low
echo [low] Instagram like (low priority). Expect: stored silently, no announcement.
call :send com.instagram.android Instagram "priya.k" "liked your photo"
exit /b 0

:email
echo [email] Gmail (normal priority). Expect: listen sound unless the LLM rates it urgent.
call :send com.google.android.gm Gmail "Rahul Mehta" "Q3 review moved to Monday 10am, please confirm"
exit /b 0

:all
call :low
call :email
echo Waiting 6s so the email settles before the next sender...
timeout /t 6 /nobreak >nul
call :burst
exit /b 0

:send
curl -s -o nul -w "  %%{http_code}  %~2 / %~3: %~4\n" -X POST "%URL%/activate" ^
  -H "Content-Type: application/json" -H "X-IFA-Token: %IFA_API_TOKEN%" ^
  -d "{\"context\":{\"packageName\":\"%~1\",\"appName\":\"%~2\",\"title\":\"%~3\",\"body\":\"%~4\"}}"
exit /b 0
