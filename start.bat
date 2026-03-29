@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON=D:\Anaconda3\envs\voice-cloning\python.exe

echo.
echo ============================================================
echo  Voice Cloning Service - Start
echo ============================================================
echo.

if not exist "%PYTHON%" (
    echo  [ERROR] Python not found at: %PYTHON%
    pause
    exit /b 1
)
echo  [OK] Python: %PYTHON%

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  [OK] .env created from .env.example
)

echo.
echo  Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 "') do (
    echo  Releasing port 8000 ^(PID %%a^)...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo.
echo ============================================================
echo  Web UI : http://localhost:8000
echo  API Doc: http://localhost:8000/docs
echo  Press Ctrl+C or close this window to stop
echo ============================================================
echo.

"%PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo.
echo  Service stopped.
pause
