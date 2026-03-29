@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

cd /d "%~dp0"

:: Direct path to the conda env Python - no activation needed
set PYTHON=D:\Anaconda3\envs\voice-cloning\Scripts\python.exe

echo.
echo ============================================================
echo  Voice Cloning Service - Start
echo ============================================================
echo.

:: Verify Python exists
if not exist "%PYTHON%" (
    echo  [ERROR] Python not found at: %PYTHON%
    echo  Please ensure the voice-cloning conda env is installed.
    pause
    exit /b 1
)
echo  [OK] Python: %PYTHON%

:: Auto-create .env if missing
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  [OK] .env created from .env.example
)

:: Free port 8000 if occupied
echo.
echo  Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R " :8000 "') do (
    echo  INFO: Releasing port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: Start service
echo.
echo ============================================================
echo  Web UI : http://localhost:8000
echo  API Doc: http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo ============================================================
echo.

"%PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo.
echo  Service stopped.
pause
