@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

cd /d "%~dp0"

:: Direct path to the conda env Python - no activation needed
set PYTHON=D:\Anaconda3\envs\voice-cloning\python.exe

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

:: Kill any previous instance recorded in .pid
if exist ".pid" (
    set /p OLD_PID=<.pid
    echo  INFO: Stopping previous instance (PID !OLD_PID!)...
    taskkill /F /PID !OLD_PID! >nul 2>&1
    del ".pid" >nul 2>&1
)

:: Also free port 8000 in case .pid is stale
echo.
echo  Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R "0\.0\.0\.0:8000 \|127\.0\.0\.1:8000 \|\[::\]:8000 "') do (
    echo  INFO: Releasing port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: Start service and record PID
echo.
echo ============================================================
echo  Web UI : http://localhost:8000
echo  API Doc: http://localhost:8000/docs
echo  Press Ctrl+C to stop  /  run stop.bat to stop from elsewhere
echo ============================================================
echo.

:: Start uvicorn, save PID for stop.bat
start /B "" "%PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
:: Give process a moment to register
timeout /t 1 /nobreak >nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr "0\.0\.0\.0:8000 "') do (
    echo %%a > .pid
    echo  [OK] Service running (PID %%a)
    goto :wait
)
:wait
echo  Press any key to stop the service...
pause >nul

:: On exit: kill by .pid
if exist ".pid" (
    set /p SVC_PID=<.pid
    echo  Stopping service (PID !SVC_PID!)...
    taskkill /F /PID !SVC_PID! >nul 2>&1
    del ".pid" >nul 2>&1
)
echo  Service stopped.
pause
