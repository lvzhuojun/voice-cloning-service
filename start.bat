@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  Voice Cloning Service - Start
echo ============================================================
echo.

:: Activate conda env
echo [1/3] Activating conda env voice-cloning...
call D:\Anaconda3\Scripts\activate.bat voice-cloning 2>nul
if %ERRORLEVEL% NEQ 0 (
    call conda activate voice-cloning 2>nul
)
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] Cannot activate voice-cloning env!
    echo  Please run setup\install.bat first.
    pause
    exit /b 1
)
echo  OK: env activated

:: Check environment
echo.
echo [2/3] Checking environment...
python setup/check_env.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] Environment check failed!
    echo  Fix the issues above, then run start.bat again.
    pause
    exit /b 1
)

:: Auto-create .env if missing
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  INFO: .env created from .env.example
)

:: Free port 8000 if occupied
echo.
echo [3/3] Starting service...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000 "') do (
    echo  INFO: Killing old process on port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
echo.
echo  Web UI : http://localhost:8000
echo  API Doc: http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo.
echo ============================================================
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000

echo.
echo Service stopped.
pause
