@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

cd /d "%~dp0"

echo.
echo ============================================================
echo  Voice Cloning Service - Stop
echo ============================================================
echo.

:: Stop by recorded PID
if exist ".pid" (
    set /p SVC_PID=<.pid
    echo  Stopping service (PID !SVC_PID!)...
    taskkill /F /PID !SVC_PID! >nul 2>&1
    if !errorlevel! equ 0 (
        echo  [OK] Service stopped.
    ) else (
        echo  [WARN] Process !SVC_PID! not found (may have already exited).
    )
    del ".pid" >nul 2>&1
) else (
    echo  No .pid file found. Scanning port 8000...
    set FOUND=0
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr "0\.0\.0\.0:8000 "') do (
        echo  Killing PID %%a on port 8000...
        taskkill /F /PID %%a >nul 2>&1
        set FOUND=1
    )
    if !FOUND! equ 0 (
        echo  No process found on port 8000. Service is not running.
    ) else (
        echo  [OK] Service stopped.
    )
)

echo.
pause
