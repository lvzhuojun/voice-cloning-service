@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0\.."

echo.
echo ============================================================
echo  Clone GPT-SoVITS
echo  Target: .\GPT-SoVITS\
echo ============================================================
echo.

:: Check if already cloned
if exist "GPT-SoVITS\.git" (
    echo [OK] GPT-SoVITS already cloned. Pulling latest changes...
    git -C GPT-SoVITS pull origin main
    echo [OK] Done.
    goto :end
)

:: -------------------------------------------------------
:: Try to find Git Bash and delegate to the .sh script
:: -------------------------------------------------------
set BASH_PATHS=^
    "C:\Program Files\Git\bin\bash.exe" ^
    "C:\Program Files\Git\usr\bin\bash.exe" ^
    "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" ^
    "%APPDATA%\Programs\Git\bin\bash.exe"

for %%B in (%BASH_PATHS%) do (
    if exist %%~B (
        echo [INFO] Found Git Bash: %%~B
        echo [INFO] Delegating to setup/clone_gptsovits.sh ...
        echo.
        %%~B setup/clone_gptsovits.sh
        if !ERRORLEVEL! EQU 0 goto :end
        echo [WARN] bash execution failed, falling back to direct git clone...
        goto :direct_clone
    )
)

:direct_clone
:: -------------------------------------------------------
:: Fallback: use git.exe directly (Git must be in PATH)
:: -------------------------------------------------------
echo [INFO] Running git clone directly...
echo.

git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] git not found in PATH.
    echo         Install Git for Windows: https://git-scm.com/download/win
    pause
    exit /b 1
)

git clone https://github.com/RVC-Boss/GPT-SoVITS.git GPT-SoVITS

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] git clone failed. Check network connection and try again.
    pause
    exit /b 1
)

echo.
echo [OK] GPT-SoVITS cloned successfully to .\GPT-SoVITS\

:end
echo.
echo ============================================================
echo  Next step: download pretrained models
echo    conda run -n voice-cloning python setup/download_models.py
echo ============================================================
echo.
pause
