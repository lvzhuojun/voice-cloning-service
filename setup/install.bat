@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0\.."

echo.
echo ============================================================
echo  Voice Cloning Service - Install
echo  ENV: voice-cloning  /  Python 3.10  /  PyTorch 2.7 cu128
echo ============================================================
echo.

:: ------------------------------------------------------------
:: Step 1: Check conda
:: ------------------------------------------------------------
echo [1/5] Checking conda...
call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] conda not found!
    echo  Please install Miniconda first:
    echo  https://docs.conda.io/en/latest/miniconda.html
    echo  Then reopen Anaconda Prompt and run this script again.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('conda --version 2^>^&1') do echo  OK: %%v

:: ------------------------------------------------------------
:: Step 2: Create conda env (skip if exists)
:: ------------------------------------------------------------
echo.
echo [2/5] Creating conda env voice-cloning (Python 3.10)...

conda env list | findstr /C:"voice-cloning" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo  OK: env already exists, skipping
    echo  To rebuild: conda env remove -n voice-cloning -y
) else (
    echo  Creating env, please wait...
    conda env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo  [ERROR] Failed to create conda env!
        echo  Check environment.yml exists and network is available.
        pause
        exit /b 1
    )
    echo  OK: conda env created
)

:: ------------------------------------------------------------
:: Step 3: Install PyTorch 2.7 cu128 (RTX 5060 Blackwell sm_120)
:: ------------------------------------------------------------
echo.
echo [3/5] Installing PyTorch 2.7 cu128...
echo  RTX 5060 is Blackwell (sm_120), requires PyTorch 2.7+
echo  cu128 = CUDA 12.8 build, works with CUDA 13.x driver
echo  Downloading ~2.5GB, please be patient...
echo.

conda run -n voice-cloning pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] PyTorch install failed!
    echo  Try without version pin:
    echo    conda run -n voice-cloning pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)
echo  OK: PyTorch installed

:: ------------------------------------------------------------
:: Step 4: Install other pip dependencies
:: ------------------------------------------------------------
echo.
echo [4/5] Installing project dependencies...

conda run -n voice-cloning pip install -r requirements-pip.txt

if %ERRORLEVEL% NEQ 0 (
    echo  [WARN] Some packages failed, retrying...
    conda run -n voice-cloning pip install -r requirements-pip.txt --ignore-requires-python
)
echo  OK: dependencies installed

:: ------------------------------------------------------------
:: Step 5: Verify environment
:: ------------------------------------------------------------
echo.
echo [5/5] Verifying environment...
echo.
conda run -n voice-cloning python setup/check_env.py

:: ------------------------------------------------------------
echo.
echo ============================================================
echo  Install complete! Next steps:
echo.
echo  1. Download model (~2GB):
echo     conda run -n voice-cloning python setup/download_models.py
echo.
echo  2. Copy config:
echo     copy .env.example .env
echo.
echo  3. Start service:
echo     start.bat
echo ============================================================
echo.
pause
