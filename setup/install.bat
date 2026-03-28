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
    echo  [ERROR] conda not found!
    echo  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
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
) else (
    echo  Creating env, please wait...
    conda env create -f environment.yml
    :: NOTE: conda env create sometimes returns non-zero even on success.
    :: So we verify by checking if the env actually exists now.
    conda env list | findstr /C:"voice-cloning" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo  [ERROR] Env creation truly failed. Check output above.
        pause
        exit /b 1
    )
    echo  OK: conda env created
)

:: ------------------------------------------------------------
:: Step 3: Install PyTorch 2.7 cu128 (RTX 5060 Blackwell sm_120)
:: ------------------------------------------------------------
echo.
echo [3/5] Installing PyTorch 2.7 cu128 (RTX 5060 Blackwell)...
echo  Downloading ~2.5GB, please wait...
echo.

conda run -n voice-cloning pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] PyTorch install failed!
    echo  Try manually:
    echo    conda activate voice-cloning
    echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)
echo  OK: PyTorch installed

:: ------------------------------------------------------------
:: Step 4: Install other pip dependencies
:: ------------------------------------------------------------
echo.
echo [4/5] Installing project dependencies...

:: Install huggingface_hub first (required by download_models.py)
conda run -n voice-cloning pip install "huggingface_hub>=0.22.0"
if %ERRORLEVEL% NEQ 0 (
    echo  [WARN] huggingface_hub install had issues, continuing...
)

conda run -n voice-cloning pip install -r requirements-pip.txt

if %ERRORLEVEL% NEQ 0 (
    echo  [WARN] Some packages failed, retrying key packages...
    conda run -n voice-cloning pip install huggingface_hub transformers fastapi uvicorn pydantic pydantic-settings librosa soundfile
)
echo  OK: dependencies installed

:: ------------------------------------------------------------
:: Step 5: Verify
:: ------------------------------------------------------------
echo.
echo [5/5] Verifying environment...
echo.
conda run -n voice-cloning python setup/check_env.py

echo.
echo ============================================================
echo  Install complete! Next steps:
echo.
echo  1. Download model (~2GB):
echo     conda run -n voice-cloning python setup/download_models.py
echo.
echo  2. Start service:
echo     start.bat
echo ============================================================
echo.
pause
