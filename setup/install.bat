@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo ════════════════════════════════════════════════════
echo   语音克隆服务 - 一键安装脚本
echo   Voice Cloning Service - Setup Script
echo ════════════════════════════════════════════════════
echo.

:: ─────────────────────────────────────────────────────
:: 步骤 1：检查 conda 是否已安装
:: ─────────────────────────────────────────────────────
echo [1/5] 检查 conda 安装...
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: 尝试常见 conda 安装路径
    set "CONDA_PATHS=%USERPROFILE%\anaconda3\Scripts\conda.exe;%USERPROFILE%\miniconda3\Scripts\conda.exe;C:\ProgramData\anaconda3\Scripts\conda.exe;C:\ProgramData\miniconda3\Scripts\conda.exe"
    set "CONDA_FOUND=0"
    for %%p in (!CONDA_PATHS!) do (
        if exist "%%p" (
            echo     找到 conda: %%p
            set "CONDA_FOUND=1"
            :: 将 conda 目录添加到临时 PATH
            for %%d in ("%%p\..") do set "CONDA_SCRIPTS=%%~fd"
            set "PATH=!CONDA_SCRIPTS!;!PATH!"
            goto :conda_found
        )
    )
    echo [错误] 未找到 conda！
    echo.
    echo 请先安装 Anaconda 或 Miniconda：
    echo   Miniconda (推荐): https://docs.conda.io/en/latest/miniconda.html
    echo   Anaconda:         https://www.anaconda.com/
    echo.
    pause
    exit /b 1
)
:conda_found
echo     ✓ conda 已找到

:: ─────────────────────────────────────────────────────
:: 步骤 2：初始化 conda（确保 conda activate 可用）
:: ─────────────────────────────────────────────────────
echo.
echo [2/5] 初始化 conda...
call conda init cmd.exe >nul 2>&1

:: 直接使用 conda run 而不是 activate（更可靠的跨环境执行方式）
:: 先检查环境是否已存在
conda env list | findstr "voice-cloning" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo     ✓ conda 环境 'voice-cloning' 已存在，跳过创建
    echo     提示：如需重建环境，请手动运行: conda env remove -n voice-cloning
    goto :install_deps
)

:: ─────────────────────────────────────────────────────
:: 步骤 3：创建 conda 环境（Python 3.10）
:: ─────────────────────────────────────────────────────
echo.
echo [3/5] 创建 conda 环境 'voice-cloning' (Python 3.10)...
echo     这可能需要几分钟，请耐心等待...
conda create -n voice-cloning python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [错误] conda 环境创建失败！
    pause
    exit /b 1
)
echo     ✓ conda 环境创建成功

:install_deps
:: ─────────────────────────────────────────────────────
:: 步骤 4：安装依赖（从 environment.yml）
:: ─────────────────────────────────────────────────────
echo.
echo [4/5] 安装项目依赖（含 PyTorch nightly for RTX 5060 Blackwell）...
echo     注意：需要下载约 3-5GB 依赖，请确保网络畅通。
echo     国内用户如遇下载慢，可提前配置 conda 镜像源。
echo.

:: 先安装 conda 依赖（排除 pip 部分，用于更快速的基础环境搭建）
conda env update -n voice-cloning -f environment.yml --prune
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [警告] environment.yml 安装出错，尝试基础安装...
    echo.

    :: 备用：手动安装基础包
    conda install -n voice-cloning -y ^
        numpy scipy librosa soundfile ffmpeg ^
        fastapi uvicorn python-multipart aiofiles ^
        pydantic python-dotenv tqdm requests ^
        -c conda-forge
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] 基础依赖安装失败！
        pause
        exit /b 1
    )

    :: 安装 PyTorch nightly（RTX 5060 Blackwell sm_120 需要 >= 2.7）
    echo.
    echo     安装 PyTorch nightly (支持 RTX 5060 Blackwell/sm_120)...
    conda run -n voice-cloning pip install torch torchaudio torchvision ^
        --index-url https://download.pytorch.org/whl/nightly/cu121
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] PyTorch 安装失败！
        pause
        exit /b 1
    )

    :: 安装其他 pip 依赖
    conda run -n voice-cloning pip install ^
        huggingface_hub transformers accelerate ^
        omegaconf hydra-core einops conformer diffusers ^
        ffmpeg-python httpx pydantic-settings
)

echo.
echo     ✓ 依赖安装完成

:: ─────────────────────────────────────────────────────
:: 步骤 5：运行环境检测脚本
:: ─────────────────────────────────────────────────────
echo.
echo [5/5] 运行环境检测...
echo.
conda run -n voice-cloning python setup/check_env.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [警告] 环境检测发现问题，请根据上方提示修复。
    echo 某些问题（如模型未下载）不影响安装完成。
)

:: ─────────────────────────────────────────────────────
:: 安装完成
:: ─────────────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════
echo   安装完成！
echo ════════════════════════════════════════════════════
echo.
echo 下一步操作：
echo.
echo   1. 下载预训练模型（约 2GB）：
echo      conda run -n voice-cloning python setup/download_models.py
echo.
echo   2. 复制配置文件：
echo      copy .env.example .env
echo.
echo   3. 启动服务：
echo      start.bat
echo.
echo   4. 打开浏览器访问：http://localhost:8000
echo.
pause
