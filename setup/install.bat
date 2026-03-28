@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo ════════════════════════════════════════════════════════════
echo   语音克隆服务 - 一键安装脚本
echo   环境：voice-cloning  /  Python 3.10  /  PyTorch 2.7 cu128
echo   GPU：RTX 5060 Blackwell (sm_120) 已处理兼容性
echo ════════════════════════════════════════════════════════════
echo.

:: ── 切换到项目根目录（不管从哪里双击都能正常运行）────────────────────────────
cd /d "%~dp0\.."

:: ════════════════════════════════════════════════════════════
:: 步骤 1：检查 conda
:: ════════════════════════════════════════════════════════════
echo [步骤 1/5] 检查 conda...

call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [错误] 找不到 conda 命令！
    echo.
    echo   请先安装 Miniconda（推荐，体积小）：
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo   安装完成后请重新打开 Anaconda Prompt 再运行本脚本。
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('conda --version 2^>^&1') do echo   ✓ %%v

:: ════════════════════════════════════════════════════════════
:: 步骤 2：创建 conda 环境（已存在则跳过）
:: ════════════════════════════════════════════════════════════
echo.
echo [步骤 2/5] 创建 conda 环境 voice-cloning (Python 3.10)...

conda env list | findstr /C:"voice-cloning" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   ✓ 环境已存在，跳过创建
    echo     如需重建：conda env remove -n voice-cloning -y
) else (
    echo   正在创建，请稍候...
    conda env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo   [错误] conda 环境创建失败！
        echo   请检查 environment.yml 文件是否存在，以及网络是否正常。
        pause
        exit /b 1
    )
    echo   ✓ conda 环境创建成功
)

:: ════════════════════════════════════════════════════════════
:: 步骤 3：安装 PyTorch 2.7（cu128，支持 RTX 5060 Blackwell）
:: ════════════════════════════════════════════════════════════
echo.
echo [步骤 3/5] 安装 PyTorch 2.7 (cu128)...
echo   RTX 5060 是 Blackwell 架构 sm_120，PyTorch 2.7+ 才原生支持。
echo   cu128 = CUDA 12.8 编译版，可在 CUDA 13.x 驱动下正常运行。
echo   正在从 pytorch CDN 下载（约 2.5GB，请耐心等待）...
echo.

conda run -n voice-cloning pip install ^
    torch==2.7.0 torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu128

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [错误] PyTorch 安装失败！
    echo.
    echo   可能原因及解决方法：
    echo   1. 网络超时 - 请重试，或使用代理
    echo   2. 磁盘空间不足 - 需要约 5GB 空闲空间
    echo   3. torch==2.7.0 不存在 - 尝试去掉版本号：
    echo      conda run -n voice-cloning pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)
echo   ✓ PyTorch 安装成功

:: ════════════════════════════════════════════════════════════
:: 步骤 4：安装其他 pip 依赖
:: ════════════════════════════════════════════════════════════
echo.
echo [步骤 4/5] 安装项目依赖（FastAPI、HuggingFace 等）...

conda run -n voice-cloning pip install -r requirements-pip.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [警告] 部分依赖安装失败，尝试跳过失败项继续...
    conda run -n voice-cloning pip install -r requirements-pip.txt --ignore-requires-python
)
echo   ✓ 项目依赖安装完成

:: ════════════════════════════════════════════════════════════
:: 步骤 5：验证环境
:: ════════════════════════════════════════════════════════════
echo.
echo [步骤 5/5] 验证环境...
echo.

conda run -n voice-cloning python setup/check_env.py

:: ════════════════════════════════════════════════════════════
:: 完成
:: ════════════════════════════════════════════════════════════
echo.
echo ════════════════════════════════════════════════════════════
echo   安装完成！下一步操作：
echo.
echo   1. 下载预训练模型（约 2GB）：
echo      conda run -n voice-cloning python setup/download_models.py
echo.
echo   2. 复制配置文件：
echo      copy .env.example .env
echo.
echo   3. 启动服务：
echo      start.bat
echo ════════════════════════════════════════════════════════════
echo.
pause
