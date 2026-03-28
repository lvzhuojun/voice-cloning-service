@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo ════════════════════════════════════════════════════
echo   语音克隆服务 - 启动脚本
echo ════════════════════════════════════════════════════
echo.

:: ─────────────────────────────────────────────────────
:: 检查 conda 是否可用
:: ─────────────────────────────────────────────────────
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: 尝试常见安装路径
    for %%p in (
        "%USERPROFILE%\anaconda3\Scripts\activate.bat"
        "%USERPROFILE%\miniconda3\Scripts\activate.bat"
        "C:\ProgramData\anaconda3\Scripts\activate.bat"
        "C:\ProgramData\miniconda3\Scripts\activate.bat"
    ) do (
        if exist %%p (
            call %%p
            goto :conda_ok
        )
    )
    echo [错误] 未找到 conda！请先安装 Anaconda 或 Miniconda，
    echo 然后运行 setup\install.bat 进行环境配置。
    pause
    exit /b 1
)
:conda_ok

:: ─────────────────────────────────────────────────────
:: 激活 conda 环境
:: ─────────────────────────────────────────────────────
echo [1/3] 激活 conda 环境 voice-cloning...
call conda activate voice-cloning
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 无法激活 conda 环境 voice-cloning！
    echo 请先运行 setup\install.bat 创建环境。
    pause
    exit /b 1
)
echo     ✓ 环境已激活

:: ─────────────────────────────────────────────────────
:: 运行环境检测
:: ─────────────────────────────────────────────────────
echo.
echo [2/3] 运行环境检测...
python setup/check_env.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 环境检测失败！
    echo 请根据上方提示修复环境问题，然后重新运行 start.bat。
    echo.
    echo 常见解决方法：
    echo   1. 重新运行 setup\install.bat 安装依赖
    echo   2. 运行 python setup\download_models.py 下载模型
    pause
    exit /b 1
)

:: ─────────────────────────────────────────────────────
:: 检查 .env 配置文件
:: ─────────────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo [提示] 未找到 .env 配置文件，使用默认配置...
    copy ".env.example" ".env" >nul 2>&1
    echo     已从 .env.example 复制默认配置
)

:: ─────────────────────────────────────────────────────
:: 启动 FastAPI 服务
:: ─────────────────────────────────────────────────────
echo.
echo [3/3] 启动语音克隆服务...
echo.
echo   Web 界面:  http://localhost:8000
echo   API 文档:  http://localhost:8000/docs
echo   健康检查:  http://localhost:8000/api/health
echo.
echo   按 Ctrl+C 停止服务
echo ════════════════════════════════════════════════════
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

:: 服务停止后显示提示
echo.
echo 服务已停止。
pause
