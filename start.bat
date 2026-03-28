@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: 切换到项目根目录
cd /d "%~dp0"

echo.
echo ════════════════════════════════════════════════════
echo   语音克隆服务 - 启动
echo ════════════════════════════════════════════════════
echo.

:: ── 激活 conda 环境 ──────────────────────────────────
echo [1/3] 激活 conda 环境 voice-cloning...
call conda activate voice-cloning
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [错误] 无法激活 voice-cloning 环境！
    echo   请先运行 setup\install.bat 安装环境。
    pause
    exit /b 1
)
echo   ✓ 已激活

:: ── 环境检测 ─────────────────────────────────────────
echo.
echo [2/3] 环境检测...
python setup/check_env.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [错误] 环境检测未通过，请根据上方提示修复后重试。
    pause
    exit /b 1
)

:: ── 自动复制 .env ────────────────────────────────────
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo   已自动创建 .env 配置文件（使用默认值）
)

:: ── 启动服务 ─────────────────────────────────────────
echo.
echo [3/3] 启动语音克隆服务...
echo.
echo   ┌─────────────────────────────────────────────┐
echo   │  Web 界面：http://localhost:8000             │
echo   │  API 文档：http://localhost:8000/docs        │
echo   │  按 Ctrl+C 停止服务                         │
echo   └─────────────────────────────────────────────┘
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo 服务已停止。
pause
