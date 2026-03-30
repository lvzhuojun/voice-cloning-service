@echo off
chcp 65001 >nul
echo Starting Voice Cloning Service (GPT-SoVITS)...
cd /d "%~dp0"
call D:\Anaconda3\Scripts\activate.bat voice-cloning
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
pause
