@echo off
@chcp 65001 > nul
cd /d %~dp0

echo Audio2Text Starting...

if not exist ".venv" (
    echo [ERROR] .venv folder not found.
    echo Please run setup.bat first to initialize environment.
    pause
    exit /b
)

:: Activate venv and run main.py
call .venv\Scripts\activate
python src\main.py

echo.
echo Process Completed.
pause
