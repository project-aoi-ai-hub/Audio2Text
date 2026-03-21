@echo off
@chcp 65001 > nul
cd /d %~dp0

echo Python Venv Setup Starting...

:: Create venv if not exists
if not exist ".venv" (
    echo Creating new virtual environment...
    python -m venv .venv
)

:: Install libraries
echo Installing libraries from requirements.txt...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup Completed!
echo Now you can run the program via run.bat.
pause
