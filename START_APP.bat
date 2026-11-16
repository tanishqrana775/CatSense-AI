@echo off
echo ============================================================
echo   MeowMind - Cat Emotion Recognition System
echo   Starting Application...
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Install/update dependencies
echo Checking dependencies...
pip install -q -r requirements.txt
echo.

REM Run the application
echo Starting Flask server...
echo.
python app.py

pause
