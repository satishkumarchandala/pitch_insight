@echo off
echo.
echo ====================================
echo  Restarting Pitch Insight Backend
echo ====================================
echo.

REM Kill any existing Python processes running on port 8000
echo Stopping existing server...
FOR /F "tokens=5" %%P IN ('netstat -aon ^| findstr :8000') DO (
    taskkill /F /PID %%P 2>nul
)

echo.
echo Starting server...
echo.

REM Activate virtual environment and start server
cd /d "%~dp0"
call ..\vevv\Scripts\activate.bat
python app.py
