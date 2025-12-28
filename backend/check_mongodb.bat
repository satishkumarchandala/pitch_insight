@echo off
REM Check MongoDB Installation and Connection

echo ========================================
echo Pitch Insight - MongoDB Setup Checker
echo ========================================
echo.

echo [1/3] Checking if MongoDB is installed...
where mongosh >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] MongoDB Shell (mongosh) not found!
    echo.
    echo Please install MongoDB from:
    echo https://www.mongodb.com/try/download/community
    echo.
    pause
    exit /b 1
) else (
    echo [OK] MongoDB Shell found
)
echo.

echo [2/3] Checking if MongoDB service is running...
sc query MongoDB | find "RUNNING" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] MongoDB service not running
    echo.
    echo Attempting to start MongoDB service...
    net start MongoDB
    if %errorlevel% neq 0 (
        echo [X] Failed to start MongoDB service
        echo Please start it manually or check installation
        pause
        exit /b 1
    )
) else (
    echo [OK] MongoDB service is running
)
echo.

echo [3/3] Testing MongoDB connection...
mongosh --eval "db.version()" --quiet >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Cannot connect to MongoDB
    echo Please check if MongoDB is running on localhost:27017
    pause
    exit /b 1
) else (
    echo [OK] Successfully connected to MongoDB
    mongosh --eval "print('MongoDB Version: ' + db.version())" --quiet
)
echo.

echo ========================================
echo All checks passed! MongoDB is ready.
echo ========================================
echo.
echo Next steps:
echo 1. Install Python dependencies: pip install -r requirements.txt
echo 2. Start backend: python app.py
echo 3. Start frontend: npm run dev (in frontend folder)
echo.
pause
