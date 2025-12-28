#!/bin/bash

# Pitch Insight - MongoDB Setup Checker

echo "========================================"
echo "Pitch Insight - MongoDB Setup Checker"
echo "========================================"
echo

echo "[1/3] Checking if MongoDB is installed..."
if ! command -v mongosh &> /dev/null; then
    echo "[X] MongoDB Shell (mongosh) not found!"
    echo
    echo "Please install MongoDB from:"
    echo "https://www.mongodb.com/try/download/community"
    echo
    exit 1
else
    echo "[OK] MongoDB Shell found"
fi
echo

echo "[2/3] Checking if MongoDB service is running..."
if ! pgrep -x "mongod" > /dev/null; then
    echo "[!] MongoDB service not running"
    echo
    echo "Attempting to start MongoDB service..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew services start mongodb-community
    else
        # Linux
        sudo systemctl start mongod
    fi
    
    if [ $? -ne 0 ]; then
        echo "[X] Failed to start MongoDB service"
        echo "Please start it manually or check installation"
        exit 1
    fi
else
    echo "[OK] MongoDB service is running"
fi
echo

echo "[3/3] Testing MongoDB connection..."
if ! mongosh --eval "db.version()" --quiet &> /dev/null; then
    echo "[X] Cannot connect to MongoDB"
    echo "Please check if MongoDB is running on localhost:27017"
    exit 1
else
    echo "[OK] Successfully connected to MongoDB"
    mongosh --eval "print('MongoDB Version: ' + db.version())" --quiet
fi
echo

echo "========================================"
echo "All checks passed! MongoDB is ready."
echo "========================================"
echo
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Start backend: python app.py"
echo "3. Start frontend: npm run dev (in frontend folder)"
echo
