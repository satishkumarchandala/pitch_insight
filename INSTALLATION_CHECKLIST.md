## 🎉 Authentication System - Installation Checklist

### Prerequisites Setup

- [ ] **MongoDB Installed**
  - Windows: Download from https://www.mongodb.com/try/download/community
  - Mac: `brew install mongodb-community`
  - Linux: `sudo apt-get install mongodb`
  
- [ ] **MongoDB Running**
  ```bash
  # Windows
  net start MongoDB
  
  # Mac
  brew services start mongodb-community
  
  # Linux
  sudo systemctl start mongod
  ```

- [ ] **Check MongoDB Connection**
  ```bash
  cd backend
  ./check_mongodb.bat  # Windows
  ./check_mongodb.sh   # Mac/Linux
  ```

### Backend Setup

- [ ] **Navigate to backend directory**
  ```bash
  cd backend
  ```

- [ ] **Install Python dependencies**
  ```bash
  pip install -r requirements.txt
  ```
  
  New packages being installed:
  - pymongo (MongoDB driver)
  - bcrypt (password hashing)
  - python-jose (JWT tokens)
  - passlib (password utilities)

- [ ] **Verify installation**
  ```bash
  python -c "import pymongo, bcrypt, jose; print('✅ All packages installed')"
  ```

- [ ] **(Optional) Configure environment**
  ```bash
  # Copy .env.example to .env
  cp .env.example .env
  
  # Edit .env and change SECRET_KEY
  # Generate secure key: python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

- [ ] **Start backend server**
  ```bash
  python app.py
  ```
  
  Expected output:
  ```
  ✓ Connected to MongoDB: pitch_insight
  🚀 Initializing ONNX Pitch Analysis Pipeline...
  INFO:     Started server process
  INFO:     Uvicorn running on http://0.0.0.0:8000
  ```

### Frontend Setup

- [ ] **Navigate to frontend directory**
  ```bash
  cd frontend
  ```

- [ ] **Install dependencies** (if not already done)
  ```bash
  npm install
  ```

- [ ] **Start development server**
  ```bash
  npm run dev
  ```
  
  Expected output:
  ```
  VITE ready in [time]
  ➜  Local:   http://localhost:5173/
  ```

### Testing Authentication

- [ ] **Run automated tests**
  ```bash
  cd backend
  python test_auth.py
  ```
  
  All tests should pass ✅

- [ ] **Test in browser**
  1. Open http://localhost:5173
  2. Click "Login" button in header
  3. Click "Sign Up" tab
  4. Create account:
     - Username: `demo`
     - Email: `demo@example.com`
     - Password: `demo123456`
  5. Submit form
  6. Switch to "Log In" tab
  7. Login with same credentials
  8. Verify user info appears in header

- [ ] **Test pitch analysis with auth**
  1. Upload a pitch image
  2. Click "Analyze Pitch"
  3. Wait for results
  4. Verify analysis completes

- [ ] **Test analysis history**
  - Use API: GET http://localhost:8000/api/auth/history
  - Or check MongoDB: `db.analysis_history.find()`

### Verification Checklist

Backend:
- [ ] MongoDB connected successfully
- [ ] Server starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Health endpoint works: http://localhost:8000/api/health

Frontend:
- [ ] Page loads without errors
- [ ] Login button visible
- [ ] Modal opens on click

Authentication:
- [ ] Can create new account
- [ ] Can login with credentials
- [ ] User info shows in header
- [ ] Can logout
- [ ] Token persists on page reload

Analysis:
- [ ] Can analyze pitch when logged in
- [ ] History is saved to database
- [ ] Can analyze pitch without login (optional)

### Troubleshooting

If something doesn't work:

**MongoDB Connection Failed**
```bash
# Check if MongoDB is running
mongosh

# If not, start it
net start MongoDB  # Windows
brew services start mongodb-community  # Mac
sudo systemctl start mongod  # Linux
```

**Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Port Already in Use**
```bash
# Backend (8000)
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -i :8000

# Kill process or change port in app.py

# Frontend (5173)
# Kill existing npm process
```

**Token/Auth Issues**
- Clear browser localStorage
- Login again
- Check browser console for errors
- Verify backend logs

**Frontend Not Updating**
- Clear browser cache
- Hard refresh (Ctrl+Shift+R)
- Restart dev server

### MongoDB Quick Commands

```bash
# Connect to MongoDB
mongosh

# Switch to pitch_insight database
use pitch_insight

# View users
db.users.find().pretty()

# View analysis history
db.analysis_history.find().pretty()

# Count users
db.users.countDocuments()

# Delete all users (development only!)
db.users.deleteMany({})

# Delete all history
db.analysis_history.deleteMany({})
```

### Success Indicators

You're ready when you see:
- ✅ MongoDB connected
- ✅ Backend running on port 8000
- ✅ Frontend running on port 5173
- ✅ Can create account
- ✅ Can login
- ✅ Can analyze pitch
- ✅ History saved in database

### Next Steps

Once everything is working:
1. Read [AUTHENTICATION_SETUP.md](AUTHENTICATION_SETUP.md) for details
2. Check [QUICKSTART_AUTH.md](QUICKSTART_AUTH.md) for usage
3. Review API docs at http://localhost:8000/docs
4. Start building your pitch analysis workflow!

### Support Resources

- **Setup Guide**: [AUTHENTICATION_SETUP.md](AUTHENTICATION_SETUP.md)
- **Quick Start**: [QUICKSTART_AUTH.md](QUICKSTART_AUTH.md)
- **Implementation Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Main README**: [README (1).md](README%20(1).md)

---

**Last Updated**: December 27, 2025
**Status**: Ready for use ✅
