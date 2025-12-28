# Quick Start - Authentication

## 🚀 Getting Started in 5 Minutes

### Step 1: Check Prerequisites
```bash
# Make sure MongoDB is installed and running
cd backend
check_mongodb.bat  # Windows
```

### Step 2: Install Dependencies (First Time Only)
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Step 3: Start Everything
```bash
# Terminal 1 - Start Backend
cd backend
python app.py

# Terminal 2 - Start Frontend
cd frontend
npm run dev
```

### Step 4: Test Authentication
```bash
# Terminal 3 - Run Test Script
cd backend
python test_auth.py
```

### Step 5: Use the App
1. Open browser: http://localhost:5173
2. Click **"Login"** button in header
3. Click **"Sign Up"** to create account
4. Fill in:
   - Username: `demo`
   - Email: `demo@example.com`
   - Password: `demo123456`
5. Click **"Sign Up"**
6. Now login with same credentials
7. Upload a pitch image and analyze
8. Your history is automatically saved!

## 📝 Key Endpoints

### Authentication
- **POST** `/api/auth/signup` - Create account
- **POST** `/api/auth/login` - Login (returns JWT token)
- **GET** `/api/auth/me` - Get current user (requires auth)
- **GET** `/api/auth/history` - Get analysis history (requires auth)

### Analysis (Works with or without auth)
- **POST** `/api/analyze` - Full analysis
- **POST** `/api/quick-analyze` - Quick analysis

## 🔑 Default Configuration

| Setting | Value |
|---------|-------|
| MongoDB URL | `mongodb://localhost:27017/` |
| Database | `pitch_insight` |
| Backend | http://localhost:8000 |
| Frontend | http://localhost:5173 |
| Token Expiry | 7 days |

## ⚠️ Common Issues

### "Failed to connect to MongoDB"
**Solution:** Start MongoDB service
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
```

### "Email already registered"
**Solution:** Use different email or clear database:
```bash
mongosh
use pitch_insight
db.users.deleteMany({})
```

### Token expired
**Solution:** Login again to get new token

## 📚 Documentation

- **Full Setup Guide:** [AUTHENTICATION_SETUP.md](AUTHENTICATION_SETUP.md)
- **Main README:** [README.md](README%20(1).md)

## 🎯 Testing Checklist

- [ ] MongoDB running
- [ ] Backend starts without errors
- [ ] Frontend loads in browser
- [ ] Can create account
- [ ] Can login
- [ ] Can view profile
- [ ] Can analyze pitch (with auth)
- [ ] History is saved
- [ ] Can logout

## 💡 Pro Tips

1. **Keep token safe:** Don't share your JWT token
2. **Use strong passwords:** Minimum 6 characters
3. **Check history:** Use GET `/api/auth/history` endpoint
4. **Logout properly:** Clears token from localStorage
5. **Works without login:** App functions without authentication too!

---

**Need Help?** Check [AUTHENTICATION_SETUP.md](AUTHENTICATION_SETUP.md) for detailed guide.
