# 🐛 Network Error Troubleshooting

## Issue: "Login Failed: Network Error"

### Most Common Causes

1. **Backend is sleeping (cold start)** - Most likely
2. **No internet connection**
3. **Wrong backend URL**
4. **Backend server down**

---

## 🔥 Quick Fixes

### Fix 1: Wait for Cold Start (60 seconds)

Render free tier sleeps after inactivity. First request takes up to 60 seconds.

**Steps:**
1. Try logging in
2. Wait 60 seconds if you see network error
3. Try again - should work now!

**The app will now show:**
```
"Server is waking up (cold start). 
This can take up to 60 seconds. 
Please try again."
```

---

### Fix 2: Test Backend First

Before logging in, wake up the backend:

**Method 1: Browser**
1. Open browser
2. Go to: `https://pitch-insight.onrender.com/api/health`
3. Wait for response
4. Now try login in app

**Method 2: Terminal**
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://pitch-insight.onrender.com/api/health"
```

---

### Fix 3: Check Internet Connection

**On Phone:**
- Make sure WiFi/mobile data is on
- Try opening a website in browser
- Restart Expo Go if needed

---

### Fix 4: Verify Environment Setting

Open: `src/constants/config.js`

Check line 7:
```javascript
const ENVIRONMENT = 'production';  // Should be 'production'
```

If using local backend, make sure it's running:
```bash
cd d:/pitch_insight/backend
python app.py
```

---

## 🔍 Check What's Wrong

### In Expo Terminal

Look for console logs:
```
🌐 API Environment: Production (Render)
🔗 Base URL: https://pitch-insight.onrender.com
```

### After Login Attempt

Check for error details in terminal. Look for:
- `ECONNABORTED` → Timeout (cold start)
- `ERR_NETWORK` → No internet
- `401` → Wrong credentials
- `404` → Wrong API endpoint

---

## ✅ Recommended Solution

**For Testing:**
1. **Use test credentials** to bypass login:
   - Use **Guest Mode** button
   - Skip authentication entirely
   - Test other features first

2. **Or wake backend first:**
   ```bash
   # Open this in browser FIRST
   https://pitch-insight.onrender.com/api/health
   
   # Wait for response
   # Then try login in app
   ```

3. **Or use local backend:**
   ```javascript
   // In src/constants/config.js
   const ENVIRONMENT = 'development';
   ```
   
   Then start backend locally:
   ```bash
   cd d:/pitch_insight/backend
   python app.py
   ```

---

## 🎯 Expected Behavior

### First Login Attempt (Cold Start)
```
❌ Login failed: Network error
...wait 60 seconds...
```

### Second Login Attempt
```
✅ Login successful!
```

---

## 💡 Pro Tips

1. **Use Guest Mode** for testing UI without login
2. **Keep backend awake** by visiting it periodically
3. **Use local backend** for faster testing
4. **First login of the day** always takes 60 seconds

---

## 🚑 Emergency Bypass

If you just want to test the app UI without backend:

**Option 1: Guest Mode**
- Click "Continue as Guest" on login screen
- Explore app features

**Option 2: Mock Auth (Development)**
- Temporarily bypass auth check
- Test UI components

---

## 📋 Checklist

Before reporting "network error":

- [ ] Waited 60 seconds for cold start?
- [ ] Internet connection working?
- [ ] Backend URL correct in config.js?
- [ ] Tried guest mode?
- [ ] Checked terminal for detailed error?
- [ ] Tried waking backend first (browser)?

---

**The error is most likely a cold start. Just wait 60 seconds and try again!** ⏳
