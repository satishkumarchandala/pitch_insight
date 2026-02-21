# 🌐 Environment Configuration Guide

## Quick Start - Switch Between Backends

Your mobile app can connect to different backend servers:

1. **Production** - Deployed backend on Render
2. **Development** - Local backend on your computer (localhost)
3. **Local** - Local backend accessible on your network (for testing on physical devices)

---

## 🔧 How to Switch Environments

### Step 1: Open Config File

Open: `src/constants/config.js`

### Step 2: Change ENVIRONMENT Variable

Find this line (around line 6):

```javascript
const ENVIRONMENT = 'production';
```

**Change to:**
- `'production'` - Use Render backend (deployed)
- `'development'` - Use localhost backend (when running backend locally)
- `'local'` - Use your computer's IP address (for phone testing)

### Step 3: Restart Expo

```bash
# Stop current server (Ctrl+C)
npx expo start --clear
```

---

## 📋 Environment Details

### Option 1: Production (Default)

```javascript
const ENVIRONMENT = 'production';
```

**Uses:**
- Backend: `https://pitch-insight.onrender.com`
- Timeout: 60 seconds (for cold starts)

**When to use:**
- Testing with deployed backend
- Don't have backend running locally
- Want to test production data

---

### Option 2: Development

```javascript
const ENVIRONMENT = 'development';
```

**Uses:**
- Backend: `http://localhost:8000`
- Timeout: 30 seconds

**When to use:**
- Backend running locally on your computer
- Testing on **web** (press `w` in Expo)
- Testing on **Android emulator** on same computer

**Requirements:**
1. Backend must be running:
   ```bash
   cd d:/pitch_insight/backend
   python app.py
   ```
2. Backend must be on `http://localhost:8000`

---

### Option 3: Local (Your Computer's IP)

```javascript
const ENVIRONMENT = 'local';
```

**Uses:**
- Backend: `http://192.168.1.100:8000` (your IP)
- Timeout: 30 seconds

**When to use:**
- Testing on **physical phone** (Expo Go)
- Backend running locally on your computer
- Phone and computer on **same WiFi network**

**Setup:**

1. **Find your computer's local IP:**
   ```bash
   # Windows
   ipconfig
   # Look for "IPv4 Address" (e.g., 192.168.1.100)
   
   # Mac/Linux
   ifconfig
   # Look for "inet" address
   ```

2. **Update config.js:**
   ```javascript
   local: {
     name: 'Local (Your IP)',
     baseURL: 'http://192.168.1.100:8000', // Your actual IP
     timeout: 30000,
   },
   ```

3. **Start backend:**
   ```bash
   cd d:/pitch_insight/backend
   python app.py
   # Must listen on 0.0.0.0, not localhost
   ```

4. **Ensure firewall allows connections**

---

## 🔍 How to Know Which Environment is Active

When you start Expo, check the terminal logs:

```
🌐 API Environment: Production (Render)
🔗 Base URL: https://pitch-insight.onrender.com
```

This tells you which backend the app will use.

---

## 🎯 Common Scenarios

### Scenario 1: Testing with Expo Go on Phone + Production Backend

```javascript
const ENVIRONMENT = 'production';
```

✅ Works out of the box
✅ No local backend needed
⚠️ May have cold start delays (60s)

---

### Scenario 2: Testing on Web Browser + Local Backend

```javascript
const ENVIRONMENT = 'development';
```

**Steps:**
1. Start backend locally: `python app.py`
2. Change environment to `'development'`
3. Restart Expo: `npx expo start --clear`
4. Press `w` for web

✅ Fast responses
✅ See backend logs in real-time
⚠️ Need to run backend locally

---

### Scenario 3: Testing on Phone + Local Backend

```javascript
const ENVIRONMENT = 'local';
```

**Steps:**
1. Find your IP: `ipconfig` → e.g., 192.168.1.100
2. Update `baseURL` in config.js: `http://192.168.1.100:8000`
3. Start backend: `python app.py` (must bind to 0.0.0.0)
4. Restart Expo: `npx expo start --clear`
5. Scan QR code with Expo Go

✅ Fast responses
✅ Test on real device
⚠️ Phone and computer must be on same WiFi
⚠️ Firewall must allow connections

---

## 🐛 Troubleshooting

### "Network Error" on Phone with 'local' Environment

**Problem:** Can't connect to local backend from phone

**Solutions:**
1. **Check backend is running** on 0.0.0.0:
   ```python
   # In backend/app.py or backend/config.py
   HOST = "0.0.0.0"  # Not "localhost"
   ```

2. **Verify IP address is correct** in config.js

3. **Ensure same WiFi network:**
   - Phone WiFi = Computer WiFi

4. **Check Windows Firewall:**
   ```bash
   # Allow Python through firewall
   # Or temporarily disable firewall for testing
   ```

---

### CORS Error on Web

**Problem:** CORS blocked when testing on web with local backend

**Solution:**
Backend already configured to allow all origins (`*`). Make sure backend is running.

---

### Cold Start Timeout on Production

**Problem:** Request timeout after 60 seconds

**Why:** Render free tier spins down after inactivity

**Solutions:**
1. **Wait up to 60 seconds** - Render is waking up
2. **Try again** - Subsequent requests will be fast
3. **Use development** environment while testing

---

## 📝 Quick Reference

| Environment | Backend URL | Use Case |
|------------|-------------|----------|
| `production` | `https://pitch-insight.onrender.com` | Default, deployed backend |
| `development` | `http://localhost:8000` | Web testing, emulator |
| `local` | `http://YOUR_IP:8000` | Phone + local backend |

---

## 🚀 Best Practices

1. **Use `production` by default** - No local backend needed
2. **Switch to `development`** when debugging API issues locally
3. **Use `local`** when testing on physical phone with local backend
4. **Commit with `production`** - Don't commit `development` or `local` settings

---

## 💡 Pro Tips

### Tip 1: Add Your Own Environments

You can add custom environments in `config.js`:

```javascript
const ENVIRONMENTS = {
  production: { /* ... */ },
  development: { /* ... */ },
  local: { /* ... */ },
  staging: {
    name: 'Staging Server',
    baseURL: 'https://staging.yourdomain.com',
    timeout: 45000,
  },
};
```

### Tip 2: Environment-Specific Features

Use the current environment to toggle features:

```javascript
import { API_ENV_NAME } from './constants/config';

if (API_ENV_NAME.includes('Development')) {
  console.log('Debug mode enabled');
}
```

---

**Now you can easily switch between production and development backends!** 🎉
