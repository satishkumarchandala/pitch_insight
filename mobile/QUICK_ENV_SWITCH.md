# 🌐 Backend Environment Switcher

## Quick Switch Guide

Open: `src/constants/config.js`

Change line 6:

```javascript
const ENVIRONMENT = 'production';  // ← Change this
```

---

## 🎯 Three Options

### 1️⃣ Production (Render) - DEFAULT

```javascript
const ENVIRONMENT = 'production';
```

- ✅ Uses deployed backend
- ✅ No local setup needed
- ⚠️ May have cold start delays

**Backend:** `https://pitch-insight.onrender.com`

---

### 2️⃣ Development (Localhost)

```javascript
const ENVIRONMENT = 'development';
```

- ✅ Fast responses
- ✅ Test locally
- ⚠️ Requires local backend running

**Backend:** `http://localhost:8000`

**Start backend first:**
```bash
cd d:/pitch_insight/backend
python app.py
```

---

### 3️⃣ Local (Your IP)

```javascript
const ENVIRONMENT = 'local';
```

- ✅ Test on physical phone
- ✅ Fast responses
- ⚠️ Requires same WiFi network

**Backend:** `http://192.168.1.100:8000` (update with your IP)

**Setup:**
1. Find IP: `ipconfig` (Windows) or `ifconfig` (Mac)
2. Update IP in `config.js` line 25
3. Start backend: `python app.py`
4. Test on phone with Expo Go

---

## 🔄 After Changing Environment

```bash
# Restart Expo server
npx expo start --clear
```

Check terminal for:
```
🌐 API Environment: Production (Render)
🔗 Base URL: https://pitch-insight.onrender.com
```

---

## 📋 When to Use Each

| Scenario | Environment | Why |
|----------|-------------|-----|
| Testing with Expo Go | `production` | No setup needed |
| Debugging API issues | `development` | See backend logs |
| Testing on phone + local backend | `local` | Fast + real device |
| Web testing | `development` | Localhost works |

---

**That's it! Easy environment switching! 🚀**

See `ENVIRONMENT_GUIDE.md` for detailed instructions.
