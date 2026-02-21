# 🐛 Connection Fix: Wrong Backend URL

## The Issue
Your mobile app was trying to connect to:
`https://pitch-insight.onrender.com`

But the actual backend is at:
`https://pitch-insight-backend.onrender.com`

## ✅ How It Was Fixed
I updated `src/constants/config.js` with the correct production URL.

## 🚀 Next Steps
1. **Restart Expo**: `npx expo start --clear`
2. **Wait for Backend**: Render might still need a cold start (60s)
3. **Login should work!**

---

**This was the exact cause of "Network Error"!**
