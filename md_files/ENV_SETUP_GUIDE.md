# Environment Variables Setup Guide

## 🔐 Backend Environment Variables (Render)

Copy these to your Render Web Service settings:

```env
# Python Version
PYTHON_VERSION=3.12.0

# Environment
ENVIRONMENT=production

# Frontend URL (Update after Vercel deployment)
FRONTEND_URL=https://your-app.vercel.app

# Weather API (Get from https://openweathermap.org/api)
WEATHER_API_KEY=your_openweathermap_api_key_here

# MongoDB Atlas (Get from MongoDB Atlas connection dialog)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/pitch_insight?retryWrites=true&w=majority
MONGODB_DB_NAME=pitch_insight

# Feature Flags
ENABLE_DATABASE=true
ENABLE_WEATHER=true
```

### 🔑 How to Get Each Value:

#### WEATHER_API_KEY
1. Go to https://openweathermap.org/api
2. Sign up for free account
3. Create API key
4. Copy the key

#### MONGODB_URI
1. MongoDB Atlas → Database → Connect
2. Choose "Connect your application"
3. Copy connection string
4. Replace `<password>` with your actual password
5. Add database name: `/pitch_insight`

#### FRONTEND_URL
1. Deploy frontend to Vercel first
2. Get the URL (e.g., https://pitch-insight.vercel.app)
3. Come back and update this in Render

---

## 🌐 Frontend Environment Variables (Vercel)

Copy these to your Vercel project settings:

```env
# Backend API URL (Update with your Render backend URL)
VITE_API_URL=https://pitch-insight-backend.onrender.com

# App Configuration
VITE_APP_NAME=Pitch Insight
VITE_ENABLE_ANALYTICS=false
```

### 🔑 How to Get Each Value:

#### VITE_API_URL
1. Deploy backend to Render first
2. Get the URL from Render dashboard
3. Copy the full URL (e.g., https://pitch-insight-backend.onrender.com)
4. DO NOT include trailing slash

---

## 📝 Local Development Setup

### Backend (.env file)

Create `backend/.env`:

```env
# Server
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=development

# CORS
FRONTEND_URL=http://localhost:5173

# Weather API
WEATHER_API_KEY=your_key_here

# MongoDB (Optional for local dev)
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=pitch_insight
ENABLE_DATABASE=false

# Features
ENABLE_WEATHER=true
```

### Frontend (.env file)

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=Pitch Insight
VITE_ENABLE_ANALYTICS=false
```

---

## ⚠️ Security Notes

1. **NEVER commit `.env` files to Git**
   - They're already in `.gitignore`
   - Use `.env.example` as template

2. **Rotate credentials regularly**
   - Change API keys every 3-6 months
   - Update MongoDB passwords periodically

3. **Use environment-specific values**
   - Different keys for dev/staging/production
   - Never use production keys in development

4. **Restrict MongoDB access in production**
   - Change from 0.0.0.0/0 to specific IPs
   - Use Render's outbound IPs

---

## ✅ Verification

### Test Backend Variables

```bash
# In Render Shell or local terminal
cd backend
python3 << EOF
import os
from dotenv import load_dotenv
load_dotenv()

print("Environment:", os.getenv("ENVIRONMENT"))
print("Weather API:", "✓" if os.getenv("WEATHER_API_KEY") else "✗")
print("MongoDB:", "✓" if os.getenv("MONGODB_URI") else "✗")
print("Frontend URL:", os.getenv("FRONTEND_URL"))
EOF
```

### Test Frontend Variables

```bash
# In frontend directory
npm run build
# Should complete without errors
```

---

## 🔄 Updating Environment Variables

### Render
1. Dashboard → Your Service → Environment
2. Add/Edit/Delete variables
3. Save changes
4. Service will auto-restart

### Vercel
1. Project Settings → Environment Variables
2. Add/Edit/Delete variables
3. Redeploy to apply changes

---

## 📚 References

- [Render Environment Variables](https://render.com/docs/environment-variables)
- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Vite Env Variables](https://vitejs.dev/guide/env-and-mode.html)
- [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/)
