# 🚀 Pitch Insight - Complete Deployment Guide

## 📊 Application Overview

**Tech Stack:**
- **Backend**: FastAPI + ONNX Runtime (Python)
- **Frontend**: React + Vite
- **Database**: MongoDB Atlas (cloud)
- **ML Models**: ONNX format (~54MB total)

## ⚠️ Critical: Render Free Tier (512MB RAM) Considerations

Your application has **ONNX models totaling ~54MB** which will be loaded into memory. With proper optimization, this can work on Render's free tier.

### Memory Optimization Strategies:

1. **Lazy Loading** ✅ (Already implemented in `utils.py`)
   - Models load only when first analysis is requested
   - Not loaded during startup

2. **Reduce Dependencies** (Implemented below)
   - Remove unnecessary packages
   - Use lightweight alternatives

3. **Implement Model Unloading**
   - Unload models after period of inactivity
   - Reload on next request

---

## 🎯 Deployment Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Netlify/      │         │   Render.com    │
│   Vercel        │────────▶│   (Backend)     │
│   (Frontend)    │  API    │   FastAPI       │
└─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  MongoDB Atlas  │
                            │   (Database)    │
                            └─────────────────┘
```

---

## 📦 Part 1: Backend Deployment (Render)

### Step 1: Optimize requirements.txt for Render

**Create `backend/requirements.txt` (optimized):**

```txt
# Core Framework
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20

# Minimal Dependencies
pydantic==2.10.5
requests==2.32.3
pillow==11.0.0

# Lightweight OpenCV (remove full opencv-python)
opencv-python-headless==4.12.0.88

# NumPy (required)
numpy==2.2.1

# ONNX Runtime (CPU only for free tier)
onnxruntime==1.16.0

# Auth & Database
pymongo==4.6.0
bcrypt==4.1.0
python-jose[cryptography]==3.3.0
passlib==1.7.4

# Payment & AI
razorpay==1.4.0
google-genai==0.3.0

# Environment
python-dotenv==1.0.0
```

### Step 2: Create `render.yaml`

```yaml
services:
  - type: web
    name: pitch-insight-backend
    env: python
    region: oregon  # Choose region closest to your users
    plan: free
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: MONGODB_URL
        sync: false
      - key: SECRET_KEY
        generateValue: true
      - key: RAZORPAY_KEY_ID
        sync: false
      - key: RAZORPAY_KEY_SECRET
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: WEATHER_API_KEY
        value: b2ad62736cbd4e52aaa133601252712
      - key: ALLOWED_ORIGINS
        value: https://your-frontend-domain.netlify.app,https://your-frontend-domain.vercel.app
```

### Step 3: Create `runtime.txt`

```txt
python-3.11.0
```

### Step 4: Optimize Backend Code for Memory

**Create `backend/memory_optimizer.py`:**

```python
"""
Memory optimization utilities for Render free tier
"""
import gc
import time
from threading import Timer

class ModelManager:
    """Manages model loading/unloading to save memory"""
    
    def __init__(self, timeout=300):  # 5 minutes
        self.pipeline = None
        self.last_used = 0
        self.timeout = timeout
        self.cleanup_timer = None
    
    def get_pipeline(self):
        """Get pipeline, loading if needed"""
        from complete_pipeline_onnx import CompletePitchPipeline
        
        if self.pipeline is None:
            print("🚀 Loading ONNX models...")
            self.pipeline = CompletePitchPipeline(
                yolo_model_path="pitch_yolov8_best.onnx",
                classifier_model_path="pitch_classifier.onnx",
                use_gpu=False
            )
        
        self.last_used = time.time()
        self._schedule_cleanup()
        return self.pipeline
    
    def _schedule_cleanup(self):
        """Schedule model cleanup after timeout"""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        self.cleanup_timer = Timer(self.timeout, self._cleanup_if_idle)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def _cleanup_if_idle(self):
        """Unload models if idle for too long"""
        if time.time() - self.last_used >= self.timeout:
            print("🧹 Unloading models to free memory...")
            self.pipeline = None
            gc.collect()
            print("✅ Memory freed")
    
    def force_cleanup(self):
        """Force immediate cleanup"""
        if self.pipeline:
            print("🧹 Forcing model cleanup...")
            self.pipeline = None
            gc.collect()

# Global instance
model_manager = ModelManager(timeout=300)  # 5 minutes
```

### Step 5: Update `utils.py` to use ModelManager

```python
# Replace get_pipeline() function with:
from memory_optimizer import model_manager

def get_pipeline():
    """Get pipeline with memory management"""
    return model_manager.get_pipeline()
```

---

## 🌐 Part 2: Frontend Deployment (Netlify/Vercel)

### Option A: Netlify Deployment

**Create `netlify.toml` in project root:**

```toml
[build]
  base = "frontend"
  command = "npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[build.environment]
  NODE_VERSION = "18"
```

**Create `frontend/.env.production`:**

```env
VITE_API_URL=https://your-backend.onrender.com
```

### Option B: Vercel Deployment

**Create `vercel.json` in frontend folder:**

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    { "source": "/(.*)", "destination": "/" }
  ]
}
```

---

## 🗄️ Part 3: MongoDB Atlas Setup

1. **Create Free Cluster**: https://www.mongodb.com/cloud/atlas/register
2. **Network Access**: Add `0.0.0.0/0` (allow all IPs)
3. **Database User**: Create user with password
4. **Get Connection String**: 
   ```
   mongodb+srv://username:password@cluster.mongodb.net/pitch_insight?retryWrites=true&w=majority
   ```

---

## 🔧 Part 4: Environment Variables

### Backend (.env on Render):

```env
# MongoDB
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/pitch_insight

# Security
SECRET_KEY=<generate-strong-random-key>
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# APIs
WEATHER_API_KEY=b2ad62736cbd4e52aaa133601252712
GEMINI_API_KEY=your_gemini_api_key
RAZORPAY_KEY_ID=your_razorpay_key
RAZORPAY_KEY_SECRET=your_razorpay_secret

# CORS
ALLOWED_ORIGINS=https://your-app.netlify.app,https://your-app.vercel.app

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

---

## 🚀 Deployment Steps

### Backend (Render)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/pitch-insight.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Select `backend` as root directory
   - Use build command: `pip install -r requirements.txt`
   - Use start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Add environment variables
   - Click "Create Web Service"

3. **Note Backend URL**: `https://your-app-name.onrender.com`

### Frontend (Netlify)

1. **Update API URL**:
   - Edit `frontend/src/main.jsx` or create `.env.production`
   - Set: `VITE_API_URL=https://your-backend.onrender.com`

2. **Deploy on Netlify**:
   - Go to https://netlify.com
   - Drag & drop `frontend` folder OR connect GitHub
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Click "Deploy"

3. **Update CORS**:
   - Copy frontend URL from Netlify
   - Update `ALLOWED_ORIGINS` in Render environment variables
   - Add: `https://your-app.netlify.app`

---

## 🧪 Testing Deployed Application

1. **Health Check**:
   ```bash
   curl https://your-backend.onrender.com/api/health
   ```

2. **Test Frontend**: Open `https://your-app.netlify.app`

3. **Test Full Flow**:
   - Register a user
   - Upload an image
   - Check analysis results

---

## 📊 Monitoring & Troubleshooting

### Common Issues:

1. **Out of Memory (OOM) on Render**:
   - Reduce model cache size in `routes/analysis.py`
   - Implement model unloading (done above)
   - Consider upgrading to paid tier ($7/month for 512MB → 2GB)

2. **Cold Start Delays**:
   - Render free tier sleeps after 15 min inactivity
   - First request after sleep takes 30-60 seconds
   - Solution: Use a cron job to ping every 10 minutes

3. **CORS Errors**:
   - Ensure frontend URL is in `ALLOWED_ORIGINS`
   - Check browser console for exact error

### Performance Tips:

1. **Keep Service Awake** (create `keep-alive.yaml`):
   ```yaml
   # Use with GitHub Actions or cron-job.org
   # Ping every 10 minutes
   curl https://your-backend.onrender.com/api/health
   ```

2. **Monitor Memory Usage**:
   - Check Render dashboard
   - If consistently >400MB, consider optimizations

3. **Database Indexing**:
   ```python
   # Add to database.py startup
   analysis_collection.create_index([("user_id", 1), ("created_at", -1)])
   analysis_collection.create_index([("analysis_id", 1)])
   ```

---

## 💰 Cost Breakdown

- **Render (Backend)**: FREE (512MB RAM, sleeps after 15min)
- **Netlify/Vercel (Frontend)**: FREE (100GB bandwidth)
- **MongoDB Atlas**: FREE (512MB storage)
- **Total**: $0/month

**Upgrade Path** (if needed):
- Render Starter: $7/month (512MB RAM always on)
- MongoDB Shared: $0 (can upgrade later)

---

## 🔐 Security Checklist

- [ ] Generate strong `SECRET_KEY`
- [ ] Use environment variables for all secrets
- [ ] Enable MongoDB authentication
- [ ] Add proper CORS origins only
- [ ] Remove API keys from code
- [ ] Use HTTPS only
- [ ] Enable rate limiting (if traffic grows)

---

## 📱 Post-Deployment

1. **Custom Domain** (optional):
   - Netlify: Add custom domain in settings
   - Render: Add custom domain in settings

2. **SSL Certificate**: Automatic on both Render & Netlify

3. **Analytics**: Add Google Analytics to frontend

4. **Error Monitoring**: Consider Sentry for production errors

---

## 🎉 Success!

Your Pitch Insight app is now live! 

**What's Next?**
- Share with users and gather feedback
- Monitor performance and errors
- Plan for scaling if usage grows
- Consider paid tiers for better performance

---

## 🆘 Need Help?

Common resources:
- Render Docs: https://render.com/docs
- Netlify Docs: https://docs.netlify.com
- FastAPI Docs: https://fastapi.tiangolo.com

**Questions?** Check logs on Render dashboard for debugging.
