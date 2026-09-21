# 🚀 Deployment Configuration Files

This directory contains all configuration files needed to deploy Pitch Insight to production.

## 📁 File Structure

```
pitch_insight/
├── .gitignore                      # Git ignore rules
├── DEPLOYMENT_GUIDE.md             # Complete deployment guide
├── DEPLOYMENT_CHECKLIST.md         # Quick checklist
├── ENV_SETUP_GUIDE.md              # Environment variables guide
├── backend/
│   ├── .env.example                # Example environment variables
│   ├── .env.template               # Environment template
│   ├── Procfile                    # Render process file
│   ├── render.yaml                 # Render configuration
│   ├── requirements.txt            # Python dependencies (updated)
│   ├── database.py                 # MongoDB connection & models
│   ├── app.py                      # Main API (updated for production)
│   └── ...
└── frontend/
    ├── .env.example                # Example environment variables
    ├── .gitignore                  # Frontend git ignore
    ├── vercel.json                 # Vercel configuration
    ├── src/
    │   ├── config.js               # API configuration
    │   └── components/             # Updated components
    └── ...
```

## 🎯 Quick Start

### Option 1: Follow Full Guide (Recommended for first-time)
```bash
# Read the comprehensive guide
cat DEPLOYMENT_GUIDE.md
```

### Option 2: Use Checklist (For experienced users)
```bash
# Use the quick checklist
cat DEPLOYMENT_CHECKLIST.md
```

### Option 3: Environment Variables Only
```bash
# Just need to set up env vars
cat ENV_SETUP_GUIDE.md
```

## 🔧 What's Been Configured

### ✅ Backend Changes

1. **Production-ready configurations**
   - Environment variable support via `python-dotenv`
   - Dynamic CORS based on `FRONTEND_URL`
   - MongoDB integration with Motor (async)
   - Lifecycle management (startup/shutdown)
   - Error handling and logging

2. **New files created**
   - `database.py` - MongoDB connection manager
   - `Procfile` - Render process configuration
   - `render.yaml` - Render service configuration
   - `.env.template` - Environment variable template

3. **Dependencies added**
   - `motor` - Async MongoDB driver
   - `pymongo` - MongoDB tools
   - `python-dotenv` - Environment management

4. **New API endpoints**
   - `GET /api/analysis/{id}` - Get specific analysis
   - `GET /api/recent-analyses` - Get recent analyses
   - `GET /api/stats` - Database statistics
   - Enhanced health check

### ✅ Frontend Changes

1. **Environment-based configuration**
   - `config.js` - Centralized API configuration
   - Vite environment variables support
   - Dynamic API URL based on environment

2. **New files created**
   - `vercel.json` - Vercel deployment config
   - `.env.example` - Environment template
   - `.gitignore` - Frontend-specific ignore rules
   - `src/config.js` - Configuration module

3. **Updated components**
   - All components now use `config.js`
   - No hardcoded localhost URLs
   - Production-ready API calls

### ✅ Database Integration

MongoDB Atlas will store:
- Complete analysis results
- Pitch classifications
- Weather data snapshots
- Timestamps and metadata
- User upload statistics

Collections:
- `analyses` - All pitch analysis results

## 🌐 Deployment Stack

| Layer | Service | Purpose |
|-------|---------|---------|
| **Frontend** | Vercel | React app hosting, CDN, SSL |
| **Backend** | Render | FastAPI server, ML models |
| **Database** | MongoDB Atlas | Analysis results storage |
| **Weather** | OpenWeatherMap | Weather data API |
| **VCS** | GitHub | Version control, CI/CD |

## 📋 Deployment Steps Summary

1. **Prepare Repository**
   - Commit all changes
   - Push to GitHub

2. **Setup MongoDB Atlas**
   - Create free cluster
   - Get connection string

3. **Deploy Backend (Render)**
   - Connect GitHub
   - Configure environment
   - Upload model files
   - Deploy

4. **Deploy Frontend (Vercel)**
   - Connect GitHub
   - Configure environment
   - Deploy

5. **Connect Services**
   - Update CORS in backend
   - Test end-to-end

## 🔐 Required Credentials

Before deploying, obtain:

1. **OpenWeatherMap API Key**
   - Sign up: https://openweathermap.org/api
   - Free tier: 60 calls/min

2. **MongoDB Connection String**
   - From MongoDB Atlas
   - Format: `mongodb+srv://user:pass@cluster.mongodb.net/`

3. **Backend URL** (after Render deployment)
   - Format: `https://your-app.onrender.com`

4. **Frontend URL** (after Vercel deployment)
   - Format: `https://your-app.vercel.app`

## ⚙️ Configuration Files Explained

### backend/Procfile
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```
Tells Render how to start the web server.

### backend/render.yaml
YAML configuration for Render service deployment.

### backend/database.py
MongoDB connection manager with async support.

### frontend/vercel.json
Vercel deployment configuration with routing rules.

### frontend/src/config.js
Centralized API configuration using environment variables.

## 🧪 Testing Locally

### Test Backend
```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app:app --reload
# Visit: http://localhost:8000/docs
```

### Test Frontend
```bash
cd frontend
npm install
npm run dev
# Visit: http://localhost:5173
```

## 📊 Cost Estimates

**Free Tier (Recommended for Testing)**
- Vercel: Free
- Render: Free (with cold starts)
- MongoDB Atlas: Free (512MB)
- OpenWeatherMap: Free (60 calls/min)
- **Total: $0/month**

**Production Tier**
- Vercel: $20/mo (Pro)
- Render: $7/mo (Starter)
- MongoDB Atlas: $57/mo (M10)
- OpenWeatherMap: Free tier sufficient
- **Total: $84/month**

## 🐛 Common Issues

### "Module not found" on Render
```bash
# Ensure requirements.txt has all dependencies
pip freeze > requirements.txt
```

### "Connection refused" from frontend
```bash
# Check VITE_API_URL is correct
# Verify CORS allows your frontend URL
```

### "Database connection failed"
```bash
# Verify MongoDB connection string
# Check IP whitelist (0.0.0.0/0)
# Test credentials
```

## 📚 Documentation

- **Full Guide**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Checklist**: [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)
- **Environment**: [ENV_SETUP_GUIDE.md](./ENV_SETUP_GUIDE.md)

## 🔄 Updates & Maintenance

### Update Backend
```bash
git add backend/
git commit -m "Update backend"
git push
# Render auto-deploys
```

### Update Frontend
```bash
git add frontend/
git commit -m "Update frontend"
git push
# Vercel auto-deploys
```

### Update Environment Variables
- Render: Dashboard → Environment → Save
- Vercel: Settings → Environment Variables → Save → Redeploy

## 🎉 Success Criteria

Your deployment is successful when:
- [ ] Frontend loads at Vercel URL
- [ ] Backend health check passes
- [ ] Image upload works
- [ ] Analysis completes successfully
- [ ] Results display correctly
- [ ] Database stores data
- [ ] Weather integration works

## 🆘 Getting Help

If you encounter issues:

1. Check logs in Render/Vercel dashboards
2. Verify all environment variables
3. Test each service individually
4. Review [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
5. Check service status pages

## ✨ Next Steps After Deployment

1. **Custom Domain** - Add your own domain
2. **Monitoring** - Set up error tracking
3. **Analytics** - Add usage analytics
4. **Optimization** - Enable caching
5. **Security** - Restrict MongoDB IPs
6. **Backup** - Configure database backups

---

**Ready to deploy?** Start with [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)! 🚀
