# 🏏 Pitch Insight - Production Deployment Package

> AI-powered cricket pitch analysis with weather integration, optimized for deployment on Render's free tier (512MB RAM)

[![Deploy Status](https://img.shields.io/badge/deploy-ready-brightgreen)]()
[![Memory Optimized](https://img.shields.io/badge/memory-512MB%20optimized-blue)]()
[![Cost](https://img.shields.io/badge/cost-FREE-success)]()

---

## 📋 Quick Links

- 🚀 **[Quick Start Deployment](QUICKSTART_DEPLOY.md)** - Deploy in 15 minutes
- 📖 **[Complete Deployment Guide](DEPLOYMENT_GUIDE.md)** - Detailed instructions
- ✅ **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)** - Step-by-step checklist
- 📊 **[Deployment Summary](DEPLOYMENT_SUMMARY.md)** - What's included
- 🔄 **[Keep-Alive Guide](KEEP_ALIVE_GUIDE.md)** - Prevent cold starts (optional)

---

## 🎯 What's This?

This application analyzes cricket pitch images using computer vision and machine learning to provide:

- **Pitch Type Classification**: Batting-friendly, bowling-friendly, seam-friendly, spin-friendly
- **Feature Analysis**: Grass coverage, cracks, moisture levels, color patterns
- **Weather Integration**: Real-time weather impact on pitch behavior
- **Match Strategy**: Data-driven recommendations for team selection and tactics
- **AI Chatbot**: Interactive Q&A about pitch analysis results

---

## ⚡ Features

✅ **User Authentication** - JWT-based secure login/registration
✅ **Image Analysis** - ONNX-optimized ML models (54MB)
✅ **Weather API** - Real-time weather data integration
✅ **Subscription System** - Free & Pro tiers with Razorpay
✅ **Chat Assistant** - Google Gemini AI integration
✅ **Analysis History** - MongoDB-backed user history
✅ **Memory Optimized** - Auto-unloading models for 512MB RAM
✅ **Production Ready** - Complete deployment configuration

---

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │
│  (Netlify/      │
│   Vercel)       │
└────────┬────────┘
         │ HTTPS/REST API
         │
┌────────▼────────────┐         ┌──────────────┐
│  FastAPI Backend    │────────▶│  MongoDB     │
│  (Render.com)       │         │  Atlas       │
│  - ONNX Models      │         │              │
│  - Auto Memory Mgmt │         │  Users       │
│  - 512MB RAM        │         │  Analyses    │
└─────────────────────┘         └──────────────┘
```

---

## 📦 What's Included

### Backend (`/backend`)
- ✅ FastAPI REST API
- ✅ ONNX models (YOLO + Classifier)
- ✅ Memory-optimized pipeline
- ✅ JWT authentication
- ✅ MongoDB integration
- ✅ Razorpay payment gateway
- ✅ Google Gemini chatbot
- ✅ Weather API integration

### Frontend (`/frontend`)
- ✅ React + Vite
- ✅ Modern UI with Lucide icons
- ✅ Authentication pages
- ✅ Image upload & analysis
- ✅ Results visualization
- ✅ Chat widget
- ✅ Subscription management

### Deployment Files
- ✅ `render.yaml` - Render configuration
- ✅ `netlify.toml` - Netlify configuration
- ✅ `vercel.json` - Vercel alternative
- ✅ `.gitignore` - Security
- ✅ `runtime.txt` - Python version

### Documentation
- ✅ Complete deployment guides
- ✅ Interactive checklists
- ✅ Troubleshooting tips
- ✅ Security best practices

---

## 🚀 Deployment Overview

### Prerequisites
- GitHub account
- Render.com account (free)
- Netlify/Vercel account (free)
- MongoDB Atlas account (free)
- API keys: Gemini, Razorpay

### Deployment Steps

1. **Setup Database** (5 min)
   - Create MongoDB Atlas cluster
   - Get connection string

2. **Deploy Backend** (5 min)
   - Push to GitHub
   - Deploy on Render
   - Configure environment variables

3. **Deploy Frontend** (3 min)
   - Build with Vite
   - Deploy on Netlify/Vercel
   - Update API URL

4. **Configure CORS** (2 min)
   - Update backend with frontend URL
   - Redeploy

**Total Time: ~15 minutes**

See [QUICKSTART_DEPLOY.md](QUICKSTART_DEPLOY.md) for detailed steps.

---

## 💾 Memory Optimization

### The Challenge
- Render free tier: **512MB RAM**
- ML models: **54MB**
- Loaded models + deps: **~300-400MB**
- Risk: Out of Memory errors

### The Solution
Custom **ModelManager** class that:
1. ✅ Lazy loads models (not at startup)
2. ✅ Auto-unloads after 5 minutes idle
3. ✅ Reloads on next request
4. ✅ Garbage collection to free RAM

### Result
- **Startup**: ~150MB (no models)
- **Active**: ~300-400MB (models loaded)
- **Idle**: ~150MB (models unloaded)
- **Safe margin**: ~100-150MB

See [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) for details.

---

## 📊 Performance

### Backend (Render Free Tier)
- **Cold Start**: 30-60s (first request after sleep)
- **Warm Request**: 2-5s (models cached)
- **Model Load**: 3-5s (on-demand)
- **Auto-Unload**: After 5 min idle
- **Memory**: 150-400MB depending on usage

### Frontend (Netlify/Vercel)
- **Load Time**: <2s (static CDN)
- **Bandwidth**: 100GB/month (free)
- **SSL**: Automatic

### Database (MongoDB Atlas)
- **Storage**: 512MB free
- **Queries**: Fast (<100ms)
- **Backup**: Automatic

---

## 💰 Cost Breakdown

| Service | Tier | Cost |
|---------|------|------|
| **Render** (Backend) | Free - 512MB RAM | $0 |
| **Netlify/Vercel** (Frontend) | Free - 100GB bandwidth | $0 |
| **MongoDB Atlas** (Database) | M0 - 512MB storage | $0 |
| **Total** | | **$0/month** |

### Optional Upgrades
- **Render Starter**: $7/month (always-on, no cold starts)
- **MongoDB Shared**: Still free until 512MB

---

## 🛡️ Security Features

✅ **Environment Variables** - All secrets in env vars
✅ **JWT Authentication** - Secure token-based auth
✅ **Password Hashing** - bcrypt with salt
✅ **CORS Protection** - Restricted origins
✅ **HTTPS Only** - Automatic SSL
✅ **Input Validation** - Pydantic schemas
✅ **File Size Limits** - Max 5MB uploads
✅ **MongoDB Auth** - Username/password required

---

## 🧪 Testing

### Health Check
```bash
curl https://your-backend.onrender.com/api/health
```

### Memory Stats
```bash
curl https://your-backend.onrender.com/api/stats
```

### API Documentation
Open: `https://your-backend.onrender.com/docs`

---

## 📈 Monitoring

### Check These Regularly:

1. **Memory Usage** (Render Dashboard)
   - Should stay <450MB
   - Models should auto-unload when idle

2. **Response Times**
   - Cold start: 30-60s (normal)
   - Warm: 2-5s (good)

3. **Error Rates** (Render Logs)
   - Check for OOM errors
   - MongoDB connection issues
   - Missing environment variables

4. **Database Usage** (MongoDB Atlas)
   - Storage usage
   - Query performance

---

## 🐛 Common Issues & Solutions

### 1. Out of Memory
**Solution**: Already optimized with auto-unload. Check `/api/stats` to verify models unload when idle.

### 2. CORS Errors
**Solution**: Verify `ALLOWED_ORIGINS` in Render matches your frontend URL exactly.

### 3. Cold Start Delays
**Solution**: Normal for free tier. See [KEEP_ALIVE_GUIDE.md](KEEP_ALIVE_GUIDE.md) for prevention.

### 4. MongoDB Connection Failed
**Solution**: Check connection string, password, and IP whitelist (0.0.0.0/0).

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) troubleshooting section for more.

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART_DEPLOY.md](QUICKSTART_DEPLOY.md) | Fast deployment (15 min) |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Complete guide (detailed) |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Interactive checklist |
| [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) | Technical overview |
| [KEEP_ALIVE_GUIDE.md](KEEP_ALIVE_GUIDE.md) | Prevent cold starts |
| [backend/README.md](backend/README.md) | Backend API docs |
| [frontend/README.md](frontend/README.md) | Frontend setup |

---

## 🔧 Tech Stack

### Backend
- **Framework**: FastAPI 0.115
- **ML**: ONNX Runtime (CPU)
- **Database**: MongoDB + Motor
- **Auth**: JWT + bcrypt
- **Payment**: Razorpay
- **AI**: Google Gemini
- **Server**: Uvicorn

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite 6
- **HTTP**: Axios
- **Icons**: Lucide React
- **Styling**: CSS Modules

### ML Models
- **Detection**: YOLOv8 ONNX (11.7MB)
- **Classification**: Custom CNN ONNX (42.6MB)
- **Total**: 54.3MB

---

## 📝 Environment Variables Reference

### Backend (.env)
```env
# Database
MONGODB_URL=mongodb+srv://...
DATABASE_NAME=pitch_insight

# Security
SECRET_KEY=<generate-random>
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# APIs
WEATHER_API_KEY=<provided>
GEMINI_API_KEY=<your-key>
RAZORPAY_KEY_ID=<your-key>
RAZORPAY_KEY_SECRET=<your-secret>

# Server
ALLOWED_ORIGINS=https://your-app.netlify.app
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### Frontend (.env.production)
```env
VITE_API_URL=https://your-backend.onrender.com
```

---

## 🎯 Next Steps

1. ✅ **Read** [QUICKSTART_DEPLOY.md](QUICKSTART_DEPLOY.md)
2. ✅ **Setup** MongoDB Atlas
3. ✅ **Gather** API keys
4. ✅ **Deploy** backend to Render
5. ✅ **Deploy** frontend to Netlify
6. ✅ **Test** complete application
7. ✅ **Monitor** performance

---

## 🤝 Support

### Resources:
- **Render Docs**: https://render.com/docs
- **Netlify Docs**: https://docs.netlify.com
- **MongoDB Atlas**: https://docs.atlas.mongodb.com
- **FastAPI**: https://fastapi.tiangolo.com

### Need Help?
1. Check deployment documentation
2. Review troubleshooting sections
3. Check service logs (Render/Netlify)
4. Verify environment variables

---

## 📄 License

This project is ready for deployment. Ensure you comply with all third-party service terms:
- Render.com Terms
- Netlify/Vercel Terms
- MongoDB Atlas Terms
- API provider terms (Gemini, Razorpay, Weather)

---

## ✨ Features Roadmap

### Currently Implemented:
- ✅ Complete ML pipeline
- ✅ Authentication system
- ✅ Payment integration
- ✅ Weather analysis
- ✅ AI chatbot
- ✅ Memory optimization
- ✅ Production deployment config

### Future Enhancements:
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] Advanced analytics
- [ ] Team management
- [ ] API rate limiting
- [ ] Advanced caching

---

## 🎉 Ready to Deploy!

Your application is **production-ready** and optimized for Render's free tier.

**Start here**: [QUICKSTART_DEPLOY.md](QUICKSTART_DEPLOY.md)

**Deployment time**: ~15 minutes
**Total cost**: $0/month
**Memory**: Optimized for 512MB
**Performance**: Production-grade

### What You Get:
✅ Live backend API
✅ Deployed frontend
✅ Cloud database
✅ SSL/HTTPS
✅ Automatic deploys
✅ Error monitoring

**Let's deploy! 🚀**

---

## 📞 Quick Support

**Issue**: Out of memory
**Fix**: Check [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md#memory-management-strategy)

**Issue**: CORS errors
**Fix**: Update `ALLOWED_ORIGINS` in Render

**Issue**: Cold starts
**Fix**: See [KEEP_ALIVE_GUIDE.md](KEEP_ALIVE_GUIDE.md)

**Issue**: Models not loading
**Fix**: Check Render logs, verify `.onnx` files committed

---

Made with ❤️ for cricket analytics
