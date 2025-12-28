# 🚀 Pre-Deployment Checklist

Complete this checklist before deploying to production.

## ✅ Code Preparation

- [x] Optimized `requirements.txt` for 512MB RAM
- [x] Removed matplotlib dependency
- [x] Changed opencv-python to opencv-python-headless
- [x] Implemented memory management (ModelManager)
- [x] Created deployment configuration files
- [ ] Test application locally with optimized dependencies
- [ ] Commit all changes to Git

## ✅ MongoDB Atlas Setup

- [ ] Create free MongoDB Atlas account
- [ ] Create a cluster (M0 Free tier)
- [ ] Create database user with password
- [ ] Add IP whitelist: `0.0.0.0/0` (all IPs)
- [ ] Get connection string
- [ ] Test connection locally

## ✅ Environment Variables

### Required Variables:
- [ ] `MONGODB_URL` - From MongoDB Atlas
- [ ] `SECRET_KEY` - Generate strong random key
- [ ] `GEMINI_API_KEY` - From Google AI Studio
- [ ] `RAZORPAY_KEY_ID` - From Razorpay Dashboard
- [ ] `RAZORPAY_KEY_SECRET` - From Razorpay Dashboard

### Optional Variables (have defaults):
- [ ] `WEATHER_API_KEY` - Default provided
- [ ] `DATABASE_NAME` - Default: pitch_insight
- [ ] `ACCESS_TOKEN_EXPIRE_MINUTES` - Default: 10080

## ✅ Backend Deployment (Render)

- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Sign up for Render.com
- [ ] Create new Web Service
- [ ] Connect GitHub repository
- [ ] Set root directory to `backend`
- [ ] Configure build command: `pip install -r requirements.txt`
- [ ] Configure start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] Add all environment variables
- [ ] Deploy and wait for success
- [ ] Note the backend URL: `https://________.onrender.com`
- [ ] Test health endpoint: `/api/health`

## ✅ Frontend Deployment (Netlify/Vercel)

### Netlify:
- [ ] Update `frontend/.env.production` with backend URL
- [ ] Sign up for Netlify
- [ ] Drag & drop `frontend` folder OR connect GitHub
- [ ] Configure build:
  - Build command: `npm run build`
  - Publish directory: `dist`
  - Base directory: `frontend`
- [ ] Deploy
- [ ] Note frontend URL: `https://________.netlify.app`

### OR Vercel:
- [ ] Update `frontend/.env.production` with backend URL
- [ ] Sign up for Vercel
- [ ] Import GitHub repository
- [ ] Configure:
  - Framework: Vite
  - Root directory: `frontend`
  - Build command: `npm run build`
  - Output directory: `dist`
- [ ] Deploy
- [ ] Note frontend URL: `https://________.vercel.app`

## ✅ CORS Configuration

- [ ] Copy frontend URL from Netlify/Vercel
- [ ] Update `ALLOWED_ORIGINS` in Render environment variables
- [ ] Add frontend URL to ALLOWED_ORIGINS
- [ ] Redeploy backend if needed

## ✅ Testing Deployed Application

### Backend Tests:
- [ ] Test health: `curl https://your-backend.onrender.com/api/health`
- [ ] Check logs for errors
- [ ] Verify models load successfully on first analysis

### Frontend Tests:
- [ ] Open frontend URL
- [ ] Test user registration
- [ ] Test login
- [ ] Upload test pitch image
- [ ] Verify analysis results
- [ ] Test weather integration
- [ ] Test chat functionality
- [ ] Test subscription flow (if implemented)

### Integration Tests:
- [ ] Verify CORS working correctly
- [ ] Check database connections
- [ ] Test payment gateway (if using real keys)
- [ ] Test all API endpoints from frontend

## ✅ Performance & Monitoring

- [ ] Monitor memory usage on Render dashboard
- [ ] Check response times
- [ ] Verify model auto-unloading works
- [ ] Set up keep-alive ping (optional)
- [ ] Monitor error rates

## ✅ Security

- [ ] All secrets in environment variables (not in code)
- [ ] Strong SECRET_KEY generated
- [ ] MongoDB has authentication enabled
- [ ] CORS restricted to actual frontend domains
- [ ] HTTPS enabled (automatic on Render/Netlify)
- [ ] No sensitive data in logs
- [ ] API keys not exposed in frontend

## ✅ Documentation

- [ ] Document deployment process
- [ ] Note all URLs and credentials (securely)
- [ ] Create user guide (if needed)
- [ ] Document API endpoints (auto-generated at /docs)

## ✅ Post-Deployment

- [ ] Share app with test users
- [ ] Gather initial feedback
- [ ] Monitor errors and crashes
- [ ] Plan for scaling if needed

---

## 🔧 Commands Reference

### Generate Secret Key:
```python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Test Backend Locally:
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### Test Frontend Locally:
```bash
cd frontend
npm install
npm run dev
```

### Build Frontend:
```bash
cd frontend
npm run build
```

---

## 🆘 Troubleshooting

### Backend Issues:
- Check Render logs for errors
- Verify environment variables are set
- Test MongoDB connection separately
- Check model file sizes

### Frontend Issues:
- Verify API URL in `.env.production`
- Check browser console for errors
- Test API endpoints directly with curl
- Verify CORS settings

### Memory Issues on Render:
- Monitor dashboard for OOM errors
- Check if models are unloading properly
- Consider reducing cache size
- Upgrade to paid tier if needed ($7/month)

---

## 📊 Expected Resource Usage

**Backend (Render Free Tier):**
- Memory: ~300-400MB with models loaded
- Memory: ~100-150MB with models unloaded
- Cold start: ~30-60 seconds (first request after sleep)
- Warm request: ~2-5 seconds

**Frontend (Netlify/Vercel):**
- Build size: ~1-2MB
- Bandwidth: <100GB/month (free tier)

**Database (MongoDB Atlas):**
- Storage: <100MB (free tier allows 512MB)

---

## ✅ Deployment Complete!

Once all items are checked:

1. **Backend URL**: https://________.onrender.com
2. **Frontend URL**: https://________.netlify.app
3. **API Docs**: https://________.onrender.com/docs

**Next Steps:**
- Monitor application health
- Collect user feedback
- Plan feature enhancements
- Consider analytics integration

Good luck! 🚀
