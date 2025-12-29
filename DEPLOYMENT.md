# 🚀 Deployment Guide - Pitch Insight

This guide provides step-by-step instructions for deploying the Pitch Insight application to production.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Backend Deployment (Render)](#backend-deployment-render)
- [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
- [Environment Variables](#environment-variables)
- [Post-Deployment Testing](#post-deployment-testing)
- [Troubleshooting](#troubleshooting)
- [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🔧 Prerequisites

Before deploying, ensure you have:

### Required Accounts
- [GitHub](https://github.com) account
- [Render](https://render.com) account (for backend)
- [Vercel](https://vercel.com) account (for frontend)
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account (for production database)

### Required API Keys
- **Weather API**: Get from [WeatherAPI.com](https://www.weatherapi.com/)
- **Gemini AI**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Razorpay**: Get from [Razorpay Dashboard](https://dashboard.razorpay.com/)

### Local Tools
- Git
- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)

---

## 🏗️ Architecture Overview

```
┌─────────────┐         ┌──────────────┐         ┌───────────────┐
│   Vercel    │  HTTPS  │    Render    │  HTTPS  │  MongoDB      │
│  (Frontend) ├────────►│   (Backend)  ├────────►│    Atlas      │
│   React     │         │   FastAPI    │         │  (Database)   │
└─────────────┘         └──────────────┘         └───────────────┘
      │                        │
      │                        │
      ▼                        ▼
┌─────────────┐         ┌──────────────┐
│  CDN/Edge   │         │  External    │
│  Network    │         │  APIs        │
└─────────────┘         └──────────────┘
```

---

## 🎯 Backend Deployment (Render)

### Step 1: Prepare MongoDB Atlas

1. **Create a MongoDB Atlas Cluster**
   - Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
   - Create a new cluster (free tier is fine for testing)
   - Click "Connect" → "Connect your application"
   - Copy the connection string

2. **Configure Network Access**
   - Go to "Network Access"
   - Add IP Address: `0.0.0.0/0` (Allow from anywhere)
   - This is required for Render to connect

3. **Create Database User**
   - Go to "Database Access"
   - Add new database user with read/write permissions
   - Save username and password

### Step 2: Push Code to GitHub

```bash
# Navigate to your project
cd pitch_insight

# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for deployment"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/pitch-insight.git

# Push to GitHub
git push -u origin main
```

### Step 3: Deploy on Render

1. **Create New Web Service**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `pitch_insight` repository

2. **Configure Build Settings**
   ```
   Name: pitch-insight-backend
   Region: Oregon (US West)
   Branch: main
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

3. **Add Environment Variables**
   Click "Advanced" → "Add Environment Variable"

   ```env
   # Server
   HOST=0.0.0.0
   PORT=8000
   DEBUG=False

   # Database (paste your MongoDB Atlas connection string)
   MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   DATABASE_NAME=pitch_insight_prod

   # Security (generate a strong secret)
   SECRET_KEY=<generate-random-32-char-string>
   ACCESS_TOKEN_EXPIRE_MINUTES=10080

   # External APIs
   WEATHER_API_KEY=<your-weather-api-key>
   GEMINI_API_KEY=<your-gemini-api-key>
   RAZORPAY_KEY_ID=<your-razorpay-key>
   RAZORPAY_KEY_SECRET=<your-razorpay-secret>

   # CORS (will update after frontend deployment)
   ALLOWED_ORIGINS=https://pitch-insight.vercel.app
   ```

4. **Create Service**
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)
   - Note your backend URL: `https://pitch-insight-backend.onrender.com`

5. **Verify Deployment**
   - Visit: `https://your-app.onrender.com/api/health`
   - Should return: `{"status": "healthy"}`
   - Visit: `https://your-app.onrender.com/docs`
   - Should show API documentation

---

## 🌐 Frontend Deployment (Vercel)

### Step 1: Update Environment Variables

1. **Update Frontend Environment**
   Edit `frontend/.env.production`:
   ```env
   VITE_API_URL=https://your-backend-app.onrender.com
   VITE_RAZORPAY_KEY_ID=<your-razorpay-key>
   ```

2. **Commit Changes**
   ```bash
   git add frontend/.env.production
   git commit -m "Update production API URL"
   git push
   ```

### Step 2: Deploy on Vercel

1. **Import Project**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Select `pitch_insight`

2. **Configure Project**
   ```
   Framework Preset: Vite
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

3. **Add Environment Variables**
   In Vercel dashboard:
   ```env
   VITE_API_URL=https://your-backend-app.onrender.com
   VITE_RAZORPAY_KEY_ID=<your-razorpay-key>
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete (2-3 minutes)
   - Note your frontend URL: `https://pitch-insight.vercel.app`

### Step 3: Update Backend CORS

1. **Update Backend Environment on Render**
   - Go to Render Dashboard → Your backend service
   - Environment → Edit
   - Update `ALLOWED_ORIGINS`:
     ```
     ALLOWED_ORIGINS=https://pitch-insight.vercel.app,https://pitch-insight-frontend.vercel.app
     ```
   - Save and redeploy

---

## 🔐 Environment Variables

### Backend (.env)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `MONGODB_URL` | MongoDB connection string | ✅ | `mongodb+srv://user:pass@cluster.mongodb.net/` |
| `DATABASE_NAME` | Database name | ✅ | `pitch_insight_prod` |
| `SECRET_KEY` | JWT secret key | ✅ | `<32-char-random-string>` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry | ✅ | `10080` (7 days) |
| `WEATHER_API_KEY` | Weather API key | ✅ | From WeatherAPI.com |
| `GEMINI_API_KEY` | Gemini AI key | ✅ | From Google AI Studio |
| `RAZORPAY_KEY_ID` | Razorpay key | ✅ | From Razorpay Dashboard |
| `RAZORPAY_KEY_SECRET` | Razorpay secret | ✅ | From Razorpay Dashboard |
| `ALLOWED_ORIGINS` | CORS origins | ✅ | `https://your-app.vercel.app` |
| `HOST` | Server host | ✅ | `0.0.0.0` |
| `PORT` | Server port | ✅ | `8000` |
| `DEBUG` | Debug mode | ✅ | `False` |

### Frontend (.env.production)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `VITE_API_URL` | Backend API URL | ✅ | `https://your-app.onrender.com` |
| `VITE_RAZORPAY_KEY_ID` | Razorpay public key | ✅ | From Razorpay Dashboard |

---

## ✅ Post-Deployment Testing

### 1. Health Check
```bash
curl https://your-backend.onrender.com/api/health
```
Expected: `{"status": "healthy"}`

### 2. API Documentation
Visit: `https://your-backend.onrender.com/docs`

### 3. Frontend Access
Visit: `https://your-app.vercel.app`

### 4. Test User Flow
1. **Sign Up**: Create a new account
2. **Login**: Authenticate
3. **Quick Analysis**: Upload an image (free feature)
4. **Payment**: Test subscription purchase
5. **Complete Analysis**: Test Pro feature

### 5. Check Logs

**Render Logs:**
- Render Dashboard → Your Service → Logs

**Vercel Logs:**
- Vercel Dashboard → Your Project → Deployments → View Function Logs

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CORS Errors
**Symptom:** "Access-Control-Allow-Origin" error in browser console

**Solution:**
- Ensure `ALLOWED_ORIGINS` in backend includes your frontend URL
- Check for trailing slashes in URLs
- Verify the URL protocol (https vs http)

#### 2. Database Connection Failed
**Symptom:** "Connection refused" or "Authentication failed"

**Solution:**
- Verify MongoDB Atlas connection string
- Check network access settings (whitelist 0.0.0.0/0)
- Confirm database user credentials

#### 3. API Requests Failing
**Symptom:** Network errors or 404 responses

**Solution:**
- Verify `VITE_API_URL` in frontend matches backend URL
- Check backend deployment status on Render
- Look for errors in Render logs

#### 4. Payment Not Working
**Symptom:** Razorpay popup doesn't appear

**Solution:**
- Verify `RAZORPAY_KEY_ID` matches in backend and frontend
- Check if using correct keys (test vs live)
- Ensure Razorpay account is activated

#### 5. Model Files Missing
**Symptom:** "Model file not found" errors

**Solution:**
- Ensure model files are included in git (check .gitignore)
- Verify files are in `backend/` directory
- Check file paths in code

---

## 📊 Monitoring & Maintenance

### Performance Monitoring

**Render:**
- Monitor CPU/Memory usage in dashboard
- Set up alerts for downtime
- Review logs regularly

**Vercel:**
- Monitor build times
- Check bandwidth usage
- Review function execution logs

### Scaling Considerations

**Free Tier Limitations:**
- Render: Service spins down after inactivity (cold starts)
- Vercel: Limited bandwidth and function executions

**Upgrading:**
- Render: Starter plan ($7/month) for always-on service
- Vercel: Pro plan ($20/month) for higher limits

### Backup Strategy

**Database:**
- MongoDB Atlas has automatic backups
- Configure backup schedule in Atlas

**Code:**
- Keep GitHub repository up to date
- Tag releases for easy rollback

### Security Best Practices

1. **Rotate secrets regularly**
   - Update SECRET_KEY quarterly
   - Rotate API keys if compromised

2. **Monitor API usage**
   - Track external API quotas
   - Set up usage alerts

3. **Review logs**
   - Check for suspicious activity
   - Monitor error rates

4. **Keep dependencies updated**
   ```bash
   # Backend
   pip list --outdated
   
   # Frontend
   npm outdated
   ```

---

## 🔄 Continuous Deployment

Both Render and Vercel support automatic deployments:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Update feature"
   git push
   ```

2. **Automatic Deployment**
   - Render: Auto-deploys on push to main branch
   - Vercel: Auto-deploys on push to main branch

3. **Rollback if Needed**
   - Render: Redeploy previous version from dashboard
   - Vercel: Instant rollback to any previous deployment

---

## 📞 Support

For deployment issues:
- Render: [docs.render.com](https://docs.render.com)
- Vercel: [vercel.com/docs](https://vercel.com/docs)
- MongoDB Atlas: [docs.atlas.mongodb.com](https://docs.atlas.mongodb.com)

---

## ✨ Production Checklist

Before going live:

- [ ] All environment variables configured
- [ ] MongoDB Atlas cluster created and connected
- [ ] Backend deployed and health check passing
- [ ] Frontend deployed and accessible
- [ ] CORS configured correctly
- [ ] Payment gateway tested
- [ ] All external APIs working
- [ ] Error monitoring set up
- [ ] Backup strategy in place
- [ ] Custom domain configured (optional)
- [ ] SSL/TLS enabled (automatic on Render/Vercel)
- [ ] Security headers configured
- [ ] Rate limiting implemented
- [ ] Logging enabled
- [ ] Documentation updated

---

**🎉 Congratulations! Your application is now live in production!**
