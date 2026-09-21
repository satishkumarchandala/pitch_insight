# 🚀 Pitch Insight Deployment Guide

Complete step-by-step guide to deploy Pitch Insight using Vercel (Frontend), Render (Backend), and MongoDB Atlas (Database).

---

## 📋 Prerequisites

Before deploying, ensure you have:
- [ ] GitHub account
- [ ] Vercel account (free - [vercel.com](https://vercel.com))
- [ ] Render account (free - [render.com](https://render.com))
- [ ] MongoDB Atlas account (free - [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas))
- [ ] OpenWeatherMap API key (free - [openweathermap.org/api](https://openweathermap.org/api))
- [ ] Git installed locally
- [ ] Your project files

---

## 🗂️ Phase 1: Prepare Your Repository

### Step 1: Initialize Git Repository

```bash
# Navigate to project directory
cd e:\pitch_insight

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Ready for deployment"
```

### Step 2: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click **New Repository**
3. Name it: `pitch-insight`
4. Make it **Public** (required for free Render tier) or **Private** (requires paid tier)
5. Don't initialize with README
6. Click **Create Repository**

### Step 3: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/pitch-insight.git

# Push code
git branch -M main
git push -u origin main
```

⚠️ **IMPORTANT**: The model files (`*.pt`, `*.pth`) are ignored by `.gitignore` because they're too large for GitHub (100MB+ limit).

---

## 📦 Phase 2: MongoDB Atlas Setup (Database)

### Step 1: Create MongoDB Cluster

1. Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up or log in
3. Click **Create a New Cluster**
4. Choose **FREE** M0 tier
5. Select a cloud provider (AWS/Google/Azure) and region closest to you
6. Name your cluster: `pitch-insight-cluster`
7. Click **Create Cluster** (takes 3-5 minutes)

### Step 2: Create Database User

1. In left sidebar, click **Database Access**
2. Click **Add New Database User**
3. Choose **Password** authentication
4. Username: `pitch_admin` (or your choice)
5. Password: Generate a secure password **SAVE THIS!**
6. Database User Privileges: **Atlas Admin**
7. Click **Add User**

### Step 3: Configure Network Access

1. In left sidebar, click **Network Access**
2. Click **Add IP Address**
3. Click **Allow Access from Anywhere** (0.0.0.0/0)
   - ⚠️ For production, restrict to your Render IPs
4. Click **Confirm**

### Step 4: Get Connection String

1. Go to **Database** → **Clusters**
2. Click **Connect** on your cluster
3. Choose **Connect your application**
4. Driver: **Python**, Version: **3.12 or later**
5. Copy the connection string:
   ```
   mongodb+srv://pitch_admin:<password>@pitch-insight-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. Replace `<password>` with your actual password
7. **SAVE THIS CONNECTION STRING** - you'll need it later!

---

## 🖥️ Phase 3: Deploy Backend to Render

### Step 1: Upload Model Files

Since model files are too large for GitHub, you need to upload them manually to Render:

**Option A: Use GitHub Releases (Recommended)**
1. Go to your GitHub repo → **Releases** → **Create a new release**
2. Upload `pitch_yolov8_best.pt` and `best_pitch_classifier.pth`
3. Note the download URLs

**Option B: Use Render Disk Storage**
- Upload via Render's persistent disk feature (requires paid plan)

**Option C: Host on Cloud Storage**
- Upload to AWS S3, Google Cloud Storage, or Dropbox
- Get public URLs

### Step 2: Create Render Web Service

1. Go to [render.com](https://render.com) and sign in
2. Click **New +** → **Web Service**
3. Connect your GitHub account if not already
4. Select your `pitch-insight` repository
5. Configure:
   - **Name**: `pitch-insight-backend`
   - **Region**: Choose closest to your MongoDB region
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free` (or paid for better performance)

### Step 3: Configure Environment Variables

In the **Environment** section, add these variables:

| Key | Value | Note |
|-----|-------|------|
| `PYTHON_VERSION` | `3.12.0` | Python version |
| `WEATHER_API_KEY` | `your_key_here` | From OpenWeatherMap |
| `MONGODB_URI` | `mongodb+srv://...` | From Step 2-4 above |
| `MONGODB_DB_NAME` | `pitch_insight` | Database name |
| `FRONTEND_URL` | Leave blank for now | We'll update this later |
| `ENVIRONMENT` | `production` | Production mode |
| `ENABLE_DATABASE` | `true` | Enable MongoDB |
| `ENABLE_WEATHER` | `true` | Enable weather API |

### Step 4: Deploy Backend

1. Click **Create Web Service**
2. Wait for deployment (10-15 minutes first time)
3. Watch the logs for any errors
4. Once deployed, you'll get a URL: `https://pitch-insight-backend.onrender.com`
5. **SAVE THIS URL!**

### Step 5: Download Model Files to Render

**After deployment**, you need to add model files:

1. Go to your service → **Shell** tab
2. Run commands to download models:

```bash
# If using GitHub releases
cd /opt/render/project/src
wget https://github.com/YOUR_USERNAME/pitch-insight/releases/download/v1.0/pitch_yolov8_best.pt
wget https://github.com/YOUR_USERNAME/pitch-insight/releases/download/v1.0/best_pitch_classifier.pth

# Or use curl
curl -L -o pitch_yolov8_best.pt "YOUR_MODEL_URL"
curl -L -o best_pitch_classifier.pth "YOUR_CLASSIFIER_URL"
```

3. Restart the service: **Manual Deploy** → **Clear build cache & deploy**

### Step 6: Test Backend

Visit: `https://pitch-insight-backend.onrender.com/api/health`

You should see:
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "timestamp": "..."
}
```

Also test docs: `https://pitch-insight-backend.onrender.com/docs`

---

## 🌐 Phase 4: Deploy Frontend to Vercel

### Step 1: Create Production Environment File

1. In your local `frontend/` directory, create `.env.production`:

```env
VITE_API_URL=https://pitch-insight-backend.onrender.com
VITE_APP_NAME=Pitch Insight
VITE_ENABLE_ANALYTICS=false
```

Replace with your actual Render backend URL!

2. Commit and push:

```bash
git add frontend/.env.production
git commit -m "Add production environment config"
git push
```

### Step 2: Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **Add New** → **Project**
3. Import your `pitch-insight` GitHub repository
4. Configure:
   - **Project Name**: `pitch-insight`
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

### Step 3: Configure Environment Variables

In **Environment Variables** section, add:

| Key | Value |
|-----|-------|
| `VITE_API_URL` | `https://pitch-insight-backend.onrender.com` |
| `VITE_APP_NAME` | `Pitch Insight` |
| `VITE_ENABLE_ANALYTICS` | `false` |

### Step 4: Deploy

1. Click **Deploy**
2. Wait 2-3 minutes for deployment
3. You'll get a URL: `https://pitch-insight.vercel.app`
4. **SAVE THIS URL!**

### Step 5: Update Backend CORS

Go back to Render:

1. Open your backend service
2. **Environment** → Add/Update:
   - `FRONTEND_URL` = `https://pitch-insight.vercel.app`
3. Save and redeploy

---

## ✅ Phase 5: Verification & Testing

### 1. Test Frontend

Visit: `https://pitch-insight.vercel.app`

- [ ] Page loads correctly
- [ ] Upload section visible
- [ ] No console errors

### 2. Test Upload

1. Upload a cricket pitch image
2. Add location (optional)
3. Click Analyze
4. Verify results appear

### 3. Test Database

Visit: `https://pitch-insight-backend.onrender.com/api/stats`

Should show:
```json
{
  "total_analyses": 1,
  "pitch_type_distribution": {...},
  "connected": true
}
```

### 4. Test Weather Integration

Analyze with location data and verify weather appears in results.

---

## 🔧 Phase 6: Post-Deployment Configuration

### Custom Domain (Optional)

**For Vercel:**
1. **Settings** → **Domains**
2. Add your custom domain
3. Update DNS records as instructed

**For Render:**
1. **Settings** → **Custom Domain**
2. Add domain and configure DNS

### Enable HTTPS

Both Vercel and Render provide free SSL certificates automatically!

### Monitoring

**Render:**
- Check **Logs** tab regularly
- Set up **Health Checks** in settings
- Monitor resource usage

**Vercel:**
- **Analytics** tab for traffic
- **Logs** for deployment issues

**MongoDB Atlas:**
- Check **Metrics** for database performance
- Set up alerts for high usage

---

## 📊 Cost Breakdown

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **MongoDB Atlas** | 512MB storage, Shared RAM | $0.08/hr (~$57/mo) M10 cluster |
| **Render** | 512MB RAM, Sleeps after 15min inactivity | $7/mo - 512MB persistent |
| **Vercel** | 100GB bandwidth, Unlimited sites | $20/mo Pro plan |
| **OpenWeatherMap** | 60 calls/min, 1M calls/mo | $40/mo for more |
| **TOTAL** | **$0/month** | $84/month for production |

### Free Tier Limitations:
- **Render**: Service sleeps after 15 min inactivity (30s cold start)
- **MongoDB**: 512MB storage limit
- **Vercel**: 100GB bandwidth/month

---

## 🐛 Troubleshooting

### Backend Issues

**Error: "Module not found"**
```bash
# In Render Shell
pip install -r requirements.txt
```

**Error: "Model files not found"**
- Re-download model files as shown in Phase 3, Step 5

**Error: "MongoDB connection failed"**
- Verify connection string is correct
- Check MongoDB IP whitelist includes 0.0.0.0/0
- Verify database user credentials

### Frontend Issues

**Error: "Network Error"**
- Check `VITE_API_URL` is correct
- Verify backend is running
- Check browser console for CORS errors

**API calls fail**
- Verify `FRONTEND_URL` is set in Render backend
- Check backend CORS configuration

### Database Issues

**"Database not connected"**
- Verify `MONGODB_URI` environment variable
- Check MongoDB Atlas cluster status
- Test connection string locally

---

## 🔄 Continuous Deployment

Both platforms support automatic deployments:

### Vercel
- Auto-deploys on push to `main` branch
- Preview deployments for pull requests

### Render
- Auto-deploys on push to `main` branch
- Configure in **Settings** → **Build & Deploy**

To disable auto-deploy:
- Vercel: **Settings** → **Git** → Disable
- Render: **Settings** → **Build & Deploy** → Set to Manual

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Production Build](https://vitejs.dev/guide/build.html)

---

## 🆘 Support

If you encounter issues:

1. Check logs in Render/Vercel dashboard
2. Verify all environment variables are set
3. Test each component individually
4. Check MongoDB Atlas cluster status

---

## ✨ Success!

Your Pitch Insight application is now live! 🎉

- **Frontend**: https://pitch-insight.vercel.app
- **Backend API**: https://pitch-insight-backend.onrender.com
- **API Docs**: https://pitch-insight-backend.onrender.com/docs

Share your deployment and start analyzing cricket pitches! 🏏
