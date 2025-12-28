# 🎯 Quick Start Deployment Guide

## TL;DR - Deploy in 15 Minutes

This is the fastest path to deployment. For detailed instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

---

## Step 1: MongoDB Atlas (3 minutes)

1. Go to https://www.mongodb.com/cloud/atlas/register
2. Create free cluster → Choose AWS → Region closest to you
3. Create user: `pitch_user` / `[generate-password]`
4. Network: Add `0.0.0.0/0`
5. Copy connection string:
   ```
   mongodb+srv://pitch_user:PASSWORD@cluster.mongodb.net/pitch_insight
   ```

---

## Step 2: Prepare Code (2 minutes)

```bash
# Generate secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Note this key for later
```

**Required API Keys:**
- Gemini API: https://aistudio.google.com/app/apikey
- Razorpay: https://dashboard.razorpay.com/app/keys

---

## Step 3: Deploy Backend to Render (5 minutes)

1. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Ready for deployment"
   git branch -M main
   git remote add origin https://github.com/YOURUSERNAME/pitch-insight.git
   git push -u origin main
   ```

2. Go to https://render.com → Sign up → New Web Service

3. Configure:
   - **Repository**: Connect your GitHub repo
   - **Name**: pitch-insight-backend
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. Add Environment Variables:
   ```
   MONGODB_URL=mongodb+srv://pitch_user:PASSWORD@...
   SECRET_KEY=<your-generated-key>
   GEMINI_API_KEY=<your-gemini-key>
   RAZORPAY_KEY_ID=<your-razorpay-key>
   RAZORPAY_KEY_SECRET=<your-razorpay-secret>
   ALLOWED_ORIGINS=http://localhost:5173
   DEBUG=False
   ```

5. Click "Create Web Service"

6. **COPY YOUR BACKEND URL**: `https://pitch-insight-backend.onrender.com`

---

## Step 4: Deploy Frontend to Netlify (3 minutes)

1. Update `frontend/.env.production`:
   ```env
   VITE_API_URL=https://pitch-insight-backend.onrender.com
   ```

2. Build frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

3. Go to https://netlify.com → Sign up → Drag & drop `frontend/dist` folder

4. **COPY YOUR FRONTEND URL**: `https://your-app-name.netlify.app`

---

## Step 5: Update CORS (2 minutes)

1. Go back to Render dashboard
2. Find `ALLOWED_ORIGINS` variable
3. Update to: `https://your-app-name.netlify.app,http://localhost:5173`
4. Save → Redeploy

---

## Step 6: Test! 🎉

1. Open your frontend URL
2. Register a new account
3. Upload a pitch image
4. See the magic! ✨

---

## 🚨 Common Issues

### "Out of Memory" on Render
- ✅ Already optimized for 512MB
- Models auto-unload after 5 minutes of inactivity
- First request may be slow (model loading)

### "CORS Error"
- Check ALLOWED_ORIGINS includes your frontend URL
- No trailing slash in URLs
- Redeploy backend after changing

### "Connection to MongoDB failed"
- Check connection string has correct password
- Verify 0.0.0.0/0 in MongoDB network access
- Test connection locally first

### "Cold Start Delay"
- Free tier sleeps after 15 minutes
- First request takes 30-60 seconds
- Use a cron job to ping every 10 minutes (optional)

---

## 📊 What You Get (FREE)

- ✅ Backend API on Render (512MB RAM)
- ✅ Frontend on Netlify (100GB bandwidth)
- ✅ MongoDB Atlas (512MB storage)
- ✅ SSL/HTTPS automatic
- ✅ Automatic deployments from GitHub

**Total Cost: $0/month** 💰

---

## 🚀 Production URLs

After deployment, save these:

- **Frontend**: https://________.netlify.app
- **Backend**: https://________.onrender.com
- **API Docs**: https://________.onrender.com/docs
- **Admin**: MongoDB Atlas dashboard

---

## 🔒 Security Reminders

✅ All secrets in environment variables
✅ MongoDB has authentication
✅ CORS properly configured
✅ HTTPS enabled everywhere
✅ No API keys in code

---

## 🎯 Next Steps

1. **Test thoroughly** with real users
2. **Monitor** Render dashboard for memory usage
3. **Collect feedback** and iterate
4. **Consider upgrading** if you get traction:
   - Render: $7/month for always-on + 512MB
   - MongoDB: Still free until 512MB

---

## 📚 Full Documentation

- [Complete Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [Backend API Docs](backend/README.md)

---

## 🆘 Need Help?

**Check logs:**
- Render: Dashboard → Logs tab
- Browser: Developer Console (F12)
- Network: Developer Tools → Network tab

**Still stuck?** 
- Read the detailed [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Check Render documentation
- Verify all environment variables are set

---

**Good luck! Your app will be live in minutes! 🚀**
