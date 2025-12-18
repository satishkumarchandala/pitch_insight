# 🚀 Quick Deployment Checklist

Use this checklist to deploy Pitch Insight in under 1 hour!

## ☑️ Pre-Deployment

- [ ] Create GitHub account
- [ ] Create Render account  
- [ ] Create Vercel account
- [ ] Create MongoDB Atlas account
- [ ] Get OpenWeatherMap API key
- [ ] Have model files ready (`pitch_yolov8_best.pt`, `best_pitch_classifier.pth`)

## ☑️ Step 1: Git & GitHub (5 min)

- [ ] `git init`
- [ ] `git add .`
- [ ] `git commit -m "Initial commit"`
- [ ] Create GitHub repository
- [ ] `git remote add origin https://github.com/YOUR_USERNAME/pitch-insight.git`
- [ ] `git push -u origin main`

## ☑️ Step 2: MongoDB Atlas (10 min)

- [ ] Create free M0 cluster
- [ ] Create database user (save username & password!)
- [ ] Allow access from anywhere (0.0.0.0/0)
- [ ] Copy connection string
- [ ] Replace `<password>` in connection string
- [ ] Save connection string for later

## ☑️ Step 3: Backend on Render (20 min)

- [ ] Create new Web Service
- [ ] Connect GitHub repo
- [ ] Set Root Directory: `backend`
- [ ] Build: `pip install -r requirements.txt`
- [ ] Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] Add environment variables:
  - [ ] `PYTHON_VERSION` = `3.12.0`
  - [ ] `WEATHER_API_KEY` = `your_key`
  - [ ] `MONGODB_URI` = `mongodb+srv://...`
  - [ ] `MONGODB_DB_NAME` = `pitch_insight`
  - [ ] `ENVIRONMENT` = `production`
  - [ ] `ENABLE_DATABASE` = `true`
  - [ ] `ENABLE_WEATHER` = `true`
- [ ] Deploy
- [ ] Upload model files via Shell or GitHub releases
- [ ] Save backend URL
- [ ] Test: Visit `/api/health` endpoint

## ☑️ Step 4: Frontend on Vercel (10 min)

- [ ] Create `.env.production` in `frontend/`:
  ```
  VITE_API_URL=https://your-backend.onrender.com
  ```
- [ ] Commit and push
- [ ] Create new Vercel project
- [ ] Connect GitHub repo
- [ ] Set Root Directory: `frontend`
- [ ] Framework: Vite
- [ ] Add environment variable: `VITE_API_URL`
- [ ] Deploy
- [ ] Save frontend URL

## ☑️ Step 5: Connect Everything (5 min)

- [ ] Update Render backend env:
  - [ ] `FRONTEND_URL` = `https://your-app.vercel.app`
- [ ] Redeploy backend

## ☑️ Step 6: Test Everything (10 min)

- [ ] Visit frontend URL
- [ ] Upload test image
- [ ] Verify analysis works
- [ ] Check database: Visit `/api/stats`
- [ ] Test weather integration
- [ ] Check API docs work

## 🎉 Done!

Total time: ~60 minutes

Your app is live at:
- Frontend: `https://pitch-insight.vercel.app`
- Backend: `https://pitch-insight-backend.onrender.com`
- Docs: `https://pitch-insight-backend.onrender.com/docs`

---

## 🆘 Quick Troubleshooting

**Backend not starting?**
- Check logs in Render dashboard
- Verify all environment variables are set
- Ensure model files are uploaded

**Frontend can't connect?**
- Check `VITE_API_URL` is correct
- Verify CORS is configured
- Check browser console

**Database errors?**
- Test MongoDB connection string
- Verify IP whitelist (0.0.0.0/0)
- Check user credentials

---

## 📞 Need Help?

See full guide: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
