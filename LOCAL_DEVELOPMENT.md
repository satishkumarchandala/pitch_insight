# Local Development Setup

## Quick Start

### 1. Activate Virtual Environment
```powershell
# Windows PowerShell
.\vevv\Scripts\Activate.ps1
```

### 2. Start Backend (Terminal 1)
```powershell
cd backend
python app.py
```
Backend will run at: **http://localhost:8000**

### 3. Start Frontend (Terminal 2)
```powershell
cd frontend
npm run dev
```
Frontend will run at: **http://localhost:5173**

## Production URLs
- **Backend**: https://pitch-insight-backend.onrender.com
- **Frontend**: https://pitch-insight-frontend.vercel.app

## Environment Configuration

### Backend
- **Local**: Uses `backend/.env` (points to localhost)
- **Production**: Uses `backend/.env.production` (configured on Render)

### Frontend
- **Local**: Uses `frontend/.env.development` → `http://localhost:8000`
- **Production**: Uses `frontend/.env.production` → `https://pitch-insight-backend.onrender.com`

## CORS Configuration
Backend CORS is configured to accept requests from:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://localhost:5174`
- `https://pitch-insight-frontend.vercel.app` (production)

## Recent Updates (Latest Commit)
✅ Removed 56 unnecessary files (codebase cleanup)
✅ Fixed color data display issue
✅ Implemented match strategy generation
✅ Fixed weather API integration
✅ Fixed AI chat feature
✅ Fixed signup validation
✅ Updated production configurations
✅ Verified ONNX Runtime usage (no PyTorch in runtime)

## Key Features Working
- ✅ Pitch Analysis (ONNX models)
- ✅ Weather Integration (WeatherAPI)
- ✅ AI Chatbot (Gemini)
- ✅ Authentication (JWT)
- ✅ Match Strategy Generation
- ✅ Subscription Management (Razorpay)

## Deployment
When ready to deploy changes:
1. Make your changes locally
2. Test thoroughly
3. Commit: `git add -A && git commit -m "your message"`
4. Push: `git push origin main`
5. Render will auto-deploy backend
6. Vercel will auto-deploy frontend

## Important Notes
- Backend uses ONNX Runtime exclusively (no PyTorch)
- ONNX models are ~200MB (ensure deployment supports large files)
- API keys are in `.env` files (never commit actual keys)
- MongoDB connection needed for production
