# 🏏 Pitch Insight - Project Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Architecture](#architecture)
5. [How It Works](#how-it-works)
6. [Getting Started](#getting-started)
7. [Project Structure](#project-structure)
8. [API Endpoints](#api-endpoints)
9. [User Journey](#user-journey)
10. [Subscription System](#subscription-system)
11. [Deployment](#deployment)

---

## 🎯 Project Overview

**Pitch Insight** is an AI-powered cricket pitch analysis platform that helps teams, coaches, and cricket enthusiasts make data-driven decisions. The system analyzes pitch images using computer vision and machine learning to predict pitch behavior and provide match strategies.

### What Problem Does It Solve?
- **Manual pitch assessment** is subjective and prone to errors
- **Weather impact** on pitch conditions is often overlooked
- **Strategic planning** requires expert cricket knowledge
- **Real-time analysis** is difficult during match preparations

### Solution
Pitch Insight uses advanced AI models (YOLOv8 + ResNet18) to:
1. Detect and analyze cricket pitches from images
2. Extract 6+ pitch features (grass, cracks, moisture, etc.)
3. Predict pitch type with high accuracy
4. Generate weather-adjusted match strategies
5. Provide AI chatbot for cricket insights (Pro feature)

---

## ✨ Key Features

### Core Features
- 🖼️ **Pitch Detection**: YOLOv8-based pitch detection with bounding box
- 🔍 **Feature Extraction**: Analyzes grass coverage, cracks, moisture, color, texture, brightness
- 🎯 **Pitch Classification**: Predicts 4 types (Batting/Bowling/Spin/Seam Friendly)
- 🌤️ **Weather Integration**: Real-time weather impact on pitch behavior
- 📊 **Match Strategy**: Toss decision, batting/bowling/team composition tips
- 📈 **Confidence Scores**: Shows prediction confidence for all pitch types

### Premium Features (Pro Subscription)
- ⚡ **Complete Analysis**: Full feature extraction + weather integration
- 💬 **AI Chatbot**: Cricket expert assistant powered by Google Gemini
- 📜 **Analysis History**: Save and retrieve past analyses
- 🔒 **Unlimited Analyses**: No daily/monthly limits
- 📥 **Detailed Reports**: Export analysis as PDF/JSON

### User Features
- 🔐 **Authentication**: Secure JWT-based login/signup
- 💳 **Payment Integration**: Razorpay for subscriptions (Monthly/Yearly)
- 👤 **User Profile**: View subscription status, analysis history
- 🎨 **Dark Theme**: Modern, responsive UI with smooth animations
- 📱 **Mobile Responsive**: Works on all devices

---

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **AI/ML**: 
  - YOLOv8 (ONNX Runtime) - Pitch detection
  - ResNet18 (ONNX Runtime) - Pitch classification
  - OpenCV - Feature extraction
- **Database**: MongoDB (with PyMongo)
- **Authentication**: JWT tokens (python-jose)
- **Payment**: Razorpay API
- **AI Chat**: Google Gemini API
- **Weather**: WeatherAPI.com
- **Server**: Uvicorn ASGI

### Frontend
- **Framework**: React 18 (with Vite)
- **Styling**: Custom CSS with CSS Variables
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Build Tool**: Vite (ultra-fast HMR)

### DevOps & Deployment
- **Containerization**: Docker (Backend + Frontend)
- **Deployment**: 
  - Backend: Render.com / Railway
  - Frontend: Vercel
- **Environment**: Python virtual env (venv)
- **Version Control**: Git

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                         │
│  (React Frontend - Vercel)                                   │
│  - Upload pitch images                                       │
│  - View analysis results                                     │
│  - Chat with AI assistant                                    │
│  - Manage subscriptions                                      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS/REST API
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                         API LAYER                            │
│  (FastAPI Backend - Render)                                  │
│  Routes: /api/analyze, /api/auth, /api/chat, etc.          │
└───┬──────────┬──────────┬──────────┬────────────┬──────────┘
    │          │          │          │            │
    ↓          ↓          ↓          ↓            ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ YOLO   │ │Feature │ │ResNet18│ │Weather │ │ Gemini   │
│Detection│ │Extract │ │Classify│ │  API   │ │ AI Chat  │
│(ONNX)  │ │(OpenCV)│ │(ONNX)  │ │        │ │          │
└────────┘ └────────┘ └────────┘ └────────┘ └──────────┘
    │          │          │          │            │
    └──────────┴──────────┴──────────┴────────────┘
                     │
                     ↓
         ┌──────────────────────┐
         │   DATABASE LAYER     │
         │  (MongoDB Atlas)     │
         │  - Users collection  │
         │  - Analyses history  │
         └──────────────────────┘
```

### Data Flow

1. **Image Upload** → Frontend uploads image to `/api/analyze`
2. **Detection** → YOLOv8 detects pitch region
3. **Feature Extraction** → OpenCV analyzes grass, cracks, moisture, etc.
4. **Classification** → ResNet18 predicts pitch type
5. **Weather** → Fetches weather data (if location provided)
6. **Adjustment** → Combines ML + features + weather for final prediction
7. **Strategy** → Generates match tactics based on analysis
8. **Storage** → Saves to MongoDB (if user authenticated)
9. **Response** → Returns JSON with complete analysis

---

## 🔄 How It Works

### 1. Pitch Analysis Pipeline

```python
Image Upload (JPG/PNG)
    ↓
YOLO Detection (YOLOv8)
    ├─ Detects pitch region
    └─ Returns bounding box + confidence
    ↓
Feature Extraction (OpenCV)
    ├─ Grass Coverage: HSV color space analysis
    ├─ Crack Detection: Canny edge detection + contours
    ├─ Moisture Level: V-channel (brightness) analysis
    ├─ Color Profile: Dominant colors (K-means)
    ├─ Texture: Gray-level co-occurrence matrix
    └─ Brightness: Mean luminosity
    ↓
ML Classification (ResNet18)
    ├─ Input: 224x224 RGB image
    ├─ Output: 4 probabilities (batting/bowling/spin/seam)
    └─ Confidence: Max probability
    ↓
Feature-Based Adjustment
    ├─ Grass ≥ 20% → +15% bowling/seam
    ├─ Cracks ≥ 5 → +20% spin/bowling
    ├─ Moisture > 50% → +10% seam
    └─ Recalculates final prediction
    ↓
Weather Impact (if enabled)
    ├─ Temperature, humidity, wind, rainfall
    └─ Adjusts predictions based on conditions
    ↓
Strategy Generation
    ├─ Toss decision (bat/bowl first)
    ├─ Batting tips (3 strategies)
    ├─ Bowling tips (3 strategies)
    └─ Team composition (3 recommendations)
```

### 2. Authentication Flow

```
User Signup
    ↓
Password Hashing (bcrypt)
    ↓
Store in MongoDB (users collection)
    ↓
Generate JWT Token (7-day expiry)
    ↓
Return token + user data
    ↓
Frontend stores in localStorage
    ↓
Include token in Authorization header for all requests
```

### 3. Subscription Flow

```
User clicks "Upgrade to Pro"
    ↓
Frontend shows pricing modal
    ↓
User selects plan (Monthly ₹199 / Yearly ₹1999)
    ↓
Backend creates Razorpay order
    ↓
Frontend opens Razorpay checkout
    ↓
User completes payment
    ↓
Razorpay webhook → Backend verifies signature
    ↓
Update user's subscription in MongoDB
    ↓
Grant Pro features (complete analysis, chatbot, history)
```

### 4. AI Chatbot Flow

```
User asks cricket question
    ↓
Check authentication + Pro subscription
    ↓
Build context from latest analysis (if available)
    ↓
Send to Google Gemini API with system prompt
    ↓
Gemini generates cricket-focused response
    ↓
Return answer to user
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 16+
- MongoDB (local or Atlas)
- API Keys:
  - WeatherAPI.com (free)
  - Google Gemini AI (free)
  - Razorpay (for payments)

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Edit .env and add your API keys

# 5. Ensure ONNX models exist
# - pitch_yolov8_best.onnx
# - pitch_classifier.onnx

# 6. Start server
python app.py
# Backend runs at: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Create .env file (optional)
echo "VITE_API_URL=http://localhost:8000" > .env

# 4. Start dev server
npm run dev
# Frontend runs at: http://localhost:5173
```

### Environment Variables

**Backend (.env)**:
```env
# MongoDB
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=pitch_insight

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# APIs
WEATHER_API_KEY=your-weather-api-key
GEMINI_API_KEY=your-gemini-api-key
RAZORPAY_KEY_ID=your-razorpay-key
RAZORPAY_KEY_SECRET=your-razorpay-secret

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Frontend (.env)**:
```env
VITE_API_URL=http://localhost:8000
```

---

## 📁 Project Structure

```
pitch_insight/
├── backend/                    # FastAPI Backend
│   ├── app.py                 # Main FastAPI application
│   ├── config.py              # Configuration & environment variables
│   ├── database.py            # MongoDB connection & queries
│   ├── models.py              # Pydantic data models
│   ├── schemas.py             # API request/response schemas
│   ├── auth.py                # JWT authentication logic
│   ├── utils.py               # Helper functions
│   ├── complete_pipeline_onnx.py  # ONNX inference pipeline
│   ├── pitch_analyzer.py      # OpenCV feature extraction
│   ├── weather_forecast_analyzer.py  # Weather API integration
│   ├── razorpay_handler.py    # Payment processing
│   ├── requirements.txt       # Python dependencies
│   ├── pitch_yolov8_best.onnx # YOLOv8 model
│   ├── pitch_classifier.onnx  # ResNet18 model
│   └── routes/                # API route handlers
│       ├── analysis.py        # Pitch analysis endpoints
│       ├── auth.py            # Authentication endpoints
│       ├── chat.py            # AI chatbot endpoints
│       ├── subscription.py    # Payment & subscription
│       ├── weather.py         # Weather endpoints
│       └── health.py          # Health check
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── main.jsx           # Entry point
│   │   ├── components/        # Reusable components
│   │   │   ├── Header.jsx     # Navigation header
│   │   │   ├── Footer.jsx     # Footer
│   │   │   ├── Auth.jsx       # Login/Signup modal
│   │   │   ├── UploadSection.jsx     # Image upload
│   │   │   ├── ResultsSection.jsx    # Analysis results
│   │   │   ├── HistorySection.jsx    # Past analyses
│   │   │   ├── ChatWidget.jsx        # AI chatbot
│   │   │   ├── PaymentModal.jsx      # Razorpay checkout
│   │   │   ├── SubscriptionBadge.jsx # Pro badge
│   │   │   └── UpgradePrompt.jsx     # Upgrade modal
│   │   ├── pages/             # Page components
│   │   │   ├── Home.jsx       # Landing page
│   │   │   ├── Analysis.jsx   # Analysis page
│   │   │   ├── Profile.jsx    # User profile
│   │   │   ├── Pricing.jsx    # Subscription plans
│   │   │   └── Settings.jsx   # App settings
│   │   └── services/
│   │       └── api.js         # Axios API client
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite configuration
│   └── index.html             # HTML template
│
├── vevv/                       # Python virtual environment
├── render.yaml                 # Render deployment config
├── requirements.txt            # Root dependencies
└── runtime.txt                 # Python version
```

---

## 🌐 API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health status

### Authentication
- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info
- `GET /api/auth/subscription-status` - Get subscription details

### Analysis
- `POST /api/analyze` - Upload and analyze pitch image
  - **Body**: `multipart/form-data`
  - **Fields**: image, city, latitude, longitude, include_weather
  - **Response**: Complete analysis JSON
- `GET /api/analysis/history` - Get user's past analyses (Pro only)
- `GET /api/analysis/{analysis_id}` - Get specific analysis

### Chatbot
- `POST /api/chat` - Chat with AI assistant (Pro only)
  - **Body**: `{ "message": "question", "analysis_id": "optional" }`
  - **Response**: AI-generated answer

### Subscription
- `POST /api/subscription/create-order` - Create Razorpay order
  - **Body**: `{ "plan_type": "monthly" | "yearly" }`
- `POST /api/subscription/verify-payment` - Verify payment signature
- `GET /api/subscription/status` - Get subscription status

### Weather
- `GET /api/weather` - Get current weather
  - **Query**: city OR lat+lon
- `GET /api/weather/forecast` - Get 5-day forecast

---

## 👤 User Journey

### New User (Free Tier)
1. **Visit Landing Page** → See features and sample analyses
2. **Click "Get Started"** → Navigate to Analysis page
3. **Quick Analysis** → Upload image without login (limited features)
   - Shows only pitch type + confidence
   - No weather, no strategy, no history
4. **See Upgrade Prompts** → Encouraged to sign up for Pro
5. **Sign Up** → Create account (free tier)
6. **Limited Features** → 5 analyses per day, no complete analysis

### Pro User
1. **Login** → Access Pro features
2. **Complete Analysis** → Full feature extraction + weather
3. **View Results** → Detailed reports with strategy
4. **Ask AI Questions** → Chat with cricket expert bot
5. **Check History** → Review past analyses
6. **Manage Subscription** → View/cancel subscription in Profile

### Payment Flow
1. **Click "Upgrade to Pro"**
2. **Select Plan** (Monthly ₹199 / Yearly ₹1999)
3. **Razorpay Checkout** → Enter payment details
4. **Payment Success** → Instant Pro access
5. **Confirmation Email** → Receipt sent (if configured)

---

## 💳 Subscription System

### Plans
| Feature | Free | Pro Monthly | Pro Yearly |
|---------|------|-------------|------------|
| Quick Analysis | ✅ 5/day | ✅ Unlimited | ✅ Unlimited |
| Complete Analysis | ❌ | ✅ | ✅ |
| Weather Integration | ❌ | ✅ | ✅ |
| Match Strategy | ❌ | ✅ | ✅ |
| AI Chatbot | ❌ | ✅ | ✅ |
| Analysis History | ❌ | ✅ | ✅ |
| Export Reports | ❌ | ✅ | ✅ |
| Priority Support | ❌ | ✅ | ✅ |
| **Price** | Free | ₹199/month | ₹1999/year |

### Database Schema (MongoDB)

**users collection**:
```json
{
  "_id": "ObjectId",
  "username": "string",
  "email": "string",
  "full_name": "string",
  "password_hash": "string",
  "subscription_type": "free | pro",
  "subscription_status": "active | expired | cancelled",
  "subscription_start_date": "ISODate",
  "subscription_end_date": "ISODate",
  "created_at": "ISODate",
  "updated_at": "ISODate"
}
```

**analyses collection**:
```json
{
  "_id": "ObjectId",
  "analysis_id": "PITCH_20260103_142531",
  "user_id": "ObjectId",
  "image_hash": "string",
  "pitch_type": "batting_friendly",
  "confidence": 0.87,
  "features": { /* grass, cracks, moisture, etc. */ },
  "weather_data": { /* optional */ },
  "match_strategy": { /* optional */ },
  "created_at": "ISODate"
}
```

---

## 🚀 Deployment

### Backend (Render.com)

1. **Create Web Service** on Render
2. **Connect GitHub repo**
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `python backend/app.py`
5. **Environment Variables**: Add all from .env
6. **Deploy** → Get live URL

### Frontend (Vercel)

1. **Import GitHub repo** on Vercel
2. **Framework**: Vite
3. **Root Directory**: `frontend`
4. **Build Command**: `npm run build`
5. **Output Directory**: `dist`
6. **Environment Variable**: `VITE_API_URL=<backend-url>`
7. **Deploy** → Get live URL

### Docker Deployment

```bash
# Build backend
docker build -t pitch-insight-backend ./backend

# Build frontend
docker build -t pitch-insight-frontend ./frontend

# Run with docker-compose
docker-compose up -d
```

---

## 📊 Performance & Optimization

### Backend Optimizations
- **ONNX Runtime**: 3-5x faster than PyTorch
- **Image Caching**: SHA256 hash-based deduplication
- **Connection Pooling**: MongoDB connection reuse
- **Async/Await**: Non-blocking I/O operations
- **Memory Limit**: Optimized for 512MB RAM

### Frontend Optimizations
- **Code Splitting**: Vite automatic chunking
- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Client-side compression
- **CDN Delivery**: Vercel Edge Network
- **Caching**: LocalStorage for auth tokens

---

## 🔐 Security Features

1. **Password Hashing**: bcrypt with salt
2. **JWT Tokens**: Secure authentication
3. **CORS**: Restricted origins
4. **Rate Limiting**: Prevent API abuse (TODO)
5. **Input Validation**: Pydantic models
6. **SQL Injection**: None (using MongoDB)
7. **XSS Protection**: React auto-escaping
8. **Payment Security**: Razorpay signature verification

---

## 📈 Future Enhancements

- [ ] Video analysis (ball-by-ball pitch tracking)
- [ ] Multi-language support (Hindi, Tamil, etc.)
- [ ] Mobile app (React Native)
- [ ] Pitch comparison tool
- [ ] Historical match data integration
- [ ] Team performance analytics
- [ ] Live match predictions
- [ ] Social sharing of analyses
- [ ] Admin dashboard
- [ ] Webhook notifications

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is proprietary. All rights reserved.

---

## 📞 Support

For issues, questions, or feature requests:
- **Email**: support@pitchinsight.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/pitch-insight/issues)
- **Pro Users**: Priority email support

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **ResNet18**: PyTorch pretrained models
- **OpenCV**: Computer vision library
- **FastAPI**: Modern Python web framework
- **React**: UI library
- **MongoDB**: Database
- **Razorpay**: Payment gateway
- **Google Gemini**: AI chatbot

---

**Built with ❤️ for cricket enthusiasts worldwide 🏏**
