# 🔧 Pitch Insight - Technical Documentation

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [Backend Deep Dive](#backend-deep-dive)
3. [Frontend Deep Dive](#frontend-deep-dive)
4. [AI/ML Pipeline](#aiml-pipeline)
5. [Database Design](#database-design)
6. [API Reference](#api-reference)
7. [Authentication & Authorization](#authentication--authorization)
8. [Payment Integration](#payment-integration)
9. [Deployment Guide](#deployment-guide)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  React Frontend (Vite + React 18)                      │  │
│  │  - Component-based UI                                   │  │
│  │  - Client-side routing                                  │  │
│  │  - Axios HTTP client                                    │  │
│  │  - LocalStorage for persistence                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │ REST API (JSON over HTTPS)
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI Backend (Python 3.11+ with Uvicorn)          │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  API Routes (Modular Router System)              │ │  │
│  │  │  - Health Check                                   │ │  │
│  │  │  - Authentication                                 │ │  │
│  │  │  - Analysis                                       │ │  │
│  │  │  - Chat (AI)                                      │ │  │
│  │  │  - Subscription                                   │ │  │
│  │  │  - Weather                                        │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  Middleware                                       │ │  │
│  │  │  - CORS (Cross-Origin Resource Sharing)          │ │  │
│  │  │  - Exception Handling                            │ │  │
│  │  │  - Request Logging                               │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┴───────────┬────────────────────┐
          ↓                        ↓                    ↓
┌─────────────────┐    ┌──────────────────┐   ┌──────────────┐
│  AI/ML LAYER    │    │  EXTERNAL APIs   │   │ DATA LAYER   │
│                 │    │                  │   │              │
│ ┌─────────────┐ │    │ ┌──────────────┐│   │ ┌──────────┐ │
│ │ YOLOv8      │ │    │ │ WeatherAPI   ││   │ │ MongoDB  │ │
│ │ (ONNX)      │ │    │ └──────────────┘│   │ │ Atlas    │ │
│ └─────────────┘ │    │                  │   │ └──────────┘ │
│                 │    │ ┌──────────────┐│   │              │
│ ┌─────────────┐ │    │ │ Google       ││   │ Collections: │
│ │ OpenCV      │ │    │ │ Gemini AI    ││   │ - users      │
│ │ Features    │ │    │ └──────────────┘│   │ - analyses   │
│ └─────────────┘ │    │                  │   │              │
│                 │    │ ┌──────────────┐│   │              │
│ ┌─────────────┐ │    │ │ Razorpay     ││   │              │
│ │ ResNet18    │ │    │ │ Payment      ││   │              │
│ │ (ONNX)      │ │    │ └──────────────┘│   │              │
│ └─────────────┘ │    │                  │   │              │
└─────────────────┘    └──────────────────┘   └──────────────┘
```

### Technology Choices & Rationale

| Component | Technology | Why? |
|-----------|------------|------|
| **Backend** | FastAPI | Async support, automatic API docs, Pydantic validation |
| **Frontend** | React + Vite | Component reusability, fast HMR, modern tooling |
| **Database** | MongoDB | Flexible schema, JSON-native, easy scaling |
| **ML Runtime** | ONNX | Cross-platform, faster than PyTorch, low memory |
| **Auth** | JWT | Stateless, scalable, standard protocol |
| **Payment** | Razorpay | India-focused, good UX, webhook support |
| **AI Chat** | Gemini | Free tier, strong reasoning, cricket knowledge |
| **Server** | Uvicorn | ASGI server, async support, production-ready |

---

## 🐍 Backend Deep Dive

### Project Structure

```
backend/
├── app.py                          # FastAPI app initialization
├── config.py                       # Configuration management
├── database.py                     # MongoDB connection & queries
├── models.py                       # Pydantic data models
├── schemas.py                      # API request/response schemas
├── auth.py                         # JWT authentication
├── utils.py                        # Helper functions
├── complete_pipeline_onnx.py       # ML inference pipeline
├── pitch_analyzer.py               # Feature extraction
├── weather_forecast_analyzer.py    # Weather integration
├── razorpay_handler.py             # Payment processing
└── routes/
    ├── __init__.py
    ├── health.py                   # Health check endpoints
    ├── auth.py                     # Auth endpoints
    ├── analysis.py                 # Analysis endpoints
    ├── chat.py                     # Chatbot endpoints
    ├── subscription.py             # Payment endpoints
    └── weather.py                  # Weather endpoints
```

### Core Components

#### 1. FastAPI Application (`app.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown event handler"""
    # Startup: Initialize connections, load models
    print("🏏 Pitch Insight API - Starting Up")
    yield
    # Shutdown: Close connections
    print("🔄 Shutting down gracefully...")
    close_database_connection()

app = FastAPI(
    title="Pitch Insight API",
    description="AI-powered cricket pitch analysis",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
# ... more routers
```

**Key Features**:
- **Lifespan Events**: Graceful startup/shutdown
- **CORS**: Configured for frontend origins
- **Modular Routers**: Separate files for each domain
- **Auto-docs**: Swagger UI at `/docs`

#### 2. Configuration (`config.py`)

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables with defaults
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "pitch_insight")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Subscription plans
SUBSCRIPTION_PLANS = {
    "monthly": {"price": 199, "duration": 30},
    "yearly": {"price": 1999, "duration": 365}
}
```

**Best Practices**:
- ✅ Use `.env` for sensitive data
- ✅ Provide sensible defaults
- ✅ Type conversion (int, bool)
- ✅ Split comma-separated values

#### 3. Database Layer (`database.py`)

```python
from pymongo import MongoClient
from typing import Optional

# Global MongoDB client
mongodb_client: Optional[MongoClient] = None

def get_database():
    """Get database instance"""
    global mongodb_client
    if mongodb_client is None:
        mongodb_client = MongoClient(MONGODB_URL)
    return mongodb_client[DATABASE_NAME]

def get_users_collection():
    """Get users collection"""
    return get_database()["users"]

def get_analysis_collection():
    """Get analyses collection"""
    return get_database()["analyses"]

def close_database_connection():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client is not None:
        mongodb_client.close()
        mongodb_client = None
```

**Key Points**:
- **Lazy Loading**: Connection created on first use
- **Singleton Pattern**: Reuses same connection
- **Graceful Shutdown**: Closes connection on exit
- **Collection Helpers**: Easy access to collections

#### 4. Authentication (`auth.py`)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Fetch user from database
        user = get_users_collection().find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Security Features**:
- **bcrypt Hashing**: Industry-standard password hashing
- **JWT Tokens**: Stateless authentication
- **Token Expiry**: 7-day default (configurable)
- **Bearer Auth**: Standard HTTP Authorization header
- **Dependency Injection**: FastAPI's `Depends()`

#### 5. Analysis Pipeline (`complete_pipeline_onnx.py`)

```python
import onnxruntime as ort
import cv2
import numpy as np
from pitch_analyzer import PitchAnalyzer

class CompletePitchPipeline:
    def __init__(self, yolo_path, classifier_path, use_gpu=False):
        """Initialize ONNX models"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Load YOLOv8 ONNX model
        self.yolo_session = ort.InferenceSession(yolo_path, providers=providers)
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        
        # Load ResNet18 ONNX classifier
        self.classifier_session = ort.InferenceSession(classifier_path, providers=providers)
        self.classifier_input_name = self.classifier_session.get_inputs()[0].name
        
        # Initialize feature analyzer
        self.feature_analyzer = PitchAnalyzer()
        
        # Classes
        self.classes = ['batting_friendly', 'bowling_friendly', 'seam_friendly', 'spin_friendly']
    
    def detect_pitch(self, image_path):
        """Detect pitch using YOLOv8"""
        image = cv2.imread(image_path)
        preprocessed, scale, (pad_left, pad_top) = self.preprocess_yolo_image(image)
        
        # Run ONNX inference
        outputs = self.yolo_session.run(None, {self.yolo_input_name: preprocessed})
        
        # Parse detections
        detection = self.parse_yolo_output(outputs[0], image.shape, scale, (pad_left, pad_top))
        return detection
    
    def classify_pitch(self, image_path, bbox=None):
        """Classify pitch type using ResNet18"""
        image = cv2.imread(image_path)
        
        # Crop to bbox if provided
        if bbox:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
        
        # Preprocess for ResNet
        preprocessed = self.preprocess_classifier_image(image)
        
        # Run ONNX inference
        outputs = self.classifier_session.run(None, {self.classifier_input_name: preprocessed})
        
        # Softmax probabilities
        probs = self.softmax(outputs[0][0])
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pitch_type = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        return pitch_type, confidence, probs
    
    def analyze_complete(self, image_path, location=None):
        """Complete analysis pipeline"""
        # Step 1: Detect pitch
        detection = self.detect_pitch(image_path)
        
        # Step 2: Extract features
        features = self.feature_analyzer.analyze(image_path)
        
        # Step 3: Classify pitch
        bbox = detection.get('bbox') if detection['detected'] else None
        pitch_type, confidence, probs = self.classify_pitch(image_path, bbox)
        
        # Step 4: Feature-based adjustments
        adjusted_pitch, adjustments = self.apply_feature_adjustments(
            pitch_type, confidence, features
        )
        
        # Step 5: Weather integration (if location provided)
        weather_data = None
        if location:
            weather_data = fetch_weather(location)
            adjusted_pitch = apply_weather_adjustments(adjusted_pitch, weather_data)
        
        return {
            'detection': detection,
            'features': features,
            'pitch_type': adjusted_pitch,
            'confidence': confidence,
            'probabilities': probs,
            'adjustments': adjustments,
            'weather': weather_data
        }
```

**Pipeline Steps**:
1. **YOLO Detection**: Locates pitch region
2. **Feature Extraction**: OpenCV analysis
3. **ML Classification**: ResNet18 prediction
4. **Feature Adjustment**: Rule-based corrections
5. **Weather Impact**: External API integration

---

## ⚛️ Frontend Deep Dive

### Project Structure

```
frontend/src/
├── main.jsx                    # Entry point (ReactDOM.render)
├── App.jsx                     # Root component (router logic)
├── App.css                     # Global styles
├── components/                 # Reusable components
│   ├── Header.jsx / .css
│   ├── Footer.jsx / .css
│   ├── Auth.jsx / .css         # Login/Signup modal
│   ├── UploadSection.jsx / .css
│   ├── ResultsSection.jsx / .css
│   ├── HistorySection.jsx / .css
│   ├── ChatWidget.jsx / .css
│   ├── PaymentModal.jsx / .css
│   ├── SubscriptionBadge.jsx / .css
│   └── UpgradePrompt.jsx / .css
├── pages/                      # Page components
│   ├── Home.jsx / .css
│   ├── Analysis.jsx / .css
│   ├── Profile.jsx / .css
│   ├── Pricing.jsx / .css
│   └── Settings.jsx / .css
└── services/
    └── api.js                  # Axios API client
```

### Key Components

#### 1. App Component (`App.jsx`)

```jsx
import React, { useState, useEffect } from 'react'
import { authAPI } from './services/api'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [theme, setTheme] = useState('light')
  
  // Check localStorage for auth on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    const storedUser = localStorage.getItem('user')
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
  }, [])
  
  const handleLogin = (userData, authToken) => {
    setUser(userData)
    setToken(authToken)
    localStorage.setItem('token', authToken)
    localStorage.setItem('user', JSON.stringify(userData))
  }
  
  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
    setToken(null)
  }
  
  const renderPage = () => {
    switch (currentPage) {
      case 'home': return <Home />
      case 'analysis': return <Analysis token={token} user={user} />
      case 'profile': return <Profile user={user} onLogout={handleLogout} />
      // ... more pages
    }
  }
  
  return (
    <div className="app">
      <Header user={user} onNavigate={setCurrentPage} />
      {renderPage()}
      <Footer />
    </div>
  )
}
```

**State Management**:
- **Local State**: React useState for UI state
- **Persistence**: localStorage for auth tokens
- **Props Drilling**: Pass user/token to children
- **Navigation**: Simple page switching (no React Router needed)

#### 2. API Client (`services/api.js`)

```javascript
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with base config
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Add auth token to requests
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle errors globally
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      // Unauthorized - clear auth and redirect
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

export const authAPI = {
  signup: (data) => apiClient.post('/api/auth/signup', data),
  login: (data) => apiClient.post('/api/auth/login', data),
  getMe: () => apiClient.get('/api/auth/me'),
}

export const analysisAPI = {
  analyze: (formData) => apiClient.post('/api/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  getHistory: () => apiClient.get('/api/analysis/history'),
}

export const chatAPI = {
  sendMessage: (message, analysisId) => apiClient.post('/api/chat', {
    message,
    analysis_id: analysisId
  })
}

export const subscriptionAPI = {
  createOrder: (planType) => apiClient.post('/api/subscription/create-order', {
    plan_type: planType
  }),
  verifyPayment: (paymentData) => apiClient.post('/api/subscription/verify-payment', paymentData),
  getStatus: () => apiClient.get('/api/subscription/status')
}
```

**Features**:
- **Axios Instance**: Centralized configuration
- **Interceptors**: Auto-attach auth token
- **Error Handling**: Global 401 handling
- **Modular APIs**: Grouped by domain

#### 3. Upload Component (`components/UploadSection.jsx`)

```jsx
function UploadSection({ onAnalyze }) {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [city, setCity] = useState('')
  const [includeWeather, setIncludeWeather] = useState(false)
  
  const handleImageSelect = (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }
    
    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      alert('File size must be less than 5MB')
      return
    }
    
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }
  
  const handleSubmit = async () => {
    const formData = new FormData()
    formData.append('image', image)
    formData.append('city', city)
    formData.append('include_weather', includeWeather)
    
    await onAnalyze(formData)
  }
  
  return (
    <div className="upload-section">
      <div className="drag-drop-area" onClick={() => fileInput.click()}>
        {preview ? (
          <img src={preview} alt="Preview" />
        ) : (
          <p>Click to upload or drag and drop</p>
        )}
      </div>
      
      <input 
        ref={fileInput}
        type="file" 
        accept="image/*"
        onChange={handleImageSelect}
        style={{ display: 'none' }}
      />
      
      <div className="weather-options">
        <label>
          <input 
            type="checkbox" 
            checked={includeWeather}
            onChange={(e) => setIncludeWeather(e.target.checked)}
          />
          Include weather analysis
        </label>
        
        {includeWeather && (
          <input 
            type="text"
            placeholder="Enter city name"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
        )}
      </div>
      
      <button onClick={handleSubmit} disabled={!image}>
        Analyze Pitch
      </button>
    </div>
  )
}
```

**Features**:
- **Drag & Drop**: File input with custom UI
- **Image Preview**: URL.createObjectURL()
- **Validation**: File type & size checks
- **Conditional Inputs**: Weather options toggle

---

## 🤖 AI/ML Pipeline

### 1. YOLO Pitch Detection

**Model**: YOLOv8 (Ultralytics)
**Format**: ONNX (converted from .pt)
**Input**: 640x640 RGB image
**Output**: Bounding boxes + confidence scores

```python
def detect_pitch(image):
    # Preprocess
    preprocessed = letterbox_resize(image, (640, 640))
    preprocessed = preprocessed.transpose(2, 0, 1)  # HWC -> CHW
    preprocessed = preprocessed / 255.0  # Normalize
    preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dim
    
    # Inference
    outputs = yolo_session.run(None, {input_name: preprocessed})
    
    # Post-process (NMS, scaling)
    boxes, scores, classes = parse_yolo_output(outputs)
    
    return {
        'detected': len(boxes) > 0,
        'bbox': boxes[0] if boxes else None,
        'confidence': scores[0] if scores else 0
    }
```

### 2. Feature Extraction (OpenCV)

**Grass Coverage**:
```python
def detect_grass(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate percentage
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = cv2.countNonZero(mask)
    percentage = (green_pixels / total_pixels) * 100
    
    return {
        'percentage': round(percentage, 2),
        'level': 'High' if percentage > 20 else 'Medium' if percentage > 5 else 'Minimal'
    }
```

**Crack Detection**:
```python
def detect_cracks(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter cracks (length > 30px, aspect ratio > 3:1)
    cracks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        aspect_ratio = max(w, h) / (min(w, h) + 1)
        
        if length > 30 and aspect_ratio > 3:
            cracks.append(contour)
    
    return {
        'num_cracks': len(cracks),
        'severity': 'High' if len(cracks) > 10 else 'Medium' if len(cracks) > 5 else 'Low'
    }
```

**Moisture Analysis**:
```python
def analyze_moisture(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # V-channel (brightness)
    v_channel = hsv[:, :, 2]
    
    # Calculate mean brightness
    mean_brightness = np.mean(v_channel)
    
    # Lower brightness = Higher moisture
    moisture_score = 100 - (mean_brightness / 255 * 100)
    
    return {
        'score': round(moisture_score, 2),
        'level': 'Wet' if moisture_score > 60 else 'Damp' if moisture_score > 30 else 'Dry'
    }
```

### 3. ML Classification (ResNet18)

**Model**: ResNet18 (PyTorch pretrained, fine-tuned on cricket pitches)
**Format**: ONNX
**Input**: 224x224 RGB image (ImageNet normalized)
**Output**: 4 class probabilities

```python
def classify_pitch(image):
    # Preprocess
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    
    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_float - mean) / std
    
    # Transpose and add batch dim
    image_tensor = image_normalized.transpose(2, 0, 1)
    image_batch = np.expand_dims(image_tensor, axis=0)
    
    # Inference
    outputs = classifier_session.run(None, {input_name: image_batch})
    
    # Softmax
    probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
    
    # Get prediction
    classes = ['batting_friendly', 'bowling_friendly', 'seam_friendly', 'spin_friendly']
    pred_idx = np.argmax(probs)
    
    return {
        'pitch_type': classes[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probabilities': {classes[i]: float(probs[i]) for i in range(4)}
    }
```

### 4. Feature-Based Adjustment

```python
def apply_feature_adjustments(pitch_type, confidence, features):
    adjustments = []
    probs = {'batting_friendly': 0.25, 'bowling_friendly': 0.25, 'seam_friendly': 0.25, 'spin_friendly': 0.25}
    
    # Grass coverage adjustment
    grass_pct = features['grass_coverage']['percentage']
    if grass_pct >= 20:
        probs['bowling_friendly'] += 0.15
        probs['seam_friendly'] += 0.15
        adjustments.append(f"High grass coverage ({grass_pct}%) increases bowling/seam advantage")
    
    # Crack adjustment
    num_cracks = features['crack_analysis']['num_cracks']
    if num_cracks >= 5:
        probs['spin_friendly'] += 0.20
        probs['bowling_friendly'] += 0.10
        adjustments.append(f"Multiple cracks ({num_cracks}) favor spin bowlers")
    
    # Moisture adjustment
    moisture_score = features['moisture_level']['score']
    if moisture_score > 50:
        probs['seam_friendly'] += 0.10
        adjustments.append(f"High moisture ({moisture_score}%) assists seam movement")
    
    # Normalize probabilities
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}
    
    # Get final prediction
    final_pitch_type = max(probs, key=probs.get)
    final_confidence = probs[final_pitch_type]
    
    return final_pitch_type, final_confidence, probs, adjustments
```

---

## 💾 Database Design

### MongoDB Collections

#### users Collection

```json
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "password_hash": "$2b$12$...",
  "subscription_type": "pro",  // "free" | "pro"
  "subscription_status": "active",  // "active" | "expired" | "cancelled"
  "subscription_start_date": ISODate("2026-01-01T00:00:00Z"),
  "subscription_end_date": ISODate("2027-01-01T00:00:00Z"),
  "razorpay_subscription_id": "sub_...",
  "created_at": ISODate("2025-12-15T10:30:00Z"),
  "updated_at": ISODate("2026-01-03T14:25:00Z")
}
```

**Indexes**:
```javascript
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "username": 1 }, { unique: true })
db.users.createIndex({ "subscription_end_date": 1 })
```

#### analyses Collection

```json
{
  "_id": ObjectId("..."),
  "analysis_id": "PITCH_20260103_142531",
  "user_id": ObjectId("..."),  // null for anonymous
  "image_hash": "sha256:abc123...",
  
  "pitch_detection": {
    "detected": true,
    "confidence": 0.93,
    "bbox": [196, 270, 440, 478]
  },
  
  "features": {
    "grass_coverage": {
      "percentage": 5.0,
      "level": "Minimal"
    },
    "crack_analysis": {
      "num_cracks": 12,
      "severity": "Medium"
    },
    "moisture_level": {
      "score": 45,
      "level": "Damp"
    },
    "color_profile": {
      "dominant_colors": [[180, 140, 100], [160, 120, 80]],
      "brown_percentage": 65
    },
    "texture_analysis": {
      "type": "Rough",
      "score": 72
    },
    "brightness": {
      "level": "Medium",
      "value": 128
    }
  },
  
  "ml_prediction": {
    "pitch_type": "spin_friendly",
    "confidence": 0.68,
    "probabilities": {
      "batting_friendly": 0.15,
      "bowling_friendly": 0.25,
      "seam_friendly": 0.12,
      "spin_friendly": 0.68
    }
  },
  
  "final_prediction": {
    "pitch_type": "spin_friendly",
    "confidence": 0.75,
    "adjustments": ["Multiple cracks favor spin bowlers", "..."]
  },
  
  "weather_data": {
    "city": "Mumbai",
    "temperature": 28,
    "humidity": 75,
    "wind_speed": 12,
    "rainfall": 0,
    "conditions": "Partly Cloudy",
    "impact_score": 7,
    "impact_description": "High humidity may assist swing bowling"
  },
  
  "match_strategy": {
    "toss_decision": "🏏 Bat First",
    "batting_strategy": ["...", "...", "..."],
    "bowling_strategy": ["...", "...", "..."],
    "team_composition": ["...", "...", "..."],
    "key_factors": ["...", "..."]
  },
  
  "created_at": ISODate("2026-01-03T14:25:31Z")
}
```

**Indexes**:
```javascript
db.analyses.createIndex({ "analysis_id": 1 }, { unique: true })
db.analyses.createIndex({ "user_id": 1, "created_at": -1 })
db.analyses.createIndex({ "image_hash": 1 })
db.analyses.createIndex({ "created_at": -1 })
```

**Queries**:
```python
# Get user's analysis history (paginated)
analyses = db.analyses.find(
    {"user_id": ObjectId(user_id)},
    projection={"image_hash": 0}  # Exclude large fields
).sort("created_at", -1).limit(20)

# Get analysis by ID
analysis = db.analyses.find_one({"analysis_id": analysis_id})

# Check duplicate image (by hash)
existing = db.analyses.find_one({"image_hash": hash_value})
```

---

## 🔐 Authentication & Authorization

### JWT Token Flow

```
┌─────────┐                                      ┌─────────┐
│ Client  │                                      │ Server  │
└────┬────┘                                      └────┬────┘
     │                                                │
     │  POST /api/auth/login                         │
     │  { email, password } ────────────────────────>│
     │                                                │
     │                                          ┌─────▼─────┐
     │                                          │ Verify    │
     │                                          │ Password  │
     │                                          └─────┬─────┘
     │                                                │
     │                                          ┌─────▼─────┐
     │                                          │ Generate  │
     │                                          │ JWT Token │
     │                                          └─────┬─────┘
     │                                                │
     │  { token: "eyJ...", user: {...} } <───────────│
     │<──────────────────────────────────────────────│
     │                                                │
     │  Store in localStorage                        │
     │                                                │
     │  GET /api/analysis/history                    │
     │  Authorization: Bearer eyJ... ────────────────>│
     │                                                │
     │                                          ┌─────▼─────┐
     │                                          │ Verify    │
     │                                          │ JWT Token │
     │                                          └─────┬─────┘
     │                                                │
     │                                          ┌─────▼─────┐
     │                                          │ Get User  │
     │                                          │ From DB   │
     │                                          └─────┬─────┘
     │                                                │
     │  { analyses: [...] } <─────────────────────────│
     │<──────────────────────────────────────────────│
```

### Token Structure

```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "user_id": "64abc123...",
    "email": "john@example.com",
    "exp": 1735910400  // Unix timestamp (7 days from issue)
  },
  "signature": "..." // HMAC SHA256(header + payload, SECRET_KEY)
}
```

### Authorization Levels

| Endpoint | Authentication | Subscription |
|----------|---------------|--------------|
| `POST /api/auth/signup` | ❌ None | - |
| `POST /api/auth/login` | ❌ None | - |
| `GET /api/auth/me` | ✅ Required | Any |
| `POST /api/analyze` (quick) | ⚠️ Optional | Free/Pro |
| `POST /api/analyze` (complete) | ✅ Required | Pro Only |
| `GET /api/analysis/history` | ✅ Required | Pro Only |
| `POST /api/chat` | ✅ Required | Pro Only |
| `POST /api/subscription/create-order` | ✅ Required | Any |

---

## 💳 Payment Integration

### Razorpay Flow

```
┌────────┐              ┌──────────┐              ┌───────────┐
│ Client │              │  Backend │              │ Razorpay  │
└───┬────┘              └────┬─────┘              └─────┬─────┘
    │                        │                          │
    │  1. Create Order       │                          │
    │  POST /api/subscription│                          │
    │     /create-order      │                          │
    │  { plan_type }         │                          │
    │───────────────────────>│                          │
    │                        │  2. Create Order         │
    │                        │  POST /v1/orders         │
    │                        │  { amount, currency }    │
    │                        │─────────────────────────>│
    │                        │                          │
    │                        │  3. Order Created        │
    │                        │  { id, amount, ... }     │
    │                        │<─────────────────────────│
    │                        │                          │
    │  4. Order Details      │                          │
    │  { order_id, amount }  │                          │
    │<───────────────────────│                          │
    │                        │                          │
    │  5. Open Checkout      │                          │
    │  (Razorpay.js)         │                          │
    │───────────────────────────────────────────────────>│
    │                        │                          │
    │  6. Payment Success    │                          │
    │  { payment_id,         │                          │
    │    order_id,           │                          │
    │    signature }         │                          │
    │<───────────────────────────────────────────────────│
    │                        │                          │
    │  7. Verify Payment     │                          │
    │  POST /api/subscription│                          │
    │     /verify-payment    │                          │
    │  { payment_id, ... }   │                          │
    │───────────────────────>│                          │
    │                        │  8. Verify Signature     │
    │                        │  HMAC(order_id|          │
    │                        │       payment_id,        │
    │                        │       secret)            │
    │                        │─────────────────────────>│
    │                        │                          │
    │                        │  9. Signature Valid      │
    │                        │<─────────────────────────│
    │                        │                          │
    │                        │  10. Update User         │
    │                        │  subscription_type=pro   │
    │                        │  (MongoDB)               │
    │                        │                          │
    │  11. Success           │                          │
    │  { subscription: {} }  │                          │
    │<───────────────────────│                          │
```

### Backend Implementation

```python
# Create Razorpay order
@router.post("/create-order")
async def create_order(plan_type: str, current_user: dict = Depends(get_current_user)):
    amount = get_plan_amount(plan_type)  # in paise
    
    order_data = {
        "amount": amount,
        "currency": "INR",
        "payment_capture": 1
    }
    
    order = razorpay_client.order.create(data=order_data)
    
    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"]
    }

# Verify payment
@router.post("/verify-payment")
async def verify_payment(
    payment_id: str,
    order_id: str,
    signature: str,
    plan_type: str,
    current_user: dict = Depends(get_current_user)
):
    # Verify signature
    generated_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256
    ).hexdigest()
    
    if generated_signature != signature:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Update user subscription
    subscription_end = datetime.utcnow() + timedelta(days=get_plan_duration(plan_type))
    
    get_users_collection().update_one(
        {"_id": current_user["_id"]},
        {
            "$set": {
                "subscription_type": "pro",
                "subscription_status": "active",
                "subscription_start_date": datetime.utcnow(),
                "subscription_end_date": subscription_end,
                "razorpay_payment_id": payment_id,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    return {"success": True, "subscription_end": subscription_end}
```

### Frontend Implementation

```javascript
// Load Razorpay script
const loadRazorpay = () => {
  return new Promise((resolve) => {
    const script = document.createElement('script')
    script.src = 'https://checkout.razorpay.com/v1/checkout.js'
    script.onload = () => resolve(true)
    script.onerror = () => resolve(false)
    document.body.appendChild(script)
  })
}

// Handle payment
const handlePayment = async (planType) => {
  // Load Razorpay
  const loaded = await loadRazorpay()
  if (!loaded) {
    alert('Failed to load Razorpay. Please try again.')
    return
  }
  
  try {
    // Create order
    const { data } = await subscriptionAPI.createOrder(planType)
    
    // Razorpay options
    const options = {
      key: 'rzp_test_...',  // Razorpay key
      amount: data.amount,
      currency: data.currency,
      order_id: data.order_id,
      name: 'Pitch Insight',
      description: 'Pro Subscription',
      handler: async (response) => {
        // Payment successful
        try {
          await subscriptionAPI.verifyPayment({
            payment_id: response.razorpay_payment_id,
            order_id: response.razorpay_order_id,
            signature: response.razorpay_signature,
            plan_type: planType
          })
          
          alert('Payment successful! You are now a Pro member.')
          window.location.reload()
        } catch (error) {
          alert('Payment verification failed. Please contact support.')
        }
      },
      prefill: {
        name: user.full_name,
        email: user.email
      },
      theme: {
        color: '#10b981'
      }
    }
    
    // Open checkout
    const razorpay = new window.Razorpay(options)
    razorpay.open()
    
  } catch (error) {
    console.error('Payment error:', error)
    alert('Failed to initiate payment. Please try again.')
  }
}
```

---

## 🚀 Deployment Guide

### Backend (Render.com)

1. **Create Web Service**:
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect GitHub repository

2. **Configure Service**:
   - **Name**: pitch-insight-backend
   - **Environment**: Python 3
   - **Region**: Singapore (closest to India)
   - **Branch**: main
   - **Root Directory**: backend
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`

3. **Environment Variables**:
   ```env
   MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
   DATABASE_NAME=pitch_insight
   SECRET_KEY=your-production-secret-key
   WEATHER_API_KEY=your-weather-api-key
   GEMINI_API_KEY=your-gemini-api-key
   RAZORPAY_KEY_ID=your-razorpay-key-id
   RAZORPAY_KEY_SECRET=your-razorpay-secret
   HOST=0.0.0.0
   PORT=8000
   DEBUG=False
   ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app
   ```

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Get your backend URL: `https://pitch-insight-backend.onrender.com`

### Frontend (Vercel)

1. **Import Project**:
   - Go to Vercel Dashboard
   - Click "Add New..." → "Project"
   - Import from GitHub

2. **Configure Project**:
   - **Framework Preset**: Vite
   - **Root Directory**: frontend
   - **Build Command**: `npm run build`
   - **Output Directory**: dist
   - **Install Command**: `npm install`

3. **Environment Variables**:
   ```env
   VITE_API_URL=https://pitch-insight-backend.onrender.com
   ```

4. **Deploy**:
   - Click "Deploy"
   - Wait for build (2-3 minutes)
   - Get your frontend URL: `https://pitch-insight.vercel.app`

5. **Custom Domain** (Optional):
   - Go to Project Settings → Domains
   - Add your custom domain
   - Configure DNS records

### Database (MongoDB Atlas)

1. **Create Cluster**:
   - Sign up at mongodb.com
   - Create free M0 cluster
   - Choose region (Mumbai for India)

2. **Configure Access**:
   - Database Access → Add Database User
   - Network Access → Add IP Address (0.0.0.0/0 for Render)

3. **Get Connection String**:
   - Click "Connect" → "Connect your application"
   - Copy connection string
   - Add to backend environment variables

### Docker Deployment (Alternative)

```bash
# Build images
docker build -t pitch-insight-backend ./backend
docker build -t pitch-insight-frontend ./frontend

# Run containers
docker run -d -p 8000:8000 \
  -e MONGODB_URL="mongodb://..." \
  -e SECRET_KEY="..." \
  pitch-insight-backend

docker run -d -p 3000:80 \
  -e VITE_API_URL="http://backend:8000" \
  pitch-insight-frontend
```

---

## 🧪 Testing

### Backend Tests

```python
# tests/test_analysis.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_analyze_endpoint():
    # Test image upload
    with open("test_pitch.jpg", "rb") as f:
        response = client.post(
            "/api/analyze",
            files={"image": ("test.jpg", f, "image/jpeg")},
            data={"city": "Mumbai", "include_weather": "false"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "pitch_type" in data
    assert "confidence" in data

def test_auth_signup():
    response = client.post(
        "/api/auth/signup",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "user" in data

# Run tests
# pytest tests/ -v
```

### Frontend Tests (Example with Vitest)

```javascript
// tests/Auth.test.jsx
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Auth from '../src/components/Auth'

describe('Auth Component', () => {
  it('renders login form by default', () => {
    render(<Auth />)
    expect(screen.getByText('Login')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Email')).toBeInTheDocument()
  })
  
  it('switches to signup form', () => {
    render(<Auth />)
    const signupButton = screen.getByText('Sign Up')
    fireEvent.click(signupButton)
    expect(screen.getByPlaceholderText('Username')).toBeInTheDocument()
  })
})

// Run tests
// npm run test
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **ONNX model not found** | Ensure `.onnx` files are in backend root directory |
| **MongoDB connection failed** | Check MONGODB_URL and network access in Atlas |
| **401 Unauthorized** | Token expired or invalid. Clear localStorage and re-login |
| **CORS error** | Add frontend URL to ALLOWED_ORIGINS in backend .env |
| **Weather API not working** | Verify WEATHER_API_KEY and check API quota |
| **Payment not verifying** | Check Razorpay webhook signature calculation |
| **High memory usage** | Use ONNX models (not PyTorch) for lower memory |
| **Slow inference** | Consider GPU provider in ONNX Runtime |

### Debug Mode

Enable debug logging:

```python
# config.py
DEBUG = True

# app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

```javascript
// api.js
axios.interceptors.request.use(config => {
  console.log('API Request:', config)
  return config
})
```

---

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **MongoDB Docs**: https://www.mongodb.com/docs/
- **ONNX Runtime**: https://onnxruntime.ai/
- **Razorpay Docs**: https://razorpay.com/docs/
- **Vercel Docs**: https://vercel.com/docs
- **Render Docs**: https://render.com/docs

---

**Last Updated**: January 3, 2026  
**Version**: 2.0.0
