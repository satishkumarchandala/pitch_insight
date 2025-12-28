# 🏏 Pitch Insight – AI Cricket Pitch Analyzer

## 📌 Project Overview
**Pitch Insight** is an AI-powered web application that analyzes **cricket pitch images** along with **real-time weather conditions** to generate an intelligent **pitch report**.
The system is designed using **open-source technologies** and follows a scalable, real-world ML + Web architecture.

---

## 🎯 Objective
To build an end-to-end system that:
- Accepts **cricket pitch images** captured from local grounds
- Fetches **location-based weather data**
- Analyzes pitch characteristics using **Computer Vision & ML**
- Generates a **predictive pitch report** useful for match analysis

---

## 🧠 Key Features

### 1. Pitch Image Analysis
- Detect pitch region from ground images
- Analyze:
  - Pitch color (dry/wet)
  - Cracks
  - Grass coverage
  - Surface roughness
- Technologies: **OpenCV, CNN, YOLO**

### 2. Weather Data Integration
- Fetch real-time weather:
  - Temperature
  - Humidity
  - Rainfall
- Correlate weather conditions with pitch behavior
- Uses **free weather APIs**

### 3. Pitch Report Generation
- Hybrid approach:
  - Rule-based logic (initial phase)
  - Machine Learning model (advanced phase)
- Output examples:
  - Batting-friendly
  - Bowling-friendly
  - Spin-support
  - Seam-support
  - Bounce & moisture estimation

### 4. User Authentication & History (NEW! 🆕)
- User signup and login with JWT authentication
- Password hashing with bcrypt
- MongoDB for user data storage
- Analysis history tracking for authenticated users
- Optional authentication (app works without login)
- Persistent sessions with token storage

---

## 🏗️ Tech Stack

### Frontend
- React.js
- Vite
- Axios for API calls
- Lucide React icons

### Backend
- Python + FastAPI
- ONNX Runtime for model inference
- JWT authentication (python-jose)
- Password hashing (bcrypt, passlib)

### Machine Learning
- Python
- OpenCV
- PyTorch / TensorFlow
- YOLO (for pitch detection)
- ONNX for optimized inference

### Database
- MongoDB (user authentication & analysis history)

### Authentication
- JWT (JSON Web Tokens)
- bcrypt password hashing
- Bearer token authentication

### Deployment
- Model Training: **Kaggle**
- Backend Hosting: **Render / Railway**
- Frontend Hosting: **Vercel / Netlify**
- Database: **MongoDB Atlas** (cloud) or local MongoDB
- Containerization (optional): **Docker**

---

## 🛠️ System Architecture (High-Level)

```
User Image Upload  
→ Frontend (React + Vite)  
→ Backend API (FastAPI)
→ Authentication Check (Optional - JWT)
→ Image Processing + ONNX Models  
→ Weather API Integration  
→ Pitch Analysis Engine  
→ Save to User History (if authenticated)
→ Pitch Report Output
Quick Start

### Prerequisites
1. **Python 3.8+** installed
2. **Node.js 16+** and npm installed
3. **MongoDB** installed and running locally (or MongoDB Atlas account)

### Setup Instructions

#### 1. Check MongoDB Installation
```bash
cd backend
./check_mongodb.bat  # Windows
./check_mongodb.sh   # Linux/Mac
```

#### 2. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

New packages for authentication:
- `pymongo` - MongoDB driver
- `bcrypt` - Password hashing
- `python-jose` - JWT tokens
- `passlib` - Password utilities

#### 3. Start Backend Server
```bash
python app.py
```

Backend runs on: http://localhost:8000

#### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
```

#### 5. Start Frontend Development Server
```bash
npm run dev
```

Frontend runs on: http://localhost:5173

### Authentication Setup

For detailed authentication setup, see: [AUTHENTICATION_SETUP.md](AUTHENTICATION_SETUP.md)

**Quick Test:**
1. Open http://localhost:5173
2. Click "Login" in the header
3. Sign up with email and password
4. Analyze a pitch image
5. View your analysis history

---

## 🚀 
Database: MongoDB
- User accounts (email, password hash, profile)
- Analysis history (user_id, pitch type, confidence, timestamp)
```

---

## 🚀 Development Roadmap

### Phase 1: Foundation
- Project setup
- Image upload API
- Weather API integration
- Basic pitch rules

### Phase 2: ML Integration
- Dataset collection
- Pitch region detection
- Feature extraction
- Model training on Kaggle

### Phase 3: Prediction Engine
- Combine image + weather features
- Pitch classification
- Confidence scoring

### Phase 4: Deployment & Optimization
- Deploy backend & frontend
- Performance tuning
- Accuracy improvements

---

## 📂 Dataset Strategy
- Capture real pitch images from local grounds
- Augment data (rotation, brightness, contrast)
- Label:
  - Grass level
  - Cracks
  - Moisture indicators
- Optional: Use synthetic or public sports datasets

---

## ⚠️ Constraints
- Only **free & open-source tools**
- No paid APIs
- Focus on real-world feasibility
- Beginner-friendly but production-oriented

---

## 🤝 Contribution
Contributions, suggestions, and improvements are welcome.
Feel free to open issues or submit pull requests.

---

## 📜 License
This project is open-source and intended for educational and research purposes.
