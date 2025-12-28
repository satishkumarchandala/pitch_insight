"""
Configuration settings for Pitch Insight Backend
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "pitch_insight")

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production-123456789")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))  # 7 days

# Weather API Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "b2ad62736cbd4e52aaa133601252712")

# Razorpay Configuration
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_RwZJe3KOgTNbo6")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "dV0pnEvxDeLKU6mndB7aFeYv")

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDOnBpIiOt-eNSqlsO3HhlQdyo1sdoTf8A")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS Configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,https://pitch-insight-backend.onrender.com/,https://pitch-insight-frontend.vercel.app/"
).split(",")

# Subscription Plans
SUBSCRIPTION_PLANS = {
    "monthly": {
        "name": "Pro Monthly",
        "price": 499,  # in INR
        "currency": "INR",
        "duration": 30,  # days
        "features": [
            "Unlimited analyses",
            "Weather integration",
            "Detailed reports",
            "Priority support"
        ]
    },
    "yearly": {
        "name": "Pro Yearly",
        "price": 4999,  # in INR (save ~17%)
        "currency": "INR",
        "duration": 365,  # days
        "features": [
            "Unlimited analyses",
            "Weather integration",
            "Detailed reports",
            "Priority support",
            "Best value - Save 17%"
        ]
    }
}

# Model paths
YOLO_MODEL_PATH = "pitch_yolov8_best.onnx"
CLASSIFIER_MODEL_PATH = "best_pitch_classifier.onnx"
