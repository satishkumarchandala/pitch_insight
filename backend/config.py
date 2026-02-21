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
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))  # 7 days

# Weather API Configuration
# Get your free API key from: https://www.weatherapi.com/signup.aspx
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

# Razorpay Configuration
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

# Gemini AI Configuration
# Get your free API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS Configuration
# Allow web frontend, mobile app (Expo), localhost for development
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"  # Default to allow all in development
).split(",") if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]

# For development, explicitly allow common localhost ports
if DEBUG or "*" in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "*",  # Allow all origins
    ]

# Subscription Plans
SUBSCRIPTION_PLANS = {
    "monthly": {
        "name": "Pro Monthly",
        "price": 199,  # in INR
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
CLASSIFIER_MODEL_PATH = "pitch_classifier.onnx"  # Corrected to match actual file
