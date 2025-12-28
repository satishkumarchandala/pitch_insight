"""
Data Models for Pitch Insight
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# ============================================
# User Models
# ============================================

class UserSignup(BaseModel):
    """User signup request model"""
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (underscores and hyphens allowed)')
        return v.lower()
    
    @validator('email')
    def email_lowercase(cls, v):
        return v.lower()


class UserLogin(BaseModel):
    """User login request model"""
    email: EmailStr
    password: str
    
    @validator('email')
    def email_lowercase(cls, v):
        return v.lower()


class UserResponse(BaseModel):
    """User response model (without password)"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    is_active: bool = True
    subscription_type: str = "free"  # "free" or "pro"
    subscription_status: str = "active"  # "active", "expired", "cancelled"
    subscription_end_date: Optional[datetime] = None


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[str] = None
    email: Optional[str] = None


# ============================================
# Analysis Models
# ============================================

class AnalysisHistory(BaseModel):
    """Analysis history record"""
    user_id: str
    image_name: str
    pitch_type: str
    confidence: float
    weather_data: Optional[dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# Subscription Models
# ============================================

class PaymentHistory(BaseModel):
    """Payment history record"""
    payment_id: str
    order_id: str
    amount: int  # in paise
    currency: str = "INR"
    status: str  # "success", "failed"
    date: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionPlan(BaseModel):
    """Subscription plan details"""
    plan_type: str  # "monthly", "yearly"
    amount: int  # in paise
    currency: str = "INR"


class SubscriptionStatus(BaseModel):
    """User subscription status response"""
    subscription_type: str  # "free" or "pro"
    subscription_status: str  # "active", "expired", "cancelled"
    subscription_end_date: Optional[datetime] = None
    days_remaining: Optional[int] = None
    can_access_complete_analysis: bool = False


# ============================================
# Database User Model
# ============================================

def user_helper(user) -> dict:
    """Convert MongoDB user document to dict"""
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "full_name": user.get("full_name"),
        "created_at": user["created_at"],
        "is_active": user.get("is_active", True),
        "subscription_type": user.get("subscription_type", "free"),
        "subscription_status": user.get("subscription_status", "active"),
        "subscription_end_date": user.get("subscription_end_date"),
    }
