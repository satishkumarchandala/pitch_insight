"""
Authentication routes for Pitch Insight API
"""
from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime
from pymongo.errors import DuplicateKeyError

from models import UserSignup, UserLogin, Token, UserResponse, user_helper
from auth import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_current_active_user
)
from database import get_users_collection, get_analysis_collection

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    """Register a new user"""
    users_collection = get_users_collection()
    
    # Check if user already exists
    existing_user = users_collection.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username}
        ]
    })
    
    if existing_user:
        if existing_user["email"] == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": get_password_hash(user_data.password),
        "created_at": datetime.utcnow(),
        "is_active": True,
        "subscription_type": "free",
        "subscription_status": "active",
        "subscription_start_date": datetime.utcnow(),
        "subscription_end_date": None,
        "razorpay_customer_id": None,
        "razorpay_subscription_id": None,
        "payment_history": []
    }
    
    try:
        result = users_collection.insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        
        print(f"✓ New user registered: {user_data.email}")
        return UserResponse(**user_helper(user_dict))
    
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return JWT token"""
    user = await authenticate_user(user_credentials.email, user_credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": str(user["_id"]),
            "email": user["email"]
        }
    )
    
    print(f"✓ User logged in: {user['email']}")
    
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_active_user)):
    """Get current authenticated user information"""
    return current_user


@router.get("/history")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """Get analysis history for current user"""
    analysis_collection = get_analysis_collection()
    
    # Get user's analysis history
    history = list(analysis_collection.find(
        {"user_id": str(current_user["_id"])},
        {
            "analysis_id": 1,
            "image_name": 1,
            "pitch_type": 1,
            "confidence": 1,
            "match_info": 1,
            "weather_included": 1,
            "location": 1,
            "processing_time": 1,
            "created_at": 1,
            "timestamp": 1
        }
    ).sort("created_at", -1).limit(100))
    
    # Convert ObjectId to string
    for item in history:
        item["_id"] = str(item["_id"])
    
    return {
        "success": True,
        "count": len(history),
        "history": history
    }


@router.get("/history/{analysis_id}")
async def get_analysis_detail(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed analysis by ID"""
    analysis_collection = get_analysis_collection()
    
    analysis = analysis_collection.find_one({
        "analysis_id": analysis_id,
        "user_id": str(current_user["_id"])
    })
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    analysis["_id"] = str(analysis["_id"])
    return {"success": True, "analysis": analysis}


@router.get("/subscription-status")
async def get_subscription_status(current_user: dict = Depends(get_current_user)):
    """Get current subscription status"""
    subscription_type = current_user.get("subscription_type", "free")
    subscription_status = current_user.get("subscription_status", "active")
    
    # Check if user can access complete analysis (Pro feature)
    can_access = (
        subscription_type == "pro" and 
        subscription_status == "active"
    )
    
    return {
        "success": True,
        "subscription_type": subscription_type,
        "subscription_status": subscription_status,
        "subscription_start_date": current_user.get("subscription_start_date"),
        "subscription_end_date": current_user.get("subscription_end_date"),
        "can_access_complete_analysis": can_access
    }
