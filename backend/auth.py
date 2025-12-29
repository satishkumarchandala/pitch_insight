"""
Authentication Utilities
Handles password hashing, JWT tokens, and user authentication
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import TokenData, UserResponse, user_helper
from database import get_users_collection
from config import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES

# Security Configuration
ALGORITHM = "HS256"

# Bearer token security
security = HTTPBearer()


# ============================================
# Password Utilities
# ============================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Hash a password"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


# ============================================
# JWT Token Utilities
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        
        return TokenData(user_id=user_id, email=email)
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


# ============================================
# User Authentication
# ============================================

async def authenticate_user(email: str, password: str):
    """Authenticate a user by email and password"""
    users_collection = get_users_collection()
    user = users_collection.find_one({"email": email.lower()})
    
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    return user


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Get current authenticated user from JWT token
    Use as a dependency in protected routes
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(token)
    except HTTPException:
        raise credentials_exception
    
    # Get user from database
    users_collection = get_users_collection()
    user = users_collection.find_one({"_id": ObjectId(token_data.user_id)})
    
    if user is None:
        raise credentials_exception
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """Get current active user as UserResponse model"""
    return UserResponse(**user_helper(current_user))


async def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[dict]:
    """
    Get current user if token is provided, otherwise return None
    Use this for endpoints that work with or without authentication
    """
    if credentials is None:
        return None
    
    try:
        token_data = verify_token(credentials.credentials)
        users_collection = get_users_collection()
        user = users_collection.find_one({"_id": ObjectId(token_data.user_id)})
        
        if user and user.get("is_active", True):
            return user
    except:
        pass
    
    return None


# Import ObjectId here to avoid circular imports
from bson import ObjectId
