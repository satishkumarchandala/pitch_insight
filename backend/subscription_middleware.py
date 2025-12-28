"""
Subscription Middleware for Pitch Insight
Checks user subscription status and access rights
"""

from fastapi import HTTPException, status
from datetime import datetime
from typing import Dict


def check_pro_subscription(current_user: Dict) -> None:
    """
    Middleware to check if user has active Pro subscription
    Raises HTTPException if user doesn't have access
    
    Args:
        current_user: User dict from get_current_user()
        
    Raises:
        HTTPException: If user doesn't have pro subscription or it's expired
    """
    subscription_type = current_user.get("subscription_type", "free")
    subscription_status = current_user.get("subscription_status", "active")
    subscription_end_date = current_user.get("subscription_end_date")
    
    # Check if user is on free plan
    if subscription_type != "pro":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "pro_subscription_required",
                "message": "Complete analysis requires Pro subscription. Upgrade to Pro to access all features including weather data, detailed insights, and match strategies.",
                "subscription_type": subscription_type,
                "upgrade_url": "/pricing"
            }
        )
    
    # Check if subscription is cancelled
    if subscription_status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "subscription_cancelled",
                "message": "Your Pro subscription has been cancelled. Please renew to continue accessing complete analysis features.",
                "subscription_type": subscription_type,
                "subscription_status": subscription_status,
                "upgrade_url": "/pricing"
            }
        )
    
    # Check if subscription is expired
    if subscription_status == "expired":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "subscription_expired",
                "message": "Your Pro subscription has expired. Renew now to regain access to complete analysis features.",
                "subscription_type": subscription_type,
                "subscription_status": subscription_status,
                "upgrade_url": "/pricing"
            }
        )
    
    # Check expiry date if provided
    if subscription_end_date:
        if isinstance(subscription_end_date, str):
            subscription_end_date = datetime.fromisoformat(subscription_end_date.replace('Z', '+00:00'))
        
        if datetime.utcnow() > subscription_end_date:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "subscription_expired",
                    "message": "Your Pro subscription has expired. Renew now to continue accessing complete analysis features.",
                    "subscription_type": subscription_type,
                    "subscription_status": "expired",
                    "expired_on": subscription_end_date.isoformat(),
                    "upgrade_url": "/pricing"
                }
            )


def get_subscription_access(current_user: Dict) -> Dict:
    """
    Get user's subscription access details
    
    Args:
        current_user: User dict from get_current_user()
        
    Returns:
        Dict with access rights and subscription info
    """
    subscription_type = current_user.get("subscription_type", "free")
    subscription_status = current_user.get("subscription_status", "active")
    subscription_end_date = current_user.get("subscription_end_date")
    
    # Calculate days remaining
    days_remaining = None
    if subscription_end_date:
        if isinstance(subscription_end_date, str):
            subscription_end_date = datetime.fromisoformat(subscription_end_date.replace('Z', '+00:00'))
        
        delta = subscription_end_date - datetime.utcnow()
        days_remaining = max(0, delta.days)
        
        # Auto-expire if past end date
        if days_remaining == 0 and subscription_status == "active":
            subscription_status = "expired"
    
    # Determine access rights
    can_access_complete_analysis = (
        subscription_type == "pro" and
        subscription_status == "active" and
        (days_remaining is None or days_remaining > 0)
    )
    
    return {
        "subscription_type": subscription_type,
        "subscription_status": subscription_status,
        "subscription_end_date": subscription_end_date.isoformat() if subscription_end_date else None,
        "days_remaining": days_remaining,
        "can_access_complete_analysis": can_access_complete_analysis,
        "can_access_quick_analysis": True,  # Always available
        "features": {
            "quick_analysis": True,
            "complete_analysis": can_access_complete_analysis,
            "weather_integration": can_access_complete_analysis,
            "match_strategy": can_access_complete_analysis,
            "history_storage": True,  # Available for all
            "priority_support": can_access_complete_analysis
        }
    }


def check_subscription_status_and_update(user_collection, user_id: str) -> Dict:
    """
    Check and update subscription status if expired
    
    Args:
        user_collection: MongoDB users collection
        user_id: User ObjectId as string
        
    Returns:
        Updated user dict
    """
    from bson import ObjectId
    
    user = user_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None
    
    subscription_end_date = user.get("subscription_end_date")
    subscription_status = user.get("subscription_status", "active")
    
    # Check if subscription should be expired
    if subscription_end_date and subscription_status == "active":
        if isinstance(subscription_end_date, str):
            subscription_end_date = datetime.fromisoformat(subscription_end_date.replace('Z', '+00:00'))
        
        if datetime.utcnow() > subscription_end_date:
            # Update status to expired
            user_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"subscription_status": "expired"}}
            )
            user["subscription_status"] = "expired"
    
    return user
