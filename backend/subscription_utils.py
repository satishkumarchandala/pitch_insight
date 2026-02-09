"""
Subscription Validation Utilities
Centralized subscription checking and auto-expiration logic
"""

from datetime import datetime
from typing import Dict, Tuple, Optional
from bson import ObjectId


def check_and_update_subscription_status(user: dict, users_collection) -> Tuple[bool, str, Optional[dict]]:
    """
    Check if user has active Pro subscription and auto-expire if needed
    
    Returns:
        Tuple of (is_pro: bool, status_message: str, updated_user: dict or None)
    """
    subscription_type = user.get("subscription_type", "free")
    subscription_status = user.get("subscription_status", "active")
    subscription_end = user.get("subscription_end_date")
    
    # Free users - no validation needed
    if subscription_type == "free":
        return False, "User has free subscription", None
    
    # Pro users - check expiration
    if subscription_type == "pro":
        # Check if already marked as expired/cancelled
        if subscription_status in ["expired", "cancelled"]:
            return False, f"Subscription is {subscription_status}", None
        
        # Check if subscription has actually expired
        if subscription_end and datetime.utcnow() > subscription_end:
            # AUTO-UPDATE DATABASE to mark as expired
            update_result = users_collection.update_one(
                {"_id": ObjectId(user["_id"])},
                {
                    "$set": {
                        "subscription_type": "free",
                        "subscription_status": "expired",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if update_result.modified_count > 0:
                print(f"⚠️ Auto-expired subscription for user: {user.get('email')} (expired on {subscription_end})")
                
                # Return updated user data
                updated_user = user.copy()
                updated_user["subscription_type"] = "free"
                updated_user["subscription_status"] = "expired"
                
                return False, f"Subscription expired on {subscription_end.strftime('%Y-%m-%d')}", updated_user
            else:
                print(f"❌ Failed to auto-expire subscription for user: {user.get('email')}")
                return False, "Subscription expired but database update failed", None
        
        # Pro subscription is valid and active
        return True, "Pro subscription is active", None
    
    # Unknown subscription type
    return False, f"Unknown subscription type: {subscription_type}", None


def require_pro_subscription(user: dict, users_collection, feature_name: str = "this feature") -> dict:
    """
    Validate Pro subscription and raise HTTPException if not valid
    Returns updated user dict if subscription was auto-expired
    
    Args:
        user: User document from database
        users_collection: MongoDB users collection
        feature_name: Name of feature being accessed (for error message)
    
    Returns:
        Updated user dict (or original if no update)
        
    Raises:
        HTTPException: If user doesn't have valid Pro subscription
    """
    from fastapi import HTTPException
    
    is_pro, message, updated_user = check_and_update_subscription_status(user, users_collection)
    
    if not is_pro:
        # Use updated user data in error message if available
        user_data = updated_user if updated_user else user
        
        if user_data.get("subscription_status") == "expired":
            raise HTTPException(
                status_code=403,
                detail=f"Your Pro subscription has expired. {message}. Please renew to access {feature_name}."
            )
        elif user_data.get("subscription_status") == "cancelled":
            raise HTTPException(
                status_code=403,
                detail=f"Your subscription was cancelled. Please subscribe again to access {feature_name}."
            )
        else:
            raise HTTPException(
                status_code=403,
                detail=f"Pro subscription required to access {feature_name}. Please upgrade your plan."
            )
    
    # Return updated user if subscription was checked and updated
    return updated_user if updated_user else user


def get_subscription_info(user: dict, users_collection) -> Dict:
    """
    Get detailed subscription information with auto-expiration check
    
    Returns dict with:
        - subscription_type: "free" or "pro"
        - subscription_status: "active", "expired", "cancelled"
        - subscription_end_date: datetime or None
        - is_pro: bool
        - days_remaining: int or None
        - can_access_complete_analysis: bool
        - auto_expired: bool (if subscription was just auto-expired)
    """
    is_pro, message, updated_user = check_and_update_subscription_status(user, users_collection)
    
    # Use updated user data if available
    user_data = updated_user if updated_user else user
    
    subscription_end = user_data.get("subscription_end_date")
    days_remaining = None
    
    if is_pro and subscription_end:
        delta = subscription_end - datetime.utcnow()
        days_remaining = max(0, delta.days)
    
    return {
        "subscription_type": user_data.get("subscription_type", "free"),
        "subscription_status": user_data.get("subscription_status", "active"),
        "subscription_start_date": user_data.get("subscription_start_date"),
        "subscription_end_date": subscription_end,
        "is_pro": is_pro,
        "days_remaining": days_remaining,
        "can_access_complete_analysis": is_pro,
        "can_access_history": is_pro,
        "can_access_chatbot": is_pro,
        "auto_expired": updated_user is not None,  # True if we just auto-expired
        "message": message
    }
