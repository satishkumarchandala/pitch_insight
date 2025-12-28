"""
Subscription and payment routes for Pitch Insight API
"""
from fastapi import APIRouter, HTTPException, Depends, Form
from datetime import datetime, timedelta
from bson import ObjectId

from auth import get_current_user
from database import get_users_collection
from config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, SUBSCRIPTION_PLANS

router = APIRouter(prefix="/api/subscription", tags=["subscription"])

# Import Razorpay handler
try:
    from razorpay_handler import RazorpayHandler
    razorpay_handler = RazorpayHandler(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
except ImportError:
    print("⚠️ Razorpay handler not found")
    razorpay_handler = None


def get_plan_amount(plan_type: str) -> int:
    """Get plan amount in paise"""
    plan = SUBSCRIPTION_PLANS.get(plan_type)
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan type")
    return plan["price"] * 100  # Convert to paise


def calculate_subscription_end_date(duration_months: int) -> datetime:
    """Calculate subscription end date"""
    return datetime.utcnow() + timedelta(days=duration_months * 30)


def get_subscription_access(user: dict) -> dict:
    """Get user's subscription access details"""
    subscription_type = user.get("subscription_type", "free")
    subscription_status = user.get("subscription_status", "active")
    subscription_end = user.get("subscription_end_date")
    
    # Check if subscription has expired
    if subscription_type == "pro" and subscription_end:
        if datetime.utcnow() > subscription_end:
            subscription_type = "free"
            subscription_status = "expired"
    
    is_pro = subscription_type == "pro" and subscription_status == "active"
    
    return {
        "subscription_type": subscription_type,
        "subscription_status": subscription_status,
        "subscription_start_date": user.get("subscription_start_date"),
        "subscription_end_date": subscription_end,
        "is_pro": is_pro,
        "features": {
            "unlimited_analyses": is_pro,
            "weather_integration": is_pro,
            "detailed_reports": is_pro,
            "priority_support": is_pro
        }
    }


@router.post("/create-order")
async def create_subscription_order(
    plan_type: str = "monthly",
    current_user: dict = Depends(get_current_user)
):
    """Create Razorpay order for subscription payment"""
    if not razorpay_handler:
        raise HTTPException(status_code=503, detail="Payment service unavailable")
    
    try:
        # Get plan amount
        amount = get_plan_amount(plan_type)
        
        # Create order with short receipt (max 40 chars)
        short_id = str(current_user['_id'])[-8:]
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        order_result = razorpay_handler.create_order(
            amount=amount,
            receipt=f"sub_{short_id}_{timestamp}"
        )
        
        if not order_result.get('success'):
            print(f"❌ Order creation failed: {order_result}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create payment order. Please try again."
            )
        
        return {
            "success": True,
            "order_id": order_result['order_id'],
            "amount": order_result['amount'],
            "currency": order_result['currency'],
            "razorpay_key": RAZORPAY_KEY_ID,
            "plan_type": plan_type,
            "user": {
                "email": current_user["email"],
                "username": current_user.get("username", "")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in create_subscription_order: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/verify-payment")
async def verify_subscription_payment(
    order_id: str = Form(...),
    payment_id: str = Form(...),
    signature: str = Form(...),
    plan_type: str = Form("monthly"),
    current_user: dict = Depends(get_current_user)
):
    """Verify Razorpay payment and upgrade user to Pro"""
    if not razorpay_handler:
        raise HTTPException(status_code=503, detail="Payment service unavailable")
    
    # Verify payment signature
    is_valid = razorpay_handler.verify_payment_signature(
        order_id=order_id,
        payment_id=payment_id,
        signature=signature
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail="Payment verification failed. Invalid signature."
        )
    
    # Fetch payment details
    payment_details = razorpay_handler.fetch_payment_details(payment_id)
    
    if not payment_details or payment_details['status'] != 'captured':
        raise HTTPException(
            status_code=400,
            detail="Payment not successful"
        )
    
    # Calculate subscription dates
    subscription_start = datetime.utcnow()
    duration_months = 12 if plan_type == "yearly" else 1
    subscription_end = calculate_subscription_end_date(duration_months)
    
    # Update user subscription
    users_collection = get_users_collection()
    
    payment_record = {
        "payment_id": payment_id,
        "order_id": order_id,
        "amount": payment_details['amount'],
        "currency": payment_details['currency'],
        "status": "success",
        "plan_type": plan_type,
        "date": datetime.utcnow()
    }
    
    update_result = users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {
            "$set": {
                "subscription_type": "pro",
                "subscription_status": "active",
                "subscription_start_date": subscription_start,
                "subscription_end_date": subscription_end,
            },
            "$push": {
                "payment_history": payment_record
            }
        }
    )
    
    if update_result.modified_count == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to update subscription"
        )
    
    print(f"✓ User upgraded to Pro: {current_user['email']} - Valid until {subscription_end}")
    
    return {
        "success": True,
        "message": "Payment successful! You are now a Pro member.",
        "subscription": {
            "type": "pro",
            "status": "active",
            "start_date": subscription_start.isoformat(),
            "end_date": subscription_end.isoformat(),
            "plan_type": plan_type
        }
    }


@router.post("/cancel")
async def cancel_subscription(current_user: dict = Depends(get_current_user)):
    """Cancel subscription"""
    users_collection = get_users_collection()
    
    result = users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"subscription_status": "cancelled"}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No active subscription to cancel"
        )
    
    print(f"✓ Subscription cancelled for: {current_user['email']}")
    
    return {
        "success": True,
        "message": "Subscription cancelled. You will have access until the end of your billing period.",
        "subscription_end_date": current_user.get("subscription_end_date")
    }


@router.get("/payment-history")
async def get_payment_history(current_user: dict = Depends(get_current_user)):
    """Get user's payment history"""
    payment_history = current_user.get("payment_history", [])
    
    return {
        "success": True,
        "count": len(payment_history),
        "payments": payment_history
    }
