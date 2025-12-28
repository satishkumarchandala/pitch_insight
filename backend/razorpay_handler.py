import razorpay
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional

class RazorpayHandler:
    def __init__(self, key_id: str, key_secret: str):
        """
        Initialize Razorpay client with API credentials
        
        Test credentials (replace with your own):
        key_id: rzp_test_XXXXXXXXXXXXXXXX
        key_secret: YOUR_SECRET_KEY
        """
        self.key_id = key_id
        self.key_secret = key_secret
        self.client = razorpay.Client(auth=(key_id, key_secret))
    
    def create_order(self, amount: int, currency: str = "INR", receipt: str = None) -> Dict:
        """
        Create a Razorpay order for one-time payment
        
        Args:
            amount: Amount in smallest currency unit (paise for INR)
            currency: Currency code (default: INR)
            receipt: Optional receipt ID for your reference
            
        Returns:
            Dict with order details including order_id
        """
        try:
            order_data = {
                'amount': amount,  # Amount in paise (199 * 100 = 19900 paise)
                'currency': currency,
                'receipt': receipt or f'receipt_{datetime.now().strftime("%Y%m%d%H%M%S")}',
                'payment_capture': 1  # Auto capture payment
            }
            
            order = self.client.order.create(data=order_data)
            return {
                'success': True,
                'order_id': order['id'],
                'amount': order['amount'],
                'currency': order['currency'],
                'receipt': order['receipt']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment_signature(self, order_id: str, payment_id: str, signature: str) -> bool:
        """
        Verify the payment signature to ensure payment authenticity
        
        Args:
            order_id: Razorpay order ID
            payment_id: Razorpay payment ID
            signature: Payment signature from Razorpay
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Create signature verification string
            message = f"{order_id}|{payment_id}"
            
            # Generate expected signature
            generated_signature = hmac.new(
                self.key_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(generated_signature, signature)
        except Exception as e:
            print(f"Signature verification error: {e}")
            return False
    
    def fetch_payment_details(self, payment_id: str) -> Optional[Dict]:
        """
        Fetch payment details from Razorpay
        
        Args:
            payment_id: Razorpay payment ID
            
        Returns:
            Payment details dict or None if error
        """
        try:
            payment = self.client.payment.fetch(payment_id)
            return {
                'id': payment['id'],
                'amount': payment['amount'],
                'currency': payment['currency'],
                'status': payment['status'],
                'method': payment['method'],
                'email': payment.get('email'),
                'contact': payment.get('contact'),
                'created_at': payment['created_at']
            }
        except Exception as e:
            print(f"Error fetching payment: {e}")
            return None
    
    def create_subscription(self, plan_id: str, customer_id: str, total_count: int = 12) -> Dict:
        """
        Create a recurring subscription (for future use)
        
        Args:
            plan_id: Razorpay plan ID
            customer_id: Razorpay customer ID
            total_count: Number of billing cycles
            
        Returns:
            Subscription details
        """
        try:
            subscription_data = {
                'plan_id': plan_id,
                'customer_id': customer_id,
                'total_count': total_count,
                'quantity': 1
            }
            
            subscription = self.client.subscription.create(data=subscription_data)
            return {
                'success': True,
                'subscription_id': subscription['id'],
                'status': subscription['status']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_subscription(self, subscription_id: str, cancel_at_end: bool = True) -> Dict:
        """
        Cancel a subscription
        
        Args:
            subscription_id: Razorpay subscription ID
            cancel_at_end: If True, cancels at end of billing cycle
            
        Returns:
            Cancellation status
        """
        try:
            if cancel_at_end:
                subscription = self.client.subscription.cancel(subscription_id)
            else:
                subscription = self.client.subscription.cancel(subscription_id, data={'cancel_at_cycle_end': 0})
            
            return {
                'success': True,
                'status': subscription['status']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_subscription_status(self, subscription_id: str) -> Optional[Dict]:
        """
        Get current status of a subscription
        
        Args:
            subscription_id: Razorpay subscription ID
            
        Returns:
            Subscription details or None
        """
        try:
            subscription = self.client.subscription.fetch(subscription_id)
            return {
                'id': subscription['id'],
                'status': subscription['status'],
                'current_start': subscription['current_start'],
                'current_end': subscription['current_end']
            }
        except Exception as e:
            print(f"Error fetching subscription: {e}")
            return None


def calculate_subscription_end_date(duration_months: int = 1) -> datetime:
    """
    Calculate subscription end date
    
    Args:
        duration_months: Subscription duration in months
        
    Returns:
        End date as datetime
    """
    return datetime.utcnow() + timedelta(days=30 * duration_months)


def get_plan_amount(plan_type: str = "monthly") -> int:
    """
    Get plan amount in paise
    
    Args:
        plan_type: "monthly" or "yearly"
        
    Returns:
        Amount in paise (smallest currency unit)
    """
    plans = {
        "monthly": 19900,  # ₹199 in paise
        "yearly": 199900   # ₹1999 in paise (optional)
    }
    return plans.get(plan_type, 19900)
