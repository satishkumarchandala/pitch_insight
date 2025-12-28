# Free vs Pro Subscription Setup Guide

## ✅ Implementation Complete

The Free vs Pro subscription system with Razorpay payment integration has been successfully implemented!

## 🔧 Configuration Required

Before the subscription system works, you need to configure Razorpay API credentials:

### 1. Get Razorpay API Credentials

1. **Sign up for Razorpay**:
   - Go to https://razorpay.com/
   - Create an account
   - Complete KYC verification (for live mode)

2. **Get Test API Keys** (for testing):
   - Login to Razorpay Dashboard
   - Go to Settings → API Keys
   - Switch to "Test Mode"
   - Generate/Copy:
     - **Key ID**: `rzp_test_XXXXXXXXXXXXXXXX`
     - **Key Secret**: `YOUR_SECRET_KEY`

### 2. Update Backend Configuration

Edit `backend/app.py` (around line 1029):

```python
# Replace these with your Razorpay credentials
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_XXXXXXXXXXXXXXXX")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "YOUR_SECRET_KEY")
```

**Option A: Hardcode (for testing only)**
```python
RAZORPAY_KEY_ID = "rzp_test_YOUR_KEY_HERE"
RAZORPAY_KEY_SECRET = "your_secret_key_here"
```

**Option B: Environment Variables (recommended)**
Create `backend/.env` file:
```
RAZORPAY_KEY_ID=rzp_test_YOUR_KEY_HERE
RAZORPAY_KEY_SECRET=your_secret_key_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

Add to top of `backend/app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Testing Payment Flow

Use Razorpay Test Cards:

**Test Card Details:**
- Card Number: `4111 1111 1111 1111`
- Expiry: Any future date
- CVV: Any 3 digits
- Name: Any name

**Test UPI:**
- VPA: `success@razorpay`

**Test Netbanking:**
- Select any bank → Use credentials shown

## 📋 System Features

### Free Plan (Default)
- ✅ Quick Analysis
- ✅ Basic Pitch Classification
- ✅ Confidence Scores  
- ✅ Analysis History
- ❌ Complete Analysis (Pro only)
- ❌ Weather Integration (Pro only)
- ❌ Match Strategy (Pro only)

### Pro Plan (₹199/month)
- ✅ All Free features
- ✅ Complete Analysis
- ✅ Real-time Weather Data
- ✅ Weather Impact Analysis
- ✅ Match Strategy Recommendations
- ✅ Detailed Pitch Features
- ✅ Unlimited History Storage
- ✅ Priority Support

## 🚀 Usage Flow

### New User Journey
1. User signs up → Automatically gets **FREE** plan
2. User sees **FREE** badge in header
3. User goes to Analysis page
4. User can do **Quick Analysis** (no restrictions)
5. User clicks **Complete Analysis** → Blocked with upgrade prompt
6. User clicks "Upgrade to Pro" → Redirected to Pricing page
7. User selects plan → Payment modal opens (Razorpay)
8. User completes payment → Upgraded to **PRO**
9. User now has access to Complete Analysis

### Pro User Journey
1. User logs in → Sees **PRO** badge with days remaining
2. Full access to both Quick and Complete Analysis
3. Can view subscription details in Profile page
4. Can cancel subscription (active till end date)

## 📁 Files Created/Modified

### Backend Files
- ✅ `backend/razorpay_handler.py` - Payment processing
- ✅ `backend/subscription_middleware.py` - Access control
- ✅ `backend/models.py` - Added subscription fields
- ✅ `backend/app.py` - Added 5 new endpoints:
  - `GET /api/auth/subscription-status`
  - `POST /api/subscription/create-order`
  - `POST /api/subscription/verify-payment`
  - `POST /api/subscription/cancel`
  - `GET /api/subscription/payment-history`
- ✅ `backend/requirements.txt` - Added razorpay package

### Frontend Files
- ✅ `frontend/src/pages/Pricing.jsx` - Pricing page
- ✅ `frontend/src/pages/Pricing.css` - Pricing styles
- ✅ `frontend/src/components/PaymentModal.jsx` - Payment interface
- ✅ `frontend/src/components/PaymentModal.css` - Payment styles
- ✅ `frontend/src/components/SubscriptionBadge.jsx` - FREE/PRO badge
- ✅ `frontend/src/components/SubscriptionBadge.css` - Badge styles
- ✅ `frontend/src/components/UpgradePrompt.jsx` - Upgrade modal
- ✅ `frontend/src/components/UpgradePrompt.css` - Prompt styles
- ✅ `frontend/src/pages/Analysis.jsx` - Added subscription checks
- ✅ `frontend/src/pages/Analysis.css` - Added pro-required badge style
- ✅ `frontend/src/components/Header.jsx` - Added Pricing nav + badge
- ✅ `frontend/src/App.jsx` - Added Pricing route
- ✅ `frontend/index.html` - Added Razorpay script

## 🧪 Testing Checklist

- [ ] **Sign up new user** → Check FREE badge appears
- [ ] **Try Quick Analysis** → Should work for free users
- [ ] **Try Complete Analysis** → Should show upgrade prompt
- [ ] **Click Upgrade** → Should redirect to Pricing page
- [ ] **Click Upgrade to Pro** → Payment modal opens
- [ ] **Complete test payment** → User upgraded to Pro
- [ ] **Check PRO badge** → Should show in header
- [ ] **Try Complete Analysis** → Should now work
- [ ] **Check subscription status** → In Profile page
- [ ] **View payment history** → Should show transaction
- [ ] **Cancel subscription** → Should mark as cancelled

## 🔒 Security Features

1. **Payment Verification**: All payments verified via signature on backend
2. **User Isolation**: Users can only access their own data
3. **Middleware Protection**: /api/analyze endpoint checks subscription
4. **Token Authentication**: All subscription APIs require JWT
5. **PCI Compliance**: Razorpay handles all card data

## 💡 Important Notes

### Database Changes
New users automatically get these fields:
```javascript
{
  subscription_type: "free",
  subscription_status: "active",
  subscription_start_date: DateTime,
  subscription_end_date: null,
  razorpay_customer_id: null,
  razorpay_subscription_id: null,
  payment_history: []
}
```

### Existing Users
Existing users in database will default to:
- `subscription_type`: "free" (via `.get("subscription_type", "free")`)
- Can upgrade normally through Pricing page

### Subscription Expiry
- Pro subscriptions expire after 30 days
- System automatically checks expiry on each request
- Expired users lose access to Complete Analysis
- Email notifications not implemented (future enhancement)

## 🐛 Troubleshooting

### Payment Modal Not Opening
- Check Razorpay script is loaded: Open browser console
- Verify `window.Razorpay` exists
- Check RAZORPAY_KEY_ID in backend response

### Payment Verification Failed
- Check RAZORPAY_KEY_SECRET is correct
- Verify signature verification logic
- Check backend logs for errors

### User Still Sees Free After Payment
- Check MongoDB users collection
- Verify subscription_type updated to "pro"
- Check subscription_end_date is set
- Refresh user token (logout/login)

### Complete Analysis Still Blocked
- Check subscription_status is "active"
- Verify subscription_end_date is in future
- Check middleware logic in backend
- Verify token includes updated user data

## 🚀 Next Steps

1. **Set up Razorpay account** and get test credentials
2. **Update backend** with your API keys
3. **Test payment flow** with test cards
4. **Verify subscription** system works end-to-end
5. **Go live** with production Razorpay keys when ready

## 📞 Support

If you need help:
1. Check Razorpay Dashboard for payment logs
2. Check MongoDB for subscription data
3. Check backend terminal for error logs
4. Check browser console for frontend errors

## 🎉 Ready to Launch!

The system is fully implemented and ready to use. Just configure your Razorpay API keys and test the payment flow!
