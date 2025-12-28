import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { X, CreditCard, Lock, CheckCircle } from 'lucide-react'
import './PaymentModal.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function PaymentModal({ planType = 'monthly', user, token, onClose, onSuccess }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [orderData, setOrderData] = useState(null)

  const planDetails = {
    monthly: {
      amount: 199,
      name: 'Pro Monthly',
      description: 'Monthly Pro Subscription',
      duration: '1 month'
    },
    yearly: {
      amount: 1999,
      name: 'Pro Yearly',
      description: 'Yearly Pro Subscription (Save ₹389)',
      duration: '12 months'
    }
  }

  const plan = planDetails[planType]

  useEffect(() => {
    createOrder()
  }, [])

  const createOrder = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(
        `${API_URL}/api/subscription/create-order?plan_type=${planType}`,
        {},
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (response.data.success) {
        setOrderData(response.data)
      } else {
        setError('Failed to create order. Please try again.')
      }
    } catch (err) {
      console.error('Error creating order:', err)
      setError(err.response?.data?.detail || 'Failed to create payment order')
    } finally {
      setLoading(false)
    }
  }

  const handlePayment = () => {
    if (!orderData) {
      setError('Order data not available. Please try again.')
      return
    }

    const options = {
      key: orderData.razorpay_key,
      amount: orderData.amount,
      currency: orderData.currency,
      name: 'Pitch Insight Pro',
      description: plan.description,
      order_id: orderData.order_id,
      handler: async function (response) {
        // Payment successful - verify on backend
        await verifyPayment(response)
      },
      prefill: {
        email: user.email,
        name: user.username || user.full_name || user.email
      },
      theme: {
        color: '#10b981'
      },
      modal: {
        ondismiss: function () {
          // User closed the payment modal
          setError('Payment cancelled')
        }
      }
    }

    const razorpay = new window.Razorpay(options)
    razorpay.open()
  }

  const verifyPayment = async (paymentResponse) => {
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('order_id', orderData.order_id)
      formData.append('payment_id', paymentResponse.razorpay_payment_id)
      formData.append('signature', paymentResponse.razorpay_signature)
      formData.append('plan_type', planType)

      const response = await axios.post(
        `${API_URL}/api/subscription/verify-payment`,
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      if (response.data.success) {
        // Payment verified successfully
        if (onSuccess) {
          onSuccess(response.data)
        }
      } else {
        setError('Payment verification failed')
      }
    } catch (err) {
      console.error('Error verifying payment:', err)
      setError(err.response?.data?.detail || 'Payment verification failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="payment-modal-overlay" onClick={onClose}>
      <div className="payment-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="payment-modal-content">
          {loading ? (
            <div className="payment-loading">
              <div className="spinner"></div>
              <p>{orderData ? 'Processing payment...' : 'Creating order...'}</p>
            </div>
          ) : error ? (
            <div className="payment-error">
              <div className="error-icon">⚠️</div>
              <h3>Payment Error</h3>
              <p>{error}</p>
              <div className="error-actions">
                <button className="btn-retry" onClick={createOrder}>
                  Try Again
                </button>
                <button className="btn-cancel" onClick={onClose}>
                  Cancel
                </button>
              </div>
            </div>
          ) : orderData ? (
            <>
              <div className="payment-header">
                <div className="payment-icon">
                  <CreditCard size={48} />
                </div>
                <h2>Complete Your Purchase</h2>
                <p>Upgrade to {plan.name}</p>
              </div>

              <div className="payment-details">
                <div className="detail-row">
                  <span>Plan</span>
                  <strong>{plan.name}</strong>
                </div>
                <div className="detail-row">
                  <span>Duration</span>
                  <strong>{plan.duration}</strong>
                </div>
                <div className="detail-row total">
                  <span>Total Amount</span>
                  <strong>₹{plan.amount}</strong>
                </div>
              </div>

              <button className="payment-button" onClick={handlePayment}>
                <Lock size={20} />
                Proceed to Secure Payment
              </button>

              <div className="payment-features">
                <h4>What you'll get:</h4>
                <ul>
                  <li>
                    <CheckCircle size={16} />
                    Complete Pitch Analysis
                  </li>
                  <li>
                    <CheckCircle size={16} />
                    Real-time Weather Integration
                  </li>
                  <li>
                    <CheckCircle size={16} />
                    Match Strategy Insights
                  </li>
                  <li>
                    <CheckCircle size={16} />
                    Priority Support
                  </li>
                </ul>
              </div>

              <div className="payment-security">
                <Lock size={16} />
                <span>Secured by Razorpay - Your payment information is encrypted and secure</span>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export default PaymentModal
