import React, { useState } from 'react'
import { Check, X, Zap, Crown, ArrowRight } from 'lucide-react'
import PaymentModal from '../components/PaymentModal'
import './Pricing.css'

function Pricing({ user, token, onNavigate, onUserUpdate }) {
  const [showPaymentModal, setShowPaymentModal] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState('monthly')

  const handleUpgradeClick = (planType) => {
    if (!user) {
      // Redirect to login/signup
      alert('Please login or signup to upgrade to Pro')
      return
    }

    setSelectedPlan(planType)
    setShowPaymentModal(true)
  }

  const handlePaymentSuccess = async () => {
    setShowPaymentModal(false)
    // Refresh user data to get updated subscription
    if (onUserUpdate) {
      await onUserUpdate()
    }
    // Give a small delay to ensure state updates propagate
    setTimeout(() => {
      // Redirect to analysis page
      if (onNavigate) {
        onNavigate('analysis')
      }
    }, 500)
  }

  const features = {
    free: [
      { text: 'Quick Analysis', available: true },
      { text: 'Basic Pitch Classification', available: true },
      { text: 'Confidence Scores', available: true },
      { text: 'Analysis History', available: true },
      { text: 'Complete Analysis', available: false },
      { text: 'Weather Integration', available: false },
      { text: 'Match Strategy Insights', available: false },
      { text: 'Detailed Features Analysis', available: false },
      { text: 'Priority Support', available: false }
    ],
    pro: [
      { text: 'Quick Analysis', available: true },
      { text: 'Complete Analysis', available: true },
      { text: 'Real-time Weather Data', available: true },
      { text: 'Weather Impact Analysis', available: true },
      { text: 'Match Strategy Recommendations', available: true },
      { text: 'Detailed Pitch Features', available: true },
      { text: 'Grass, Cracks, Moisture Analysis', available: true },
      { text: 'Unlimited History Storage', available: true },
      { text: 'Priority Support', available: true }
    ]
  }

  return (
    <div className="pricing-page">
      <div className="pricing-hero">
        <h1>Choose Your Plan</h1>
        <p>Get the most out of Pitch Insight with our Pro subscription</p>
      </div>

      <div className="pricing-cards">
        {/* Free Plan */}
        <div className="pricing-card free-card">
          <div className="plan-header">
            <div className="plan-icon free-icon">
              <Zap size={32} />
            </div>
            <h2>Free</h2>
            <div className="plan-price">
              <span className="currency">₹</span>
              <span className="amount">0</span>
              <span className="period">/month</span>
            </div>
            <p className="plan-description">Perfect for quick assessments</p>
          </div>

          <ul className="features-list">
            {features.free.map((feature, index) => (
              <li key={index} className={feature.available ? 'available' : 'unavailable'}>
                {feature.available ? (
                  <Check size={20} className="feature-icon" />
                ) : (
                  <X size={20} className="feature-icon" />
                )}
                <span>{feature.text}</span>
              </li>
            ))}
          </ul>

          <button 
            className="plan-button free-button"
            disabled={user}
          >
            {user ? 'Current Plan' : 'Sign Up Free'}
          </button>
        </div>

        {/* Pro Plan */}
        <div className="pricing-card pro-card featured">
          <div className="popular-badge">
            <Crown size={16} />
            Most Popular
          </div>

          <div className="plan-header">
            <div className="plan-icon pro-icon">
              <Crown size={32} />
            </div>
            <h2>Pro</h2>
            <div className="plan-price">
              <span className="currency">₹</span>
              <span className="amount">199</span>
              <span className="period">/month</span>
            </div>
            <p className="plan-description">Complete analysis with all features</p>
          </div>

          <ul className="features-list">
            {features.pro.map((feature, index) => (
              <li key={index} className="available">
                <Check size={20} className="feature-icon" />
                <span>{feature.text}</span>
              </li>
            ))}
          </ul>

          <button 
            className="plan-button free-button"
            onClick={() => handleUpgradeClick('monthly')}
            disabled={user?.subscription_type === 'pro'}
          >
            {user?.subscription_type === 'pro' ? (
              'Current Plan'
            ) : (
              <>
                Upgrade to Pro
                <ArrowRight size={20} />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Feature Comparison */}
      <div className="feature-comparison">
        <h2>What's Included in Pro?</h2>
        <div className="comparison-grid">
          <div className="comparison-item">
            <div className="comparison-icon">🎯</div>
            <h3>Complete Analysis</h3>
            <p>Get comprehensive pitch analysis with detailed features including grass coverage, cracks, moisture levels, and hardness assessment.</p>
          </div>

          <div className="comparison-item">
            <div className="comparison-icon">🌤️</div>
            <h3>Weather Integration</h3>
            <p>Real-time weather data with impact analysis on swing, spin, drying rate, and dew likelihood for strategic insights.</p>
          </div>

          <div className="comparison-item">
            <div className="comparison-icon">📊</div>
            <h3>Match Strategy</h3>
            <p>AI-powered match strategy recommendations including optimal approach, key considerations, and toss advantage insights.</p>
          </div>

          <div className="comparison-item">
            <div className="comparison-icon">⚡</div>
            <h3>Priority Support</h3>
            <p>Get faster response times and dedicated support for all your pitch analysis needs.</p>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="pricing-faq">
        <h2>Frequently Asked Questions</h2>
        <div className="faq-list">
          <div className="faq-item">
            <h3>Can I switch from Free to Pro anytime?</h3>
            <p>Yes! You can upgrade to Pro at any time. Your Pro subscription will start immediately after payment.</p>
          </div>

          <div className="faq-item">
            <h3>Can I cancel my Pro subscription?</h3>
            <p>Yes, you can cancel anytime from your profile page. You'll continue to have Pro access until the end of your billing period.</p>
          </div>

          <div className="faq-item">
            <h3>What payment methods do you accept?</h3>
            <p>We accept all major credit/debit cards, UPI, net banking, and wallets through Razorpay secure payment gateway.</p>
          </div>

          <div className="faq-item">
            <h3>Is my payment information secure?</h3>
            <p>Absolutely! All payments are processed through Razorpay, a PCI DSS compliant payment gateway. We never store your card details.</p>
          </div>
        </div>
      </div>

      {/* Payment Modal */}
      {showPaymentModal && (
        <PaymentModal
          planType={selectedPlan}
          user={user}
          token={token}
          onClose={() => setShowPaymentModal(false)}
          onSuccess={handlePaymentSuccess}
        />
      )}
    </div>
  )
}

export default Pricing
