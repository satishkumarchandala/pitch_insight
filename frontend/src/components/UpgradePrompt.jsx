import React from 'react'
import { X, Crown, Check, ArrowRight } from 'lucide-react'
import './UpgradePrompt.css'

function UpgradePrompt({ onClose, onUpgrade }) {
  return (
    <div className="upgrade-prompt-overlay" onClick={onClose}>
      <div className="upgrade-prompt" onClick={(e) => e.stopPropagation()}>
        <button className="close-btn" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="upgrade-content">
          <div className="upgrade-icon">
            <Crown size={64} />
          </div>

          <h2>Upgrade to Pro</h2>
          <p className="upgrade-subtitle">
            Complete Analysis requires a Pro subscription
          </p>

          <div className="upgrade-features">
            <h3>Unlock Pro Features:</h3>
            <ul>
              <li>
                <Check size={20} />
                <span>Complete Pitch Analysis</span>
              </li>
              <li>
                <Check size={20} />
                <span>Real-time Weather Integration</span>
              </li>
              <li>
                <Check size={20} />
                <span>Match Strategy Recommendations</span>
              </li>
              <li>
                <Check size={20} />
                <span>Detailed Features Analysis</span>
              </li>
              <li>
                <Check size={20} />
                <span>Priority Support</span>
              </li>
            </ul>
          </div>

          <div className="upgrade-pricing">
            <div className="price-tag">
              <span className="currency">₹</span>
              <span className="amount">199</span>
              <span className="period">/month</span>
            </div>
            <p className="price-note">Cancel anytime • Secure payment</p>
          </div>

          <div className="upgrade-actions">
            <button className="btn-upgrade" onClick={onUpgrade}>
              <Crown size={20} />
              Upgrade to Pro
              <ArrowRight size={20} />
            </button>
            <button className="btn-continue-free" onClick={onClose}>
              Continue with Free
            </button>
          </div>

          <div className="upgrade-note">
            <p>✨ Join thousands of cricket professionals using Pro features</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default UpgradePrompt
