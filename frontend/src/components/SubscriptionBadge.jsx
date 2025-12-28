import React from 'react'
import { Crown, Zap } from 'lucide-react'
import './SubscriptionBadge.css'

function SubscriptionBadge({ user }) {
  if (!user) return null

  const isPro = user.subscription_type === 'pro'
  const status = user.subscription_status
  
  // Calculate days remaining if pro
  let daysRemaining = null
  if (isPro && user.subscription_end_date) {
    const endDate = new Date(user.subscription_end_date)
    const now = new Date()
    const diff = endDate - now
    daysRemaining = Math.max(0, Math.ceil(diff / (1000 * 60 * 60 * 24)))
  }

  return (
    <div className={`subscription-badge ${isPro ? 'pro' : 'free'} ${status}`}>
      {isPro ? (
        <>
          <Crown size={16} />
          <span className="badge-text">PRO</span>
          {status === 'active' && daysRemaining !== null && (
            <span className="badge-days">{daysRemaining}d</span>
          )}
          {status === 'expired' && (
            <span className="badge-status">Expired</span>
          )}
          {status === 'cancelled' && (
            <span className="badge-status">Cancelled</span>
          )}
        </>
      ) : (
        <>
          <Zap size={16} />
          <span className="badge-text">FREE</span>
        </>
      )}
    </div>
  )
}

export default SubscriptionBadge
