import React, { useState } from 'react'
import { Activity, Home, User, LogOut, ChevronLeft, ChevronRight, DollarSign, Settings } from 'lucide-react'
import SubscriptionBadge from './SubscriptionBadge'
import './Header.css'

function Header({ user, onLoginClick, onLogout, currentPage, onNavigate, onSidebarToggle }) {
  const [isOpen, setIsOpen] = useState(true)

  const toggleSidebar = () => {
    const newState = !isOpen
    setIsOpen(newState)
    if (onSidebarToggle) {
      onSidebarToggle(newState)
    }
  }

  return (
    <>
      <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-content">
          <div className="logo" onClick={() => onNavigate('home')}>
            <img src="/logo.png" alt="Pitch Insight" className="logo-icon" />
            {isOpen && (
              <div className="logo-text-container">
                <h1 className="logo-text">Pitch Insight</h1>
                <p className="logo-subtitle">AI-Powered Cricket Pitch Analyzer</p>
              </div>
            )}
          </div>

          <nav className="nav">
            <button 
              className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
              onClick={() => onNavigate('home')}
              title="Home"
            >
              <Home size={20} />
              {isOpen && <span>Home</span>}
            </button>
            <button 
              className={`nav-link ${currentPage === 'analysis' ? 'active' : ''}`}
              onClick={() => onNavigate('analysis')}
              title="Analysis"
            >
              <Activity size={20} />
              {isOpen && <span>Analysis</span>}
            </button>
            <button 
              className={`nav-link ${currentPage === 'pricing' ? 'active' : ''}`}
              onClick={() => onNavigate('pricing')}
              title="Pricing"
            >
              <DollarSign size={20} />
              {isOpen && <span>Pricing</span>}
            </button>
            {user && (
              <button 
                className={`nav-link ${currentPage === 'profile' ? 'active' : ''}`}
                onClick={() => onNavigate('profile')}
                title="Profile"
              >
                <User size={20} />
                {isOpen && <span>Profile</span>}
              </button>
            )}
            <button 
              className={`nav-link ${currentPage === 'settings' ? 'active' : ''}`}
              onClick={() => onNavigate('settings')}
              title="Settings"
            >
              <Settings size={20} />
              {isOpen && <span>Settings</span>}
            </button>
          </nav>
        </div>
      </aside>

      <button className="sidebar-toggle" onClick={toggleSidebar}>
        {isOpen ? <ChevronLeft size={24} /> : <ChevronRight size={24} />}
      </button>

      {/* Profile in top-right corner */}
      {user ? (
        <div className="profile-corner">
          <SubscriptionBadge user={user} />
          <div className="profile-avatar" onClick={() => onNavigate('profile')} title={user.username}>
            <User size={20} />
          </div>
        </div>
      ) : (
        <button onClick={onLoginClick} className="profile-corner login-corner" title="Login">
          <User size={20} />
        </button>
      )}
    </>
  )
}

export default Header
