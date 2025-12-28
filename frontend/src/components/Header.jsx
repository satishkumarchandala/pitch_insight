import React from 'react'
import { Activity, Github, User, LogOut } from 'lucide-react'
import SubscriptionBadge from './SubscriptionBadge'
import './Header.css'

function Header({ user, onLoginClick, onLogout, currentPage, onNavigate }) {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo" onClick={() => onNavigate('home')}>
            <Activity className="logo-icon" />
            <div>
              <h1 className="logo-text">Pitch Insight</h1>
              <p className="logo-subtitle">AI-Powered Cricket Pitch Analyzer</p>
            </div>
          </div>

          <nav className="nav">
            <button 
              className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
              onClick={() => onNavigate('home')}
            >
              Home
            </button>
            <button 
              className={`nav-link ${currentPage === 'analysis' ? 'active' : ''}`}
              onClick={() => onNavigate('analysis')}
            >
              Analysis
            </button>
            <button 
              className={`nav-link ${currentPage === 'pricing' ? 'active' : ''}`}
              onClick={() => onNavigate('pricing')}
            >
              Pricing
            </button>
            {user && (
              <button 
                className={`nav-link ${currentPage === 'profile' ? 'active' : ''}`}
                onClick={() => onNavigate('profile')}
              >
                Profile
              </button>
            )}
            
            {user ? (
              <div className="user-menu">
                <SubscriptionBadge user={user} />
                <div className="user-info">
                  <User size={18} />
                  <span>{user.username}</span>
                </div>
                <button onClick={onLogout} className="nav-link logout-btn">
                  <LogOut size={18} />
                  <span>Logout</span>
                </button>
              </div>
            ) : (
              <button onClick={onLoginClick} className="nav-link login-btn">
                <User size={18} />
                <span>Login</span>
              </button>
            )}
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header
