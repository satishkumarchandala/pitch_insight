import React from 'react'
import { Activity, Github } from 'lucide-react'
import { API_ENDPOINTS } from '../config'
import './Header.css'

function Header() {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <Activity className="logo-icon" />
            <div>
              <h1 className="logo-text">Pitch Insight</h1>
              <p className="logo-subtitle">AI-Powered Cricket Pitch Analyzer</p>
            </div>
          </div>

          <nav className="nav">
            <a 
              href={API_ENDPOINTS.docs} 
              target="_blank" 
              rel="noopener noreferrer"
              className="nav-link"
            >
              API Docs
            </a>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="nav-link"
            >
              <Github size={20} />
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header
