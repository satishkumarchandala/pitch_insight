import React from 'react'
import { Heart, Github } from 'lucide-react'
import './Footer.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function Footer({ sidebarOpen = true }) {
  return (
    <footer className="footer" style={{ marginLeft: sidebarOpen ? '280px' : '80px', transition: 'margin-left 0.3s ease' }}>
      <div className="container">
        <div className="footer-content">
          <p className="footer-text">
            Made with <Heart size={16} className="heart-icon" /> for Cricket Analytics
          </p>
          
          <div className="footer-links">
            <a href={`${API_URL}/docs`} target="_blank" rel="noopener noreferrer">
              API Docs
            </a>
            <span className="separator">•</span>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github size={16} />
              GitHub
            </a>
          </div>

          <p className="footer-copyright">
            © 2025 Pitch Insight. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer
