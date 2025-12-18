import React from 'react'
import { Heart, Github } from 'lucide-react'
import { API_ENDPOINTS } from '../config'
import './Footer.css'

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <p className="footer-text">
            Made with <Heart size={16} className="heart-icon" /> for Cricket Analytics
          </p>
          
          <div className="footer-links">
            <a href={API_ENDPOINTS.docs} target="_blank" rel="noopener noreferrer">
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
