import React from 'react'
import { Heart, Github } from 'lucide-react'
import './Footer.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function Footer({ sidebarOpen = true }) {
  return (
    <footer className={`footer ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <div className="container">
        <div className="footer-content">
          <p className="footer-text">
            Made with <Heart size={16} className="heart-icon" /> for Cricket Analytics
          </p>
          
          <div className="footer-links">
            <span>Developed by Satish Chandala</span>
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
