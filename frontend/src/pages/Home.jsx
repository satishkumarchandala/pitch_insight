import React from 'react'
import { Activity, TrendingUp, FileText } from 'lucide-react'
import './Home.css'

function Home({ user, onNavigate }) {
  return (
    <div className="home-page">
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-icon">
            <Activity size={64} />
          </div>
          <h1>Welcome to Pitch Insight</h1>
          <p className="hero-subtitle">
            AI-powered cricket pitch analysis with weather integration
          </p>
          {user && (
            <p className="hero-greeting">
              Hello, <strong>{user.username}</strong>! Ready to analyze some pitches?
            </p>
          )}
        </div>

        <div className="action-cards">
          <div className="action-card" onClick={() => onNavigate('history')}>
            <div className="card-icon">
              <FileText size={40} />
            </div>
            <h3>Your Analysis</h3>
            <p>View your previous pitch analysis history and insights</p>
            <button className="card-btn">
              View History
            </button>
          </div>

          <div className="action-card primary" onClick={() => onNavigate('analysis')}>
            <div className="card-icon">
              <TrendingUp size={40} />
            </div>
            <h3>New Analysis</h3>
            <p>Upload a pitch image and get instant AI-powered analysis</p>
            <button className="card-btn primary">
              Start Analysis
            </button>
          </div>
        </div>
      </div>

      <div className="features-section">
        <h2>Why Choose Pitch Insight?</h2>
        <div className="features-grid">
          <div className="feature-item">
            <div className="feature-icon">🎯</div>
            <h4>Accurate Detection</h4>
            <p>Advanced YOLO-based pitch detection with 95%+ accuracy</p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">🌤️</div>
            <h4>Weather Integration</h4>
            <p>Real-time weather data for comprehensive analysis</p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">⚡</div>
            <h4>Fast Results</h4>
            <p>Get detailed analysis in seconds with ONNX optimization</p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">📊</div>
            <h4>Detailed Reports</h4>
            <p>Comprehensive pitch characteristics and match strategies</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home
