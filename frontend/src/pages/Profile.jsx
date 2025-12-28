import React, { useState, useEffect } from 'react'
import { User, Mail, Calendar, LogOut, History, TrendingUp } from 'lucide-react'
import axios from 'axios'
import './Profile.css'

const API_URL = 'http://localhost:8000'

function Profile({ user, token, onLogout, onNavigate }) {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    recentAnalyses: 0,
    favoriteType: 'N/A'
  })

  useEffect(() => {
    if (token) {
      fetchHistory()
    }
  }, [token])

  const fetchHistory = async () => {
    setLoading(true)
    try {
      const response = await axios.get(`${API_URL}/api/auth/history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      const historyData = response.data.history || []
      setHistory(historyData)
      
      // Calculate stats
      const total = historyData.length
      const recent = historyData.filter(item => {
        const date = new Date(item.created_at)
        const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000)
        return date > dayAgo
      }).length
      
      // Find most common pitch type
      const typeCounts = {}
      historyData.forEach(item => {
        typeCounts[item.pitch_type] = (typeCounts[item.pitch_type] || 0) + 1
      })
      const favoriteType = Object.keys(typeCounts).length > 0
        ? Object.keys(typeCounts).reduce((a, b) => typeCounts[a] > typeCounts[b] ? a : b)
        : 'N/A'
      
      setStats({
        totalAnalyses: total,
        recentAnalyses: recent,
        favoriteType
      })
    } catch (error) {
      console.error('Failed to fetch history:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#4CAF50'
    if (confidence >= 0.7) return '#FF9800'
    return '#f44336'
  }

  return (
    <div className="profile-page">
      <div className="profile-header">
        <div className="profile-avatar">
          <User size={48} />
        </div>
        <div className="profile-info">
          <h1>{user?.username || 'User'}</h1>
          <p className="profile-email">
            <Mail size={16} />
            {user?.email || 'email@example.com'}
          </p>
          {user?.full_name && (
            <p className="profile-fullname">{user.full_name}</p>
          )}
          <p className="profile-joined">
            <Calendar size={16} />
            Joined {new Date(user?.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
          </p>
        </div>
        <button className="logout-btn" onClick={onLogout}>
          <LogOut size={20} />
          Logout
        </button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <History size={32} />
          </div>
          <div className="stat-info">
            <h3>{stats.totalAnalyses}</h3>
            <p>Total Analyses</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <TrendingUp size={32} />
          </div>
          <div className="stat-info">
            <h3>{stats.recentAnalyses}</h3>
            <p>Last 24 Hours</p>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <span className="stat-emoji">🏏</span>
          </div>
          <div className="stat-info">
            <h3>{stats.favoriteType}</h3>
            <p>Most Common Type</p>
          </div>
        </div>
      </div>

      <div className="history-section">
        <div className="section-header">
          <h2>Analysis History</h2>
          {history.length > 0 && (
            <button className="view-all-btn" onClick={() => onNavigate('history')}>
              View All
            </button>
          )}
        </div>

        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading history...</p>
          </div>
        ) : history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <h3>No Analysis Yet</h3>
            <p>Start analyzing pitches to build your history</p>
            <button className="btn btn-primary" onClick={() => onNavigate('analysis')}>
              Start First Analysis
            </button>
          </div>
        ) : (
          <div className="history-list">
            {history.slice(0, 5).map((item, index) => (
              <div key={item._id || index} className="history-item">
                <div className="history-main">
                  <div className="history-icon">🏏</div>
                  <div className="history-details">
                    <h4>{item.pitch_type}</h4>
                    <p className="history-filename">{item.image_name}</p>
                    <p className="history-date">{formatDate(item.created_at)}</p>
                  </div>
                </div>
                <div 
                  className="history-confidence"
                  style={{ color: getConfidenceColor(item.confidence) }}
                >
                  {(item.confidence * 100).toFixed(0)}% confident
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default Profile
