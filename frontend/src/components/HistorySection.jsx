import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Clock, Trash2, Eye, MapPin, Image as ImageIcon, CloudSun } from 'lucide-react';
import './HistorySection.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const HistorySection = ({ onViewDetails, authToken }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  useEffect(() => {
    if (authToken) {
      fetchHistory();
    }
  }, [authToken]);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/api/auth/history`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      setHistory(response.data.history || []);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError('Failed to load analysis history');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (analysisId) => {
    if (!window.confirm('Are you sure you want to delete this analysis?')) {
      return;
    }

    try {
      await axios.delete(`${API_URL}/api/auth/history/${analysisId}`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      // Remove from local state
      setHistory(history.filter(item => item.analysis_id !== analysisId));
    } catch (err) {
      console.error('Error deleting analysis:', err);
      alert('Failed to delete analysis');
    }
  };

  const handleViewDetails = async (analysisId) => {
    try {
      const response = await axios.get(`${API_URL}/api/auth/history/${analysisId}`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      
      // If full_result is available (saved analysis), pass it to the parent
      if (response.data.full_result) {
        setSelectedAnalysis(response.data.full_result);
        if (onViewDetails) {
          onViewDetails(response.data.full_result);
        }
      } else {
        // Legacy analysis without full result
        setSelectedAnalysis(response.data.analysis);
        if (onViewDetails) {
          onViewDetails(response.data.analysis);
        }
      }
    } catch (err) {
      console.error('Error fetching analysis details:', err);
      alert('Failed to load analysis details');
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getPitchTypeColor = (pitchType) => {
    const colors = {
      'batting_friendly': '#4CAF50',
      'bowling_friendly': '#FF5722',
      'spin_friendly': '#FF9800',
      'seam_friendly': '#2196F3'
    };
    return colors[pitchType] || '#9E9E9E';
  };

  if (!authToken) {
    return (
      <div className="history-section">
        <div className="history-empty">
          <p>Please login to view your analysis history</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="history-section">
        <div className="history-loading">
          <div className="spinner"></div>
          <p>Loading your analysis history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="history-section">
        <div className="history-error">
          <p>{error}</p>
          <button onClick={fetchHistory} className="btn-retry">Retry</button>
        </div>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="history-section">
        <div className="history-empty">
          <ImageIcon size={48} />
          <p>No analysis history yet</p>
          <p className="history-empty-subtitle">Your pitch analyses will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-section">
      <div className="history-header">
        <h2>Analysis History</h2>
        <button onClick={fetchHistory} className="btn-refresh">
          Refresh
        </button>
      </div>

      <div className="history-grid">
        {history.map((item) => (
          <div key={item.analysis_id} className="history-card">
            {/* Image Preview */}
            {item.image_data && (
              <div className="history-card-image">
                <img 
                  src={item.image_data} 
                  alt={item.image_name}
                  loading="lazy"
                />
                <div className="image-overlay">
                  <button
                    onClick={() => handleViewDetails(item.analysis_id)}
                    className="view-full-btn"
                  >
                    <Eye size={20} />
                    View Full Analysis
                  </button>
                </div>
              </div>
            )}
            
            <div className="history-card-header">
              <div className="history-card-title">
                <ImageIcon size={16} />
                <span className="history-image-name">{item.image_name}</span>
              </div>
              <div className="history-card-actions">
                <button
                  onClick={() => handleViewDetails(item.analysis_id)}
                  className="btn-icon"
                  title="View Details"
                >
                  <Eye size={18} />
                </button>
                <button
                  onClick={() => handleDelete(item.analysis_id)}
                  className="btn-icon btn-delete"
                  title="Delete"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>

            <div className="history-card-body">
              <div className="history-pitch-type">
                <div 
                  className="pitch-type-badge"
                  style={{ backgroundColor: getPitchTypeColor(item.pitch_type) }}
                >
                  {item.pitch_type?.replace(/_/g, ' ').toUpperCase()}
                </div>
                <span className="confidence-text">
                  {item.confidence?.toFixed(1)}% confident
                </span>
              </div>

              {item.match_info && (
                <div className="history-info-row">
                  <span className="info-label">Match:</span>
                  <span className="info-value">{item.match_info.format_description}</span>
                </div>
              )}

              {item.location && (
                <div className="history-info-row">
                  <MapPin size={14} />
                  <span className="info-value">{item.location}</span>
                  {item.weather_included && (
                    <CloudSun size={14} className="weather-icon" />
                  )}
                </div>
              )}

              <div className="history-card-footer">
                <div className="history-timestamp">
                  <Clock size={14} />
                  <span>{formatDate(item.saved_at || item.created_at)}</span>
                </div>
                {item.processing_time && (
                  <span className="processing-time">
                    {item.processing_time.toFixed(2)}s
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistorySection;
