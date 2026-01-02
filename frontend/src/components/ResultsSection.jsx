import React, { useState } from 'react'
import { ArrowLeft, Clock, Target, Activity, Droplets, Thermometer, Wind, CloudRain, Save, Check } from 'lucide-react'
import WeatherForecastDisplay from './WeatherForecastDisplay'
import axios from 'axios'
import './ResultsSection.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function ResultsSection({ result, onReset, authToken, uploadedImage }) {
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  
  const { 
    final_classification, 
    features, 
    weather,
    weather_forecast,
    match_strategy,
    match_info,
    processing_time,
    ml_classification,
    image_data // This will be present when viewing from history
  } = result

  const handleSaveAnalysis = async () => {
    if (!authToken) {
      alert('Please login to save analysis')
      return
    }

    setSaving(true)
    try {
      // Convert uploaded image to base64 if available
      let imageData = null
      
      // If result already has image_data (from history), use it
      if (image_data) {
        imageData = image_data
      } else if (uploadedImage) {
        // If uploadedImage is already base64, use it
        if (typeof uploadedImage === 'string' && uploadedImage.startsWith('data:image')) {
          imageData = uploadedImage
        } else if (uploadedImage instanceof File) {
          // Convert File to base64
          imageData = await new Promise((resolve) => {
            const reader = new FileReader()
            reader.onloadend = () => resolve(reader.result)
            reader.readAsDataURL(uploadedImage)
          })
        }
      }

      // Prepare analysis data for saving
      const analysisData = {
        final_classification,
        features,
        weather,
        weather_forecast,
        match_strategy,
        match_info,
        processing_time,
        ml_classification,
        image_name: uploadedImage?.name || 'pitch_analysis.jpg',
        image_data: imageData,
        timestamp: new Date().toISOString()
      }

      const response = await axios.post(
        `${API_URL}/api/save-analysis`,
        analysisData,
        {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (response.data.success) {
        setSaved(true)
        setTimeout(() => setSaved(false), 3000) // Reset after 3 seconds
      }
    } catch (error) {
      console.error('Error saving analysis:', error)
      alert(error.response?.data?.detail || 'Failed to save analysis. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const getPitchTypeColor = (type) => {
    const colors = {
      'batting_friendly': '#10b981',
      'bowling_friendly': '#3b82f6',
      'spin_friendly': '#f59e0b',
      'seam_friendly': '#ef4444'
    }
    return colors[type] || '#6b7280'
  }

  const getPitchTypeIcon = (type) => {
    const icons = {
      'batting_friendly': '🏏',
      'bowling_friendly': '🎯',
      'spin_friendly': '🌀',
      'seam_friendly': '⚡'
    }
    return icons[type] || '🏏'
  }

  return (
    <div className="results-section fade-in">
      <div className="results-header">
        <button className="btn btn-secondary" onClick={onReset}>
          <ArrowLeft size={20} />
          Analyze Another
        </button>
        
        <div className="header-right">
          <div className="processing-info">
            <Clock size={16} />
            <span>{processing_time?.toFixed(2)}s</span>
          </div>
          
          {authToken && (
            <button 
              className={`btn btn-save ${saved ? 'saved' : ''}`}
              onClick={handleSaveAnalysis}
              disabled={saving || saved}
            >
              {saved ? (
                <>
                  <Check size={18} />
                  Saved
                </>
              ) : (
                <>
                  <Save size={18} />
                  {saving ? 'Saving...' : 'Save Analysis'}
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Display uploaded image if available */}
      {(uploadedImage || image_data) && (
        <div className="uploaded-image-section slide-in-left" style={{ '--delay': '0.05s' }}>
          <h3>📸 Analyzed Pitch Image</h3>
          <div className="uploaded-image-container">
            <img 
              src={image_data || (typeof uploadedImage === 'string' ? uploadedImage : URL.createObjectURL(uploadedImage))}
              alt="Analyzed pitch" 
              className="uploaded-image"
            />
          </div>
        </div>
      )}

      {/* Main Prediction Card */}
      <div className="prediction-card slide-in-left" style={{ '--delay': '0.1s' }}>
        <div className="prediction-header">
          <div className="pitch-icon" style={{ background: getPitchTypeColor(final_classification.prediction) }}>
            {getPitchTypeIcon(final_classification.prediction)}
          </div>
          <div>
            <h2 className="pitch-type">
              {final_classification.prediction.replace('_', ' ').toUpperCase()}
            </h2>
            <p className="confidence-text">
              {final_classification.confidence.toFixed(1)}% Confidence
            </p>
            {match_info && (
              <p className="match-format">
                🏏 {match_info.format_description}
              </p>
            )}
          </div>
        </div>

        <div className="confidence-bar">
          <div 
            className="confidence-fill"
            style={{ 
              width: `${final_classification.confidence}%`,
              background: getPitchTypeColor(final_classification.prediction)
            }}
          />
        </div>

        {final_classification?.adjustments && final_classification.adjustments.length > 0 && (
          <div className="adjustments">
            <h4>🔬 Feature-Based Adjustments</h4>
            {final_classification.reasons?.map((reason, i) => (
              <div key={i} className="adjustment-item">
                <span className="adjustment-badge">{final_classification.adjustments[i]}</span>
                <span>{reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="results-grid">
        {/* Probabilities */}
        <div className="card slide-in-left" style={{ '--delay': '0.2s' }}>
          <h3>📊 Classification Probabilities</h3>
          <div className="probabilities">
            {Object.entries(final_classification.probabilities).map(([type, prob]) => (
              <div key={type} className="probability-item">
                <div className="prob-header">
                  <span className="prob-type">{type.replace('_', ' ')}</span>
                  <span className="prob-value">{(prob).toFixed(1)}%</span>
                </div>
                <div className="prob-bar">
                  <div 
                    className="prob-fill"
                    style={{ 
                      width: `${prob}%`,
                      background: getPitchTypeColor(type)
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Features */}
        {features && (
          <div className="card slide-in-right" style={{ '--delay': '0.2s' }}>
            <h3>🔬 Pitch Features</h3>
            <div className="features-grid">
              <div className="feature-item">
                <div className="feature-icon grass">🌱</div>
                <div className="feature-content">
                  <span className="feature-label">Grass Coverage</span>
                  <span className="feature-value">
                    {features.grass_coverage.percentage.toFixed(1)}%
                  </span>
                  <span className="feature-level">{features.grass_coverage.level}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon cracks">⚡</div>
                <div className="feature-content">
                  <span className="feature-label">Cracks</span>
                  <span className="feature-value">{features.crack_analysis?.num_cracks || 0}</span>
                  <span className="feature-level">{features.crack_analysis?.severity || 'None'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon moisture">💧</div>
                <div className="feature-content">
                  <span className="feature-label">Moisture</span>
                  <span className="feature-value">{features.moisture_level?.score?.toFixed(0) || 0}/100</span>
                  <span className="feature-level">{features.moisture_level?.level || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon color">🎨</div>
                <div className="feature-content">
                  <span className="feature-label">Color</span>
                  <span className="feature-value">{features.color_profile?.color_type || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon texture">🔲</div>
                <div className="feature-content">
                  <span className="feature-label">Texture</span>
                  <span className="feature-value">{features.texture_analysis?.type || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon brightness">☀️</div>
                <div className="feature-content">
                  <span className="feature-label">Brightness</span>
                  <span className="feature-value">{features.brightness?.level || 'Unknown'}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Weather */}
        {weather && (
          <div className="card slide-in-left" style={{ '--delay': '0.3s' }}>
            <h3>🌤️ Weather Conditions</h3>
            <div className="weather-location">
              <span className="location-icon">📍</span>
              <span className="location-name">{weather.location}</span>
            </div>
            <div className="weather-grid">
              <div className="weather-item">
                <Thermometer className="weather-icon" />
                <span className="weather-label">Temperature</span>
                <span className="weather-value">{weather.temperature}°C</span>
              </div>
              <div className="weather-item">
                <Droplets className="weather-icon" />
                <span className="weather-label">Humidity</span>
                <span className="weather-value">{weather.humidity}%</span>
              </div>
              <div className="weather-item">
                <Wind className="weather-icon" />
                <span className="weather-label">Wind Speed</span>
                <span className="weather-value">{weather.wind_speed} km/h</span>
              </div>
              <div className="weather-item">
                <CloudRain className="weather-icon" />
                <span className="weather-label">Rainfall</span>
                <span className="weather-value">{weather.rainfall} mm</span>
              </div>
            </div>
            <div className="weather-conditions">
              <span className="conditions-badge">{weather.conditions}</span>
            </div>

            {weather.impact && (
              <div className="weather-impact">
                <h4>⚡ Weather Impact Analysis</h4>
                <div className="impact-scores">
                  <div className="impact-score-item">
                    <span className="score-label">Swing Potential</span>
                    <div className="score-bar">
                      <div 
                        className="score-fill swing"
                        style={{ width: `${weather.impact.swing_score}%` }}
                      />
                    </div>
                    <span className="score-value">{weather.impact.swing_score}/100</span>
                  </div>
                  <div className="impact-score-item">
                    <span className="score-label">Spin Assistance</span>
                    <div className="score-bar">
                      <div 
                        className="score-fill spin"
                        style={{ width: `${weather.impact.spin_score}%` }}
                      />
                    </div>
                    <span className="score-value">{weather.impact.spin_score}/100</span>
                  </div>
                </div>
                <div className="impact-details">
                  <p><strong>Pitch Drying:</strong> {weather.impact.pitch_drying_rate}</p>
                  <p><strong>Dew Likelihood:</strong> {weather.impact.dew_likelihood}</p>
                </div>
                {weather.impact.key_factors && weather.impact.key_factors.length > 0 && (
                  <div className="key-factors-weather">
                    <h5>🔑 Key Factors:</h5>
                    <ul>
                      {weather.impact.key_factors.map((factor, i) => (
                        <li key={i}>{factor}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Comprehensive Weather Forecast */}
        {weather_forecast && (
          <div className="card slide-in-left" style={{ '--delay': '0.35s' }}>
            <WeatherForecastDisplay 
              forecast={weather_forecast} 
              matchFormat={match_info?.format || 'odi'} 
            />
          </div>
        )}

        {/* Match Strategy */}
        {match_strategy && (
          <div className="card slide-in-right" style={{ '--delay': '0.3s' }}>
            <h3>🏏 Match Strategy</h3>
            
            <div className="toss-decision">
              <div className="toss-icon">🎲</div>
              <div>
                <h4>Toss Decision</h4>
                <p>{match_strategy.toss_decision}</p>
              </div>
            </div>

            <div className="strategy-section">
              <h4>🏏 Batting Strategy</h4>
              <ul>
                {(match_strategy.batting_strategy || []).slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
                {(!match_strategy.batting_strategy || match_strategy.batting_strategy.length === 0) && (
                  <li>Strategy generation coming soon...</li>
                )}
              </ul>
            </div>

            <div className="strategy-section">
              <h4>🎳 Bowling Strategy</h4>
              <ul>
                {(match_strategy.bowling_strategy || []).slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
                {(!match_strategy.bowling_strategy || match_strategy.bowling_strategy.length === 0) && (
                  <li>Strategy generation coming soon...</li>
                )}
              </ul>
            </div>

            <div className="strategy-section">
              <h4>👥 Team Composition</h4>
              <ul>
                {(match_strategy.team_composition || []).slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
                {(!match_strategy.team_composition || match_strategy.team_composition.length === 0) && (
                  <li>Strategy generation coming soon...</li>
                )}
              </ul>
            </div>

            {match_strategy.key_factors && match_strategy.key_factors.length > 0 && (
              <div className="key-factors">
                <h4>⚠️ Key Factors</h4>
                {match_strategy.key_factors.map((factor, i) => (
                  <div key={i} className="factor-badge">{factor}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ResultsSection
