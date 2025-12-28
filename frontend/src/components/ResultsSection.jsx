import React from 'react'
import { ArrowLeft, Clock, Target, Activity, Droplets, Thermometer, Wind, CloudRain } from 'lucide-react'
import './ResultsSection.css'

function ResultsSection({ result, onReset }) {
  // Add safety checks for all destructured values
  const { 
    final_classification = {}, 
    features = {}, 
    weather = null, 
    match_strategy = {},
    match_info = null,
    processing_time = 0,
    ml_classification = {}
  } = result || {}

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
        
        <div className="processing-info">
          <Clock size={16} />
          <span>{processing_time?.toFixed(2)}s</span>
        </div>
      </div>

      {/* Main Prediction Card */}
      <div className="prediction-card slide-in-left" style={{ '--delay': '0.1s' }}>
        <div className="prediction-header">
          <div className="pitch-icon" style={{ background: getPitchTypeColor(final_classification.prediction) }}>
            {getPitchTypeIcon(final_classification.prediction)}
          </div>
          <div>
            <h2 className="pitch-type">
              {final_classification.prediction?.replace('_', ' ').toUpperCase() || 'Unknown'}
            </h2>
            <p className="confidence-text">
              {final_classification.confidence?.toFixed(1) || 0}% Confidence
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
              width: `${final_classification.confidence || 0}%`,
              background: getPitchTypeColor(final_classification.prediction)
            }}
          />
        </div>

        {final_classification.adjustments && final_classification.adjustments.length > 0 && (
          <div className="adjustments">
            <h4>🔬 Feature-Based Adjustments</h4>
            {final_classification.reasons.map((reason, i) => (
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
            {Object.entries(final_classification.probabilities || {}).map(([type, prob]) => (
              <div key={type} className="probability-item">
                <div className="prob-header">
                  <span className="prob-type">{type?.replace('_', ' ') || type}</span>
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
                    {features.grass_coverage?.percentage?.toFixed(1) || 0}%
                  </span>
                  <span className="feature-level">{features.grass_coverage?.level || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon cracks">⚡</div>
                <div className="feature-content">
                  <span className="feature-label">Cracks</span>
                  <span className="feature-value">{features.cracks?.count || 0}</span>
                  <span className="feature-level">{features.cracks?.severity || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon moisture">💧</div>
                <div className="feature-content">
                  <span className="feature-label">Moisture</span>
                  <span className="feature-value">{features.moisture?.score?.toFixed(0) || 0}/100</span>
                  <span className="feature-level">{features.moisture?.level || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon color">🎨</div>
                <div className="feature-content">
                  <span className="feature-label">Color</span>
                  <span className="feature-value">{features.color?.type || 'Unknown'}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon texture">🔲</div>
                <div className="feature-content">
                  <span className="feature-label">Texture</span>
                  <span className="feature-value">{features.texture?.type || 'Unknown'}</span>
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
        {weather && weather.current && (
          <div className="card slide-in-left" style={{ '--delay': '0.3s' }}>
            <h3>🌤️ Weather Conditions</h3>
            <div className="weather-location">
              <span className="location-icon">📍</span>
              <span className="location-name">{weather.current.location}</span>
            </div>
            <div className="weather-grid">
              <div className="weather-item">
                <Thermometer className="weather-icon" />
                <span className="weather-label">Temperature</span>
                <span className="weather-value">{weather.current.temperature}°C</span>
              </div>
              <div className="weather-item">
                <Droplets className="weather-icon" />
                <span className="weather-label">Humidity</span>
                <span className="weather-value">{weather.current.humidity}%</span>
              </div>
              <div className="weather-item">
                <Wind className="weather-icon" />
                <span className="weather-label">Wind Speed</span>
                <span className="weather-value">{weather.current.wind_speed} m/s</span>
              </div>
              <div className="weather-item">
                <CloudRain className="weather-icon" />
                <span className="weather-label">Rainfall</span>
                <span className="weather-value">{weather.current.rainfall} mm</span>
              </div>
            </div>
            <div className="weather-conditions">
              <span className="conditions-badge">{weather.current.conditions}</span>
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
              </ul>
            </div>

            <div className="strategy-section">
              <h4>🎳 Bowling Strategy</h4>
              <ul>
                {(match_strategy.bowling_strategy || []).slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>

            <div className="strategy-section">
              <h4>👥 Team Composition</h4>
              <ul>
                {(match_strategy.team_composition || []).slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
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
