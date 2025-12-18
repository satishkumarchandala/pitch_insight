import React, { useState } from 'react'
import { ArrowLeft, Clock, Target, Activity, Droplets, Thermometer, Wind, CloudRain, Download, FileText } from 'lucide-react'
import SpecialistAnalysis from './SpecialistAnalysis'
import './ResultsSection.css'

const API_URL = 'http://localhost:8000'

function ResultsSection({ result, onReset, uploadedImage, weatherData }) {
  const [downloadingReport, setDownloadingReport] = useState(false)
  
  const { 
    final_classification, 
    features, 
    weather, 
    match_strategy,
    processing_time,
    ml_classification,
    specialist_analysis
  } = result

  const handleDownloadReport = async () => {
    if (!uploadedImage) {
      alert('Original image not available for report generation')
      return
    }

    setDownloadingReport(true)
    
    try {
      const formData = new FormData()
      formData.append('image', uploadedImage)
      
      if (weatherData?.latitude) {
        formData.append('latitude', weatherData.latitude)
      }
      if (weatherData?.longitude) {
        formData.append('longitude', weatherData.longitude)
      }
      if (weatherData?.city) {
        formData.append('city', weatherData.city)
      }

      const response = await fetch(API_ENDPOINTS.generateReport, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to generate report')
      }

      // Download the report
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `pitch_analysis_report_${new Date().getTime()}.png`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
    } catch (error) {
      console.error('Report download error:', error)
      alert('Failed to generate report. Please try again.')
    } finally {
      setDownloadingReport(false)
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
        
        <div className="header-actions">
          <button 
            className="btn btn-report" 
            onClick={handleDownloadReport}
            disabled={downloadingReport}
          >
            {downloadingReport ? (
              <>
                <Activity size={20} className="spin-icon" />
                Generating...
              </>
            ) : (
              <>
                <FileText size={20} />
                Download Full Report
              </>
            )}
          </button>
          
          <div className="processing-info">
            <Clock size={16} />
            <span>{processing_time?.toFixed(2)}s</span>
          </div>
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
              {final_classification.prediction.replace('_', ' ').toUpperCase()}
            </h2>
            <p className="confidence-text">
              {final_classification.confidence.toFixed(1)}% Confidence
            </p>
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
                  <span className="feature-value">{features.cracks.count}</span>
                  <span className="feature-level">{features.cracks.severity}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon moisture">💧</div>
                <div className="feature-content">
                  <span className="feature-label">Moisture</span>
                  <span className="feature-value">{features.moisture.score.toFixed(0)}/100</span>
                  <span className="feature-level">{features.moisture.level}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon color">🎨</div>
                <div className="feature-content">
                  <span className="feature-label">Color</span>
                  <span className="feature-value">{features.color.type}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon texture">🔲</div>
                <div className="feature-content">
                  <span className="feature-label">Texture</span>
                  <span className="feature-value">{features.texture.type}</span>
                </div>
              </div>

              <div className="feature-item">
                <div className="feature-icon brightness">☀️</div>
                <div className="feature-content">
                  <span className="feature-label">Brightness</span>
                  <span className="feature-value">{features.brightness.level}</span>
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
                <span className="weather-value">{weather.wind_speed} m/s</span>
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

            {match_strategy?.weather_impact && (
              <div className="weather-impact">
                <h4>Weather Impact Analysis</h4>
                <div className={`impact-severity impact-${match_strategy.weather_impact.severity}`}>
                  {match_strategy.weather_impact.severity.toUpperCase()} Impact
                </div>
                <ul className="impact-list">
                  {match_strategy.weather_impact.impacts.map((impact, i) => (
                    <li key={i}>{impact}</li>
                  ))}
                </ul>
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
                {match_strategy.batting_strategy.slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>

            <div className="strategy-section">
              <h4>🎳 Bowling Strategy</h4>
              <ul>
                {match_strategy.bowling_strategy.slice(0, 3).map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>

            <div className="strategy-section">
              <h4>👥 Team Composition</h4>
              <ul>
                {match_strategy.team_composition.slice(0, 3).map((tip, i) => (
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

      {/* Specialist Analysis Section */}
      {specialist_analysis && (
        <SpecialistAnalysis 
          specialistData={specialist_analysis} 
          features={features}
        />
      )}
    </div>
  )
}

export default ResultsSection
