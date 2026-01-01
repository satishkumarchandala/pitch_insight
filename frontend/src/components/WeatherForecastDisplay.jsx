import React, { useState } from 'react'
import { 
  CloudRain, Thermometer, Droplets, Wind, Sun, Cloud, 
  AlertTriangle, TrendingUp, Target, ChevronDown, ChevronUp 
} from 'lucide-react'
import './WeatherForecastDisplay.css'

function WeatherForecastDisplay({ forecast, matchFormat }) {
  const [expandedDay, setExpandedDay] = useState(null)
  const [expandedInnings, setExpandedInnings] = useState(null)

  if (!forecast) return null

  const { 
    current, 
    historical, 
    daily_forecasts, 
    innings_forecasts,
    pitch_behavior_trend,
    key_risks,
    phase_wise_advantage,
    match_condition_summary,
    recommendations 
  } = forecast

  const toggleDay = (dayNum) => {
    setExpandedDay(expandedDay === dayNum ? null : dayNum)
  }

  const toggleInnings = (inningsNum) => {
    setExpandedInnings(expandedInnings === inningsNum ? null : inningsNum)
  }

  const getAdvantageColor = (advantage) => {
    if (advantage === 'Bowlers') return '#3b82f6'
    if (advantage === 'Batters') return '#10b981'
    return '#f59e0b'
  }

  const getSwingColor = (level) => {
    if (level === 'Very High' || level === 'High') return '#ef4444'
    if (level === 'Medium') return '#f59e0b'
    return '#10b981'
  }

  return (
    <div className="weather-forecast-container">
      <div className="forecast-header">
        <h2>🌦️ Comprehensive Weather Analysis</h2>
        <p className="forecast-location">{forecast.location}</p>
      </div>

      {/* Current Weather */}
      <div className="current-weather-card">
        <h3>Current Conditions</h3>
        <div className="current-grid">
          <div className="current-item">
            <Thermometer size={24} />
            <div>
              <span className="current-label">Temperature</span>
              <span className="current-value">{current.temperature}°C</span>
            </div>
          </div>
          <div className="current-item">
            <Droplets size={24} />
            <div>
              <span className="current-label">Humidity</span>
              <span className="current-value">{current.humidity}%</span>
            </div>
          </div>
          <div className="current-item">
            <Wind size={24} />
            <div>
              <span className="current-label">Wind</span>
              <span className="current-value">{current.wind_speed} kph</span>
            </div>
          </div>
          <div className="current-item">
            <Cloud size={24} />
            <div>
              <span className="current-label">Cloud Cover</span>
              <span className="current-value">{current.cloud_cover}%</span>
            </div>
          </div>
        </div>
        <p className="current-conditions">{current.conditions}</p>
      </div>

      {/* Historical Context */}
      {historical && (
        <div className="historical-card">
          <h3>📊 Historical Context (Past 3 Days)</h3>
          <div className="historical-grid">
            <div className="historical-item">
              <span className="hist-label">Rainfall (72h)</span>
              <span className="hist-value">{historical.rainfall_72h}mm</span>
            </div>
            <div className="historical-item">
              <span className="hist-label">Avg Temperature</span>
              <span className="hist-value">{historical.avg_temp_3d}°C</span>
            </div>
            <div className="historical-item">
              <span className="hist-label">Conditions</span>
              <span className="hist-value">{historical.recent_conditions}</span>
            </div>
          </div>
          <div className="historical-inference">
            <strong>Pitch Inference:</strong>
            <p>{historical.pitch_moisture_inference}</p>
            <p>{historical.interpretation}</p>
          </div>
        </div>
      )}

      {/* Test Match - Daily Forecasts with Sessions */}
      {daily_forecasts && daily_forecasts.length > 0 && (
        <div className="test-match-forecast">
          <h3>📅 5-Day Test Match Forecast</h3>
          {daily_forecasts.map((day) => (
            <div key={day.day_number} className="day-forecast-card">
              <div 
                className="day-header"
                onClick={() => toggleDay(day.day_number)}
              >
                <div className="day-info">
                  <h4>Day {day.day_number}</h4>
                  <span className="day-date">{new Date(day.date).toLocaleDateString()}</span>
                  <span 
                    className="day-advantage"
                    style={{ background: getAdvantageColor(day.overall_advantage) }}
                  >
                    {day.overall_advantage}
                  </span>
                </div>
                <div className="day-summary">
                  <span>{day.min_temp}°C - {day.max_temp}°C</span>
                  <span>{day.conditions_summary}</span>
                  {expandedDay === day.day_number ? <ChevronUp /> : <ChevronDown />}
                </div>
              </div>

              {expandedDay === day.day_number && (
                <div className="day-details">
                  <div className="pitch-condition">
                    <p><strong>Deterioration:</strong> {day.pitch_deterioration_rate}</p>
                    <p><strong>Cracks:</strong> {day.crack_development}</p>
                    <p><strong>Outfield:</strong> {day.outfield_condition}</p>
                  </div>

                  {/* Sessions */}
                  <div className="sessions-container">
                    <h5>Session Breakdown</h5>
                    {day.sessions.map((session, idx) => (
                      <div key={idx} className="session-card">
                        <div className="session-header">
                          <span className="session-name">{session.session_name}</span>
                          <span className="session-time">{session.time_range}</span>
                        </div>
                        
                        <div className="session-weather">
                          <span>🌡️ {session.avg_temperature}°C</span>
                          <span>💧 {session.avg_humidity}%</span>
                          <span>☁️ {session.avg_cloud_cover}%</span>
                          {session.chance_of_rain > 0 && (
                            <span>🌧️ {session.chance_of_rain}%</span>
                          )}
                        </div>

                        <div className="session-impact">
                          <div className="impact-item">
                            <span>Swing Potential:</span>
                            <span 
                              className="impact-badge"
                              style={{ background: getSwingColor(session.swing_potential) }}
                            >
                              {session.swing_potential}
                            </span>
                          </div>
                          <div className="impact-item">
                            <span>Seam Movement:</span>
                            <span className="impact-value">{session.seam_movement}</span>
                          </div>
                          <div className="impact-item">
                            <span>Spin Assistance:</span>
                            <span className="impact-value">{session.spin_assistance}</span>
                          </div>
                        </div>

                        <div className="advantage-bars">
                          <div className="advantage-bar">
                            <span>Bowling</span>
                            <div className="bar">
                              <div 
                                className="bar-fill bowling"
                                style={{ width: `${session.bowling_advantage}%` }}
                              />
                            </div>
                            <span>{session.bowling_advantage}%</span>
                          </div>
                          <div className="advantage-bar">
                            <span>Batting</span>
                            <div className="bar">
                              <div 
                                className="bar-fill batting"
                                style={{ width: `${session.batting_advantage}%` }}
                              />
                            </div>
                            <span>{session.batting_advantage}%</span>
                          </div>
                        </div>

                        {session.key_factors && session.key_factors.length > 0 && (
                          <div className="session-factors">
                            {session.key_factors.map((factor, i) => (
                              <span key={i} className="factor-tag">{factor}</span>
                            ))}
                          </div>
                        )}

                        <p className="session-strategy">
                          <strong>Strategy:</strong> {session.recommended_strategy}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Limited Overs - Innings Forecasts */}
      {innings_forecasts && innings_forecasts.length > 0 && (
        <div className="innings-forecast">
          <h3>⚡ Innings-Wise Forecast</h3>
          {innings_forecasts.map((innings, idx) => (
            <div key={idx} className="innings-card">
              <div 
                className="innings-header"
                onClick={() => toggleInnings(idx)}
              >
                <h4>{innings.session_name}</h4>
                <span className="innings-time">{innings.time_range}</span>
                {expandedInnings === idx ? <ChevronUp /> : <ChevronDown />}
              </div>

              {expandedInnings === idx && (
                <div className="innings-details">
                  <div className="innings-weather">
                    <div className="weather-item">
                      <Thermometer size={20} />
                      <span>{innings.avg_temperature}°C</span>
                    </div>
                    <div className="weather-item">
                      <Droplets size={20} />
                      <span>{innings.avg_humidity}%</span>
                    </div>
                    <div className="weather-item">
                      <Wind size={20} />
                      <span>{innings.avg_wind_speed} kph</span>
                    </div>
                    <div className="weather-item">
                      <Cloud size={20} />
                      <span>{innings.avg_cloud_cover}%</span>
                    </div>
                  </div>

                  <div className="innings-impact">
                    <div className="impact-grid">
                      <div className="impact-box">
                        <label>Swing Potential</label>
                        <span 
                          className="impact-badge large"
                          style={{ background: getSwingColor(innings.swing_potential) }}
                        >
                          {innings.swing_potential}
                        </span>
                      </div>
                      <div className="impact-box">
                        <label>Seam Movement</label>
                        <span className="impact-value large">{innings.seam_movement}</span>
                      </div>
                      <div className="impact-box">
                        <label>Spin Assistance</label>
                        <span className="impact-value large">{innings.spin_assistance}</span>
                      </div>
                      <div className="impact-box">
                        <label>Dew Likelihood</label>
                        <span className="impact-value large">{innings.dew_likelihood}</span>
                      </div>
                    </div>
                  </div>

                  <div className="innings-advantage">
                    <div className="advantage-bar">
                      <span>Bowling</span>
                      <div className="bar">
                        <div 
                          className="bar-fill bowling"
                          style={{ width: `${innings.bowling_advantage}%` }}
                        />
                      </div>
                      <span>{innings.bowling_advantage}%</span>
                    </div>
                    <div className="advantage-bar">
                      <span>Batting</span>
                      <div className="bar">
                        <div 
                          className="bar-fill batting"
                          style={{ width: `${innings.batting_advantage}%` }}
                        />
                      </div>
                      <span>{innings.batting_advantage}%</span>
                    </div>
                  </div>

                  {innings.key_factors && innings.key_factors.length > 0 && (
                    <div className="innings-factors">
                      <strong>Key Factors:</strong>
                      {innings.key_factors.map((factor, i) => (
                        <span key={i} className="factor-tag">{factor}</span>
                      ))}
                    </div>
                  )}

                  <div className="innings-strategy">
                    <Target size={20} />
                    <p>{innings.recommended_strategy}</p>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Match Summary */}
      <div className="match-summary-card">
        <h3>📋 Match Condition Summary</h3>
        <p className="summary-text">{match_condition_summary}</p>
        
        {pitch_behavior_trend && (
          <div className="pitch-trend">
            <TrendingUp size={20} />
            <p><strong>Pitch Behavior:</strong> {pitch_behavior_trend}</p>
          </div>
        )}

        {key_risks && key_risks.length > 0 && (
          <div className="key-risks">
            <h4><AlertTriangle size={20} /> Key Risks</h4>
            <ul>
              {key_risks.map((risk, idx) => (
                <li key={idx}>{risk}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations && recommendations.length > 0 && (
          <div className="recommendations">
            <h4>💡 Recommendations</h4>
            <ul>
              {recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {phase_wise_advantage && Object.keys(phase_wise_advantage).length > 0 && (
          <div className="phase-advantages">
            <h4>⚖️ Phase-Wise Advantage</h4>
            <div className="phase-grid">
              {Object.entries(phase_wise_advantage).map(([phase, advantage]) => (
                <div key={phase} className="phase-item">
                  <span className="phase-name">{phase}</span>
                  <span 
                    className="phase-badge"
                    style={{ background: getAdvantageColor(advantage) }}
                  >
                    {advantage}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default WeatherForecastDisplay
