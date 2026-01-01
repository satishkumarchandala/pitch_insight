import React, { useState, useRef } from 'react'
import { Upload, MapPin, Cloud, Activity } from 'lucide-react'
import axios from 'axios'
import './UploadSection.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function UploadSection({ onAnalysisComplete, onError, loading, setLoading, token }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [includeWeather, setIncludeWeather] = useState(false)
  const [useForecast, setUseForecast] = useState(false)
  const [matchType, setMatchType] = useState('odi')
  const [customOvers, setCustomOvers] = useState('')
  const [matchStartTime, setMatchStartTime] = useState('')
  const [location, setLocation] = useState({
    latitude: '',
    longitude: '',
    city: ''
  })
  const [gettingLocation, setGettingLocation] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      onError('Geolocation is not supported by your browser')
      return
    }

    setGettingLocation(true)

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude.toFixed(4)
        const lon = position.coords.longitude.toFixed(4)
        
        setLocation({
          latitude: lat,
          longitude: lon,
          city: location.city || 'Current Location'
        })
        
        setGettingLocation(false)
        
        // Optional: Reverse geocode to get city name
        try {
          const response = await fetch(
            `https://api.weatherapi.com/v1/current.json?key=b2ad62736cbd4e52aaa133601252712&q=${lat},${lon}`
          )
          const data = await response.json()
          if (data.location) {
            setLocation({
              latitude: lat,
              longitude: lon,
              city: `${data.location.name}, ${data.location.country}`
            })
          }
        } catch (err) {
          console.log('Could not fetch city name:', err)
          // Keep the location even if city name fetch fails
        }
      },
      (error) => {
        setGettingLocation(false)
        let errorMessage = 'Unable to get your location'
        
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = 'Location permission denied. Please enable location access in your browser settings.'
            break
          case error.POSITION_UNAVAILABLE:
            errorMessage = 'Location information is currently unavailable. Please try again or enter location manually.'
            break
          case error.TIMEOUT:
            errorMessage = 'Location request timed out. Please try again or enter location manually.'
            break
          default:
            errorMessage = 'An error occurred while getting your location. Please enter location manually.'
        }
        
        onError(errorMessage)
      },
      {
        enableHighAccuracy: false, // Changed to false for faster response
        timeout: 30000, // Increased to 30 seconds
        maximumAge: 60000 // Allow cached position up to 1 minute old
      }
    )
  }

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(file)
    } else {
      onError('Please select a valid image file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedImage) {
      onError('Please select an image first')
      return
    }

    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('image', selectedImage)
      formData.append('include_weather', includeWeather)
      formData.append('use_forecast', useForecast)
      formData.append('match_type', matchType)

      if (matchType === 'custom' && customOvers) {
        formData.append('custom_overs', customOvers)
      }
      
      if (matchStartTime) {
        formData.append('match_start_time', matchStartTime)
      }

      if (includeWeather) {
        if (location.latitude) formData.append('latitude', location.latitude)
        if (location.longitude) formData.append('longitude', location.longitude)
        if (location.city) formData.append('city', location.city)
      }

      const headers = {
        'Content-Type': 'multipart/form-data'
      }
      
      // Add authorization header if user is logged in
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers
      })

      onAnalysisComplete(response.data)
    } catch (error) {
      console.error('Analysis error:', error)
      onError(error.response?.data?.detail || 'Failed to analyze pitch. Please ensure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleQuickAnalyze = async () => {
    if (!selectedImage) {
      onError('Please select an image first')
      return
    }

    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('image', selectedImage)

      const headers = {
        'Content-Type': 'multipart/form-data'
      }
      
      // Add authorization header if user is logged in
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await axios.post(`${API_URL}/api/quick-analyze`, formData, {
        headers
      })

      // Convert quick analysis to full format
      const fullResult = {
        success: response.data.success,
        analysis_id: `QUICK_${Date.now()}`,
        pitch_detection: { detected: true, confidence: 0.9 },
        features: null,
        ml_classification: {
          prediction: response.data.prediction,
          confidence: response.data.confidence,
          probabilities: response.data.probabilities
        },
        final_classification: {
          prediction: response.data.prediction,
          confidence: response.data.confidence,
          probabilities: response.data.probabilities,
          adjustments: [],
          reasons: []
        },
        weather: null,
        match_strategy: null,
        timestamp: new Date().toISOString(),
        processing_time: response.data.processing_time
      }

      onAnalysisComplete(fullResult)
    } catch (error) {
      console.error('Quick analysis error:', error)
      onError(error.response?.data?.detail || 'Failed to analyze pitch')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-section fade-in">
      <div className="upload-header">
        <h2>Analyze Cricket Pitch</h2>
        <p>Upload a pitch image for AI-powered analysis</p>
      </div>

      <div className="upload-grid">
        {/* Upload Area */}
        <div className="upload-card">
          <div 
            className={`upload-dropzone ${dragActive ? 'active' : ''} ${previewUrl ? 'has-preview' : ''}`}
            onDrop={handleDrop}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onClick={() => fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <div className="preview-overlay">
                  <Upload size={32} />
                  <p>Click or drag to change image</p>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <Upload size={48} className="upload-icon" />
                <h3>Drop pitch image here</h3>
                <p>or click to browse</p>
                <span className="upload-hint">Supports: JPG, PNG, JPEG (Max 10MB)</span>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              style={{ display: 'none' }}
            />
          </div>

          {selectedImage && (
            <div className="file-info">
              <p className="file-name">📄 {selectedImage.name}</p>
              <p className="file-size">{(selectedImage.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          )}
        </div>

        {/* Options Card */}
        <div className="options-card">
          <h3>Analysis Options</h3>

          <div className="option-group">
            <label className="option-label">
              <span className="label-text">🏏 Match Format</span>
            </label>
            <select 
              className="select-input"
              value={matchType}
              onChange={(e) => setMatchType(e.target.value)}
            >
              <option value="test">Test Match (5 days)</option>
              <option value="odi">ODI (50 overs)</option>
              <option value="t20">T20 (20 overs)</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          {matchType === 'custom' && (
            <div className="option-group fade-in">
              <input
                type="number"
                placeholder="Number of overs"
                value={customOvers}
                onChange={(e) => setCustomOvers(e.target.value)}
                className="input"
                min="1"
                max="200"
              />
            </div>
          )}

          <div className="option-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={includeWeather}
                onChange={(e) => setIncludeWeather(e.target.checked)}
              />
              <Cloud size={18} />
              <span>Include Weather Analysis</span>
            </label>
            <p className="option-hint">Get weather-adjusted match strategies</p>
          </div>

          {includeWeather && (
            <>
              <div className="option-group fade-in">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={useForecast}
                    onChange={(e) => setUseForecast(e.target.checked)}
                  />
                  <Activity size={18} />
                  <span>Comprehensive Weather Forecast</span>
                </label>
                <p className="option-hint">
                  {matchType === 'test' 
                    ? '5-day forecast with session-wise analysis' 
                    : 'Innings-wise forecast with dew analysis'}
                </p>
              </div>

              {useForecast && (matchType === 't20' || matchType === 'odi') && (
                <div className="option-group fade-in">
                  <label className="option-label">
                    <span className="label-text">⏰ Match Start Time (Optional)</span>
                  </label>
                  <input
                    type="time"
                    value={matchStartTime}
                    onChange={(e) => setMatchStartTime(e.target.value)}
                    className="input"
                    placeholder="HH:MM (e.g., 19:00)"
                  />
                  <p className="option-hint">For better dew and lighting analysis</p>
                </div>
              )}
            </>
          )}

          {includeWeather && (
            <div className="location-inputs fade-in">
              <div className="location-header">
                <button
                  type="button"
                  className="btn-location"
                  onClick={getCurrentLocation}
                  disabled={gettingLocation || loading}
                >
                  {gettingLocation ? (
                    <>
                      <div className="spinner-small" />
                      Getting Location...
                    </>
                  ) : (
                    <>
                      <MapPin size={18} />
                      Use Current Location
                    </>
                  )}
                </button>
                <span className="location-divider">or enter manually</span>
              </div>

              <div className="input-group">
                <MapPin size={18} />
                <input
                  type="text"
                  placeholder="City (e.g., Mumbai)"
                  value={location.city}
                  onChange={(e) => setLocation({ ...location, city: e.target.value })}
                  className="input"
                />
              </div>

              <div className="input-row">
                <input
                  type="number"
                  placeholder="Latitude"
                  value={location.latitude}
                  onChange={(e) => setLocation({ ...location, latitude: e.target.value })}
                  className="input"
                  step="0.0001"
                />
                <input
                  type="number"
                  placeholder="Longitude"
                  value={location.longitude}
                  onChange={(e) => setLocation({ ...location, longitude: e.target.value })}
                  className="input"
                  step="0.0001"
                />
              </div>

              <p className="location-hint">
                💡 Tip: Search "{location.city || 'your city'} coordinates" on Google
              </p>
            </div>
          )}

          <div className="action-buttons">
            <button
              className="btn btn-primary btn-full"
              onClick={handleAnalyze}
              disabled={!selectedImage || loading}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity size={20} />
                  Complete Analysis
                </>
              )}
            </button>

            <button
              className="btn btn-secondary btn-full"
              onClick={handleQuickAnalyze}
              disabled={!selectedImage || loading}
            >
              ⚡ Quick Analysis
            </button>
          </div>

          <div className="info-box">
            <p><strong>Complete Analysis:</strong> Full features + weather (2-4s)</p>
            <p><strong>Quick Analysis:</strong> Classification only (~1s)</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default UploadSection
