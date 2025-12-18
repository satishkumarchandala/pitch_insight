import React, { useState, useRef } from 'react'
import { Upload, MapPin, Cloud, Activity } from 'lucide-react'
import axios from 'axios'
import './UploadSection.css'

const API_URL = 'http://localhost:8000'

function UploadSection({ onAnalysisComplete, onError, loading, setLoading }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [includeWeather, setIncludeWeather] = useState(false)
  const [location, setLocation] = useState({
    latitude: '',
    longitude: '',
    city: ''
  })
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

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

      if (includeWeather) {
        if (location.latitude) formData.append('latitude', location.latitude)
        if (location.longitude) formData.append('longitude', location.longitude)
        if (location.city) formData.append('city', location.city)
      }

      const response = await axios.post(API_ENDPOINTS.analyze, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      // Pass image file and weather data for report generation
      const weatherInfo = includeWeather ? {
        latitude: location.latitude,
        longitude: location.longitude,
        city: location.city
      } : null
      
      onAnalysisComplete(response.data, selectedImage, weatherInfo)
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

      const response = await axios.post(API_ENDPOINTS.quickAnalyze, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
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

      onAnalysisComplete(fullResult, selectedImage, null)
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
            <div className="location-inputs fade-in">
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
