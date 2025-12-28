import React, { useState, useEffect } from 'react'
import { Zap, Target, History, PlusCircle } from 'lucide-react'
import UploadSection from '../components/UploadSection'
import ResultsSection from '../components/ResultsSection'
import HistorySection from '../components/HistorySection'
import UpgradePrompt from '../components/UpgradePrompt'
import axios from 'axios'
import './Analysis.css'

function Analysis({ token, onNavigate, user }) {
  const [activeTab, setActiveTab] = useState('new') // 'new' or 'history'
  const [analysisType, setAnalysisType] = useState(null) // 'quick' or 'complete'
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showUpgradePrompt, setShowUpgradePrompt] = useState(false)
  const [subscriptionStatus, setSubscriptionStatus] = useState(null)

  // Fetch subscription status when component mounts
  useEffect(() => {
    if (token) {
      fetchSubscriptionStatus()
    }
  }, [token])

  const fetchSubscriptionStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/auth/subscription-status`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (response.data.success) {
        setSubscriptionStatus(response.data)
      }
    } catch (err) {
      console.error('Error fetching subscription:', err)
    }
  }

  const handleAnalysisTypeClick = (type) => {
    // Check if user needs Pro for complete analysis
    if (type === 'complete' && user) {
      const canAccess = subscriptionStatus?.can_access_complete_analysis
      if (!canAccess) {
        setShowUpgradePrompt(true)
        return
      }
    }
    setAnalysisType(type)
  }

  const handleAnalysisComplete = (data) => {
    setResult(data)
    setError(null)
  }

  const handleError = (err) => {
    setError(err)
    setResult(null)
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setAnalysisType(null)
    setActiveTab('new')
  }

  const handleViewHistoryDetails = (analysisData) => {
    // When viewing history details, show the result
    setResult(analysisData)
    setActiveTab('new') // Switch to results view
  }

  // Render results view
  if (result) {
    return (
      <div className="analysis-page">
        <ResultsSection result={result} onReset={handleReset} />
      </div>
    )
  }

  if (analysisType) {
    return (
      <div className="analysis-page">
        <div className="analysis-header">
          <button className="back-btn" onClick={() => setAnalysisType(null)}>
            ← Back to Options
          </button>
          <h2>{analysisType === 'quick' ? 'Quick Analysis' : 'Complete Analysis'}</h2>
          <p>
            {analysisType === 'quick' 
              ? 'Fast pitch classification without detailed features'
              : 'Comprehensive analysis with weather data and detailed insights'}
          </p>
        </div>
        
        {error && (
          <div className="error-container fade-in">
            <div className="error-card">
              <div className="error-icon">⚠️</div>
              <h3>Analysis Failed</h3>
              <p>{error}</p>
              <button className="btn btn-primary" onClick={() => setError(null)}>
                Try Again
              </button>
            </div>
          </div>
        )}

        {!error && (
          <UploadSection
            onAnalysisComplete={handleAnalysisComplete}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
            token={token}
            analysisType={analysisType}
          />
        )}
      </div>
    )
  }

  // Render main analysis page with tabs
  return (
    <div className="analysis-page">
      {/* Tab Navigation */}
      <div className="analysis-tabs">
        <button 
          className={`tab-btn ${activeTab === 'new' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('new')
            setAnalysisType(null)
            setError(null)
          }}
        >
          <PlusCircle size={20} />
          New Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <History size={20} />
          History
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'history' ? (
        <HistorySection 
          authToken={token}
          onViewDetails={handleViewHistoryDetails}
        />
      ) : (
        <>
          {analysisType ? (
            <>
              <div className="analysis-header">
                <button className="back-btn" onClick={() => setAnalysisType(null)}>
                  ← Back to Options
                </button>
                <h2>{analysisType === 'quick' ? 'Quick Analysis' : 'Complete Analysis'}</h2>
                <p>
                  {analysisType === 'quick' 
                    ? 'Fast pitch classification without detailed features'
                    : 'Comprehensive analysis with weather data and detailed insights'}
                </p>
              </div>
              
              {error && (
                <div className="error-container fade-in">
                  <div className="error-card">
                    <div className="error-icon">⚠️</div>
                    <h3>Analysis Failed</h3>
                    <p>{error}</p>
                    <button className="btn btn-primary" onClick={() => setError(null)}>
                      Try Again
                    </button>
                  </div>
                </div>
              )}

              {!error && (
                <UploadSection
                  onAnalysisComplete={handleAnalysisComplete}
                  onError={handleError}
                  loading={loading}
                  setLoading={setLoading}
                  token={token}
                  analysisType={analysisType}
                />
              )}
            </>
          ) : (
            <>
              <div className="analysis-hero">
                <h1>Choose Analysis Type</h1>
                <p>Select the type of analysis you want to perform on your pitch image</p>
              </div>

              <div className="analysis-options">
                <div className="option-card" onClick={() => handleAnalysisTypeClick('quick')}>
                  <div className="option-icon quick">
                    <Zap size={48} />
                  </div>
                  <h3>Quick Analysis</h3>
                  <div className="option-features">
                    <div className="feature-badge">⚡ Fast Results</div>
                    <div className="feature-badge">🎯 Pitch Classification</div>
                    <div className="feature-badge">📊 Basic Insights</div>
                  </div>
                  <p>Get instant pitch type classification with confidence scores. Perfect for quick assessments.</p>
                  <ul className="option-list">
                    <li>✓ Pitch type prediction</li>
                    <li>✓ Confidence scores</li>
                    <li>✓ Processing time: ~2-3 seconds</li>
                  </ul>
                  <button className="option-btn quick">
                    Start Quick Analysis
                  </button>
                </div>

<div className="option-card featured" onClick={() => handleAnalysisTypeClick('complete')}>
                  <div className="featured-badge">Recommended</div>
                  <div className="option-icon complete">
                    <Target size={48} />
                  </div>
                  <h3>Complete Analysis</h3>
                  <div className="option-features">
                    <div className="feature-badge">🔍 Detailed Report</div>
                    <div className="feature-badge">🌤️ Weather Data</div>
                    <div className="feature-badge">📈 Match Strategy</div>
                    {user && subscriptionStatus?.subscription_type !== 'pro' && (
                      <div className="feature-badge pro-required">👑 Pro Required</div>
                    )}
                  </div>
                  <p>Comprehensive analysis with detailed features, weather integration, and strategic insights.</p>
                  <ul className="option-list">
                    <li>✓ Pitch detection & features</li>
                    <li>✓ Grass, cracks, moisture analysis</li>
                    <li>✓ Real-time weather integration</li>
                    <li>✓ Match strategy recommendations</li>
                    <li>✓ Processing time: ~5-7 seconds</li>
                  </ul>
                  <button className="option-btn complete">
                    {user && subscriptionStatus?.subscription_type !== 'pro' 
                      ? '👑 Upgrade to Pro' 
                      : 'Start Complete Analysis'}
                  </button>
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* Upgrade Prompt Modal */}
      {showUpgradePrompt && (
        <UpgradePrompt
          onClose={() => setShowUpgradePrompt(false)}
          onUpgrade={() => {
            setShowUpgradePrompt(false)
            if (onNavigate) {
              onNavigate('pricing')
            }
          }}
        />
      )}
    </div>
  )
}

export default Analysis
