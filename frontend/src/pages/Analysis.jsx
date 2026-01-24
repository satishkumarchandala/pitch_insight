import React, { useState, useEffect } from 'react'
import { Zap, Target, History, PlusCircle } from 'lucide-react'
import UploadSection from '../components/UploadSection'
import ResultsSection from '../components/ResultsSection'
import HistorySection from '../components/HistorySection'
import UpgradePrompt from '../components/UpgradePrompt'
import axios from 'axios'
import './Analysis.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Analysis({ token, onNavigate, user, onUserUpdate }) {
  const [activeTab, setActiveTab] = useState('new') // 'new' or 'history'
  const [analysisType, setAnalysisType] = useState(null) // 'quick' or 'complete'
  const [result, setResult] = useState(null)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showUpgradePrompt, setShowUpgradePrompt] = useState(false)
  const [subscriptionStatus, setSubscriptionStatus] = useState(null)

  // Fetch subscription status when component mounts or user changes
  useEffect(() => {
    if (token && user) {
      fetchSubscriptionStatus()
    }
  }, [token, user?.subscription_type, user?.subscription_status])

  const fetchSubscriptionStatus = async () => {
    if (!token) return // Don't fetch if not authenticated
    
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
      // Only log error if it's not a 401 (which is expected when not logged in)
      if (err.response?.status !== 401) {
        console.error('Error fetching subscription:', err)
      }
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

  const handlePaymentSuccess = async () => {
    setShowUpgradePrompt(false)
    // Refresh user data to get updated subscription
    if (onUserUpdate) {
      await onUserUpdate()
    }
    // Refetch subscription status to update UI
    await fetchSubscriptionStatus()
  }

  const handleAnalysisComplete = (data, image) => {
    setResult(data)
    setUploadedImage(image) // Store the uploaded image
    setError(null)
  }

  const handleError = (err) => {
    setError(err)
    setResult(null)
  }

  const handleReset = () => {
    setResult(null)
    setUploadedImage(null)
    setError(null)
    setAnalysisType(null)
    // If we were viewing from history, go back to history tab
    if (activeTab === 'history-detail') {
      setActiveTab('history')
    } else {
      setActiveTab('new')
    }
  }

  const handleViewHistoryDetails = (analysisData) => {
    console.log('Viewing history details:', analysisData) // Debug log
    // When viewing history details, show the result
    // If analysisData has image_data, convert it to uploadedImage format
    if (analysisData.image_data) {
      setUploadedImage(analysisData.image_data)
    }
    setResult(analysisData)
    setActiveTab('history-detail') // New state to indicate viewing from history
  }

  // Render results view
  if (result) {
    return (
      <div className="analysis-page">
        {/* Tab Navigation - Show even when viewing results */}
        <div className="analysis-tabs">
          <button 
            className={`tab-btn ${activeTab === 'new' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('new')
              setResult(null)
              setUploadedImage(null)
              setAnalysisType(null)
            }}
          >
            <PlusCircle size={20} />
            New Analysis
          </button>
          <button 
            className={`tab-btn ${activeTab === 'history' || activeTab === 'history-detail' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('history')
              setResult(null)
              setUploadedImage(null)
              setAnalysisType(null)
            }}
          >
            <History size={20} />
            History
          </button>
        </div>

        <ResultsSection 
          result={result} 
          onReset={handleReset} 
          authToken={token}
          uploadedImage={uploadedImage}
        />
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
            setResult(null)
            setUploadedImage(null)
            setAnalysisType(null)
          }}
        >
          <PlusCircle size={20} />
          New Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 'history' || activeTab === 'history-detail' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('history')
            setResult(null)
            setUploadedImage(null)
            setAnalysisType(null)
          }}
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
                 
                  <button className="option-btn quick">
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
          onUpgrade={handlePaymentSuccess}
        />
      )}
    </div>
  )
}

export default Analysis
