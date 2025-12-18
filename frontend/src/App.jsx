import React, { useState } from 'react'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultsSection from './components/ResultsSection'
import Footer from './components/Footer'
import './App.css'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [weatherData, setWeatherData] = useState(null)

  const handleAnalysisComplete = (data, imageFile, weather) => {
    setResult(data)
    setUploadedImage(imageFile)
    setWeatherData(weather)
    setError(null)
  }

  const handleError = (err) => {
    setError(err)
    setResult(null)
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setUploadedImage(null)
    setWeatherData(null)
  }

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        <div className="container">
          {!result && !error && (
            <UploadSection 
              onAnalysisComplete={handleAnalysisComplete}
              onError={handleError}
              loading={loading}
              setLoading={setLoading}
            />
          )}

          {error && (
            <div className="error-container fade-in">
              <div className="error-card">
                <div className="error-icon">⚠️</div>
                <h2>Analysis Failed</h2>
                <p>{error}</p>
                <button className="btn btn-primary" onClick={handleReset}>
                  Try Again
                </button>
              </div>
            </div>
          )}

          {result && (
            <ResultsSection 
              result={result} 
              onReset={handleReset}
              uploadedImage={uploadedImage}
              weatherData={weatherData}
            />
          )}
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default App
