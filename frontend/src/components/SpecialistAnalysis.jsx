import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Download, ZoomIn } from 'lucide-react'
import './SpecialistAnalysis.css'

function SpecialistAnalysis({ specialistData, features }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [selectedImage, setSelectedImage] = useState(null)

  if (!specialistData || Object.keys(specialistData).length === 0) {
    return null
  }

  const handleDownload = (imageData, filename) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${imageData}`
    link.download = filename
    link.click()
  }

  const openImageModal = (imageData, title) => {
    setSelectedImage({ imageData, title })
  }

  const closeModal = () => {
    setSelectedImage(null)
  }

  return (
    <>
      <div className="specialist-analysis-container">
        <button 
          className="specialist-toggle-btn"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="toggle-content">
            <span className="toggle-icon">🔬</span>
            <span className="toggle-text">Specialist Analysis</span>
            <span className="toggle-badge">Advanced</span>
          </div>
          {isExpanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>

        {isExpanded && (
          <div className="specialist-content fade-in">
            <div className="specialist-grid">
              
              {/* Grass Analysis */}
              {specialistData.grass_analysis && (
                <div className="specialist-card slide-in-up" style={{ '--delay': '0.1s' }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-icon grass-icon">🌱</span>
                      <h4>Grass Coverage Analysis</h4>
                    </div>
                    <div className="card-actions">
                      <button 
                        className="icon-btn"
                        onClick={() => openImageModal(specialistData.grass_analysis, 'Grass Coverage')}
                        title="View Full Size"
                      >
                        <ZoomIn size={18} />
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={() => handleDownload(specialistData.grass_analysis, 'grass-analysis.png')}
                        title="Download"
                      >
                        <Download size={18} />
                      </button>
                    </div>
                  </div>
                  <div className="card-image">
                    <img 
                      src={`data:image/png;base64,${specialistData.grass_analysis}`} 
                      alt="Grass Coverage Analysis"
                    />
                  </div>
                  <div className="card-stats">
                    <div className="stat-item">
                      <span className="stat-label">Coverage</span>
                      <span className="stat-value">{features.grass_coverage.percentage.toFixed(1)}%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Level</span>
                      <span className="stat-value">{features.grass_coverage.level}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Quality</span>
                      <span className="stat-value">{features.grass_coverage.quality}</span>
                    </div>
                  </div>
                  <div className="card-description">
                    <p>Green overlay indicates grass-covered areas. Higher coverage suggests better batting conditions and less wear.</p>
                  </div>
                </div>
              )}

              {/* Crack Detection */}
              {specialistData.crack_analysis && (
                <div className="specialist-card slide-in-up" style={{ '--delay': '0.2s' }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-icon crack-icon">⚡</span>
                      <h4>Crack Detection Analysis</h4>
                    </div>
                    <div className="card-actions">
                      <button 
                        className="icon-btn"
                        onClick={() => openImageModal(specialistData.crack_analysis, 'Crack Detection')}
                        title="View Full Size"
                      >
                        <ZoomIn size={18} />
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={() => handleDownload(specialistData.crack_analysis, 'crack-analysis.png')}
                        title="Download"
                      >
                        <Download size={18} />
                      </button>
                    </div>
                  </div>
                  <div className="card-image">
                    <img 
                      src={`data:image/png;base64,${specialistData.crack_analysis}`} 
                      alt="Crack Detection Analysis"
                    />
                  </div>
                  <div className="card-stats">
                    <div className="stat-item">
                      <span className="stat-label">Severity</span>
                      <span className="stat-value">{features.cracks.severity}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Density</span>
                      <span className="stat-value">{features.cracks.density.toFixed(2)}%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Count</span>
                      <span className="stat-value">{features.cracks.count}</span>
                    </div>
                  </div>
                  <div className="card-description">
                    <p>Heat map visualization shows crack intensity. More cracks indicate pitch deterioration and unpredictable bounce.</p>
                  </div>
                </div>
              )}

              {/* Moisture Analysis */}
              {specialistData.moisture_analysis && (
                <div className="specialist-card slide-in-up" style={{ '--delay': '0.3s' }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-icon moisture-icon">💧</span>
                      <h4>Moisture Distribution</h4>
                    </div>
                    <div className="card-actions">
                      <button 
                        className="icon-btn"
                        onClick={() => openImageModal(specialistData.moisture_analysis, 'Moisture Analysis')}
                        title="View Full Size"
                      >
                        <ZoomIn size={18} />
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={() => handleDownload(specialistData.moisture_analysis, 'moisture-analysis.png')}
                        title="Download"
                      >
                        <Download size={18} />
                      </button>
                    </div>
                  </div>
                  <div className="card-image">
                    <img 
                      src={`data:image/png;base64,${specialistData.moisture_analysis}`} 
                      alt="Moisture Distribution Analysis"
                    />
                  </div>
                  <div className="card-stats">
                    <div className="stat-item">
                      <span className="stat-label">Level</span>
                      <span className="stat-value">{features.moisture.level}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Score</span>
                      <span className="stat-value">{features.moisture.score.toFixed(0)}/100</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Impact</span>
                      <span className="stat-value">{features.moisture.score > 50 ? 'High' : features.moisture.score > 30 ? 'Medium' : 'Low'}</span>
                    </div>
                  </div>
                  <div className="card-description">
                    <p>Thermal map shows moisture distribution. Blue/cyan areas indicate higher moisture, affecting seam movement and grip.</p>
                  </div>
                </div>
              )}

              {/* Original Pitch Reference */}
              {specialistData.original_pitch && (
                <div className="specialist-card slide-in-up" style={{ '--delay': '0.4s' }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-icon original-icon">📸</span>
                      <h4>Original Pitch Image</h4>
                    </div>
                    <div className="card-actions">
                      <button 
                        className="icon-btn"
                        onClick={() => openImageModal(specialistData.original_pitch, 'Original Pitch')}
                        title="View Full Size"
                      >
                        <ZoomIn size={18} />
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={() => handleDownload(specialistData.original_pitch, 'original-pitch.png')}
                        title="Download"
                      >
                        <Download size={18} />
                      </button>
                    </div>
                  </div>
                  <div className="card-image">
                    <img 
                      src={`data:image/png;base64,${specialistData.original_pitch}`} 
                      alt="Original Pitch"
                    />
                  </div>
                  <div className="card-stats">
                    <div className="stat-item">
                      <span className="stat-label">Color</span>
                      <span className="stat-value">{features.color.type}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Texture</span>
                      <span className="stat-value">{features.texture.type}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Brightness</span>
                      <span className="stat-value">{features.brightness.level}</span>
                    </div>
                  </div>
                  <div className="card-description">
                    <p>Original detected pitch region used as baseline for all specialist analysis.</p>
                  </div>
                </div>
              )}

            </div>

            {/* Analysis Legend */}
            <div className="analysis-legend">
              <h4>🎨 Visualization Guide</h4>
              <div className="legend-grid">
                <div className="legend-item">
                  <span className="legend-color grass-legend"></span>
                  <span>Grass Coverage (Green overlay)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color crack-legend"></span>
                  <span>Crack Intensity (Hot colormap)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color moisture-legend"></span>
                  <span>Moisture Level (Jet colormap)</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div className="image-modal" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>×</button>
            <h3>{selectedImage.title}</h3>
            <img 
              src={`data:image/png;base64,${selectedImage.imageData}`} 
              alt={selectedImage.title}
            />
            <button 
              className="modal-download"
              onClick={() => handleDownload(selectedImage.imageData, `${selectedImage.title.toLowerCase().replace(/\s+/g, '-')}.png`)}
            >
              <Download size={20} />
              Download Full Resolution
            </button>
          </div>
        </div>
      )}
    </>
  )
}

export default SpecialistAnalysis
