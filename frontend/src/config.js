// API Configuration
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const APP_NAME = import.meta.env.VITE_APP_NAME || 'Pitch Insight'
export const ENABLE_ANALYTICS = import.meta.env.VITE_ENABLE_ANALYTICS === 'true'

// API Endpoints
export const API_ENDPOINTS = {
  analyze: `${API_URL}/api/analyze`,
  quickAnalyze: `${API_URL}/api/quick-analyze`,
  generateReport: `${API_URL}/api/generate-report`,
  weather: `${API_URL}/api/weather`,
  health: `${API_URL}/api/health`,
  docs: `${API_URL}/docs`,
  recentAnalyses: `${API_URL}/api/recent-analyses`,
  stats: `${API_URL}/api/stats`
}

export default {
  API_URL,
  APP_NAME,
  ENABLE_ANALYTICS,
  API_ENDPOINTS
}
