/**
 * Centralized API Service
 * All backend API calls go through this service
 */

import axios from 'axios'

// Get API URL from environment variables
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
})

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

// ==========================================
// AUTHENTICATION API
// ==========================================

export const authAPI = {
  // User signup
  signup: async (email, password, username, fullName = null) => {
    const response = await apiClient.post('/api/auth/signup', {
      email,
      password,
      username,
      full_name: fullName,
    })
    return response.data
  },

  // User login
  login: async (email, password) => {
    const response = await apiClient.post('/api/auth/login', {
      email,
      password,
    })
    return response.data
  },

  // Get current user info
  getMe: async () => {
    const response = await apiClient.get('/api/auth/me')
    return response.data
  },

  // Get user analysis history
  getHistory: async () => {
    const response = await apiClient.get('/api/auth/history')
    return response.data
  },

  // Get subscription status
  getSubscriptionStatus: async () => {
    const response = await apiClient.get('/api/auth/subscription-status')
    return response.data
  },
}

// ==========================================
// ANALYSIS API
// ==========================================

export const analysisAPI = {
  // Complete analysis (Pro feature)
  analyzeComplete: async (imageFile, options = {}) => {
    const formData = new FormData()
    formData.append('file', imageFile)
    
    // Add optional parameters
    if (options.weatherData) {
      formData.append('weather_data', JSON.stringify(options.weatherData))
    }
    if (options.useForecast !== undefined) {
      formData.append('use_forecast', options.useForecast)
    }
    if (options.matchType) {
      formData.append('match_type', options.matchType)
    }
    if (options.matchStartTime) {
      formData.append('match_start_time', options.matchStartTime)
    }
    if (options.city) {
      formData.append('city', options.city)
    }
    if (options.latitude) {
      formData.append('latitude', options.latitude)
    }
    if (options.longitude) {
      formData.append('longitude', options.longitude)
    }

    const response = await apiClient.post('/api/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Quick analysis (Free feature)
  analyzeQuick: async (imageFile) => {
    const formData = new FormData()
    formData.append('file', imageFile)

    const response = await apiClient.post('/api/quick-analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
}

// ==========================================
// WEATHER API
// ==========================================

export const weatherAPI = {
  // Get current weather for a location
  getWeather: async (location) => {
    const response = await apiClient.get(`/api/weather/${encodeURIComponent(location)}`)
    return response.data
  },

  // Get comprehensive weather forecast
  getForecast: async (options = {}) => {
    const params = new URLSearchParams()
    
    if (options.city) {
      params.append('city', options.city)
    }
    if (options.latitude) {
      params.append('latitude', options.latitude)
    }
    if (options.longitude) {
      params.append('longitude', options.longitude)
    }
    if (options.matchFormat) {
      params.append('match_format', options.matchFormat)
    }
    if (options.matchStartTime) {
      params.append('match_start_time', options.matchStartTime)
    }

    const response = await apiClient.get(`/api/weather/forecast?${params.toString()}`)
    return response.data
  },
}

// ==========================================
// SUBSCRIPTION API
// ==========================================

export const subscriptionAPI = {
  // Create Razorpay order
  createOrder: async (planType) => {
    const response = await apiClient.post(
      `/api/subscription/create-order?plan_type=${planType}`
    )
    return response.data
  },

  // Verify payment
  verifyPayment: async (paymentData) => {
    const response = await apiClient.post('/api/subscription/verify-payment', paymentData)
    return response.data
  },
}

// ==========================================
// CHAT API
// ==========================================

export const chatAPI = {
  // Send chat message
  sendMessage: async (message, analysisId = null) => {
    const response = await apiClient.post('/api/chat', {
      message,
      analysis_id: analysisId,
    })
    return response.data
  },
}

// ==========================================
// HEALTH CHECK API
// ==========================================

export const healthAPI = {
  // Check API health
  checkHealth: async () => {
    const response = await apiClient.get('/api/health')
    return response.data
  },
}

// Export the base URL for use in components
export { API_URL }

// Export the axios instance for custom requests
export default apiClient
