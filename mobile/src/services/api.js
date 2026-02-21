import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL, API_TIMEOUT, API_ENDPOINTS } from '../constants/config';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: API_TIMEOUT,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor to add auth token
api.interceptors.request.use(
    async (config) => {
        const token = await AsyncStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        if (error.response?.status === 401) {
            // Token expired or invalid
            await AsyncStorage.removeItem('token');
            // We might want to trigger a global logout state here
            // distinct from just removing the token
        }
        return Promise.reject(error);
    }
);

// ==========================================
// AUTHENTICATION API
// ==========================================
export const authAPI = {
    // User login
    login: async (email, password) => {
        const response = await api.post(API_ENDPOINTS.AUTH.LOGIN, { email, password });
        return response.data;
    },

    // User signup
    signup: async (userData) => {
        const response = await api.post(API_ENDPOINTS.AUTH.REGISTER, userData);
        return response.data;
    },

    // Get current user info
    getMe: async () => {
        const response = await api.get(API_ENDPOINTS.AUTH.ME);
        return response.data;
    },

    // Get subscription status
    getSubscriptionStatus: async () => {
        const response = await api.get(API_ENDPOINTS.SUBSCRIPTION.STATUS);
        return response.data;
    }
};

// ==========================================
// ANALYSIS API
// ==========================================
export const analysisAPI = {
    // Quick analysis (Free feature)
    analyzeQuick: async (imageUri) => {
        const formData = new FormData();

        // Append image file
        const filename = imageUri.split('/').pop();
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';

        formData.append('file', {
            uri: imageUri,
            name: filename,
            type: type,
        });

        const response = await api.post(API_ENDPOINTS.ANALYSIS.QUICK_ANALYZE, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Complete analysis (Pro feature)
    analyzeComplete: async (imageUri, options = {}) => {
        const formData = new FormData();

        // Append image file
        const filename = imageUri.split('/').pop();
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';

        formData.append('file', {
            uri: imageUri,
            name: filename,
            type: type, // e.g. 'image/jpg'
        });

        // Add optional parameters
        if (options.weatherData) {
            formData.append('weather_data', JSON.stringify(options.weatherData));
        }
        if (options.useForecast !== undefined) {
            formData.append('use_forecast', String(options.useForecast));
        }
        if (options.matchType) {
            formData.append('match_type', options.matchType);
        }
        if (options.matchStartTime) {
            formData.append('match_start_time', options.matchStartTime);
        }
        if (options.city) {
            formData.append('city', options.city);
        }
        // Location coordinates if available
        if (options.latitude) formData.append('latitude', String(options.latitude));
        if (options.longitude) formData.append('longitude', String(options.longitude));

        const response = await api.post(API_ENDPOINTS.ANALYSIS.ANALYZE, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    getHistory: async () => {
        const response = await api.get('/api/auth/history');
        return response.data;
    },

    // Get single analysis detail
    getAnalysisDetail: async (id) => {
        const response = await api.get(`${API_ENDPOINTS.ANALYSIS.HISTORY}/${id}`);
        return response.data;
    },

    // Delete analysis
    deleteAnalysis: async (id) => {
        const response = await api.delete(`${API_ENDPOINTS.ANALYSIS.HISTORY}/${id}`);
        return response.data;
    }
};

// ==========================================
// WEATHER API
// ==========================================
export const weatherAPI = {
    // Get current weather
    getWeather: async (location) => {
        const response = await api.get(`${API_ENDPOINTS.WEATHER.CURRENT}/${encodeURIComponent(location)}`);
        return response.data;
    },

    // Get forecast
    getForecast: async (options = {}) => {
        // Construct query string manually or use URLSearchParams (polyfill might be needed)
        const params = [];
        if (options.city) params.push(`city=${encodeURIComponent(options.city)}`);
        if (options.latitude) params.push(`latitude=${options.latitude}`);
        if (options.longitude) params.push(`longitude=${options.longitude}`);

        const queryString = params.length > 0 ? `?${params.join('&')}` : '';
        const response = await api.get(`${API_ENDPOINTS.WEATHER.FORECAST}${queryString}`);
        return response.data;
    }
};

// ==========================================
// CHAT API
// ==========================================
export const chatAPI = {
    sendMessage: async (message, analysisId = null) => {
        const payload = { message };
        if (analysisId) payload.analysis_id = analysisId;

        const response = await api.post(API_ENDPOINTS.CHAT.SEND, payload);
        return response.data;
    }
};

export default api;
