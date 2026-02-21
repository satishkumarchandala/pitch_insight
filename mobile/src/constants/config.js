// ==========================================
// API Configuration
// ==========================================

// 🔧 CHANGE THIS TO SWITCH ENVIRONMENTS
// Options: 'production' | 'development' | 'local'
const ENVIRONMENT = 'production';

// Environment configurations
const ENVIRONMENTS = {
    production: {
        name: 'Production (Render)',
        baseURL: 'https://pitch-insight-backend.onrender.com',
        timeout: 60000, // 60 seconds for cold start
    },
    development: {
        name: 'Development (Local)',
        baseURL: 'http://localhost:8000',
        timeout: 30000, // 30 seconds
    },
    local: {
        name: 'Local (Your IP)',
        // Replace with your computer's local IP address
        // Find it with: ipconfig (Windows) or ifconfig (Mac/Linux)
        // Example: 'http://192.168.1.100:8000'
        baseURL: 'http://192.168.1.100:8000',
        timeout: 30000,
    },
};

// Get current environment config
const currentEnv = ENVIRONMENTS[ENVIRONMENT];

// Export API configuration
export const API_BASE_URL = currentEnv.baseURL;
export const API_TIMEOUT = currentEnv.timeout;
export const API_ENV_NAME = currentEnv.name;

// Log current environment (helpful for debugging)
console.log(`🌐 API Environment: ${API_ENV_NAME}`);
console.log(`🔗 Base URL: ${API_BASE_URL}`);

// API Endpoints
export const API_ENDPOINTS = {
    AUTH: {
        LOGIN: '/api/auth/login',
        REGISTER: '/api/auth/signup',
        ME: '/api/auth/me',
    },
    ANALYSIS: {
        ANALYZE: '/api/analyze',
        QUICK_ANALYZE: '/api/quick-analyze',
        HISTORY: '/api/auth/history', // Corrected path based on frontend
    },
    CHAT: {
        SEND: '/api/chat',
    },
    SUBSCRIPTION: {
        STATUS: '/api/auth/subscription-status',
        CREATE_ORDER: '/api/subscription/create-order',
        VERIFY_PAYMENT: '/api/subscription/verify-payment',
    },
    WEATHER: {
        CURRENT: '/api/weather',
        FORECAST: '/api/weather/forecast',
    },
};
