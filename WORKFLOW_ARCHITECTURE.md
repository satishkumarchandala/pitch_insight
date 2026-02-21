# Pitch Insight - Workflow Architecture

## System Overview
Pitch Insight is an AI-powered cricket pitch analysis application built with FastAPI (backend) and React (frontend), featuring machine learning models for pitch detection and classification, integrated with weather data and AI chatbot capabilities.

---

## Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Browser<br/>React + Vite]
        WEB_PAGES[Pages:<br/>Home, Analysis, Profile,<br/>Pricing, Settings]
        WEB_COMPONENTS[Components:<br/>Auth, ChatWidget,<br/>Header, Footer,<br/>HistorySection, etc.]
    end

    subgraph "API Gateway & Middleware"
        FASTAPI[FastAPI Application<br/>app.py]
        CORS[CORS Middleware<br/>Allow Origins]
        AUTH_MW[JWT Authentication<br/>Bearer Token]
    end

    subgraph "Route Handlers"
        HEALTH[Health Routes<br/>/api/health]
        AUTH_ROUTE[Auth Routes<br/>/api/auth/*]
        ANALYSIS_ROUTE[Analysis Routes<br/>/api/analysis/*]
        CHAT_ROUTE[Chat Routes<br/>/api/chat]
        WEATHER_ROUTE[Weather Routes<br/>/api/weather]
        SUB_ROUTE[Subscription Routes<br/>/api/subscription/*]
    end

    subgraph "Business Logic Layer"
        AUTH_LOGIC[Authentication Logic<br/>- Signup/Login<br/>- JWT Token Generation<br/>- Password Hashing]
        ANALYSIS_LOGIC[Analysis Logic<br/>- Image Upload<br/>- ML Pipeline Execution<br/>- Result Caching<br/>- History Management]
        CHAT_LOGIC[Chat Logic<br/>- Context Building<br/>- Gemini AI Integration<br/>- Pro User Validation]
        WEATHER_LOGIC[Weather Logic<br/>- Location Detection<br/>- Weather Data Fetch<br/>- Impact Analysis]
        SUB_LOGIC[Subscription Logic<br/>- Plan Management<br/>- Payment Processing<br/>- Access Control]
    end

    subgraph "ML Pipeline - ONNX Runtime"
        PIPELINE[Complete Pipeline<br/>complete_pipeline_onnx.py]
        YOLO[YOLO Detection<br/>pitch_yolov8_best.onnx<br/>Detect pitch area]
        FEATURE[Feature Analyzer<br/>pitch_analyzer.py<br/>Extract features]
        CLASSIFIER[Pitch Classifier<br/>pitch_classifier.onnx<br/>Classify pitch type]
        STRATEGY[Strategy Generator<br/>Match recommendations]
    end

    subgraph "Data Layer - MongoDB"
        DB[(MongoDB Database)]
        USERS_COL[Users Collection<br/>- Credentials<br/>- Subscription Info<br/>- Payment History]
        ANALYSIS_COL[Analysis Collection<br/>- Image Metadata<br/>- Results<br/>- Timestamps]
    end

    subgraph "External Services"
        WEATHER_API[Weather API<br/>weatherapi.com<br/>Current conditions]
        GEMINI_API[Gemini AI<br/>Google GenAI<br/>Cricket chatbot]
        RAZORPAY_API[Razorpay<br/>Payment Gateway<br/>Subscriptions]
    end

    subgraph "Configuration & Security"
        ENV[Environment Config<br/>.env file]
        CONFIG[config.py<br/>- MongoDB URL<br/>- API Keys<br/>- Plans & Pricing]
        SECRETS[Security<br/>- JWT Secret<br/>- API Keys<br/>- Razorpay Keys]
    end

    %% Client to API
    WEB --> WEB_PAGES
    WEB_PAGES --> WEB_COMPONENTS
    WEB_COMPONENTS --> FASTAPI
    
    %% API Gateway Flow
    FASTAPI --> CORS
    CORS --> AUTH_MW
    AUTH_MW --> HEALTH
    AUTH_MW --> AUTH_ROUTE
    AUTH_MW --> ANALYSIS_ROUTE
    AUTH_MW --> CHAT_ROUTE
    AUTH_MW --> WEATHER_ROUTE
    AUTH_MW --> SUB_ROUTE
    
    %% Route to Logic
    AUTH_ROUTE --> AUTH_LOGIC
    ANALYSIS_ROUTE --> ANALYSIS_LOGIC
    CHAT_ROUTE --> CHAT_LOGIC
    WEATHER_ROUTE --> WEATHER_LOGIC
    SUB_ROUTE --> SUB_LOGIC
    
    %% Analysis Pipeline
    ANALYSIS_LOGIC --> PIPELINE
    PIPELINE --> YOLO
    YOLO --> FEATURE
    FEATURE --> CLASSIFIER
    CLASSIFIER --> STRATEGY
    STRATEGY --> ANALYSIS_LOGIC
    
    %% Database Connections
    AUTH_LOGIC --> USERS_COL
    ANALYSIS_LOGIC --> ANALYSIS_COL
    ANALYSIS_LOGIC --> USERS_COL
    SUB_LOGIC --> USERS_COL
    USERS_COL --> DB
    ANALYSIS_COL --> DB
    
    %% External Services
    CHAT_LOGIC --> GEMINI_API
    WEATHER_LOGIC --> WEATHER_API
    SUB_LOGIC --> RAZORPAY_API
    
    %% Configuration
    CONFIG --> ENV
    CONFIG --> SECRETS
    FASTAPI --> CONFIG
    AUTH_LOGIC --> CONFIG
    WEATHER_LOGIC --> CONFIG
    CHAT_LOGIC --> CONFIG
    SUB_LOGIC --> CONFIG

    style WEB fill:#61dafb,stroke:#333,stroke-width:3px
    style FASTAPI fill:#009688,stroke:#333,stroke-width:3px
    style DB fill:#4db33d,stroke:#333,stroke-width:3px
    style PIPELINE fill:#ff9800,stroke:#333,stroke-width:3px
    style GEMINI_API fill:#4285f4,stroke:#333,stroke-width:2px
    style WEATHER_API fill:#00bcd4,stroke:#333,stroke-width:2px
    style RAZORPAY_API fill:#3395ff,stroke:#333,stroke-width:2px
```

---

## Data Flow Workflows

### 1. User Authentication Flow
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant AuthAPI
    participant Database
    
    User->>Frontend: Enter credentials
    Frontend->>AuthAPI: POST /api/auth/signup or /login
    AuthAPI->>AuthAPI: Hash password (bcrypt)
    AuthAPI->>Database: Store/Verify user
    Database-->>AuthAPI: User data
    AuthAPI->>AuthAPI: Generate JWT token
    AuthAPI-->>Frontend: Return token + user data
    Frontend->>Frontend: Store in localStorage
    Frontend-->>User: Redirect to dashboard
```

### 2. Pitch Analysis Workflow
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant AnalysisAPI
    participant Pipeline
    participant YOLO
    participant FeatureExtractor
    participant Classifier
    participant Database
    
    User->>Frontend: Upload pitch image
    Frontend->>AnalysisAPI: POST /api/analyze (multipart/form-data)
    AnalysisAPI->>AnalysisAPI: Validate file size & type
    AnalysisAPI->>AnalysisAPI: Check cache (image hash)
    
    alt Cache Hit
        AnalysisAPI-->>Frontend: Return cached results
    else Cache Miss
        AnalysisAPI->>Pipeline: Process image
        Pipeline->>YOLO: Detect pitch area
        YOLO-->>Pipeline: Bounding box coordinates
        Pipeline->>FeatureExtractor: Extract features
        FeatureExtractor-->>Pipeline: Grass%, cracks, moisture, etc.
        Pipeline->>Classifier: Classify pitch type
        Classifier-->>Pipeline: Prediction + confidence
        Pipeline->>Pipeline: Generate match strategy
        Pipeline-->>AnalysisAPI: Complete analysis
        AnalysisAPI->>Database: Save analysis result
        AnalysisAPI->>AnalysisAPI: Update cache
        AnalysisAPI-->>Frontend: Return analysis
    end
    
    Frontend-->>User: Display results
```

### 3. Weather Integration Workflow
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant WeatherAPI
    participant ExternalWeatherService
    participant ForecastAnalyzer
    
    User->>Frontend: Enter location/coordinates
    Frontend->>WeatherAPI: GET /api/weather?city=X
    WeatherAPI->>ExternalWeatherService: Fetch weather data
    ExternalWeatherService-->>WeatherAPI: Current conditions
    WeatherAPI->>ForecastAnalyzer: Analyze impact
    ForecastAnalyzer-->>WeatherAPI: Pitch impact analysis
    WeatherAPI-->>Frontend: Weather + Impact data
    Frontend-->>User: Display weather conditions
```

### 4. AI Chat Workflow
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant ChatAPI
    participant GeminiAI
    participant Database
    
    User->>Frontend: Send chat message
    Frontend->>ChatAPI: POST /api/chat
    ChatAPI->>ChatAPI: Verify Pro subscription
    ChatAPI->>Database: Get recent analysis context
    Database-->>ChatAPI: Analysis data
    ChatAPI->>ChatAPI: Build context prompt
    ChatAPI->>GeminiAI: Send prompt with context
    GeminiAI-->>ChatAPI: AI response
    ChatAPI-->>Frontend: Return chat response
    Frontend-->>User: Display AI message
```

### 5. Subscription & Payment Workflow
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant SubAPI
    participant Razorpay
    participant Database
    
    User->>Frontend: Select plan (Monthly/Yearly)
    Frontend->>SubAPI: POST /api/subscription/create-order
    SubAPI->>SubAPI: Calculate amount
    SubAPI->>Razorpay: Create payment order
    Razorpay-->>SubAPI: Order ID
    SubAPI-->>Frontend: Return order details
    Frontend->>Frontend: Open Razorpay checkout
    User->>Razorpay: Complete payment
    Razorpay->>Frontend: Payment success callback
    Frontend->>SubAPI: POST /api/subscription/verify-payment
    SubAPI->>Razorpay: Verify signature
    Razorpay-->>SubAPI: Verification result
    SubAPI->>Database: Update user subscription
    Database-->>SubAPI: Confirmed
    SubAPI-->>Frontend: Success response
    Frontend->>Frontend: Refresh user data
    Frontend-->>User: Show success message
```

---

## Component Details

### Frontend (React + Vite)
- **Framework**: React 18 with Vite for fast development
- **State Management**: React Hooks (useState, useEffect)
- **Routing**: Single-page application with component-based routing
- **API Communication**: Axios with interceptors for auth
- **Key Features**:
  - Authentication UI (Auth.jsx)
  - Image upload and analysis display
  - Chat widget integration
  - Subscription management
  - Analysis history

### Backend (FastAPI)
- **Framework**: FastAPI (async Python web framework)
- **Authentication**: JWT tokens with bcrypt password hashing
- **API Documentation**: Auto-generated Swagger/OpenAPI docs at `/docs`
- **Key Features**:
  - RESTful API design
  - CORS middleware for cross-origin requests
  - File upload handling
  - Async/await for concurrent operations
  - Environment-based configuration

### Database (MongoDB)
- **Collections**:
  - **users**: User accounts, subscriptions, payment history
  - **analysis**: Pitch analysis results and metadata
- **Indexes**:
  - Unique indexes on email and username
  - Query optimization for history retrieval
- **Data Storage**:
  - JSON documents for flexible schema
  - GridFS for large images (optional)

### ML Pipeline (ONNX)
- **Models**:
  - **YOLOv8 ONNX**: Pitch detection (640x640 input)
  - **Pitch Classifier ONNX**: Type classification (4 classes)
- **Processing Steps**:
  1. Image preprocessing (resize, normalize)
  2. YOLO inference for pitch detection
  3. Feature extraction (grass %, cracks, moisture, color)
  4. Classification with confidence scores
  5. Rule-based adjustments
  6. Strategy generation
- **Optimization**: ONNX Runtime for fast CPU inference

### External Integrations
- **Weather API** (weatherapi.com):
  - Current weather conditions
  - Location-based queries
  - Temperature, humidity, wind data
  
- **Gemini AI** (Google):
  - Cricket-specific chatbot
  - Context-aware responses
  - Analysis explanation
  
- **Razorpay**:
  - Payment processing
  - Subscription management
  - Webhook handling

---

## Security Features

1. **Authentication**:
   - JWT tokens with 7-day expiration
   - Bcrypt password hashing
   - Secure token storage (localStorage with HTTPS)

2. **Authorization**:
   - Route protection with dependency injection
   - Pro feature access control
   - User ownership validation

3. **API Security**:
   - CORS configuration
   - Request validation with Pydantic schemas
   - File upload restrictions (size, type)
   - Rate limiting (recommended for production)

4. **Data Protection**:
   - Environment variables for secrets
   - No hardcoded credentials
   - MongoDB authentication (production)

---

## Deployment Architecture

### Development
```
Frontend: localhost:5173 (Vite dev server)
Backend: localhost:8000 (Uvicorn)
Database: localhost:27017 (MongoDB)
```

### Production (Render)
```
Frontend: Nginx static hosting or Vercel
Backend: Render web service (Uvicorn workers)
Database: MongoDB Atlas
External Services: Cloud APIs
```

### Configuration
- **Environment Variables**: All sensitive data in `.env`
- **CORS**: Configured for specific origins in production
- **Keep-Alive**: `keep_alive.py` script to prevent service sleep

---

## Key Technologies

| Layer | Technologies |
|-------|-------------|
| Frontend | React, Vite, Axios, CSS3 |
| Backend | FastAPI, Uvicorn, Python 3.9+ |
| Database | MongoDB, PyMongo |
| ML/AI | ONNX Runtime, OpenCV, NumPy, Gemini AI |
| Authentication | JWT, bcrypt, python-jose |
| Payment | Razorpay SDK |
| Weather | WeatherAPI.com |
| Deployment | Render, Vercel, Docker |

---

## Performance Optimizations

1. **Caching**:
   - Image hash-based result caching
   - In-memory cache for frequent queries
   - Max cache size: 100 entries

2. **ONNX Runtime**:
   - Faster than PyTorch for inference
   - Optimized for CPU execution
   - Memory efficient (512MB compatible)

3. **Async Operations**:
   - FastAPI async endpoints
   - Concurrent request handling
   - Non-blocking I/O operations

4. **Frontend**:
   - Lazy loading components
   - Axios request interceptors
   - Local storage for user data

---

## API Endpoints Summary

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user
- `GET /api/auth/history` - Get analysis history

### Analysis
- `POST /api/analyze` - Upload and analyze pitch image
- `GET /api/analysis/{id}` - Get analysis by ID
- `DELETE /api/analysis/{id}` - Delete analysis

### Chat
- `POST /api/chat` - Send message to AI chatbot (Pro)

### Weather
- `GET /api/weather` - Get weather data by location
- `GET /api/weather/forecast-impact` - Get forecast impact analysis

### Subscription
- `POST /api/subscription/create-order` - Create payment order
- `POST /api/subscription/verify-payment` - Verify payment
- `GET /api/subscription/status` - Get subscription status

### Health
- `GET /api/health` - Health check endpoint

---

## Future Enhancements

1. **Real-time Features**:
   - WebSocket support for live match updates
   - Real-time chat notifications

2. **Advanced Analytics**:
   - Historical pitch performance tracking
   - Venue-specific analysis
   - Player performance correlation

3. **Mobile Optimization**:
   - Progressive Web App (PWA)
   - Mobile-specific UI improvements

4. **Scalability**:
   - Redis caching layer
   - CDN for static assets
   - Load balancing for API

---

**Generated**: February 2026  
**Version**: 2.0.0  
**Application**: Pitch Insight - AI-Powered Cricket Pitch Analysis
