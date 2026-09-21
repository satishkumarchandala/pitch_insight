# Pitch Insight — 18-Day Structured Learning & Audit Plan
**Project:** AI-Based Cricket Pitch Analysis System ("Pitch Insight")  
**Target Audience:** Senior Engineering Mentorship & Full-Stack Code Audit  
**Author:** Senior Systems & AI Architect  

---

## Executive Summary & Architecture Blueprint

Pitch Insight is a full-stack, AI-powered SaaS platform that analyzes cricket pitch images and environmental conditions to provide explainable pitch condition diagnoses, 5-day match behavior forecasts, and tactical match strategy recommendations.

### Technology Stack Mapping
- **Backend Framework:** FastAPI (Python 3.10+) with Uvicorn ASGI server
- **Computer Vision & Image Processing:** OpenCV (`cv2`), NumPy, PIL
- **Machine Learning & Object Detection:** Ultralytics YOLOv8 (pitch auto-segmentation), ResNet-18 / MobileNetV2 ONNX Runtime & PyTorch (4-class pitch classifier)
- **Weather Intelligence:** WeatherAPI.com integration with custom hourly & multi-day degradation modeling
- **Database & State:** MongoDB via Motor (async driver) and PyMongo
- **Authentication & Payments:** PyJWT, Passlib (Bcrypt), Razorpay Payment Gateway SDK
- **Frontend Framework:** React 18 (Vite build tool), Axios, React-Image-Crop, Lucide-React icons
- **Deployment & Infra:** Docker, Nginx, Render / PaaS deployment with 512MB RAM optimizations

---

## 1. Complete Module Breakdown

Below is the complete architectural mapping of the codebase divided into 8 distinct modules:

| Module ID | Module Name | Primary Responsibilities | Mapped Code Files & Folders |
| :--- | :--- | :--- | :--- |
| **MOD-01** | **Configuration & Database** | App settings, env variables, MongoDB connections, PyMongo/Motor client lifecycle | [`config.py`](file:///e:/pitch_insight/backend/config.py)<br>[`database.py`](file:///e:/pitch_insight/backend/database.py)<br>[`check_mongodb.bat`](file:///e:/pitch_insight/backend/check_mongodb.bat) |
| **MOD-02** | **Data Models & Schemas** | MongoDB Document models, Pydantic request/response validation schemas | [`models.py`](file:///e:/pitch_insight/backend/models.py)<br>[`schemas.py`](file:///e:/pitch_insight/backend/schemas.py) |
| **MOD-03** | **Authentication & Security** | JWT generation/verification, bcrypt password hashing, user tier context | [`auth.py`](file:///e:/pitch_insight/backend/auth.py)<br>[`routes/auth.py`](file:///e:/pitch_insight/backend/routes/auth.py) |
| **MOD-04** | **Computer Vision Engine** | Color space masks (HSV/LAB), Canny edge crack detection, texture GLCM variance, moisture scoring | [`pitch_analyzer.py`](file:///e:/pitch_insight/backend/pitch_analyzer.py)<br>[`utils.py`](file:///e:/pitch_insight/backend/utils.py) |
| **MOD-05** | **ML Inference & Pipeline** | YOLOv8 pitch cropping, ResNet-18/MobileNet ONNX runtime inference, PyTorch fallback, memory monitoring | [`complete_pipeline.py`](file:///e:/pitch_insight/backend/complete_pipeline.py)<br>[`complete_pipeline_onnx.py`](file:///e:/pitch_insight/backend/complete_pipeline_onnx.py)<br>[`pitch_classifier.onnx`](file:///e:/pitch_insight/backend/pitch_classifier.onnx)<br>[`pitch_yolov8_best.onnx`](file:///e:/pitch_insight/backend/pitch_yolov8_best.onnx) |
| **MOD-06** | **Weather & Forecast Engine** | WeatherAPI fetcher, 5-day match forecast, hourly pitch degradation, dew point & pressure impact | [`weather_forecast_analyzer.py`](file:///e:/pitch_insight/backend/weather_forecast_analyzer.py)<br>[`routes/weather.py`](file:///e:/pitch_insight/backend/routes/weather.py) |
| **MOD-07** | **API & Business Logic** | Orchestration of analysis, tactical strategy rules, Razorpay payments, AI Specialist chatbot | [`app.py`](file:///e:/pitch_insight/backend/app.py)<br>[`routes/analysis.py`](file:///e:/pitch_insight/backend/routes/analysis.py)<br>[`routes/subscription.py`](file:///e:/pitch_insight/backend/routes/subscription.py)<br>[`routes/chat.py`](file:///e:/pitch_insight/backend/routes/chat.py)<br>[`razorpay_handler.py`](file:///e:/pitch_insight/backend/razorpay_handler.py) |
| **MOD-08** | **Frontend Application** | React UI hierarchy, Axios interceptors, cropping modal, dashboards, chat widget | [`frontend/src/App.jsx`](file:///e:/pitch_insight/frontend/src/App.jsx)<br>[`frontend/src/services/api.js`](file:///e:/pitch_insight/frontend/src/services/api.js)<br>[`UploadSection.jsx`](file:///e:/pitch_insight/frontend/src/components/UploadSection.jsx)<br>[`ResultsSection.jsx`](file:///e:/pitch_insight/frontend/src/components/ResultsSection.jsx)<br>[`WeatherForecastDisplay.jsx`](file:///e:/pitch_insight/frontend/src/components/WeatherForecastDisplay.jsx)<br>[`SpecialistAnalysis.jsx`](file:///e:/pitch_insight/frontend/src/components/SpecialistAnalysis.jsx)<br>[`ChatWidget.jsx`](file:///e:/pitch_insight/frontend/src/components/ChatWidget.jsx)<br>[`HistorySection.jsx`](file:///e:/pitch_insight/frontend/src/components/HistorySection.jsx)<br>[`PaymentModal.jsx`](file:///e:/pitch_insight/frontend/src/components/PaymentModal.jsx)<br>[`Auth.jsx`](file:///e:/pitch_insight/frontend/src/components/Auth.jsx) |

---

## 2. Interaction Mapping & System Data Flow

### Module Dependency Matrix

```text
[ MOD-08 React Frontend ] 
         |
         | HTTP / REST (Axios + JWT Interceptor)
         v
[ MOD-07 FastAPI API Router ] <-------------> [ MOD-03 Auth & Security ]
   |         |            |
   |         |            +-----------------> [ MOD-06 Weather Engine ]
   |         v                                         |
   |   [ MOD-05 ML Inference Pipeline ]                v
   |         |                              [ WeatherAPI.com ]
   |         v
   |   [ MOD-04 OpenCV Pitch Analyzer ]
   |
   +----------------------------------------> [ MOD-02 Schemas & Models ]
   |                                                   |
   v                                                   v
[ MOD-01 Database (MongoDB) ] <------------------------+
```

### End-to-End Request Sequence: Pitch Analysis Flow

```text
User Selects Image & City in React UI (MOD-08)
   │
   ├─► Axios Interceptor adds `Authorization: Bearer <JWT>` header (`api.js`)
   │
   ▼
POST /api/analyze (MOD-07: routes/analysis.py)
   │
   ├─► 1. Authenticate Token (MOD-03: auth.py -> get_optional_current_user)
   │
   ├─► 2. Validate Usage Limits (MOD-03/07: check user free_analyses count vs tier)
   │
   ├─► 3. Compute Image MD5 Hash (MOD-04: utils.py -> check memory cache)
   │
   ├─► 4. Run ML Pipeline (MOD-05: complete_pipeline_onnx.py)
   │      ├── YOLOv8 ONNX detects and crops pitch bounding box
   │      ├── ResNet-18 ONNX predicts 4 class probabilities: [batting, bowling, seam, spin]
   │      └── OpenCV Engine (MOD-04: pitch_analyzer.py) extracts:
   │          ├── Grass Coverage % (HSV Mask)
   │          ├── Crack Density & Count (Canny Edges + Contour Aspect Ratio)
   │          ├── Moisture Score (HSV Saturation + Luminance)
   │          └── Texture Variance (GLCM Edge Variance)
   │
   ├─► 5. Fetch Weather & Degradation (MOD-06: weather_forecast_analyzer.py)
   │      ├── Call WeatherAPI.com (current weather + 5-day forecast)
   │      └── Compute hourly pitch moisture loss, UV index impact, dew factor
   │
   ├─► 6. Execute Feature Fusion Engine (MOD-07: routes/analysis.py)
   │      ├── Apply rule-based score adjustments (e.g. Grass > 40% -> seam +15%)
   │      ├── Calculate Final Confidence Score based on model-weather agreement
   │      └── Generate Tactical Match Strategy (Toss decision, Batting, Bowling, Team XI)
   │
   ├─► 7. Store Result in MongoDB (MOD-01: database.py -> get_analysis_collection)
   │      └── Convert NumPy data types -> Python native dict via convert_numpy_types()
   │
   ▼
Return JSON Response (MOD-02: PitchAnalysisResponse Schema)
   │
   ▼
React UI (MOD-08) renders:
   ├── ResultsSection.jsx (Confidence gauge, pitch scores, strategy cards)
   ├── WeatherForecastDisplay.jsx (5-day match forecast tabs)
   ├── SpecialistAnalysis.jsx (Expert insights breakdown)
   └── ChatWidget.jsx (Passes analysis_id for interactive strategy Q&A)
```

---

## 3. Deep-Dive Day-by-Day Study & Audit Plan (Days 1–18)

---

### DAY 1: System Initialization, Configuration & Server Lifecycle
- **Focus Area:** [`config.py`](file:///e:/pitch_insight/backend/config.py), [`app.py`](file:///e:/pitch_insight/backend/app.py), [`routes/health.py`](file:///e:/pitch_insight/backend/routes/health.py)
- **What to Read & Study:**
  1. Inspect how environment variables (`MONGODB_URL`, `WEATHER_API_KEY`, `JWT_SECRET_KEY`, `RAZORPAY_*`) are parsed in `config.py`.
  2. Study the FastAPI `lifespan` async context manager in `app.py` (lines 14–32).
  3. Inspect CORS configuration (`allow_origins=ALLOWED_ORIGINS`) in `app.py` (lines 43–50).
- **Key Questions to Answer:**
  - What happens during server startup and shutdown in the `lifespan` handler?
  - How are default fallback values handled if `.env` is missing or incomplete?
  - What security risk exists with the current CORS setup if `ALLOWED_ORIGINS` defaults to `["*"]`?
- **Hands-on Exercise:**
  - Modify `config.py` to add a print log on startup showing all loaded config keys (masking secret keys).
  - Trigger the `GET /api/health` endpoint via curl or Postman and inspect system uptime and database health output.
- **Common Pitfalls & Subtle Logic:**
  - `DEBUG` defaults to `True` if `ENVIRONMENT != "production"`, which reloads Uvicorn on every file change and can duplicate model loads in memory!

---

### DAY 2: Database Driver Architecture & Schema Modeling
- **Focus Area:** [`database.py`](file:///e:/pitch_insight/backend/database.py), [`models.py`](file:///e:/pitch_insight/backend/models.py), [`schemas.py`](file:///e:/pitch_insight/backend/schemas.py)
- **What to Read & Study:**
  1. Examine PyMongo vs Motor async database connection handling in `database.py`.
  2. Study PyDantic models in `schemas.py` for request validation (`UserCreate`, `UserLogin`) and response validation (`PitchAnalysisResponse`).
  3. Study database model helpers in `models.py` for PyDantic `ObjectId` string conversions.
- **Key Questions to Answer:**
  - Is database connection pooling properly reused across incoming FastAPI request threads?
  - How does PyDantic validate missing optional fields in `PitchAnalysisResponse`?
  - Why is `PyObjectId` required when converting MongoDB `_id` field to JSON?
- **Hands-on Exercise:**
  - Run [`check_mongodb.bat`](file:///e:/pitch_insight/backend/check_mongodb.bat) locally.
  - Create a Python script in `scratch/test_db.py` to insert a dummy analysis document into MongoDB and retrieve it by `_id`.
- **Common Pitfalls & Subtle Logic:**
  - PyMongo synchronous client operations will block the FastAPI event loop if called inside `async def` routes without using Motor or `run_in_executor`.

---

### DAY 3: Authentication, Security & User Tier Authorization
- **Focus Area:** [`auth.py`](file:///e:/pitch_insight/backend/auth.py), [`routes/auth.py`](file:///e:/pitch_insight/backend/routes/auth.py)
- **What to Read & Study:**
  1. Inspect `create_access_token()`, `verify_password()`, and `get_password_hash()` in `auth.py`.
  2. Study `get_current_user` and `get_optional_current_user` FastAPI OAuth2 dependence injectables.
  3. Trace the user registration (`POST /api/auth/signup`) and login (`POST /api/auth/login`) flow.
- **Key Questions to Answer:**
  - What is the difference between `get_current_user` and `get_optional_current_user`? Where is each used?
  - How are user subscription tiers (`free`, `pro`, `elite`) stored and validated against usage limits?
  - What algorithm and expiration timeframe are used for JWT signing?
- **Hands-on Exercise:**
  - Send a POST request to `/api/auth/signup` to register a test user.
  - Decode the generated JWT token using `jwt.decode` in Python and print token claims (`sub`, `exp`).
- **Common Pitfalls & Subtle Logic:**
  - If `JWT_SECRET_KEY` falls back to a default hardcoded string when environment variable is missing, any user can forge valid JWT tokens!

---

### DAY 4: Computer Vision — Feature Extraction Engine
- **Focus Area:** [`pitch_analyzer.py`](file:///e:/pitch_insight/backend/pitch_analyzer.py)
- **What to Read & Study:**
  1. Study `PitchAnalyzer` class methods: `_detect_grass()`, `_detect_cracks()`, `_analyze_moisture()`, `_analyze_color()`, and `_analyze_texture()`.
  2. Trace color-space transformations (BGR -> HSV, BGR -> LAB, BGR -> Gray).
  3. Inspect edge detection thresholds in `_detect_cracks()` (Canny edge parameters, contour aspect ratio `> 3`).
- **Key Questions to Answer:**
  - How does the HSV mask separate green grass from brown/dry soil pixels?
  - How does `_detect_cracks()` distinguish between real pitch cracks and dark shadows or grass boundaries?
  - What metric measures surface roughness in `_analyze_texture()`?
- **Hands-on Exercise:**
  - Write a standalone test script `scratch/test_cv.py` loading an image from `complete_pitch_analysis.png`.
  - Save output binary masks (`mask_green`, dilated crack edges) to disk and print calculated grass % and crack density.
- **Common Pitfalls & Subtle Logic:**
  - OpenCV uses **BGR** channel ordering, whereas PIL and PyTorch use **RGB**. Passing BGR images to PIL/PyTorch results in corrupted color classification!

---

### DAY 5: Machine Learning Pipeline — YOLOv8 Detection & PyTorch Pipeline
- **Focus Area:** [`complete_pipeline.py`](file:///e:/pitch_insight/backend/complete_pipeline.py)
- **What to Read & Study:**
  1. Study `CompletePitchPipeline` initialization and lazy model loading (`_load_models()`).
  2. Inspect YOLO pitch bounding box detection and image cropping logic.
  3. Trace image preprocessing transforms (Resize 224x224, ImageNet normalization) for ResNet-18 / MobileNetV2 classification.
- **Key Questions to Answer:**
  - Why is lazy loading implemented for machine learning models?
  - What fallback occurs if YOLOv8 fails to detect a pitch in the image?
  - How does `MemoryMonitor` track RSS memory usage during inference?
- **Hands-on Exercise:**
  - Instantiate `CompletePitchPipeline(lazy_load=True)` in Python and log memory consumption before and after running inference on a pitch image.
- **Common Pitfalls & Subtle Logic:**
  - PyTorch CUDA memory is not automatically freed when objects are garbage collected unless `torch.cuda.empty_cache()` and `gc.collect()` are explicitly called.

---

### DAY 6: High-Performance ONNX Runtime Optimization
- **Focus Area:** [`complete_pipeline_onnx.py`](file:///e:/pitch_insight/backend/complete_pipeline_onnx.py), [`utils.py`](file:///e:/pitch_insight/backend/utils.py)
- **What to Read & Study:**
  1. Study `CompletePitchPipeline` in `complete_pipeline_onnx.py` using `onnxruntime.InferenceSession`.
  2. Inspect `preprocess_yolo_image()` letterbox padding (640x640 padding with color `(114, 114, 114)`).
  3. Compare ONNX runtime execution speed and RAM usage vs. PyTorch pipeline.
- **Key Questions to Answer:**
  - Why does ONNX Runtime consume significantly less RAM than PyTorch?
  - How does `get_pipeline()` in `utils.py` implement singleton caching for the pipeline instance?
  - What execution providers are passed to `InferenceSession` (`CPUExecutionProvider` vs `CUDAExecutionProvider`)?
- **Hands-on Exercise:**
  - Benchmark inference speed: compare execution time of `complete_pipeline.py` (PyTorch) vs `complete_pipeline_onnx.py` (ONNX) over 10 iterations.
- **Common Pitfalls & Subtle Logic:**
  - YOLOv8 ONNX output shape `[1, 5, 8400]` requires custom NMS (Non-Maximum Suppression) post-processing in Python to extract bounding box coordinates `[x, y, w, h]` and confidence scores!

---

### DAY 7: Weather Intelligence & Hourly Pitch Degradation Engine
- **Focus Area:** [`weather_forecast_analyzer.py`](file:///e:/pitch_insight/backend/weather_forecast_analyzer.py), [`routes/weather.py`](file:///e:/pitch_insight/backend/routes/weather.py)
- **What to Read & Study:**
  1. Study `WeatherForecastAnalyzer` class methods: `get_forecast()`, `_analyze_day_forecast()`, and `_calculate_pitch_impact()`.
  2. Inspect pitch moisture loss formulas based on temperature, humidity, UV index, and wind speed.
  3. Study 5-day match forecast analysis and session-by-session pitch condition degradation logic.
- **Key Questions to Answer:**
  - How does the weather engine calculate the "Dew Risk Score" for evening/night matches?
  - What formula estimates pitch surface moisture loss per hour under high UV and wind?
  - How does weather data modify the base pitch condition (e.g. converting a "Moist" pitch to "Dry & Dusty" by Day 4)?
- **Hands-on Exercise:**
  - Call `weather_forecast_analyzer.get_forecast("London", match_format="TEST")` in Python (using mock or real key) and inspect the returned 5-day session degradation dictionary.
- **Common Pitfalls & Subtle Logic:**
  - Missing WeatherAPI key or API downtime must be gracefully caught with fallback default weather metrics, preventing 500 server crashes.

---

### DAY 8: Core Analysis Orchestrator & Tactical Match Strategy Engine
- **Focus Area:** [`routes/analysis.py`](file:///e:/pitch_insight/backend/routes/analysis.py)
- **What to Read & Study:**
  1. Study `/api/analyze` and `/api/quick-analyze` route handlers.
  2. Trace `generate_match_strategy()` (lines 52–140) generating Toss Decision, Batting Strategy, Bowling Strategy, and Team Composition.
  3. Inspect `convert_numpy_types()` (lines 28–49) preventing MongoDB serialization errors.
- **Key Questions to Answer:**
  - How are OpenCV visual features, ONNX neural net probabilities, and WeatherAPI metrics fused together?
  - What specific feature rules trigger a "Bowl First" vs "Bat First" toss recommendation?
  - How does the route check and enforce user daily analysis quotas based on subscription tier?
- **Hands-on Exercise:**
  - Send a multipart image upload request to `POST /api/analyze` via Postman/curl and verify all JSON response keys match `PitchAnalysisResponse`.
- **Common Pitfalls & Subtle Logic:**
  - Storing raw NumPy arrays (`np.ndarray`) or OpenCV mask matrices directly in MongoDB throws `InvalidDocument: cannot encode object of type ndarray`.

---

### DAY 9: Monetization, Payment Verification & Subscription Management
- **Focus Area:** [`razorpay_handler.py`](file:///e:/pitch_insight/backend/razorpay_handler.py), [`routes/subscription.py`](file:///e:/pitch_insight/backend/routes/subscription.py)
- **What to Read & Study:**
  1. Study `RazorpayHandler` order creation (`create_order()`) and HMAC-SHA256 signature verification (`verify_payment()`).
  2. Inspect subscription tier definitions (`FREE`, `PRO`, `ELITE`) and quota limits.
  3. Study `/api/subscription/verify-payment` DB transaction updating user tier and expiration dates.
- **Key Questions to Answer:**
  - How does HMAC-SHA256 signature verification prevent fraudulent payment confirmation calls?
  - What happens when a user's subscription expires? How is usage fallback handled?
  - How are Razorpay key ID and secret loaded dynamically per environment?
- **Hands-on Exercise:**
  - Write a python snippet calling `RazorpayHandler.verify_signature(order_id, payment_id, signature)` with valid and invalid signatures to verify cryptographic verification logic.
- **Common Pitfalls & Subtle Logic:**
  - Verifying payment signatures using loose string comparisons instead of `hmac.compare_digest` leaves the system vulnerable to timing attack exploits!

---

### DAY 10: Interactive AI Specialist Chat Assistant
- **Focus Area:** [`routes/chat.py`](file:///e:/pitch_insight/backend/routes/chat.py)
- **What to Read & Study:**
  1. Study `POST /api/chat` route handler.
  2. Inspect how pitch analysis history context (`analysis_id`) is retrieved from MongoDB and injected into the AI chat prompt.
  3. Trace rule-based AI expert recommendations for pitch tactics and bowler selection.
- **Key Questions to Answer:**
  - How does the chat endpoint maintain context awareness about the specific pitch analyzed by the user?
  - What fallback response is generated if `analysis_id` is null or invalid?
  - How is chat message history stored or scoped per user?
- **Hands-on Exercise:**
  - Post a message to `/api/chat` passing a valid `analysis_id` and question: *"Which bowler should bowl the 15th over?"* and analyze the generated response.
- **Common Pitfalls & Subtle Logic:**
  - Unsanitized user inputs in chat prompts could lead to prompt injection or backend log manipulation if not properly escaped.

---

### DAY 11: Frontend Architecture, Axios Client & Authentication State
- **Focus Area:** [`frontend/src/services/api.js`](file:///e:/pitch_insight/frontend/src/services/api.js), [`frontend/src/App.jsx`](file:///e:/pitch_insight/frontend/src/App.jsx), [`frontend/src/components/Auth.jsx`](file:///e:/pitch_insight/frontend/src/components/Auth.jsx)
- **What to Read & Study:**
  1. Study Axios setup in `api.js`: request interceptor inserting `Authorization: Bearer <token>` and response interceptor handling 401 token expiration.
  2. Inspect global state management in `App.jsx` (`user`, `token`, `currentAnalysis`, `activeTab`).
  3. Study login/signup state, input handling, and localStorage token persistence in `Auth.jsx`.
- **Key Questions to Answer:**
  - What happens in `api.js` when the backend returns a `401 Unauthorized` status?
  - How does `App.jsx` synchronize user authentication state on page reload (`authAPI.getMe()`)?
  - How are API base URLs dynamically configured for development vs. production builds?
- **Hands-on Exercise:**
  - Open Chrome DevTools Network tab. Perform a login, inspect set headers in `api.js`, then manually clear `localStorage` token to verify redirect behavior.
- **Common Pitfalls & Subtle Logic:**
  - Storing auth tokens in `localStorage` exposes them to XSS attacks. (Consider HTTP-only cookies in production upgrades).

---

### DAY 12: Image Upload & Interactive Manual Cropping UI
- **Focus Area:** [`frontend/src/components/UploadSection.jsx`](file:///e:/pitch_insight/frontend/src/components/UploadSection.jsx)
- **What to Read & Study:**
  1. Study drag-and-drop file upload handling and image file validation (file size `< 5MB`, format check).
  2. Inspect React image cropper integration (`ReactCrop` canvas cropping).
  3. Trace how cropped image blob is packaged into `FormData` and dispatched to `analysisAPI.analyzeComplete()`.
- **Key Questions to Answer:**
  - How does `UploadSection.jsx` handle both automatic full-image uploads and user manual pitch cropping?
  - How are city names, match formats, and match start times passed along with the image file in `FormData`?
  - What user visual feedback is displayed during backend ML processing?
- **Hands-on Exercise:**
  - Upload a high-resolution pitch image in the React UI, apply a custom manual crop box, and inspect the `FormData` binary payload in DevTools Network tab.
- **Common Pitfalls & Subtle Logic:**
  - Canvas cropping without scaling image display size to natural pixel width results in low-resolution or distorted crop coordinates being sent to backend!

---

### DAY 13: Results Dashboard, Specialist & Weather Visualizations
- **Focus Area:** [`frontend/src/components/ResultsSection.jsx`](file:///e:/pitch_insight/frontend/src/components/ResultsSection.jsx), [`frontend/src/components/WeatherForecastDisplay.jsx`](file:///e:/pitch_insight/frontend/src/components/WeatherForecastDisplay.jsx), [`frontend/src/components/SpecialistAnalysis.jsx`](file:///e:/pitch_insight/frontend/src/components/SpecialistAnalysis.jsx)
- **What to Read & Study:**
  1. Study rendering of confidence gauge, pitch suitability progress bars (Batting %, Seam %, Spin %), and feature metrics cards in `ResultsSection.jsx`.
  2. Inspect 5-day match forecast tab navigation and session deterioration UI in `WeatherForecastDisplay.jsx`.
  3. Study tactical strategy recommendations UI (Toss decision, XI selection) in `SpecialistAnalysis.jsx`.
- **Key Questions to Answer:**
  - How are visual metrics (Grass %, Crack Density) formatted into color-coded progress badges?
  - How does `WeatherForecastDisplay.jsx` handle null or missing weather forecast data gracefully?
  - What visual cues distinguish high-confidence predictions from low-confidence predictions?
- **Hands-on Exercise:**
  - Modify `ResultsSection.jsx` CSS styling to add a custom badge for "Extreme Spin Warning" when `spin_suitability > 75%`.
- **Common Pitfalls & Subtle Logic:**
  - Deep nested JSON response properties (e.g. `analysis.weather.forecast.days[0].sessions`) will crash React with `TypeError: Cannot read property of undefined` if optional chaining (`?.`) is omitted!

---

### DAY 14: Chat Widget, History Tracker & Subscription Modals
- **Focus Area:** [`frontend/src/components/ChatWidget.jsx`](file:///e:/pitch_insight/frontend/src/components/ChatWidget.jsx), [`frontend/src/components/HistorySection.jsx`](file:///e:/pitch_insight/frontend/src/components/HistorySection.jsx), [`frontend/src/components/PaymentModal.jsx`](file:///e:/pitch_insight/frontend/src/components/PaymentModal.jsx)
- **What to Read & Study:**
  1. Study sliding chat widget drawer, message state array, and backend stream/response handling in `ChatWidget.jsx`.
  2. Inspect past pitch analysis listing, search filtering, and reload logic in `HistorySection.jsx`.
  3. Study Razorpay Checkout script loading and order trigger in `PaymentModal.jsx`.
- **Key Questions to Answer:**
  - How does `ChatWidget.jsx` attach `currentAnalysis._id` to outbound messages?
  - How does `PaymentModal.jsx` handle successful payment callbacks and trigger user tier refreshes?
  - How is user pitch history cached vs refetched on demand?
- **Hands-on Exercise:**
  - Open `HistorySection.jsx`, select a historical analysis record, and verify that the full results dashboard repopulates with historical data.
- **Common Pitfalls & Subtle Logic:**
  - Razorpay SDK script (`checkout.js`) must be asynchronously loaded on window object prior to user clicking "Upgrade Plan", or button click will fail silently.

---

### DAY 15: Deployment Infrastructure, Docker & Memory Optimization
- **Focus Area:** [`backend/Dockerfile`](file:///e:/pitch_insight/backend/Dockerfile), [`backend/MEMORY_OPTIMIZATION.md`](file:///e:/pitch_insight/backend/MEMORY_OPTIMIZATION.md), [`backend/render.yaml`](file:///e:/pitch_insight/backend/render.yaml), [`frontend/nginx.conf`](file:///e:/pitch_insight/frontend/nginx.conf)
- **What to Read & Study:**
  1. Study single-worker Uvicorn startup flags (`uvicorn app:app --workers 1 --limit-concurrency 10`) designed for 512MB RAM environments.
  2. Inspect multi-stage Docker build steps in `backend/Dockerfile` and `frontend/Dockerfile`.
  3. Study Nginx reverse proxy configuration (`nginx.conf`) routing `/api` traffic to FastAPI container.
- **Key Questions to Answer:**
  - Why is Uvicorn worker count limited to 1 in low-memory cloud instances (e.g. Render Free Tier)?
  - How do environment variables in `render.yaml` coordinate frontend and backend microservices?
  - What static asset caching headers are set in `nginx.conf`?
- **Hands-on Exercise:**
  - Build the backend Docker container locally (`docker build -t pitch-insight-backend ./backend`) and run it with RAM limitation flag (`--memory=512m`).
- **Common Pitfalls & Subtle Logic:**
  - Spawning multiple Uvicorn worker processes in a 512MB container loads duplicate ONNX/YOLO model copies into RAM, causing immediate `OOM Killed` crashes!

---

### DAY 16: Codebase Audit — Identifying Inefficiencies, Code Smells & Security Gaps
- **Focus Area:** Full Repository Review
- **What to Read & Study:**
  1. Review global mutable variables in [`routes/analysis.py`](file:///e:/pitch_insight/backend/routes/analysis.py#L25) (`analysis_cache = {}`).
  2. Inspect exception swallowing (`except Exception: pass`) across backend service modules.
  3. Search for synchronous blocking calls (e.g. `requests.get` or `cv2.imread`) executed directly inside async route handlers.
- **Key Findings & Vulnerability Audit:**
  - **Security Gap:** `JWT_SECRET_KEY` fallback in `config.py` uses a predictable string if env variable is absent.
  - **Performance Bottleneck:** OpenCV image processing (`cv2.imread`, Canny edge computation) in `pitch_analyzer.py` is CPU-bound and blocks the main Uvicorn async event loop during execution.
  - **Memory Leak Risk:** `analysis_cache` dictionary in `analysis.py` grows indefinitely without TTL eviction policy.
- **Hands-on Exercise:**
  - Perform a static code audit scan across backend files using `flake8` or `bandit` to list security warnings and unhandled exceptions.

---

### DAY 17: Architectural Refactoring & Robust Error Handling Proposals
- **Focus Area:** Refactoring Blueprints
- **Refactoring Strategy & Action Plan:**
  1. **Async Offloading for CV Computations:**
     Wrap blocking OpenCV feature extraction calls in FastAPI background execution threads using `asyncio.to_thread(analyzer.analyze, path)`.
  2. **Cache Eviction Policy:**
     Replace unbound python dictionary `analysis_cache` with an `TTLCache` (e.g. `cachetools.TTLCache(maxsize=100, ttl=3600)`).
  3. **Strict Environment Enforcement:**
     Raise an explicit `ValueError` on startup if critical secrets (`JWT_SECRET_KEY`, `WEATHER_API_KEY`) are missing when `ENVIRONMENT=production`.
  4. **Centralized Error Middleware:**
     Implement global FastAPI exception handlers returning uniform JSON error objects (`{"error": True, "message": "..."}`).

---

### DAY 18: System Mastery Assessment & Audit Verification
- **Focus Area:** Final End-to-End Verification & Mastery Review
- **Final Audit Task:** Verify complete application flow from initial CLI startup script [`start_server.bat`](file:///e:/pitch_insight/backend/start_server.bat) through pitch analysis, strategic match output, and database verification.

---

## 4. Testing & Isolated Verification Matrix

To verify understanding without assumptions, execute the following isolated test suites for each module:

```text
+-----------------------+----------------------------------+-------------------------------------------------+
| Module                | Isolation Testing Strategy       | Verification Command / Action                   |
+-----------------------+----------------------------------+-------------------------------------------------+
| MOD-01 Database       | Test MongoDB connection & Motor  | Run check_mongodb.bat & insert test doc         |
| MOD-03 Auth           | Unit test JWT issuance & hash    | pytest backend/tests/test_auth.py (or script)   |
| MOD-04 ComputerVision | Test CV analyzer on test images  | python backend/temp_test.py                     |
| MOD-05 ML Pipeline    | Compare ONNX vs PyTorch output   | python backend/test_onnx_vs_pt.py               |
| MOD-06 Weather Engine | Test WeatherAPI parser & mock    | Test weather_forecast_analyzer with sample city |
| MOD-07 API Router     | Full HTTP API Integration test   | python backend/test_api.py                      |
| MOD-08 Frontend       | React Component UI Render Test   | npm run dev & run Cypress/Playwright or manual  |
+-----------------------+----------------------------------+-------------------------------------------------+
```

### Sample Automated Verification Script (`backend/test_onnx_vs_pt.py`)
```python
# Verify ONNX and PyTorch outputs match within tolerance
import numpy as np
from complete_pipeline import CompletePitchPipeline as PyTorchPipeline
from complete_pipeline_onnx import CompletePitchPipeline as ONNXPipeline

pt_pipe = PyTorchPipeline(lazy_load=False)
onnx_pipe = ONNXPipeline()

test_image_path = "pitch_analysis_report_20251218_125952.png"
pt_res = pt_pipe.analyze(test_image_path)
onnx_res = onnx_pipe.analyze(test_image_path)

print("PyTorch Prediction:", pt_res['prediction'])
print("ONNX Prediction:   ", onnx_res['prediction'])
# Assert predictions align
```

---

## 5. Prioritized Codebase Improvements Roadmap

| Priority | Category | Affected File | Problem Statement | Proposed Refactoring |
| :--- | :--- | :--- | :--- | :--- |
| **P0 (Critical)** | **Security** | [`config.py`](file:///e:/pitch_insight/backend/config.py) | Insecure JWT secret fallback allows signature forging if env var missing. | Raise exception at startup in production if `JWT_SECRET_KEY` is not explicitly set. |
| **P1 (High)** | **Performance** | [`pitch_analyzer.py`](file:///e:/pitch_insight/backend/pitch_analyzer.py) | Synchronous OpenCV functions block FastAPI main event loop. | Offload CPU-heavy CV analysis using `await asyncio.to_thread()`. |
| **P1 (High)** | **Memory** | [`routes/analysis.py`](file:///e:/pitch_insight/backend/routes/analysis.py) | `analysis_cache` dictionary has no max limit or TTL eviction. | Replace with `cachetools.TTLCache(maxsize=100, ttl=1800)`. |
| **P2 (Medium)**| **Code Quality** | [`complete_pipeline.py`](file:///e:/pitch_insight/backend/complete_pipeline.py) | Duplicate pipeline logic between PyTorch & ONNX files. | Create abstract `BasePitchPipeline` class; inherit PyTorch & ONNX implementations. |
| **P2 (Medium)**| **UX/API** | [`routes/chat.py`](file:///e:/pitch_insight/backend/routes/chat.py) | Chat endpoint lacks message rate-limiting per user tier. | Add tier-based rate-limiting middleware (Free: 5 msgs/day, Pro: 50 msgs/day). |

---

## 6. System Mastery Checklist

To prove complete 100% end-to-end understanding of the **Pitch Insight** platform, you should be able to answer all 16 technical questions below without referencing the documentation:

### Backend & Infrastructure
1. How does FastAPI's async event loop handle CPU-bound OpenCV tasks versus I/O-bound MongoDB calls?
2. What specific environment configuration allows the backend to run ONNX models within a 512MB RAM container constraint without triggering OOM errors?
3. What is the exact sequence of HTTP headers and status codes returned when an unauthenticated user attempts to access a Pro-tier analysis endpoint?
4. How does `convert_numpy_types()` work, and why does PyMongo fail if NumPy `float32` or `int64` data types are passed directly to `insert_one()`?

### Computer Vision & ML Pipeline
5. What are the specific HSV lower and upper color threshold vectors used in `_detect_grass()` to isolate pitch grass coverage from soil?
6. How does Canny edge detection in `_detect_cracks()` filter out shadow lines using contour aspect ratio filtering (`aspect_ratio > 3`)?
7. What is letterbox padding in YOLOv8 image preprocessing, and why is color `(114, 114, 114)` used for margin padding in `preprocess_yolo_image()`?
8. How are the 4 ResNet-18 output logits converted into normalized class probability distributions for `batting_friendly`, `bowling_friendly`, `seam_friendly`, and `spin_friendly`?

### Business Logic & Weather Fusion Engine
9. In `generate_match_strategy()`, what visual and weather thresholds trigger a recommended toss decision of *"Bat First"* versus *"Bowl First"*?
10. How does `WeatherForecastAnalyzer` model hourly pitch moisture evaporation as a function of temperature, wind speed, and UV index?
11. What mathematical logic combines model classifier probabilities, CV visual metrics, and weather conditions into the final `confidence_score`?
12. How does HMAC-SHA256 verification validate Razorpay payment payloads in `verify_payment()`, and why is `hmac.compare_digest` required?

### Frontend & Client Integration
13. How does the Axios request interceptor in `api.js` attach JWT bearer tokens to outbound requests, and how does the response interceptor recover from `401 Unauthorized` responses?
14. How does `UploadSection.jsx` extract image crop coordinates from `ReactCrop` and package the binary image blob into a `multipart/form-data` payload?
15. How does `App.jsx` maintain persistent authentication state across browser refreshes using `localStorage` and `authAPI.getMe()`?
16. How does `ChatWidget.jsx` send pitch analysis context (`analysis_id`) to the backend to enable context-aware AI pitch strategy responses?
