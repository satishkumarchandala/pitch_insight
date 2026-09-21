# Pitch Insight

Pitch Insight is an AI-powered cricket pitch analysis platform that helps coaches, players, and cricket enthusiasts understand pitch conditions from images. The application combines computer vision, machine learning, weather data, and cricket strategy recommendations to provide actionable match insights.

## Features

- Upload cricket pitch images for automated analysis
- Detect the pitch area using computer vision
- Extract visual features such as grass coverage, cracks, moisture, texture, and brightness
- Classify pitch behavior into batting-friendly, bowling-friendly, seam-friendly, or spin-friendly categories
- Generate weather-adjusted match strategies and toss recommendations
- Provide an AI chat assistant for cricket-related questions (Pro feature)
- Support authentication, analysis history, and subscription-based premium access

## Tech Stack

### Backend
- Python
- FastAPI
- MongoDB
- PyTorch / ONNX Runtime
- OpenCV
- JWT authentication
- Razorpay integration
- Google Gemini API support

### Frontend
- React
- Vite
- Axios
- CSS

## Project Structure

```text
backend/          # FastAPI backend server
frontend/         # React frontend application
models/           # Model assets and related files (if present)
docs/             # Project documentation and guides
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.10+ installed
- Node.js 18+ installed
- MongoDB running locally or a MongoDB Atlas connection string
- API keys for:
  - Weather service
  - Gemini AI
  - Razorpay (optional for payments)

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd pitch_insight
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Create a backend environment file:

```bash
copy .env.example .env
```

Then update the values in `.env` for:

- `MONGODB_URL`
- `SECRET_KEY`
- `WEATHER_API_KEY`
- `GEMINI_API_KEY`
- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`

### 3. Run the backend

```bash
python app.py
```

The API will be available at:

- http://localhost:8000
- Swagger docs: http://localhost:8000/docs

### 4. Set up the frontend

Open a new terminal:

```bash
cd frontend
npm install
```

Create a frontend environment file if needed:

```bash
copy .env.example .env.local
```

Then start the frontend:

```bash
npm run dev
```

The frontend will be available at:

- http://localhost:5173

## Environment Notes

- The backend expects the required model files to be available in the backend directory or the configured model path.
- If you are using a local MongoDB instance, make sure it is running before starting the backend.
- The app can run without payment or AI features enabled, but some premium flows will require the relevant API keys.

## API Overview

Key API endpoints include:

- `POST /api/analyze` - Full pitch analysis with optional weather data
- `POST /api/quick-analyze` - Lightweight classification
- `GET /api/health` - Health check
- `GET /api/weather` - Weather lookup
- `GET /api/classes` - Available pitch classes

## Development Notes

- Backend logs and API docs are available via the FastAPI Swagger interface.
- Frontend is configured to communicate with the backend through the `VITE_API_URL` environment variable.
- For deployment, the backend and frontend can be hosted separately, typically on Render or Vercel.

## License

This project is intended for educational and demonstration purposes.
