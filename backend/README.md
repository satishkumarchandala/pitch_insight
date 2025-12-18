# Pitch Insight Backend API

FastAPI backend server for cricket pitch analysis with weather integration.

## Features

✅ **Complete Pitch Analysis**
- YOLO pitch detection
- OpenCV feature extraction (grass, cracks, moisture, etc.)
- ResNet18 classification
- Cricket domain intelligence

✅ **Weather Integration**
- Real-time weather data (OpenWeatherMap API)
- Weather impact analysis on pitch behavior
- Location-based forecasting

✅ **Match Strategy Generation**
- Toss decision recommendations
- Batting and bowling strategies
- Team composition suggestions
- Weather-adjusted tactics

✅ **API Documentation**
- Auto-generated Swagger UI
- Interactive API testing
- Complete request/response schemas

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your OpenWeatherMap API key:

```bash
cp .env.example .env
# Edit .env and add your API key
```

Get a free API key: https://openweathermap.org/api

### 3. Copy Model Files

Ensure these model files are in the parent directory:
- `pitch_yolov8_best.pt`
- `best_pitch_classifier.pth`

### 4. Run Server

```bash
python app.py
```

Server will start at: http://localhost:8000

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Complete Analysis
**POST** `/api/analyze`

Upload pitch image with optional location data for complete analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@pitch_image.jpg" \
  -F "latitude=28.6139" \
  -F "longitude=77.2090" \
  -F "city=Mumbai" \
  -F "include_weather=true"
```

**Response:**
```json
{
  "success": true,
  "analysis_id": "PITCH_20241217_143052",
  "pitch_detection": {
    "detected": true,
    "confidence": 0.93,
    "bbox": [196, 270, 440, 478]
  },
  "features": {
    "grass_coverage": {"percentage": 5.0, "level": "Minimal"},
    "cracks": {"severity": "Medium", "count": 12},
    "moisture": {"level": "Damp", "score": 45}
  },
  "ml_classification": {
    "prediction": "spin_friendly",
    "confidence": 81.8
  },
  "final_classification": {
    "prediction": "spin_friendly",
    "confidence": 83.7,
    "adjustments": ["+1.9%"],
    "reasons": ["Moderate cracking helps spinners"]
  },
  "weather": {
    "temperature": 28.5,
    "humidity": 65,
    "rainfall": 0.0,
    "wind_speed": 12.5,
    "conditions": "clear sky",
    "location": "Mumbai"
  },
  "match_strategy": {
    "pitch_type": "spin_friendly",
    "toss_decision": "Bat first - pitch will deteriorate",
    "batting_strategy": [...],
    "bowling_strategy": [...],
    "team_composition": [...]
  },
  "processing_time": 2.45
}
```

### 2. Quick Classification
**POST** `/api/quick-analyze`

Fast classification without detailed feature extraction.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/quick-analyze" \
  -F "image=@pitch_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "prediction": "batting_friendly",
  "confidence": 91.2,
  "probabilities": {
    "batting_friendly": 91.2,
    "bowling_friendly": 3.5,
    "seam_friendly": 2.8,
    "spin_friendly": 2.5
  },
  "processing_time": 0.85
}
```

### 3. Weather Data
**GET** `/api/weather?latitude=28.6139&longitude=77.2090&city=Mumbai`

Fetch current weather for a location.

### 4. Pitch Classes
**GET** `/api/classes`

Get list of all pitch classes with descriptions.

### 5. Health Check
**GET** `/api/health`

Check API health status.

## Python Client Example

```python
import requests

# Complete analysis
with open('pitch_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze',
        files={'image': f},
        data={
            'latitude': 28.6139,
            'longitude': 77.2090,
            'city': 'Mumbai',
            'include_weather': True
        }
    )

result = response.json()
print(f"Pitch Type: {result['final_classification']['prediction']}")
print(f"Confidence: {result['final_classification']['confidence']:.1f}%")
print(f"Toss Decision: {result['match_strategy']['toss_decision']}")
```

## Testing with cURL

### Test Complete Analysis
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@e:/pitch_insight/pitch_classification/test/spin_friendly/img000000029_jpeg.rf.5a2e387c8ed6ec3093e7282e4dfd8f5f.jpg" \
  -F "latitude=28.6139" \
  -F "longitude=77.2090" \
  -F "city=Mumbai"
```

### Test Quick Analysis
```bash
curl -X POST http://localhost:8000/api/quick-analyze \
  -F "image=@e:/pitch_insight/pitch_classification/test/batting_friendly/img000000001_jpeg.rf.0dff83aee56a34c64c14ccf10f8e9d2d.jpg"
```

## Project Structure

```
backend/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create from .env.example)
├── .env.example          # Environment template
└── README.md             # This file
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WEATHER_API_KEY` | OpenWeatherMap API key | No (weather features disabled without it) |
| `HOST` | Server host | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8000) |
| `ALLOWED_ORIGINS` | CORS allowed origins | No (default: *) |

## Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../pitch_yolov8_best.pt .
COPY ../best_pitch_classifier.pth .
COPY ../complete_pipeline.py .

EXPOSE 8000

CMD ["python", "app.py"]
```

### Deploy to Render

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: pitch-insight-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: WEATHER_API_KEY
        sync: false
```

2. Push to GitHub and connect to Render

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable (weather API)

Error response format:
```json
{
  "detail": "Error message"
}
```

## Performance

- **Quick Analysis**: ~0.8-1.5s
- **Complete Analysis** (no weather): ~2-3s
- **Complete Analysis** (with weather): ~3-4s

Processing time depends on:
- Image size and resolution
- CPU/GPU availability
- Network latency (weather API)

## Troubleshooting

### Models not found
```
FileNotFoundError: pitch_yolov8_best.pt
```
**Solution**: Copy model files to backend directory or parent directory

### Weather API not working
```
Weather service unavailable
```
**Solution**: Check your API key in `.env` file

### CORS errors
**Solution**: Add your frontend URL to `ALLOWED_ORIGINS` in app.py

### Import errors
```
ModuleNotFoundError: No module named 'complete_pipeline'
```
**Solution**: Copy `complete_pipeline.py` to backend directory

## License

MIT License - See LICENSE file for details
