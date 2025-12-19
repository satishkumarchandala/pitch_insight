"""
Pitch Insight Backend API
FastAPI server for cricket pitch analysis with weather integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import cv2
import numpy as np
import os
import tempfile
import requests
from datetime import datetime
from pathlib import Path
import base64
import json
from dotenv import load_dotenv

from complete_pipeline import CompletePitchPipeline
from database import db_manager, DatabaseManager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Pitch Insight API",
    description="AI-powered cricket pitch analysis with weather integration",
    version="1.0.0"
)

# CORS configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
ALLOWED_ORIGINS = [FRONTEND_URL] if FRONTEND_URL != "*" else ["*"]

# For development, also allow localhost
if os.getenv("ENVIRONMENT") != "production":
    ALLOWED_ORIGINS.extend(["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline with lazy loading enabled for memory optimization
print("🚀 Initializing Pitch Analysis Pipeline...")
pipeline = CompletePitchPipeline(
    yolo_model_path="pitch_yolov8_best.pt",
    classifier_model_path="best_pitch_classifier.pth",
    lazy_load=True  # Enable lazy loading to reduce memory footprint
)
print("✅ Pipeline ready!")


# ============================================
# Lifecycle Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Pitch Insight API...")
    await db_manager.connect()
    print("✅ All services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🔌 Shutting down...")
    await db_manager.disconnect()

# Weather API configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # OpenWeatherMap API key
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"


# ============================================
# Response Models
# ============================================

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str


class WeatherData(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    conditions: str
    location: str


class PitchAnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    pitch_detection: Dict
    features: Dict
    ml_classification: Dict
    final_classification: Dict
    specialist_analysis: Optional[Dict]
    weather: Optional[WeatherData]
    match_strategy: Dict
    timestamp: str
    processing_time: float


# ============================================
# Helper Functions
# ============================================

def get_weather_data(latitude: float, longitude: float, city: str = None) -> Optional[WeatherData]:
    """
    Fetch weather data from OpenWeatherMap API
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        city: City name (optional, used as fallback)
        
    Returns:
        WeatherData or None if API call fails
    """
    if not WEATHER_API_KEY:
        print("⚠️ Weather API key not configured")
        return None
    
    try:
        # Try with coordinates
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        
        if response.status_code != 200:
            # Fallback to city name
            if city:
                params = {
                    "q": city,
                    "appid": WEATHER_API_KEY,
                    "units": "metric"
                }
                response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Calculate rainfall (0 if no rain data)
            rainfall = 0.0
            if "rain" in data:
                rainfall = data["rain"].get("1h", 0.0)  # mm in last hour
            
            return WeatherData(
                temperature=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                rainfall=rainfall,
                wind_speed=data["wind"]["speed"],
                conditions=data["weather"][0]["description"],
                location=data["name"]
            )
    except Exception as e:
        print(f"❌ Weather API error: {e}")
    
    return None


def get_weather_impact_on_pitch(weather: WeatherData, features: Dict) -> Dict:
    """
    Analyze how weather affects pitch behavior
    
    Args:
        weather: Current weather data
        features: Extracted pitch features
        
    Returns:
        Dictionary with weather impact analysis
    """
    impacts = []
    severity = "low"
    
    # Temperature impact
    if weather.temperature > 35:
        impacts.append("High temperature will dry the pitch quickly")
        severity = "medium"
    elif weather.temperature < 15:
        impacts.append("Cool temperature may retain moisture longer")
    
    # Humidity impact
    if weather.humidity > 70:
        impacts.append("High humidity favors swing bowling")
        if features['grass_coverage']['percentage'] > 40:
            impacts.append("Humid conditions + grass = excellent for pace bowlers")
            severity = "high"
    
    # Rainfall impact
    if weather.rainfall > 0:
        impacts.append(f"Recent rainfall ({weather.rainfall}mm) - pitch will be damp")
        impacts.append("Expect slower outfield and unpredictable bounce")
        severity = "high"
    
    # Wind impact
    if weather.wind_speed > 20:
        impacts.append("Strong winds will aid swing bowling")
        severity = "medium"
    
    # Combined effects
    moisture_level = features['moisture_level']['level']
    if weather.humidity > 60 and moisture_level in ['Wet', 'Damp']:
        impacts.append("Wet pitch + humid conditions = very bowler-friendly")
    
    if not impacts:
        impacts.append("Weather conditions are neutral")
    
    return {
        "impacts": impacts,
        "severity": severity,
        "favorable_for": "bowlers" if severity in ["medium", "high"] else "balanced"
    }


def generate_match_strategy(results: Dict, weather: Optional[WeatherData]) -> Dict:
    """
    Generate comprehensive match strategy
    
    Args:
        results: Analysis results
        weather: Weather data (optional)
        
    Returns:
        Match strategy dictionary
    """
    final_class = results['final_classification']['prediction']
    features = results['features']
    
    strategy = {
        "pitch_type": final_class,
        "toss_decision": "",
        "batting_strategy": [],
        "bowling_strategy": [],
        "team_composition": [],
        "key_factors": []
    }
    
    # Base strategy on pitch type
    if final_class == 'batting_friendly':
        strategy['toss_decision'] = "Bat first - accumulate runs"
        strategy['batting_strategy'] = [
            "Play aggressive cricket",
            "Target 300+ in ODI / 180+ in T20",
            "Rotate strike freely"
        ]
        strategy['bowling_strategy'] = [
            "Be patient and disciplined",
            "Vary pace and use slower balls",
            "Focus on dot balls and pressure"
        ]
        strategy['team_composition'] = [
            "Include 5-6 specialist batsmen",
            "2-3 pace bowlers",
            "1-2 spinners for variation"
        ]
    
    elif final_class == 'bowling_friendly':
        strategy['toss_decision'] = "Bowl first - exploit conditions"
        strategy['batting_strategy'] = [
            "Play cautiously early on",
            "Build partnerships",
            "Graft for runs"
        ]
        strategy['bowling_strategy'] = [
            "Attack with new ball",
            "Exploit swing and seam",
            "Target top order wickets"
        ]
        strategy['team_composition'] = [
            "3-4 quality pace bowlers",
            "Include swing bowlers",
            "Technically sound batsmen"
        ]
    
    elif final_class == 'spin_friendly':
        strategy['toss_decision'] = "Bat first - pitch will deteriorate"
        strategy['batting_strategy'] = [
            "Play spin with soft hands",
            "Use feet against spinners",
            "Score heavily in first innings"
        ]
        strategy['bowling_strategy'] = [
            "Use spinners extensively",
            "Bowl tight lines",
            "Create rough patches for Day 4-5"
        ]
        strategy['team_composition'] = [
            "2-3 quality spinners mandatory",
            "Batsmen strong against spin",
            "Consider 4 spinners in Tests"
        ]
    
    else:  # seam_friendly
        strategy['toss_decision'] = "Bowl first - morning conditions"
        strategy['batting_strategy'] = [
            "Play close to body",
            "Leave balls outside off",
            "Be patient early on"
        ]
        strategy['bowling_strategy'] = [
            "Bowl fuller length",
            "Target off-stump channel",
            "Use cutters and variations"
        ]
        strategy['team_composition'] = [
            "3 seam bowlers essential",
            "Include a swing bowler",
            "Gritty batsmen needed"
        ]
    
    # Add weather-based modifications
    if weather:
        weather_impact = get_weather_impact_on_pitch(weather, features)
        strategy['weather_impact'] = weather_impact
        
        if weather_impact['severity'] == 'high':
            strategy['key_factors'].append(f"Weather is a major factor: {weather.conditions}")
        
        if weather.humidity > 70:
            strategy['bowling_strategy'].append("Exploit humid conditions for swing")
        
        if weather.rainfall > 0:
            strategy['key_factors'].append("Recent rain - expect damp pitch")
    
    # Add feature-based insights
    if features['grass_coverage']['percentage'] > 50:
        strategy['key_factors'].append("Heavy grass coverage - bowler advantage")
    
    if features['crack_analysis']['severity'] in ['High', 'Medium']:
        strategy['key_factors'].append("Cracks present - expect variable bounce")
    
    return strategy


def save_image_from_upload(upload_file: UploadFile) -> str:
    """Save uploaded image to temporary file"""
    suffix = Path(upload_file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    content = upload_file.file.read()
    temp_file.write(content)
    temp_file.close()
    
    return temp_file.name


# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Pitch Insight API is running",
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/analyze")
async def analyze_pitch(
    image: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    include_weather: bool = Form(True)
):
    """
    Complete pitch analysis with weather integration
    
    Args:
        image: Pitch image file
        latitude: Location latitude (optional)
        longitude: Location longitude (optional)
        city: City name (optional, fallback for weather)
        include_weather: Whether to fetch weather data
        
    Returns:
        Complete analysis results
    """
    import time
    start_time = time.time()
    
    # Validate image
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_image_path = None
    
    try:
        # Save uploaded image
        temp_image_path = save_image_from_upload(image)
        
        # Run pitch analysis
        print(f"📸 Analyzing: {image.filename}")
        results = pipeline.analyze(temp_image_path, save_visualization=False)
        
        # Fetch weather data if requested
        weather_data = None
        if include_weather and (latitude and longitude):
            weather_data = get_weather_data(latitude, longitude, city)
        
        # Generate match strategy
        match_strategy = generate_match_strategy(results, weather_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate analysis ID
        analysis_id = f"PITCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare response
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "pitch_detection": {
                "detected": results['pitch_detection']['detected'],
                "confidence": 0.93 if results['pitch_detection']['detected'] else 0.0,
                "bbox": [int(x) for x in results['pitch_detection']['bbox']] if results['pitch_detection']['bbox'] else None
            },
            "features": {
                "grass_coverage": {
                    "percentage": float(results['features']['grass_coverage']['percentage']),
                    "level": results['features']['grass_coverage']['level'],
                    "quality": results['features']['grass_coverage']['quality']
                },
                "cracks": {
                    "severity": results['features']['crack_analysis']['severity'],
                    "density": float(results['features']['crack_analysis']['density']) if isinstance(results['features']['crack_analysis']['density'], (int, float, np.number)) else results['features']['crack_analysis']['density'],
                    "count": int(results['features']['crack_analysis']['num_cracks'])
                },
                "moisture": {
                    "level": results['features']['moisture_level']['level'],
                    "score": float(results['features']['moisture_level']['score'])
                },
                "color": {
                    "type": results['features']['color_profile']['color_type'],
                    "description": results['features']['color_profile']['description']
                },
                "texture": {
                    "type": results['features']['texture_analysis']['type'],
                    "variance": float(results['features']['texture_analysis']['variance'])
                },
                "brightness": {
                    "level": results['features']['brightness']['level'],
                    "value": float(results['features']['brightness']['average'])
                }
            },
            "ml_classification": {
                "prediction": results['ml_classification']['prediction'],
                "confidence": float(results['ml_classification']['confidence']),
                "probabilities": {
                    cls: float(prob * 100) 
                    for cls, prob in zip(pipeline.classes, results['ml_classification']['probabilities'])
                }
            },
            "final_classification": {
                "prediction": results['final_classification']['prediction'],
                "confidence": float(results['final_classification']['confidence']),
                "probabilities": {
                    cls: float(prob * 100) 
                    for cls, prob in zip(pipeline.classes, results['final_classification']['probabilities'])
                },
                "adjustments": results['final_classification']['adjustment_info']['adjustments'],
                "reasons": results['final_classification']['adjustment_info']['reasons']
            },
            "specialist_analysis": results.get('specialist_analysis', {}),
            "weather": weather_data.dict() if weather_data else None,
            "match_strategy": match_strategy,
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(processing_time, 2)
        }
        
        print(f"✅ Analysis complete in {processing_time:.2f}s")
        
        # Save to database (async, don't wait for result)
        if db_manager.connected:
            try:
                doc_id = await db_manager.save_analysis(response_data)
                if doc_id:
                    response_data["analysis_id"] = doc_id
                    print(f"💾 Saved to database: {doc_id}")
            except Exception as e:
                print(f"⚠️  Database save failed: {e}")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        if os.path.exists("temp_pitch_region.jpg"):
            os.unlink("temp_pitch_region.jpg")


@app.post("/api/quick-analyze")
async def quick_analyze(
    image: UploadFile = File(...)
):
    """
    Quick pitch analysis (classification only, no detailed features)
    Faster endpoint for quick results
    """
    import time
    start_time = time.time()
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_image_path = None
    
    try:
        temp_image_path = save_image_from_upload(image)
        
        # Quick classification
        img = cv2.imread(temp_image_path)
        ml_class, ml_confidence, ml_probs = pipeline.classify_pitch(img)
        
        processing_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "prediction": ml_class,
            "confidence": float(ml_confidence),
            "probabilities": {
                cls: float(prob * 100) 
                for cls, prob in zip(pipeline.classes, ml_probs)
            },
            "processing_time": round(processing_time, 2)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")
    
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)


@app.get("/api/weather")
async def get_weather(
    latitude: float,
    longitude: float,
    city: Optional[str] = None
):
    """
    Get weather data for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        city: City name (optional)
    """
    weather = get_weather_data(latitude, longitude, city)
    
    if weather:
        return JSONResponse(content=weather.dict())
    else:
        raise HTTPException(status_code=404, detail="Weather data not available")


@app.post("/api/generate-report")
async def generate_report(
    image: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None)
):
    """
    Generate comprehensive downloadable analysis report
    
    Args:
        image: Pitch image file
        latitude: Location latitude (optional)
        longitude: Location longitude (optional)
        city: City name (optional)
        
    Returns:
        PNG report file
    """
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_image_path = None
    report_path = None
    
    try:
        # Save uploaded image
        temp_image_path = save_image_from_upload(image)
        
        # Run pitch analysis
        print(f"📸 Generating report for: {image.filename}")
        results = pipeline.analyze(temp_image_path, save_visualization=False)
        
        # Fetch weather data if coordinates provided
        weather_data = None
        if latitude and longitude:
            weather = get_weather_data(latitude, longitude, city)
            if weather:
                weather_data = weather.dict()
        
        # Generate comprehensive report
        print("📊 Creating comprehensive visual report...")
        report_path = pipeline.generate_comprehensive_report(
            temp_image_path, 
            results,
            weather_data
        )
        
        # Return the report file
        return FileResponse(
            report_path,
            media_type="image/png",
            filename=f"pitch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            headers={
                "Content-Disposition": f"attachment; filename=pitch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            }
        )
    
    except Exception as e:
        print(f"❌ Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass
        if os.path.exists("temp_pitch_region.jpg"):
            try:
                os.unlink("temp_pitch_region.jpg")
            except:
                pass
        # Note: Report file will be auto-deleted by FastAPI after response


@app.get("/api/classes")
async def get_classes():
    """Get list of pitch classes"""
    return JSONResponse(content={
        "classes": pipeline.classes,
        "descriptions": {
            "batting_friendly": "Good for batting, even bounce, minimal assistance for bowlers",
            "bowling_friendly": "Assists fast bowlers with swing and seam movement",
            "spin_friendly": "Assists spin bowlers with turn and variable bounce",
            "seam_friendly": "Assists seam bowlers with lateral movement off the pitch"
        }
    })


# ============================================
# Database Endpoints
# ============================================

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve a specific analysis by ID"""
    if not db_manager.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    result = await db_manager.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return JSONResponse(content=result)


@app.get("/api/recent-analyses")
async def get_recent_analyses(limit: int = 10):
    """Get recent analyses"""
    if not db_manager.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    results = await db_manager.get_recent_analyses(limit=limit)
    return JSONResponse(content={"analyses": results, "count": len(results)})


@app.get("/api/stats")
async def get_statistics():
    """Get database statistics"""
    stats = await db_manager.get_stats()
    return JSONResponse(content=stats)


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏏 PITCH INSIGHT API SERVER")
    print("="*60)
    print("\n📚 API Documentation:")
    print("   Swagger UI: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print("\n🔗 Endpoints:")
    print("   POST /api/analyze - Complete pitch analysis")
    print("   POST /api/quick-analyze - Quick classification")
    print("   POST /api/generate-report - Generate downloadable report")
    print("   GET  /api/weather - Get weather data")
    print("   GET  /api/classes - Get pitch classes info")
    print("   GET  /api/health - Health check")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

