"""
Analysis routes for Pitch Insight API
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from typing import Optional
import tempfile
import os
import uuid
import time
from datetime import datetime
from pathlib import Path

from schemas import PitchAnalysisResponse
from auth import get_optional_current_user
from database import get_analysis_collection
from utils import get_pipeline, compute_image_hash
import numpy as np

router = APIRouter(prefix="/api", tags=["analysis"])

# Global variables
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CACHE_SIZE = 100
analysis_cache = {}


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for MongoDB storage.
    Removes numpy arrays and masks that shouldn't be stored.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items() 
                if k not in ['mask', 'edges_mask']}  # Skip numpy array fields
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def generate_match_strategy(pitch_type: str, features: dict, adjustments: list) -> dict:
    """
    Generate match strategy based on pitch analysis
    
    Args:
        pitch_type: Predicted pitch type (batting_friendly, bowling_friendly, etc.)
        features: Extracted pitch features
        adjustments: Applied adjustments from analysis
        
    Returns:
        Dictionary with toss decision and strategies
    """
    grass_pct = features.get('grass_coverage', {}).get('percentage', 0)
    cracks = features.get('crack_analysis', {}).get('num_cracks', 0)
    moisture = features.get('moisture_level', {}).get('level', 'Normal')
    
    strategy = {
        'toss_decision': '',
        'batting_strategy': [],
        'bowling_strategy': [],
        'team_composition': [],
        'key_factors': []
    }
    
    # Toss Decision
    if pitch_type == 'batting_friendly':
        strategy['toss_decision'] = '🏏 Bat First - Good batting conditions early'
        strategy['batting_strategy'] = [
            'Play aggressively in powerplay overs',
            'Build partnerships with rotating strike',
            'Capitalize on shorter boundaries if available',
            'Target spinners for boundaries'
        ]
        strategy['bowling_strategy'] = [
            'Bowl tight lines in powerplay',
            'Use variations to slow scoring rate',
            'Set defensive fields after 10 overs',
            'Exploit any worn patches later'
        ]
        strategy['team_composition'] = [
            'Include 5-6 specialist batsmen',
            'Pick 2 spinners for control',
            '2-3 pace bowlers with variations',
            'All-rounders for balance'
        ]
        
    elif pitch_type == 'bowling_friendly':
        strategy['toss_decision'] = '🎳 Bowl First - Early assistance for bowlers'
        strategy['batting_strategy'] = [
            'Play cautiously in first 10 overs',
            'Focus on building partnerships',
            'Avoid risky shots early on',
            'Accelerate after pitch settles'
        ]
        strategy['bowling_strategy'] = [
            'Exploit movement with new ball',
            'Bowl fuller length for swing',
            'Use slip cordon early',
            'Maintain pressure with tight lines'
        ]
        strategy['team_composition'] = [
            'Pick 3-4 quality pace bowlers',
            'Include swing bowlers if available',
            'Solid top-order batsmen',
            'All-rounders who can bowl'
        ]
        
    elif pitch_type == 'spin_friendly':
        strategy['toss_decision'] = '🎲 Bat First - Pitch will deteriorate for spinners'
        strategy['batting_strategy'] = [
            'Score quickly while pitch is fresh',
            'Use feet against spinners',
            'Target straight boundaries',
            'Build big total while you can'
        ]
        strategy['bowling_strategy'] = [
            'Use spinners early and often',
            'Vary pace and trajectory',
            'Bowl stump-to-stump line',
            'Set attacking fields for spinners'
        ]
        strategy['team_composition'] = [
            'Pick 3 quality spinners',
            'Include a leg-spinner if available',
            '2 pace bowlers for variety',
            'Batsmen who play spin well'
        ]
        
    elif pitch_type == 'seam_friendly':
        strategy['toss_decision'] = '🎳 Bowl First - Seam movement available'
        strategy['batting_strategy'] = [
            'Leave balls outside off stump',
            'Watch the ball closely',
            'Play close to body',
            'Wait for loose deliveries'
        ]
        strategy['bowling_strategy'] = [
            'Bowl seam-up on good length',
            'Hit the pitch hard',
            'Use the crease for angles',
            'Target top of off stump'
        ]
        strategy['team_composition'] = [
            'Pick tall seam bowlers',
            '3-4 pace bowlers with seam skills',
            'Technically sound batsmen',
            'Wicketkeeper-batsman for depth'
        ]
    
    # Add key factors based on features
    if grass_pct > 50:
        strategy['key_factors'].append(f'⚠️ High grass coverage ({grass_pct:.1f}%) - expect swing and seam')
    
    if cracks > 3:
        strategy['key_factors'].append(f'⚠️ Multiple cracks ({cracks}) - variable bounce likely')
    
    if moisture == 'Wet':
        strategy['key_factors'].append('⚠️ Wet pitch - ball will swing and seam early')
    elif moisture == 'Dry':
        strategy['key_factors'].append('⚠️ Dry pitch - expect spin and uneven bounce')
    
    # Add adjustment-based factors
    for adj in adjustments:
        if 'Bowling' in adj:
            strategy['key_factors'].append('⚠️ Conditions favor bowlers - be cautious')
            break
    
    return strategy


def check_pro_subscription(user: dict):
    """Check if user has active Pro subscription"""
    subscription_type = user.get("subscription_type", "free")
    subscription_end = user.get("subscription_end_date")
    
    if subscription_type != "pro":
        raise HTTPException(
            status_code=403,
            detail="Pro subscription required for complete analysis"
        )
    
    if subscription_end and datetime.utcnow() > subscription_end:
        raise HTTPException(
            status_code=403,
            detail="Your Pro subscription has expired. Please renew to continue."
        )


@router.post("/analyze", response_model=PitchAnalysisResponse)
async def analyze_pitch(
    image: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    include_weather: bool = Form(True),
    match_type: str = Form("odi"),
    custom_overs: Optional[int] = Form(None),
    current_user: Optional[dict] = Depends(get_optional_current_user)
):
    """Complete pitch analysis with weather integration (Pro feature)"""
    # Check Pro subscription for authenticated users
    if current_user:
        check_pro_subscription(current_user)
    
    start_time = time.time()
    
    # Validate match type
    valid_match_types = ["test", "odi", "t20", "custom"]
    if match_type not in valid_match_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid match_type. Must be one of: {', '.join(valid_match_types)}"
        )
    
    # Validate file size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 5MB allowed")
    
    # Save temp file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Check cache
        image_hash = compute_image_hash(temp_path)
        if image_hash in analysis_cache:
            print(f"✓ Cache hit for image {image_hash[:8]}")
            cached_result = analysis_cache[image_hash]
            cached_result["from_cache"] = True
            return cached_result
        
        # Get pipeline and analyze
        pipe = get_pipeline()
        results = pipe.analyze(temp_path, save_visualization=False)
        
        # Fetch weather data if requested (optional feature)
        weather_data = None
        if include_weather and (city or (latitude is not None and longitude is not None)):
            try:
                import requests
                from config import WEATHER_API_KEY
                
                # Only attempt if API key is configured and not default
                if WEATHER_API_KEY and WEATHER_API_KEY != "your-weather-api-key-here":
                    location = city if city else f"{latitude},{longitude}"
                    weather_url = "https://api.weatherapi.com/v1/current.json"
                    weather_response = requests.get(
                        weather_url,
                        params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"},
                        timeout=5
                    )
                    
                    if weather_response.status_code == 200:
                        weather_json = weather_response.json()
                        current = weather_json.get("current", {})
                        location_data = weather_json.get("location", {})
                        
                        # Match the WeatherData schema exactly
                        weather_data = {
                            "temperature": current.get("temp_c", 0),
                            "feels_like": current.get("feelslike_c", 0),
                            "humidity": current.get("humidity", 0),
                            "dew_point": current.get("dewpoint_c", 0),
                            "uv_index": current.get("uv", 0),
                            "wind_speed": current.get("wind_kph", 0),
                            "wind_direction": current.get("wind_dir", "N"),
                            "wind_degree": current.get("wind_degree", 0),
                            "cloud_cover": current.get("cloud", 0),
                            "pressure": current.get("pressure_mb", 0),
                            "visibility": current.get("vis_km", 0),
                            "rainfall": current.get("precip_mm", 0),
                            "conditions": current.get("condition", {}).get("text", "Unknown"),
                            "location": f"{location_data.get('name')}, {location_data.get('country')}"
                        }
                        print(f"✅ Weather data fetched for {location}")
                    elif weather_response.status_code == 401:
                        print(f"⚠️ Weather API key is invalid or expired (401 Unauthorized)")
                        print(f"   To enable weather features, get a free API key from https://www.weatherapi.com/")
                    else:
                        print(f"⚠️ Weather API returned status {weather_response.status_code}")
                else:
                    print("ℹ️ Weather API key not configured - analysis will continue without weather data")
                    print("   To enable weather features, get a free API key from https://www.weatherapi.com/")
            except Exception as e:
                print(f"⚠️ Failed to fetch weather: {str(e)}")
                # Continue without weather data - this is not a critical failure
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Generate match strategy based on analysis
        match_strategy = generate_match_strategy(
            pitch_type=results.get("final_classification", {}).get("prediction", ""),
            features=results.get("features", {}),
            adjustments=results.get("final_classification", {}).get("adjustments", [])
        )
        
        # Build match info
        match_info = {
            "format": match_type,
            "overs": custom_overs if match_type == "custom" else (
                90 if match_type == "test" else 50 if match_type == "odi" else 20
            ),
            "format_description": {
                "test": "5-day Test Match",
                "odi": "One Day International (50 overs)",
                "t20": "Twenty20 (20 overs)",
                "custom": f"Custom Match ({custom_overs} overs)"
            }.get(match_type, "Unknown")
        }
        
        # Build response matching old format
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "pitch_detection": results.get("pitch_detection", {}),
            "features": results.get("features", {}),
            "ml_classification": results.get("ml_classification", {}),
            "final_classification": results.get("final_classification", {}),
            "match_info": match_info,
            "weather": weather_data,
            "match_strategy": match_strategy,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Convert numpy types to Python native types for the response
        response_data = convert_numpy_types(response_data)
        
        # Cache result
        if len(analysis_cache) >= MAX_CACHE_SIZE:
            # Remove oldest
            analysis_cache.pop(next(iter(analysis_cache)))
        analysis_cache[image_hash] = response_data
        
        # Save to database if user is authenticated
        if current_user:
            analysis_collection = get_analysis_collection()
            
            # Convert numpy types to Python native types for MongoDB
            clean_data = convert_numpy_types({
                "analysis_id": analysis_id,
                "user_id": str(current_user["_id"]),
                "image_name": image.filename,
                "image_hash": image_hash,
                "pitch_type": response_data["final_classification"].get("prediction"),
                "confidence": response_data["final_classification"].get("confidence"),
                "analysis_type": "complete",
                "match_info": match_info,
                "weather_included": include_weather,
                "location": city or f"{latitude},{longitude}" if latitude else None,
                "processing_time": response_data["processing_time"],
                "created_at": datetime.utcnow(),
                "timestamp": response_data["timestamp"],
                "pitch_detection": response_data["pitch_detection"],
                "features": response_data["features"],
                "ml_classification": response_data["ml_classification"],
                "final_classification": response_data["final_classification"],
                "weather": response_data["weather"],
                "match_strategy": response_data["match_strategy"]
            })
            
            analysis_collection.insert_one(clean_data)
            print(f"✓ Analysis saved for user: {current_user['email']}")
        
        return response_data
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/quick-analyze")
async def quick_analyze(
    image: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_current_user)
):
    """Quick pitch analysis (Free tier available)"""
    start_time = time.time()
    
    # Validate file size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Get pipeline and analyze (use full analysis, just return simplified response)
        pipe = get_pipeline()
        results = pipe.analyze(temp_path, save_visualization=False)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Extract just the prediction for quick analysis
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "prediction": results["final_classification"]["prediction"],
            "confidence": results["final_classification"]["confidence"],
            "probabilities": results["final_classification"]["probabilities"],
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Convert numpy types to Python native types for the response
        response_data = convert_numpy_types(response_data)
        
        # Save to database if authenticated
        if current_user:
            analysis_collection = get_analysis_collection()
            
            # Convert numpy types to Python native types
            clean_data = convert_numpy_types({
                "analysis_id": analysis_id,
                "user_id": str(current_user["_id"]),
                "image_name": image.filename,
                "pitch_type": response_data["prediction"],
                "confidence": response_data["confidence"],
                "analysis_type": "quick",
                "processing_time": response_data["processing_time"],
                "created_at": datetime.utcnow(),
                "timestamp": response_data["timestamp"],
                "prediction": response_data["prediction"],
                "probabilities": response_data["probabilities"]
            })
            
            analysis_collection.insert_one(clean_data)
        
        return response_data
        
    except Exception as e:
        print(f"❌ Quick analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.get("/classes")
async def get_pitch_classes():
    """Get available pitch classification classes"""
    return {
        "classes": [
            "batting_friendly",
            "bowling_friendly",
            "seam_friendly",
            "spin_friendly"
        ],
        "descriptions": {
            "batting_friendly": "Good for batting, minimal movement",
            "bowling_friendly": "Assists all types of bowling",
            "seam_friendly": "Favors fast bowling with seam movement",
            "spin_friendly": "Provides turn and bounce for spinners"
        }
    }


@router.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Retrieve analysis visualization image"""
    viz_path = Path("visualizations") / filename
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(viz_path)
