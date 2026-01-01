"""
Weather API routes for Pitch Insight
"""
from fastapi import APIRouter, HTTPException, Query
import requests
from config import WEATHER_API_KEY
from weather_forecast_analyzer import get_weather_analyzer

router = APIRouter(prefix="/api/weather", tags=["weather"])

WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"


@router.get("")
async def get_weather(
    city: str = Query(None, description="City name"),
    latitude: float = Query(None, description="Latitude"),
    longitude: float = Query(None, description="Longitude")
):
    """Get current weather data for a location"""
    if not WEATHER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Weather service not configured"
        )
    
    # Determine location parameter
    if city:
        location = city
    elif latitude is not None and longitude is not None:
        location = f"{latitude},{longitude}"
    else:
        raise HTTPException(
            status_code=400,
            detail="Either city or coordinates (latitude & longitude) required"
        )
    
    try:
        response = requests.get(
            WEATHER_API_URL,
            params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant data
        current = data.get("current", {})
        location_data = data.get("location", {})
        
        return {
            "success": True,
            "location": {
                "name": location_data.get("name"),
                "region": location_data.get("region"),
                "country": location_data.get("country"),
                "lat": location_data.get("lat"),
                "lon": location_data.get("lon")
            },
            "current": {
                "temperature": current.get("temp_c"),
                "feels_like": current.get("feelslike_c"),
                "humidity": current.get("humidity"),
                "wind_speed": current.get("wind_kph"),
                "wind_direction": current.get("wind_dir"),
                "wind_degree": current.get("wind_degree"),
                "pressure": current.get("pressure_mb"),
                "cloud_cover": current.get("cloud"),
                "uv_index": current.get("uv"),
                "visibility": current.get("vis_km"),
                "conditions": current.get("condition", {}).get("text"),
                "icon": current.get("condition", {}).get("icon")
            }
        }
    
    except requests.RequestException as e:
        print(f"❌ Weather API error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch weather data: {str(e)}"
        )


@router.get("/forecast")
async def get_weather_forecast(
    city: str = Query(None, description="City name"),
    latitude: float = Query(None, description="Latitude"),
    longitude: float = Query(None, description="Longitude"),
    match_format: str = Query("odi", description="Match format: test, odi, or t20"),
    match_start_time: str = Query(None, description="Match start time (HH:MM format)")
):
    """
    Get comprehensive weather forecast with cricket-specific analysis
    
    Includes:
    - Current conditions
    - Historical weather trends (past 3 days)
    - Format-specific forecasts:
      * Test: 5-day forecast with session-wise breakdown
      * ODI/T20: Innings-wise forecast
    - Cricket impact analysis (swing, seam, spin, dew)
    - Match strategy recommendations
    """
    if not WEATHER_API_KEY or WEATHER_API_KEY == "your-weather-api-key-here":
        raise HTTPException(
            status_code=503,
            detail="Weather forecast service not configured. Get a free API key from https://www.weatherapi.com/"
        )
    
    # Determine location
    if city:
        location = city
    elif latitude is not None and longitude is not None:
        location = f"{latitude},{longitude}"
    else:
        raise HTTPException(
            status_code=400,
            detail="Either city or coordinates (latitude & longitude) required"
        )
    
    # Validate match format
    valid_formats = ["test", "odi", "t20"]
    if match_format.lower() not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid match_format. Must be one of: {', '.join(valid_formats)}"
        )
    
    try:
        # Get comprehensive forecast
        analyzer = get_weather_analyzer()
        forecast = analyzer.get_comprehensive_forecast(
            location=location,
            match_format=match_format.lower(),
            match_start_time=match_start_time
        )
        
        if "error" in forecast:
            raise HTTPException(
                status_code=503,
                detail=forecast["error"]
            )
        
        return forecast
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Weather forecast error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate weather forecast: {str(e)}"
        )


@router.get("/forecast")
async def get_weather_forecast(
    city: str = Query(None, description="City name"),
    latitude: float = Query(None, description="Latitude"),
    longitude: float = Query(None, description="Longitude"),
    match_format: str = Query("odi", description="Match format: test, odi, or t20"),
    match_start_time: str = Query(None, description="Match start time (HH:MM format)")
):
    """
    Get comprehensive weather forecast with cricket-specific analysis
    
    Includes:
    - Current conditions
    - Historical weather trends (past 3 days)
    - Format-specific forecasts:
      * Test: 5-day forecast with session-wise breakdown
      * ODI/T20: Innings-wise forecast
    - Cricket impact analysis (swing, seam, spin, dew)
    - Match strategy recommendations
    """
    if not WEATHER_API_KEY or WEATHER_API_KEY == "your-weather-api-key-here":
        raise HTTPException(
            status_code=503,
            detail="Weather forecast service not configured. Get a free API key from https://www.weatherapi.com/"
        )
    
    # Determine location
    if city:
        location = city
    elif latitude is not None and longitude is not None:
        location = f"{latitude},{longitude}"
    else:
        raise HTTPException(
            status_code=400,
            detail="Either city or coordinates (latitude & longitude) required"
        )
    
    # Validate match format
    valid_formats = ["test", "odi", "t20"]
    if match_format.lower() not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid match_format. Must be one of: {', '.join(valid_formats)}"
        )
    
    try:
        # Get comprehensive forecast
        analyzer = get_weather_analyzer()
        forecast = analyzer.get_comprehensive_forecast(
            location=location,
            match_format=match_format.lower(),
            match_start_time=match_start_time
        )
        
        if "error" in forecast:
            raise HTTPException(
                status_code=503,
                detail=forecast["error"]
            )
        
        return forecast
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Weather forecast error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate weather forecast: {str(e)}"
        )
