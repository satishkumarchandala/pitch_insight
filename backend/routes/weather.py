"""
Weather API routes for Pitch Insight
"""
from fastapi import APIRouter, HTTPException, Query
import requests
from config import WEATHER_API_KEY

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
