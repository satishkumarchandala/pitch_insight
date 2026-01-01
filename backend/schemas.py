"""
Pydantic Response Models for Pitch Insight API
"""
from pydantic import BaseModel
from typing import Optional, Dict, List


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str


class CurrentWeather(BaseModel):
    """Current weather conditions"""
    temperature: float
    feels_like: float
    humidity: float
    dew_point: float
    uv_index: float
    wind_speed: float
    wind_direction: str
    wind_degree: int
    cloud_cover: int
    pressure: float
    visibility: float
    rainfall: float
    conditions: str
    is_day: int


class HourlyForecast(BaseModel):
    """Hourly weather forecast"""
    time: str
    temperature: float
    humidity: float
    cloud_cover: int
    chance_of_rain: int
    rainfall: float
    wind_speed: float
    wind_direction: str
    conditions: str
    will_it_rain: bool


class SessionWeather(BaseModel):
    """Weather conditions for a cricket session"""
    session_name: str  # "1st Session", "2nd Session", "3rd Session", "1st Innings", "2nd Innings"
    time_range: str
    avg_temperature: float
    avg_humidity: float
    avg_cloud_cover: float
    total_rainfall: float
    chance_of_rain: int
    avg_wind_speed: float
    conditions_summary: str
    
    # Cricket-specific impact
    swing_potential: str  # Low, Medium, High
    swing_score: float  # 0-100
    seam_movement: str
    spin_assistance: str
    spin_score: float
    pitch_moisture_level: str
    bowling_advantage: int  # 0-100
    batting_advantage: int  # 0-100
    dew_likelihood: str
    recommended_strategy: str
    key_factors: List[str]


class DayForecast(BaseModel):
    """Daily forecast with session breakdown"""
    date: str
    day_number: int  # Day 1, Day 2, etc.
    max_temp: float
    min_temp: float
    avg_humidity: float
    total_rainfall: float
    chance_of_rain: int
    sunrise: str
    sunset: str
    uv_index: float
    conditions_summary: str
    
    # Session-wise breakdown (for Test matches)
    sessions: List[SessionWeather]
    
    # Daily cricket impact
    pitch_deterioration_rate: str
    crack_development: str
    outfield_condition: str
    overall_advantage: str  # "Bowlers", "Batters", "Balanced"


class HistoricalWeather(BaseModel):
    """Historical weather trends"""
    rainfall_24h: float
    rainfall_48h: float
    rainfall_72h: float
    avg_temp_3d: float
    avg_temp_7d: float
    recent_conditions: str
    pitch_moisture_inference: str
    surface_hardness_inference: str
    crack_potential: str
    interpretation: str


class WeatherImpact(BaseModel):
    """Overall weather impact on cricket match"""
    swing_potential: str
    swing_score: float
    seam_movement: str
    spin_assistance: str
    spin_score: float
    pitch_drying_rate: str
    dew_likelihood: str
    dew_gap: float
    overall_severity: str
    key_factors: List[str]


class MatchWeatherForecast(BaseModel):
    """Complete match weather forecast"""
    match_format: str  # "test", "odi", "t20"
    location: str
    
    # Current conditions
    current: CurrentWeather
    
    # Historical context
    historical: HistoricalWeather
    
    # Format-specific forecasts
    daily_forecasts: Optional[List[DayForecast]] = None  # For Test matches (5 days)
    innings_forecasts: Optional[List[SessionWeather]] = None  # For limited-overs (2 innings)
    hourly_forecast: List[HourlyForecast]
    
    # Overall match impact summary
    pitch_behavior_trend: str
    key_risks: List[str]
    phase_wise_advantage: Dict[str, str]
    match_condition_summary: str
    recommendations: List[str]


class HistoricalWeatherOld(BaseModel):
    rainfall_24h: float
    rainfall_72h: float
    avg_temp_7d: float
    uv_trend: str
    interpretation: str


class MatchType(BaseModel):
    format: str  # "test", "odi", "t20", "custom"
    overs: Optional[int] = None  # For custom format


class PitchAnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    pitch_detection: Dict
    features: Dict
    ml_classification: Dict
    final_classification: Dict
    match_info: Optional[Dict] = None
    weather: Optional[Dict] = None  # Can be CurrentWeather or MatchWeatherForecast
    weather_forecast: Optional[MatchWeatherForecast] = None
    match_strategy: Dict
    timestamp: str
    processing_time: float


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    analysis_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    success: bool
    reply: str
    tokens_used: Optional[int] = None
    context_used: bool = False
    analysis_id: Optional[str] = None
