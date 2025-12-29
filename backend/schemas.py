"""
Pydantic Response Models for Pitch Insight API
"""
from pydantic import BaseModel
from typing import Optional, Dict, List


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str


class WeatherData(BaseModel):
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
    location: str


class WeatherImpact(BaseModel):
    swing_potential: str  # Low, Medium, High
    swing_score: float  # 0-100
    spin_assistance: str
    spin_score: float
    pitch_drying_rate: str  # Slow, Moderate, Fast
    dew_likelihood: str
    dew_gap: float
    overall_severity: str
    key_factors: List[str]


class SessionForecast(BaseModel):
    conditions: str
    bowling_advantage: int  # 0-100
    batting_advantage: int
    recommended_strategy: str


class HistoricalWeather(BaseModel):
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
    weather: Optional[WeatherData] = None
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
