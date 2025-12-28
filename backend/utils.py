"""
Utility functions for Pitch Insight Backend
"""
import hashlib
import time
from typing import Optional
from database import get_analysis_collection


def compute_image_hash(image_path: str) -> str:
    """Compute SHA256 hash of image file for caching"""
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def build_chat_context(analysis_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Build context from latest analysis for chatbot
    
    Args:
        analysis_id: Specific analysis ID
        user_id: User ID to fetch latest analysis
        
    Returns:
        Formatted context string
    """
    if not analysis_id and not user_id:
        return "No analysis context available."
    
    try:
        analysis_collection = get_analysis_collection()
        
        # Get specific analysis or latest for user
        if analysis_id:
            analysis = analysis_collection.find_one({"analysis_id": analysis_id})
        else:
            analysis = analysis_collection.find_one(
                {"user_id": user_id},
                sort=[("created_at", -1)]
            )
        
        if not analysis:
            return "No analysis context available."
        
        # Build structured context
        context = f"""
PITCH ANALYSIS CONTEXT:
======================

Pitch Type: {analysis.get('pitch_type', 'Unknown')}
Confidence: {analysis.get('confidence', 0):.1f}%
Analysis Type: {analysis.get('analysis_type', 'Unknown')}
Date: {analysis.get('timestamp', 'Unknown')}

PITCH FEATURES:
"""
        
        # Add features if complete analysis
        if analysis.get('features'):
            features = analysis['features']
            context += f"""
- Grass Coverage: {features.get('grass_coverage', {}).get('percentage', 'N/A')}% ({features.get('grass_coverage', {}).get('level', 'N/A')})
- Cracks: {features.get('cracks', {}).get('severity', 'N/A')} severity, {features.get('cracks', {}).get('count', 0)} detected
- Moisture: {features.get('moisture', {}).get('level', 'N/A')}
- Color: {features.get('color', {}).get('type', 'N/A')}
- Brightness: {features.get('brightness', {}).get('level', 'N/A')}
"""
        
        # Add weather data if available
        if analysis.get('weather_data'):
            weather = analysis['weather_data'].get('current', {})
            if weather:
                context += f"""
WEATHER CONDITIONS:
- Temperature: {weather.get('temperature', 'N/A')}°C
- Humidity: {weather.get('humidity', 'N/A')}%
- Wind: {weather.get('wind_speed', 'N/A')} kph
- Dew Point: {weather.get('dew_point', 'N/A')}°C
- Conditions: {weather.get('conditions', 'N/A')}
"""
            
            # Add weather impact
            impact = analysis['weather_data'].get('impact', {})
            if impact:
                context += f"""
WEATHER IMPACT:
- Swing Potential: {impact.get('swing_potential', 'N/A')} (Score: {impact.get('swing_score', 'N/A')})
- Spin Assistance: {impact.get('spin_assistance', 'N/A')} (Score: {impact.get('spin_score', 'N/A')})
- Dew Likelihood: {impact.get('dew_likelihood', 'N/A')}
"""
        
        # Add match strategy
        if analysis.get('match_strategy'):
            strategy = analysis['match_strategy']
            context += f"""
MATCH STRATEGY:
- Toss Decision: {strategy.get('toss_decision', 'N/A')}
- Key Factors: {', '.join(strategy.get('key_factors', [])[:3]) if strategy.get('key_factors') else 'N/A'}
"""
        
        return context.strip()
        
    except Exception as e:
        print(f"Error building chat context: {e}")
        return "Error loading analysis context."


# Global pipeline instance with memory management
from memory_optimizer import model_manager

def get_pipeline():
    """
    Get pipeline with automatic memory management
    Models are lazy-loaded and auto-unloaded after idle period
    Optimized for Render free tier (512MB RAM)
    """
    return model_manager.get_pipeline()
