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
from utils import get_pipeline, compute_image_hash, convert_numpy_types

router = APIRouter(prefix="/api", tags=["analysis"])

# Global variables
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CACHE_SIZE = 100
analysis_cache = {}


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
        analysis_result = pipe.analyze(temp_path)
        
        # Convert numpy types to Python native types
        analysis_result = convert_numpy_types(analysis_result)
        
        # TODO: Add weather integration if include_weather=True
        weather_data = None
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Build response
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "pitch_detection": analysis_result.get("pitch_detection", {}),
            "features": analysis_result.get("features", {}),
            "ml_classification": analysis_result.get("ml_classification", {}),
            "final_classification": analysis_result.get("final_classification", {}),
            "weather": weather_data,
            "match_strategy": analysis_result.get("strategy", {}),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Cache result
        if len(analysis_cache) >= MAX_CACHE_SIZE:
            # Remove oldest
            analysis_cache.pop(next(iter(analysis_cache)))
        analysis_cache[image_hash] = response_data
        
        # Save to database if user is authenticated
        if current_user:
            analysis_collection = get_analysis_collection()
            analysis_doc = {
                "analysis_id": analysis_id,
                "user_id": str(current_user["_id"]),
                "image_name": image.filename,
                "image_hash": image_hash,
                "pitch_type": response_data["final_classification"].get("pitch_type"),
                "confidence": response_data["final_classification"].get("confidence"),
                "analysis_type": "complete",
                "match_info": {"format": match_type},
                "weather_included": include_weather,
                "location": city or f"{latitude},{longitude}" if latitude else None,
                "processing_time": response_data["processing_time"],
                "created_at": datetime.utcnow(),
                "timestamp": response_data["timestamp"],
                **response_data
            }
            analysis_collection.insert_one(analysis_doc)
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
        
        # Get pipeline and analyze
        pipe = get_pipeline()
        analysis_result = pipe.analyze(temp_path, save_visualization=False)
        
        # Convert numpy types to Python native types
        analysis_result = convert_numpy_types(analysis_result)
        
        # Extract final classification results
        final_classification = analysis_result.get("final_classification", {})
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "pitch_type": final_classification.get("prediction"),
            "prediction": final_classification.get("prediction"),
            "confidence": float(final_classification.get("confidence", 0.0)),
            "probabilities": final_classification.get("probabilities", {}),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Save to database if authenticated
        if current_user:
            analysis_collection = get_analysis_collection()
            analysis_doc = {
                "analysis_id": analysis_id,
                "user_id": str(current_user["_id"]),
                "image_name": image.filename,
                "pitch_type": response_data["pitch_type"],
                "confidence": response_data["confidence"],
                "analysis_type": "quick",
                "processing_time": response_data["processing_time"],
                "created_at": datetime.utcnow(),
                "timestamp": response_data["timestamp"]
            }
            analysis_collection.insert_one(analysis_doc)
        
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
