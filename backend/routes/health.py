"""
Health and status routes for Pitch Insight API
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="online",
        message="Pitch Insight API is running",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/api/stats")
async def get_stats():
    """Get API performance statistics"""
    from utils import pipeline
    from routes.analysis import analysis_cache, MAX_CACHE_SIZE, MAX_FILE_SIZE
    
    return JSONResponse(content={
        "cache_size": len(analysis_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "pipeline_loaded": pipeline is not None,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "connection_pool_active": True
    })
