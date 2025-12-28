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
    """Get API performance statistics and memory status"""
    from memory_optimizer import model_manager
    from routes.analysis import analysis_cache, MAX_CACHE_SIZE, MAX_FILE_SIZE
    import sys
    
    # Get memory manager status
    manager_status = model_manager.get_status()
    
    # Try to get process memory (optional, requires psutil)
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        memory_mb = None
    
    return JSONResponse(content={
        "cache_size": len(analysis_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "models_loaded": manager_status["models_loaded"],
        "model_idle_time_seconds": round(manager_status["idle_time"], 1),
        "model_auto_unload_timeout": manager_status["idle_timeout"],
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "memory_usage_mb": round(memory_mb, 2) if memory_mb else "N/A",
        "python_version": sys.version.split()[0],
        "optimized_for_render_free_tier": True
    })
