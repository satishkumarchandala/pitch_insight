"""
Pitch Insight Backend API
FastAPI server for cricket pitch analysis with weather integration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import ALLOWED_ORIGINS, HOST, PORT, DEBUG
from database import close_database_connection

# Initialize FastAPI app
app = FastAPI(
    title="Pitch Insight API",
    description="AI-powered cricket pitch analysis with weather integration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from routes.health import router as health_router
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.subscription import router as subscription_router
from routes.analysis import router as analysis_router
from routes.weather import router as weather_router

# Include all routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(subscription_router)
app.include_router(analysis_router)
app.include_router(weather_router)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("=" * 60)
    print("🏏 Pitch Insight API - Starting Up")
    print("=" * 60)
    print(f"📍 Server: http://{HOST}:{PORT}")
    print(f"📚 Docs: http://{HOST}:{PORT}/docs")
    print(f"🔧 Debug Mode: {DEBUG}")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🔄 Shutting down gracefully...")
    close_database_connection()
    print("✅ Pitch Insight API stopped")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=DEBUG
    )
