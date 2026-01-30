"""
VoiceAuth API - FastAPI Application
===================================
Main entry point for the Voice Authentication REST API.

Run with:
    cd voiceauth-mvp
    uvicorn api.main:app --reload --port 8000

Or:
    python -m api.main

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure parent directory is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    CORS_ALLOW_ALL,
    DEBUG_MODE,
)

from .dependencies import startup_load_model, shutdown_cleanup

# Import routes
from .routes import health, challenge, enroll, auth, users


# ============================================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print(f"[VoiceAuth API] Starting {API_TITLE} v{API_VERSION}")
    print(f"[VoiceAuth API] Debug mode: {DEBUG_MODE}")
    
    # Load ML model (optional - can be lazy loaded on first request)
    # Uncomment to preload model on startup:
    # await startup_load_model()
    
    print("[VoiceAuth API] Ready to accept requests")
    print(f"[VoiceAuth API] Docs: http://{API_HOST}:{API_PORT}/docs")
    
    yield
    
    # Shutdown
    await shutdown_cleanup()
    print("[VoiceAuth API] Shutdown complete")


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

# Configure CORS for React frontend
cors_origins = ["*"] if CORS_ALLOW_ALL else CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    error_message = str(exc) if DEBUG_MODE else "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": error_message,
            "error_code": "INTERNAL_ERROR",
        }
    )


# ============================================================================
# REGISTER ROUTES
# ============================================================================

# Health check routes (no prefix)
app.include_router(health.router)

# API routes with /api/v1 prefix
API_PREFIX = "/api/v1"

app.include_router(challenge.router, prefix=API_PREFIX)
app.include_router(enroll.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(users.router, prefix=API_PREFIX)


# ============================================================================
# ROOT ROUTE
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - API info."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": f"{API_PREFIX}",
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG_MODE,
        log_level="info",
    )
