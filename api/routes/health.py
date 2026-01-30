"""
VoiceAuth API - Health Routes
=============================
Health check and system status endpoints.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from ..config import API_VERSION, EMBEDDING_DIM, Messages
from ..schemas import HealthResponse, StatsResponse
from ..dependencies import get_db, is_model_loaded, is_model_loading

from storage import VoiceprintDB


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and the ML model is loaded."
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        - status: "healthy" or "degraded"
        - message: Human-readable status message
        - version: API version
        - timestamp: Current server time
        - model_loaded: Whether the ML model is ready
    """
    model_loaded = is_model_loaded()
    model_loading = is_model_loading()
    
    if model_loaded:
        status = "healthy"
        message = Messages.SYSTEM_HEALTHY
    elif model_loading:
        status = "degraded"
        message = Messages.MODEL_LOADING
    else:
        status = "degraded"
        message = "ML model not loaded. Will load on first request."
    
    return HealthResponse(
        status=status,
        message=message,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc),
        model_loaded=model_loaded,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System Statistics",
    description="Get system statistics including user count and database info."
)
async def get_stats(db: VoiceprintDB = Depends(get_db)):
    """
    Get system statistics.
    
    Returns:
        - total_users: Number of enrolled users
        - database_mode: "sqlite" or "in-memory"
        - database_path: Path to database file
        - model_loaded: Whether the ML model is loaded
        - embedding_dim: Dimension of voice embeddings (192)
    """
    db_stats = db.get_stats()
    
    return StatsResponse(
        total_users=db_stats.get("users", 0),
        database_mode=db_stats.get("mode", "unknown"),
        database_path=db_stats.get("db_path"),
        model_loaded=is_model_loaded(),
        embedding_dim=EMBEDDING_DIM,
    )


# Removed duplicate root endpoint - main.py handles "/"
