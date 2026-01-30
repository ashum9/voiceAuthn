"""
VoiceAuth API - Dependencies
============================
Shared dependencies for FastAPI routes.

Provides:
- Database instance (VoiceprintDB)
- ML model instance (ECAPA-TDNN)
- Challenge cache for authentication
- Audio validation utilities
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
import secrets

# Add parent directory to path for imports from main project
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, HTTPException, UploadFile, status

from .config import (
    CHALLENGE_EXPIRY_SECONDS,
    SUPPORTED_AUDIO_FORMATS,
    MAX_AUDIO_SIZE_BYTES,
    Messages,
)

# Import from main project
from storage import VoiceprintDB
from utils import get_speaker_model, EMBEDDING_DIM


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

# Database instance (singleton)
_voiceprint_db: Optional[VoiceprintDB] = None

# Model loading state
_model_loaded: bool = False
_model_loading: bool = False


def get_db() -> VoiceprintDB:
    """
    Get the VoiceprintDB singleton instance.
    
    Returns:
        VoiceprintDB instance for voiceprint storage
    """
    global _voiceprint_db
    if _voiceprint_db is None:
        _voiceprint_db = VoiceprintDB()
    return _voiceprint_db


def get_model():
    """
    Get the ECAPA-TDNN speaker model.
    
    Lazy loads the model on first call (downloads ~90MB if not cached).
    
    Returns:
        SpeechBrain EncoderClassifier for speaker embeddings
    """
    global _model_loaded, _model_loading
    
    _model_loading = True
    try:
        model = get_speaker_model()
        _model_loaded = True
        _model_loading = False
        return model
    except Exception as e:
        _model_loading = False
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML model unavailable: {str(e)}"
        )


def is_model_loaded() -> bool:
    """Check if the ML model is loaded."""
    return _model_loaded


def is_model_loading() -> bool:
    """Check if the ML model is currently loading."""
    return _model_loading


# ============================================================================
# CHALLENGE CACHE
# ============================================================================

class ChallengeCache:
    """
    In-memory cache for challenge phrases.
    
    Stores issued challenges with expiration times for verification.
    Thread-safe for basic FastAPI usage.
    
    Note: In production, use Redis for multi-instance deployments.
    """
    
    def __init__(self, expiry_seconds: int = CHALLENGE_EXPIRY_SECONDS):
        self._challenges: Dict[str, dict] = {}
        self._expiry_seconds = expiry_seconds
    
    def create(self, text: str, language: str, language_display: str) -> dict:
        """
        Create a new challenge and store it.
        
        Args:
            text: The challenge phrase text
            language: Language code (e.g., "english")
            language_display: Display name (e.g., "English")
            
        Returns:
            Challenge dict with id, text, language, and expiration info
        """
        # Generate unique challenge ID
        challenge_id = f"ch_{secrets.token_hex(12)}"
        
        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._expiry_seconds)
        
        # Create challenge object
        challenge = {
            "challenge_id": challenge_id,
            "text": text,
            "language": language,
            "language_display": language_display,
            "expires_at": expires_at,
            "expires_in_seconds": self._expiry_seconds,
            "created_at": datetime.now(timezone.utc),
        }
        
        # Store challenge
        self._challenges[challenge_id] = challenge
        
        # Clean up expired challenges periodically
        self._cleanup_expired()
        
        return challenge
    
    def verify(self, challenge_id: str) -> Optional[dict]:
        """
        Verify a challenge ID is valid and not expired.
        
        Args:
            challenge_id: The challenge ID to verify
            
        Returns:
            Challenge dict if valid, None if invalid or expired
        """
        challenge = self._challenges.get(challenge_id)
        
        if challenge is None:
            return None
        
        # Check expiration
        if datetime.now(timezone.utc) > challenge["expires_at"]:
            # Remove expired challenge
            del self._challenges[challenge_id]
            return None
        
        return challenge
    
    def consume(self, challenge_id: str) -> Optional[dict]:
        """
        Verify and consume a challenge (one-time use).
        
        Args:
            challenge_id: The challenge ID to consume
            
        Returns:
            Challenge dict if valid, None if invalid or expired
        """
        challenge = self.verify(challenge_id)
        
        if challenge is not None:
            # Remove after use (one-time)
            del self._challenges[challenge_id]
        
        return challenge
    
    def _cleanup_expired(self):
        """Remove expired challenges from cache."""
        now = datetime.now(timezone.utc)
        expired_ids = [
            cid for cid, ch in self._challenges.items()
            if now > ch["expires_at"]
        ]
        for cid in expired_ids:
            del self._challenges[cid]
    
    def get_count(self) -> int:
        """Get count of active (non-expired) challenges."""
        self._cleanup_expired()
        return len(self._challenges)


# Singleton challenge cache
_challenge_cache: Optional[ChallengeCache] = None


def get_challenge_cache() -> ChallengeCache:
    """Get the challenge cache singleton."""
    global _challenge_cache
    if _challenge_cache is None:
        _challenge_cache = ChallengeCache()
    return _challenge_cache


# ============================================================================
# FILE VALIDATION
# ============================================================================

async def validate_audio_file(file: UploadFile) -> bytes:
    """
    Validate an uploaded audio file.
    
    Checks:
    - Content type is supported
    - File size is within limits
    
    Args:
        file: Uploaded file from FastAPI
        
    Returns:
        Audio bytes if valid
        
    Raises:
        HTTPException: If validation fails
    """
    # Check content type
    content_type = file.content_type or ""
    
    # Allow common audio types
    if content_type and not any(
        content_type.startswith(fmt.split('/')[0]) or content_type == fmt
        for fmt in SUPPORTED_AUDIO_FORMATS
    ):
        # Be lenient - many browsers send generic types
        if not content_type.startswith("audio/") and content_type != "application/octet-stream":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format: {content_type}. Supported: WAV, WebM, MP3, OGG"
            )
    
    # Read file content
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {str(e)}"
        )
    
    # Check file size
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum size: {MAX_AUDIO_SIZE_BYTES // (1024*1024)}MB"
        )
    
    # Check minimum size (empty or too small)
    if len(audio_bytes) < 1000:  # Less than 1KB is definitely too small
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file too small. Please record at least 3 seconds of speech."
        )
    
    return audio_bytes


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

async def startup_load_model():
    """
    Load the ML model on startup.
    
    Called during FastAPI startup event.
    """
    print("[VoiceAuth API] Loading ML model...")
    try:
        get_model()
        print("[VoiceAuth API] ML model loaded successfully!")
    except Exception as e:
        print(f"[VoiceAuth API] Warning: ML model failed to load: {e}")
        print("[VoiceAuth API] Model will be loaded on first request.")


async def shutdown_cleanup():
    """
    Cleanup on shutdown.
    
    Called during FastAPI shutdown event.
    """
    print("[VoiceAuth API] Shutting down...")
    # Add any cleanup logic here (e.g., close database connections)
