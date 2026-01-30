"""
VoiceAuth API - Configuration
=============================
Central configuration for API settings, thresholds, and constants.

All settings are defined here to make it easy to adjust without
modifying multiple files.
"""

import os
from pathlib import Path

# ============================================================================
# API SETTINGS
# ============================================================================

# API Metadata
API_TITLE = "VoiceAuth API"
API_DESCRIPTION = "Voice Authentication API using ECAPA-TDNN speaker embeddings"
API_VERSION = "1.0.0"

# Server settings
API_HOST = os.getenv("VOICEAUTH_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("VOICEAUTH_API_PORT", "8000"))
DEBUG_MODE = os.getenv("VOICEAUTH_DEBUG", "true").lower() == "true"

# CORS settings (for React frontend)
CORS_ORIGINS = [
    "http://localhost:3000",      # React dev server
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8501",      # Streamlit (for testing)
]

# Allow all origins in development
CORS_ALLOW_ALL = os.getenv("VOICEAUTH_CORS_ALLOW_ALL", "false").lower() == "true"


# ============================================================================
# VOICE AUTHENTICATION SETTINGS
# ============================================================================

# Fixed authentication threshold (50% as per requirements)
AUTH_THRESHOLD = 0.50

# Enrollment settings
MIN_ENROLLMENT_SAMPLES = 1       # Minimum samples for quick enrollment
MAX_ENROLLMENT_SAMPLES = 5       # Maximum samples for multi-enrollment
RECOMMENDED_SAMPLES = 3          # Recommended for best accuracy

# Challenge settings
CHALLENGE_EXPIRY_SECONDS = 60    # Challenge valid for 60 seconds
SUPPORTED_LANGUAGES = ["english", "hindi", "marathi"]
DEFAULT_LANGUAGE = "english"


# ============================================================================
# AUDIO SETTINGS
# ============================================================================

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = [
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mp3",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
]

# Audio constraints
MIN_AUDIO_DURATION = 1.5         # Minimum seconds
MAX_AUDIO_DURATION = 15.0        # Maximum seconds
MAX_AUDIO_SIZE_MB = 10           # Maximum file size in MB
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024


# ============================================================================
# PATHS
# ============================================================================

# Project root (parent of api folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
DATABASE_PATH = DATA_DIR / "voiceauth.db"
ENCRYPTION_KEY_FILE = DATA_DIR / ".encryption_key"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# ECAPA-TDNN settings (must match utils.py)
EMBEDDING_DIM = 192
SAMPLE_RATE = 16000


# ============================================================================
# RESPONSE MESSAGES
# ============================================================================

class Messages:
    """Standard response messages for API."""
    
    # Success messages
    ENROLLMENT_SUCCESS = "Voice enrollment completed successfully"
    AUTH_SUCCESS = "Voice authentication successful"
    USER_DELETED = "User and voiceprint deleted successfully"
    
    # Error messages
    USER_NOT_FOUND = "User not found"
    USER_ALREADY_EXISTS = "User already enrolled"
    INVALID_AUDIO = "Invalid audio file"
    AUDIO_TOO_SHORT = "Audio is too short. Please record at least 3 seconds of speech."
    AUDIO_TOO_LONG = "Audio is too long. Maximum duration is 15 seconds."
    AUTH_FAILED = "Voice authentication failed"
    CHALLENGE_EXPIRED = "Challenge phrase has expired. Please request a new one."
    INVALID_LANGUAGE = "Invalid language. Supported: english, hindi, marathi"
    
    # System messages
    SYSTEM_HEALTHY = "VoiceAuth API is running"
    MODEL_LOADED = "ML model loaded and ready"
    MODEL_LOADING = "ML model is loading..."
