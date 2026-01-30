"""
VoiceAuth API - Pydantic Schemas
================================
Request and response models for the API.

These schemas:
1. Validate incoming request data
2. Serialize outgoing response data
3. Generate OpenAPI documentation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


# ============================================================================
# HEALTH & STATUS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    message: str = Field(..., example="VoiceAuth API is running")
    version: str = Field(..., example="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., example=True)


class StatsResponse(BaseModel):
    """System statistics response."""
    total_users: int = Field(..., ge=0, example=5)
    database_mode: str = Field(..., example="sqlite")
    database_path: Optional[str] = Field(None, example="data/voiceauth.db")
    model_loaded: bool = Field(..., example=True)
    embedding_dim: int = Field(..., example=192)


# ============================================================================
# CHALLENGE PHRASES
# ============================================================================

class ChallengeRequest(BaseModel):
    """Request for a challenge phrase."""
    language: Optional[str] = Field(
        None, 
        example="english",
        description="Language for challenge phrase. Options: english, hindi, marathi. If not provided, random language is selected."
    )
    
    @validator('language')
    def validate_language(cls, v):
        if v is not None:
            v = v.lower().strip()
            valid = ["english", "hindi", "marathi"]
            if v not in valid:
                raise ValueError(f"Invalid language. Must be one of: {', '.join(valid)}")
        return v


class ChallengeResponse(BaseModel):
    """Challenge phrase response."""
    challenge_id: str = Field(..., example="ch_abc123def456")
    text: str = Field(..., example="The quick brown fox jumps over the lazy dog near the river bank every single morning")
    language: str = Field(..., example="english")
    language_display: str = Field(..., example="English")
    expires_at: datetime = Field(...)
    expires_in_seconds: int = Field(..., ge=0, example=60)


# ============================================================================
# ENROLLMENT
# ============================================================================

class EnrollmentRequest(BaseModel):
    """Enrollment request metadata (audio sent as file)."""
    user_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        example="user_123",
        description="Unique identifier for the user"
    )
    
    @validator('user_id')
    def validate_user_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("User ID cannot be empty")
        # Sanitize: only allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[\w\-]+$', v):
            raise ValueError("User ID can only contain letters, numbers, underscores, and hyphens")
        return v


class EnrollmentResponse(BaseModel):
    """Enrollment success response."""
    success: bool = Field(..., example=True)
    message: str = Field(..., example="Voice enrollment completed successfully")
    user_id: str = Field(..., example="user_123")
    samples_count: int = Field(..., ge=1, example=3)
    quality_score: Optional[float] = Field(None, ge=0, le=1, example=0.85)
    consistency_score: Optional[float] = Field(None, ge=0, le=1, example=0.92)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MultiEnrollmentSampleResponse(BaseModel):
    """Response for a single enrollment sample in multi-enrollment."""
    sample_number: int = Field(..., ge=1, example=1)
    success: bool = Field(..., example=True)
    message: Optional[str] = Field(None, example="Sample recorded successfully")
    duration_seconds: Optional[float] = Field(None, example=5.2)


# ============================================================================
# AUTHENTICATION
# ============================================================================

class AuthenticationRequest(BaseModel):
    """Authentication request metadata (audio sent as file)."""
    user_id: str = Field(
        ..., 
        min_length=1,
        max_length=100,
        example="user_123",
        description="User ID to authenticate against"
    )
    challenge_id: Optional[str] = Field(
        None,
        example="ch_abc123def456",
        description="Optional challenge ID for challenge-response verification"
    )


class AuthenticationResponse(BaseModel):
    """Authentication result response."""
    success: bool = Field(..., example=True)
    authenticated: bool = Field(..., example=True)
    user_id: str = Field(..., example="user_123")
    message: str = Field(..., example="Voice authentication successful")
    similarity_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        example=0.78,
        description="Similarity score (only shown on failure)"
    )
    threshold: float = Field(..., ge=0, le=1, example=0.50)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# USER MANAGEMENT
# ============================================================================

class UserInfo(BaseModel):
    """User information."""
    user_id: str = Field(..., example="user_123")
    enrolled_at: Optional[datetime] = Field(None)
    has_voiceprint: bool = Field(..., example=True)


class UserListResponse(BaseModel):
    """List of enrolled users."""
    users: List[str] = Field(..., example=["user_123", "user_456"])
    total_count: int = Field(..., ge=0, example=2)


class UserDeleteRequest(BaseModel):
    """Request to delete a user."""
    user_id: str = Field(
        ..., 
        min_length=1,
        max_length=100,
        example="user_123"
    )


class UserDeleteResponse(BaseModel):
    """User deletion response."""
    success: bool = Field(..., example=True)
    message: str = Field(..., example="User and voiceprint deleted successfully")
    user_id: str = Field(..., example="user_123")


class UserExistsResponse(BaseModel):
    """Check if user exists."""
    exists: bool = Field(..., example=True)
    user_id: str = Field(..., example="user_123")


# ============================================================================
# ERROR RESPONSES
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(False, example=False)
    error: str = Field(..., example="User not found")
    error_code: Optional[str] = Field(None, example="USER_NOT_FOUND")
    details: Optional[Dict[str, Any]] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    success: bool = Field(False, example=False)
    error: str = Field("Validation error", example="Validation error")
    details: List[Dict[str, Any]] = Field(...)


# ============================================================================
# LANGUAGE INFO
# ============================================================================

class LanguageInfo(BaseModel):
    """Information about a supported language."""
    code: str = Field(..., example="english")
    display_name: str = Field(..., example="English")
    phrase_count: int = Field(..., ge=1, example=15)


class LanguagesResponse(BaseModel):
    """List of supported languages."""
    languages: List[LanguageInfo]
    default: str = Field(..., example="english")
