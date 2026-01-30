"""
VoiceAuth API - Authentication Routes
=====================================
Voice authentication/verification endpoints.
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status

from ..config import AUTH_THRESHOLD, Messages
from ..schemas import (
    AuthenticationResponse,
    ErrorResponse,
)
from ..dependencies import get_db, validate_audio_file, get_challenge_cache

from ..services.voice_service import (
    authenticate_user,
    check_user_exists,
    AuthenticationError,
)

from storage import VoiceprintDB


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "",
    response_model=AuthenticationResponse,
    summary="Authenticate User",
    description="Authenticate a user by comparing their voice against their enrolled voiceprint.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio"},
        404: {"model": ErrorResponse, "description": "User not found"},
    }
)
async def authenticate(
    user_id: str = Form(..., description="User ID to authenticate against"),
    audio: UploadFile = File(..., description="Voice recording for authentication"),
    challenge_id: str = Form(None, description="Optional challenge ID for verification"),
    db: VoiceprintDB = Depends(get_db),
):
    """
    Authenticate a user by voice.
    
    Compares the provided voice recording against the user's enrolled
    voiceprint. Uses a fixed 50% similarity threshold.
    
    Response behavior:
    - SUCCESS: Returns authenticated=True, no similarity score shown
    - FAILURE: Returns authenticated=False, shows similarity score
    
    Args:
        user_id: The user ID to authenticate against
        audio: Voice recording file
        challenge_id: Optional challenge ID (validates but doesn't require)
        
    Returns:
        Authentication result with success status
        
    Raises:
        400: Invalid audio
        404: User not enrolled
    """
    # Check if user exists
    if not check_user_exists(db, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found. Please enroll first."
        )
    
    # Validate challenge if provided
    if challenge_id:
        cache = get_challenge_cache()
        challenge = cache.consume(challenge_id)
        
        if challenge is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=Messages.CHALLENGE_EXPIRED
            )
    
    # Validate audio file
    audio_bytes = await validate_audio_file(audio)
    
    # Authenticate
    try:
        result = authenticate_user(
            db=db,
            user_id=user_id,
            audio_bytes=audio_bytes,
            threshold=AUTH_THRESHOLD,
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    return AuthenticationResponse(
        success=result["success"],
        authenticated=result["authenticated"],
        user_id=result["user_id"],
        message=result["message"],
        similarity_score=result.get("similarity_score"),  # None on success
        threshold=result["threshold"],
        timestamp=result["timestamp"],
    )


@router.post(
    "/verify",
    response_model=AuthenticationResponse,
    summary="Verify User (Alias)",
    description="Alias for /auth endpoint. Verifies a user's voice.",
)
async def verify(
    user_id: str = Form(..., description="User ID to verify"),
    audio: UploadFile = File(..., description="Voice recording"),
    challenge_id: str = Form(None, description="Optional challenge ID"),
    db: VoiceprintDB = Depends(get_db),
):
    """
    Verify a user's voice (alias for /auth).
    
    Same as authenticate endpoint, provided for semantic clarity.
    "Verify" implies checking identity, "authenticate" implies granting access.
    """
    return await authenticate(user_id, audio, challenge_id, db)


@router.get(
    "/threshold",
    summary="Get Authentication Threshold",
    description="Get the current authentication similarity threshold.",
)
async def get_threshold():
    """
    Get the authentication threshold.
    
    Returns:
        threshold: Current similarity threshold (0.50 = 50%)
        description: Human-readable explanation
    """
    return {
        "threshold": AUTH_THRESHOLD,
        "percentage": f"{AUTH_THRESHOLD * 100:.0f}%",
        "description": "Voice recordings must match at least this percentage to authenticate successfully."
    }
