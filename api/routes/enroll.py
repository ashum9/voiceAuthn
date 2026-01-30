"""
VoiceAuth API - Enrollment Routes
=================================
Voice enrollment endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status

from ..config import Messages, MIN_ENROLLMENT_SAMPLES, MAX_ENROLLMENT_SAMPLES
from ..schemas import (
    EnrollmentResponse,
    MultiEnrollmentSampleResponse,
    ErrorResponse,
    UserExistsResponse,
)
from ..dependencies import get_db, validate_audio_file

from ..services.voice_service import (
    enroll_user,
    enroll_user_multi,
    check_user_exists,
    EnrollmentError,
)

from storage import VoiceprintDB


router = APIRouter(prefix="/enroll", tags=["Enrollment"])


@router.post(
    "",
    response_model=EnrollmentResponse,
    summary="Enroll User (Single Sample)",
    description="Enroll a user with a single voice sample. Quick enrollment.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio or user ID"},
        409: {"model": ErrorResponse, "description": "User already enrolled"},
    }
)
async def enroll_single(
    user_id: str = Form(..., description="Unique user identifier"),
    audio: UploadFile = File(..., description="Voice recording (WAV, WebM, MP3, OGG)"),
    overwrite: bool = Form(False, description="Overwrite existing enrollment if user exists"),
    db: VoiceprintDB = Depends(get_db),
):
    """
    Enroll a user with a single voice sample.
    
    This is the quick enrollment option. For better accuracy,
    use multi-sample enrollment with 3-5 samples.
    
    Args:
        user_id: Unique identifier for the user
        audio: Voice recording file
        overwrite: If True, replace existing enrollment
        
    Returns:
        Enrollment result with user_id and sample count
        
    Raises:
        400: Invalid audio or user ID
        409: User already enrolled (and overwrite=False)
    """
    # Validate audio file
    audio_bytes = await validate_audio_file(audio)
    
    # Check if user exists
    if check_user_exists(db, user_id) and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user_id}' is already enrolled. Set overwrite=true to re-enroll."
        )
    
    # Enroll user
    try:
        result = enroll_user(db, user_id, audio_bytes, overwrite=overwrite)
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    return EnrollmentResponse(
        success=result["success"],
        message=result["message"],
        user_id=result["user_id"],
        samples_count=result["samples_count"],
        quality_score=result.get("quality_score"),
        consistency_score=result.get("consistency_score"),
        created_at=result["created_at"],
    )


@router.post(
    "/multi",
    response_model=EnrollmentResponse,
    summary="Enroll User (Multiple Samples)",
    description="Enroll a user with multiple voice samples for better accuracy.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio or user ID"},
        409: {"model": ErrorResponse, "description": "User already enrolled"},
    }
)
async def enroll_multi(
    user_id: str = Form(..., description="Unique user identifier"),
    audio_files: List[UploadFile] = File(..., description="Multiple voice recordings (1-5 files)"),
    overwrite: bool = Form(False, description="Overwrite existing enrollment"),
    db: VoiceprintDB = Depends(get_db),
):
    """
    Enroll a user with multiple voice samples.
    
    Multi-sample enrollment averages embeddings from multiple recordings
    to create a more robust voiceprint. Research shows 10-30% improvement
    in accuracy with 3-5 samples.
    
    Args:
        user_id: Unique identifier for the user
        audio_files: List of voice recordings (1-5 files)
        overwrite: If True, replace existing enrollment
        
    Returns:
        Enrollment result with quality metrics
    """
    # Validate sample count
    if len(audio_files) < MIN_ENROLLMENT_SAMPLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {MIN_ENROLLMENT_SAMPLES} audio sample(s) required"
        )
    
    if len(audio_files) > MAX_ENROLLMENT_SAMPLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_ENROLLMENT_SAMPLES} audio samples allowed"
        )
    
    # Check if user exists
    if check_user_exists(db, user_id) and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user_id}' is already enrolled. Set overwrite=true to re-enroll."
        )
    
    # Validate and collect audio bytes
    audio_samples = []
    for i, audio in enumerate(audio_files):
        try:
            audio_bytes = await validate_audio_file(audio)
            audio_samples.append(audio_bytes)
        except HTTPException as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"Sample {i+1}: {e.detail}"
            )
    
    # Enroll with multiple samples
    try:
        result = enroll_user_multi(db, user_id, audio_samples, overwrite=overwrite)
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    return EnrollmentResponse(
        success=result["success"],
        message=result["message"],
        user_id=result["user_id"],
        samples_count=result["samples_count"],
        quality_score=result.get("quality_score"),
        consistency_score=result.get("consistency_score"),
        created_at=result["created_at"],
    )


@router.get(
    "/check/{user_id}",
    response_model=UserExistsResponse,
    summary="Check User Enrollment",
    description="Check if a user is already enrolled.",
)
async def check_enrollment(
    user_id: str,
    db: VoiceprintDB = Depends(get_db),
):
    """
    Check if a user has an enrolled voiceprint.
    
    Args:
        user_id: User ID to check
        
    Returns:
        exists: True if user is enrolled
    """
    exists = check_user_exists(db, user_id)
    
    return UserExistsResponse(
        exists=exists,
        user_id=user_id,
    )
