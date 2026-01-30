"""
VoiceAuth API - Voice Service
=============================
Core voice authentication business logic.

Handles:
- Voice embedding extraction
- Enrollment (single and multi-sample)
- Authentication/verification
- Similarity computation
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime, timezone

import numpy as np

# Import from main project
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    get_speaker_model,
    get_embedding,
    compute_similarity,
    average_embeddings,
    compute_enrollment_quality,
    validate_embedding,
    normalize_embedding,
    EMBEDDING_DIM,
    SAMPLE_RATE,
)

from storage import VoiceprintDB

from .audio_service import (
    process_audio_bytes,
    save_temp_audio,
    cleanup_temp_file,
    AudioProcessingError,
)

from ..config import AUTH_THRESHOLD, Messages


class VoiceAuthError(Exception):
    """Custom exception for voice authentication errors."""
    pass


class EnrollmentError(VoiceAuthError):
    """Error during enrollment."""
    pass


class AuthenticationError(VoiceAuthError):
    """Error during authentication."""
    pass


def extract_embedding_from_audio(audio_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    Extract speaker embedding from audio bytes.
    
    Args:
        audio_bytes: Raw audio data (WAV, WebM, etc.)
        
    Returns:
        Tuple of (192-dim embedding vector, metadata dict)
        
    Raises:
        VoiceAuthError: If extraction fails
    """
    temp_file = None
    
    try:
        # Process audio
        waveform, metadata = process_audio_bytes(audio_bytes)
        
        # Save to temp file for model processing
        # (get_embedding expects a file path)
        temp_file = save_temp_audio(audio_bytes)
        
        # Extract embedding
        embedding = get_embedding(temp_file, warn_short=True)
        
        # Validate embedding
        embedding = validate_embedding(embedding, "extracted_embedding")
        
        # Add to metadata
        metadata["embedding_dim"] = embedding.shape[0]
        
        return embedding, metadata
        
    except AudioProcessingError as e:
        raise VoiceAuthError(str(e))
    except Exception as e:
        raise VoiceAuthError(f"Failed to extract voice embedding: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file:
            cleanup_temp_file(temp_file)


def enroll_user(
    db: VoiceprintDB,
    user_id: str,
    audio_bytes: bytes,
    overwrite: bool = False
) -> dict:
    """
    Enroll a user with a single voice sample.
    
    Args:
        db: VoiceprintDB instance
        user_id: Unique user identifier
        audio_bytes: Voice recording bytes
        overwrite: If True, overwrite existing enrollment
        
    Returns:
        Enrollment result dict
        
    Raises:
        EnrollmentError: If enrollment fails
    """
    # Check if user already exists
    if db.exists(user_id) and not overwrite:
        raise EnrollmentError(f"User '{user_id}' is already enrolled. Use overwrite=True to re-enroll.")
    
    # Extract embedding
    try:
        embedding, metadata = extract_embedding_from_audio(audio_bytes)
    except VoiceAuthError as e:
        raise EnrollmentError(str(e))
    
    # Store embedding
    try:
        success = db.store(user_id, embedding)
        if not success:
            raise EnrollmentError("Failed to store voiceprint in database")
    except Exception as e:
        raise EnrollmentError(f"Database error: {str(e)}")
    
    return {
        "success": True,
        "message": Messages.ENROLLMENT_SUCCESS,
        "user_id": user_id,
        "samples_count": 1,
        "quality_score": None,  # Single sample, no quality metric
        "consistency_score": None,
        "duration": metadata.get("duration"),
        "created_at": datetime.now(timezone.utc),
    }


def enroll_user_multi(
    db: VoiceprintDB,
    user_id: str,
    audio_samples: List[bytes],
    overwrite: bool = False
) -> dict:
    """
    Enroll a user with multiple voice samples.
    
    Averages embeddings from multiple samples for better accuracy.
    Research shows 10-30% EER reduction with multi-enrollment.
    
    Args:
        db: VoiceprintDB instance
        user_id: Unique user identifier
        audio_samples: List of voice recording bytes
        overwrite: If True, overwrite existing enrollment
        
    Returns:
        Enrollment result dict
        
    Raises:
        EnrollmentError: If enrollment fails
    """
    if not audio_samples:
        raise EnrollmentError("No audio samples provided")
    
    # Check if user already exists
    if db.exists(user_id) and not overwrite:
        raise EnrollmentError(f"User '{user_id}' is already enrolled")
    
    # Extract embeddings from all samples
    embeddings = []
    total_duration = 0
    
    for i, audio_bytes in enumerate(audio_samples):
        try:
            embedding, metadata = extract_embedding_from_audio(audio_bytes)
            embeddings.append(embedding)
            total_duration += metadata.get("duration", 0)
        except VoiceAuthError as e:
            raise EnrollmentError(f"Sample {i+1} failed: {str(e)}")
    
    # Compute quality metrics
    quality_metrics = compute_enrollment_quality(embeddings)
    
    # Average embeddings
    try:
        averaged_embedding = average_embeddings(embeddings, normalize_result=True)
    except Exception as e:
        raise EnrollmentError(f"Failed to average embeddings: {str(e)}")
    
    # Store averaged embedding
    try:
        success = db.store(user_id, averaged_embedding)
        if not success:
            raise EnrollmentError("Failed to store voiceprint in database")
    except Exception as e:
        raise EnrollmentError(f"Database error: {str(e)}")
    
    return {
        "success": True,
        "message": Messages.ENROLLMENT_SUCCESS,
        "user_id": user_id,
        "samples_count": len(embeddings),
        "quality_score": quality_metrics.get("consistency", None),
        "consistency_score": quality_metrics.get("min_similarity", None),
        "total_duration": total_duration,
        "created_at": datetime.now(timezone.utc),
    }


def authenticate_user(
    db: VoiceprintDB,
    user_id: str,
    audio_bytes: bytes,
    threshold: float = AUTH_THRESHOLD
) -> dict:
    """
    Authenticate a user by voice.
    
    Args:
        db: VoiceprintDB instance
        user_id: User ID to authenticate against
        audio_bytes: Voice recording bytes
        threshold: Similarity threshold (default: 0.50 = 50%)
        
    Returns:
        Authentication result dict with:
        - success: Always True (request succeeded)
        - authenticated: Whether voice matched
        - similarity_score: Only included on failure
        - message: Human-readable result
        
    Raises:
        AuthenticationError: If authentication process fails
    """
    # Check if user exists
    if not db.exists(user_id):
        raise AuthenticationError(f"User '{user_id}' not found")
    
    # Retrieve stored embedding
    try:
        stored_embedding = db.retrieve(user_id)
        if stored_embedding is None:
            raise AuthenticationError(f"No voiceprint found for user '{user_id}'")
    except Exception as e:
        raise AuthenticationError(f"Database error: {str(e)}")
    
    # Extract embedding from audio
    try:
        new_embedding, metadata = extract_embedding_from_audio(audio_bytes)
    except VoiceAuthError as e:
        raise AuthenticationError(str(e))
    
    # Compute similarity
    try:
        similarity = compute_similarity(new_embedding, stored_embedding)
    except Exception as e:
        raise AuthenticationError(f"Similarity computation failed: {str(e)}")
    
    # Make decision
    authenticated = similarity >= threshold
    
    # Build response (as per requirements)
    result = {
        "success": True,
        "authenticated": authenticated,
        "user_id": user_id,
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc),
    }
    
    if authenticated:
        # Success: No score shown
        result["message"] = Messages.AUTH_SUCCESS
        result["similarity_score"] = None
    else:
        # Failure: Show similarity score
        result["message"] = f"{Messages.AUTH_FAILED}. Similarity: {similarity:.1%}"
        result["similarity_score"] = round(similarity, 4)
    
    return result


def check_user_exists(db: VoiceprintDB, user_id: str) -> bool:
    """Check if a user has an enrolled voiceprint."""
    return db.exists(user_id)


def delete_user(db: VoiceprintDB, user_id: str) -> dict:
    """
    Delete a user's voiceprint (GDPR right to erasure).
    
    Args:
        db: VoiceprintDB instance
        user_id: User ID to delete
        
    Returns:
        Deletion result dict
        
    Raises:
        VoiceAuthError: If user not found
    """
    if not db.exists(user_id):
        raise VoiceAuthError(f"User '{user_id}' not found")
    
    success = db.delete(user_id)
    
    return {
        "success": success,
        "message": Messages.USER_DELETED if success else "Failed to delete user",
        "user_id": user_id,
    }


def list_users(db: VoiceprintDB) -> List[str]:
    """Get list of all enrolled user IDs."""
    return db.list_users()


def get_database_stats(db: VoiceprintDB) -> dict:
    """Get database statistics."""
    return db.get_stats()
