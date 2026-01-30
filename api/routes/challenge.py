"""
VoiceAuth API - Challenge Routes
================================
Challenge phrase endpoints for authentication.
"""

from fastapi import APIRouter, HTTPException, status

from ..config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, Messages
from ..schemas import (
    ChallengeRequest,
    ChallengeResponse,
    LanguageInfo,
    LanguagesResponse,
    ErrorResponse,
)
from ..dependencies import get_challenge_cache

# Import from main project
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_random_challenge, PHRASES_BY_LANGUAGE, LANGUAGE_NAMES


router = APIRouter(prefix="/challenge", tags=["Challenge"])


@router.post(
    "",
    response_model=ChallengeResponse,
    summary="Get Challenge Phrase",
    description="Get a random challenge phrase for the user to speak during authentication.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid language"},
    }
)
async def create_challenge(request: ChallengeRequest = None):
    """
    Generate a challenge phrase for authentication.
    
    The challenge phrase is a sentence the user must speak.
    This ensures liveness (not a replay attack) and provides
    consistent audio for verification.
    
    Args:
        request: Optional language preference
        
    Returns:
        Challenge with ID, text, language, and expiration info
    """
    # Parse request
    language = None
    if request and request.language:
        language = request.language.lower()
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
    
    # Get random challenge from utils
    challenge_data = get_random_challenge(language)
    
    # Store in cache with expiration
    cache = get_challenge_cache()
    challenge = cache.create(
        text=challenge_data["text"],
        language=challenge_data["lang"],
        language_display=challenge_data["lang_display"],
    )
    
    return ChallengeResponse(
        challenge_id=challenge["challenge_id"],
        text=challenge["text"],
        language=challenge["language"],
        language_display=challenge["language_display"],
        expires_at=challenge["expires_at"],
        expires_in_seconds=challenge["expires_in_seconds"],
    )


@router.get(
    "",
    response_model=ChallengeResponse,
    summary="Get Challenge Phrase (GET)",
    description="Get a random challenge phrase (convenience GET endpoint).",
)
async def get_challenge(language: str = None):
    """
    GET version of challenge creation.
    
    Args:
        language: Optional language code (english, hindi, marathi)
        
    Returns:
        Challenge phrase with ID and expiration
    """
    request = ChallengeRequest(language=language) if language else None
    return await create_challenge(request)


@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="List Supported Languages",
    description="Get list of supported languages for challenge phrases.",
)
async def list_languages():
    """
    Get all supported languages for challenge phrases.
    
    Returns:
        List of languages with codes, display names, and phrase counts
    """
    languages = []
    
    for code in SUPPORTED_LANGUAGES:
        phrases = PHRASES_BY_LANGUAGE.get(code, [])
        display_name = LANGUAGE_NAMES.get(code, code.capitalize())
        
        languages.append(LanguageInfo(
            code=code,
            display_name=display_name,
            phrase_count=len(phrases),
        ))
    
    return LanguagesResponse(
        languages=languages,
        default=DEFAULT_LANGUAGE,
    )


@router.get(
    "/verify/{challenge_id}",
    summary="Verify Challenge",
    description="Check if a challenge ID is valid and not expired.",
    responses={
        404: {"model": ErrorResponse, "description": "Challenge not found or expired"},
    }
)
async def verify_challenge(challenge_id: str):
    """
    Verify a challenge ID is valid.
    
    Note: This does NOT consume the challenge. Use authentication
    endpoint with challenge_id to consume it.
    
    Args:
        challenge_id: The challenge ID to verify
        
    Returns:
        Challenge details if valid
        
    Raises:
        404: If challenge not found or expired
    """
    cache = get_challenge_cache()
    challenge = cache.verify(challenge_id)
    
    if challenge is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Challenge not found or expired"
        )
    
    return {
        "valid": True,
        "challenge_id": challenge["challenge_id"],
        "language": challenge["language"],
        "expires_at": challenge["expires_at"],
    }
