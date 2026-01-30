"""
VoiceAuth API - User Management Routes
======================================
User listing and deletion endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Messages
from ..schemas import (
    UserListResponse,
    UserDeleteResponse,
    UserExistsResponse,
    ErrorResponse,
)
from ..dependencies import get_db

from ..services.voice_service import (
    list_users,
    delete_user,
    check_user_exists,
    VoiceAuthError,
)

from storage import VoiceprintDB


router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "",
    response_model=UserListResponse,
    summary="List Enrolled Users",
    description="Get a list of all enrolled user IDs.",
)
async def get_users(db: VoiceprintDB = Depends(get_db)):
    """
    List all enrolled users.
    
    Returns:
        users: List of user IDs
        total_count: Number of enrolled users
    """
    users = list_users(db)
    
    return UserListResponse(
        users=users,
        total_count=len(users),
    )


@router.get(
    "/{user_id}",
    response_model=UserExistsResponse,
    summary="Check User Exists",
    description="Check if a specific user is enrolled.",
)
async def get_user(
    user_id: str,
    db: VoiceprintDB = Depends(get_db),
):
    """
    Check if a user exists.
    
    Args:
        user_id: User ID to check
        
    Returns:
        exists: True if user is enrolled
        user_id: The queried user ID
    """
    exists = check_user_exists(db, user_id)
    
    return UserExistsResponse(
        exists=exists,
        user_id=user_id,
    )


@router.delete(
    "/{user_id}",
    response_model=UserDeleteResponse,
    summary="Delete User",
    description="Delete a user and their voiceprint (GDPR right to erasure).",
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
    }
)
async def delete_user_endpoint(
    user_id: str,
    db: VoiceprintDB = Depends(get_db),
):
    """
    Delete a user's voiceprint.
    
    This permanently removes the user's voice data from the database.
    Supports GDPR right to erasure.
    
    Args:
        user_id: User ID to delete
        
    Returns:
        Deletion confirmation
        
    Raises:
        404: User not found
    """
    # Check if user exists
    if not check_user_exists(db, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found"
        )
    
    # Delete user
    try:
        result = delete_user(db, user_id)
    except VoiceAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    return UserDeleteResponse(
        success=result["success"],
        message=result["message"],
        user_id=result["user_id"],
    )


@router.delete(
    "",
    summary="Delete All Users",
    description="Delete all enrolled users. Use with caution!",
    responses={
        200: {"description": "All users deleted"},
    }
)
async def delete_all_users(
    confirm: bool = False,
    db: VoiceprintDB = Depends(get_db),
):
    """
    Delete ALL enrolled users.
    
    This is a destructive operation. Requires confirm=true parameter.
    
    Args:
        confirm: Must be True to proceed with deletion
        
    Returns:
        Deletion summary
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This will delete ALL users. Set confirm=true to proceed."
        )
    
    # Get all users
    users = list_users(db)
    deleted_count = 0
    failed_count = 0
    
    # Delete each user
    for user_id in users:
        try:
            delete_user(db, user_id)
            deleted_count += 1
        except Exception:
            failed_count += 1
    
    return {
        "success": True,
        "message": f"Deleted {deleted_count} users",
        "deleted_count": deleted_count,
        "failed_count": failed_count,
    }
