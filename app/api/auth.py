import logging


from typing import Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.entra_service import get_current_user

logger = logging.getLogger(__name__)

class UserInfoResponse(BaseModel):
    """Response model for user information."""
    user_id: str
    email: str
    name: str
    tenant_id: str

class AuthStatusResponse(BaseModel):
    """Response model for authentication status."""
    authenticated: bool
    message: str
    user_info: UserInfoResponse = None

router = APIRouter()

@router.get("/auth/me", response_model=UserInfoResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> UserInfoResponse:
    """
    Get information about the currently authenticated user.
    This endpoint requires a valid Bearer token in the Authorization header.
    """
    logger.info(f"User info requested for: {current_user.get('email')}")
    return UserInfoResponse(
        user_id=current_user.get("user_id", ""),
        email=current_user.get("email", ""),
        name=current_user.get("name", ""),
        tenant_id=current_user.get("tenant_id", "")
    )

@router.get("/auth/status", response_model=AuthStatusResponse)
async def check_auth_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> AuthStatusResponse:
    """
    Check if the user is properly authenticated.
    Returns authentication status and basic user information.
    """
    logger.info(f"Auth status check for: {current_user.get('email')}")
    
    return AuthStatusResponse(
        authenticated=True,
        message="User is authenticated",
        user_info=UserInfoResponse(
            user_id=current_user.get("user_id", ""),
            email=current_user.get("email", ""),
            name=current_user.get("name", ""),
            tenant_id=current_user.get("tenant_id", "")
        )
    )

@router.post("/auth/validate-token")
async def validate_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate a token and return user claims.
    Useful for frontend applications to verify token validity.
    """
    logger.info(f"Token validation requested for: {current_user.get('email')}")
    
    return {
        "valid": True,
        "user": current_user,
        "message": "Token is valid"
    }