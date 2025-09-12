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