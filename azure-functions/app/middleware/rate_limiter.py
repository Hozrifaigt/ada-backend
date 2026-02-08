import logging
from typing import Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.config.settings import settings

logger = logging.getLogger(__name__)


def get_user_identifier(request: Request) -> str:
    """
    Extract user identifier from request.
    Priority: user email > user ID > IP address
    """
    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, 'user'):
        user = request.state.user
        if hasattr(user, 'email'):
            return f"user:{user.email}"
        elif hasattr(user, 'id'):
            return f"user:{user.id}"
        elif hasattr(user, 'sub'):
            return f"user:{user.sub}"

    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"


# Initialize the rate limiter with in-memory storage
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=[
        f"{settings.RATE_LIMIT_PER_MINUTE}/minute",
        f"{settings.RATE_LIMIT_PER_HOUR}/hour"
    ],
    enabled=settings.RATE_LIMIT_ENABLED,
    storage_uri="memory://",
    strategy="fixed-window",
)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors.
    """
    user_id = get_user_identifier(request)
    logger.warning(
        f"Rate limit exceeded for {user_id} on {request.url.path}",
        extra={
            "user_id": user_id,
            "path": request.url.path,
            "method": request.method,
        }
    )

    raise HTTPException(
        status_code=429,
        detail={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.detail if hasattr(exc, 'detail') else "60 seconds",
        }
    )


def get_limiter() -> Limiter:
    """
    Get the configured rate limiter instance.
    """
    return limiter
