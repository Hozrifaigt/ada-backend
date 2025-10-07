import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import user, processing, admin, auth, drafts, debug
from app.api.policy import router as policy_router
from app.config.logging_config import setup_logging
from app.config.settings import settings

if os.environ.get('DEBUG_MODE') == 'true':
    import debugpy
    if not debugpy.is_client_connected():
        debugpy.listen(("0.0.0.0", 5678))
        print("🐛 Debug mode enabled. Debugger listening on port 5678...")
        print("🔗 VS Code: Connect to localhost:5678 to start debugging")

        # Wait for debugger if explicitly requested
        if os.environ.get('WAIT_FOR_DEBUGGER', 'false').lower() == 'true':
            print("⏸️  Waiting for debugger to attach...")
            debugpy.wait_for_client()
            print("✅ Debugger attached! Starting application...")

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Drafting Policy",
    description="An AI-powered system for drafting policies based on client profiles and existing documents.",
    version="1.0.0",
)

# Configure CORS - support both local development and Azure deployment
allowed_origins = settings.ALLOWED_ORIGINS.split(",") if "," in settings.ALLOWED_ORIGINS else [settings.ALLOWED_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global exception handler for HTTP exceptions, especially authentication errors.
    """
    # Log authentication-related errors
    if exc.status_code in [401, 403]:
        logger.warning(f"Authentication error on {request.url}: {exc.detail}")
    elif exc.status_code >= 500:
        logger.error(f"Server error on {request.url}: {exc.detail}")
    else:
        logger.info(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "authenticated": exc.status_code not in [401, 403],
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unexpected errors.
    """
    logger.error(f"Unexpected error on {request.url}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "authenticated": False,
            "path": str(request.url.path)
        }
    )

app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(policy_router, prefix="/api/v1", tags=["Policy Drafting"])
app.include_router(drafts.router, prefix="/api/v1", tags=["Draft Management"])
app.include_router(user.router, prefix="/api/v1", tags=["User Management"])
app.include_router(processing.router, prefix="/api/v1", tags=["Pipeline Trigger"])
app.include_router(admin.router, prefix="/api/v1", tags=["Administration"])
app.include_router(debug.router, prefix="/api/v1", tags=["Debug"])


@app.get("/healthcheck", tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Drafting Policy application started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 AI Drafting Policy application shutting down")