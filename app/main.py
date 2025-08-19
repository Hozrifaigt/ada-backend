from fastapi import FastAPI
from app.api import user, processing, admin
from app.api.policy import router as policy_router
from app.config.logging_config import setup_logging

setup_logging()

app = FastAPI(
    title="AI Drafting Policy",
    description="An AI-powered system for drafting policies based on client profiles and existing documents.",
    version="1.0.0",
)

app.include_router(policy_router, prefix="/api/v1", tags=["Policy Drafting"])
app.include_router(user.router, prefix="/api/v1", tags=["User Management"])
app.include_router(processing.router, prefix="/api/v1", tags=["Pipeline Trigger"])
app.include_router(admin.router, prefix="/api/v1", tags=["Administration"])


@app.get("/healthcheck", tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}