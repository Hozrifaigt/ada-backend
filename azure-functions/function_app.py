import azure.functions as func
import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load FastAPI and use it as the main app
try:
    from app.main import app as fastapi_app
    logger.info("Successfully imported FastAPI app")

    # Use AsgiFunctionApp for FastAPI - this should handle all routes
    app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
    logger.info("Created AsgiFunctionApp successfully")

except Exception as e:
    logger.error(f"Failed to load FastAPI: {str(e)}")

    # Fallback to basic function app
    app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

    @app.route(route="hello")
    def hello(req: func.HttpRequest) -> func.HttpResponse:
        """Simple test endpoint"""
        return func.HttpResponse(
            json.dumps({
                "message": "Hello from Azure Functions (FastAPI failed)",
                "python_version": sys.version,
                "working_dir": os.getcwd()
            }),
            status_code=200,
            mimetype="application/json"
        )

    @app.route(route="env-check")
    def env_check(req: func.HttpRequest) -> func.HttpResponse:
        """Check environment variables"""
        return func.HttpResponse(
            json.dumps({
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "NOT_SET"),
                "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT", "NOT_SET"),
                "PROMPT_MANAGER_PATH": os.getenv("PROMPT_MANAGER_PATH", "NOT_SET"),
                "DATA_PATH": os.getenv("DATA_PATH", "NOT_SET"),
            }),
            status_code=200,
            mimetype="application/json"
        )

    @app.route(route="fastapi-error")
    def fastapi_error(req: func.HttpRequest) -> func.HttpResponse:
        # Keep it simple to avoid secondary errors
        return func.HttpResponse(
            json.dumps({
                "error": "FastAPI failed to load",
                "type": type(e).__name__,
                "message": str(e)[:500]  # Limit message length
            }),
            status_code=200,
            mimetype="application/json"
        )