"""
Azure Functions wrapper for the FastAPI application.
"""
import azure.functions as func
import logging
import json
import sys
import os
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple test function first
test_app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@test_app.function_name(name="BasicTest")
@test_app.route(route="basic-test")
async def basic_test_function(req: func.HttpRequest) -> func.HttpResponse:
    """Most basic test - no dependencies"""
    return func.HttpResponse(
        json.dumps({
            "status": "Basic Azure Function works",
            "python_version": sys.version,
            "current_dir": os.getcwd(),
            "files_in_root": os.listdir('.'),
            "app_exists": os.path.exists('app'),
            "env_vars_count": len(os.environ)
        }),
        status_code=200,
        mimetype="application/json"
    )

@test_app.function_name(name="EnvTest")
@test_app.route(route="env-test")
async def env_test_function(req: func.HttpRequest) -> func.HttpResponse:
    """Check environment variables without any imports"""
    env_check = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "NOT_SET"),
        "AZURE_OPENAI_KEY": "SET" if os.getenv("AZURE_OPENAI_KEY") else "NOT_SET",
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT", "NOT_SET"),
        "AZURE_SEARCH_KEY": "SET" if os.getenv("AZURE_SEARCH_KEY") else "NOT_SET",
        "PROMPT_MANAGER_PATH": os.getenv("PROMPT_MANAGER_PATH", "NOT_SET"),
        "DATA_PATH": os.getenv("DATA_PATH", "NOT_SET"),
    }

    return func.HttpResponse(
        json.dumps({"environment": env_check}),
        status_code=200,
        mimetype="application/json"
    )

# Now try to import FastAPI
try:
    from app.main import app as fastapi_app
    logger.info("Successfully imported FastAPI app")

    # Create the Azure Functions app wrapper with ASGI
    app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
    logger.info("Successfully created AsgiFunctionApp")

except Exception as e:
    error_details = {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }
    logger.error(f"Failed to import FastAPI app: {json.dumps(error_details)}")

    # Use the test app as fallback
    app = test_app

    @app.function_name(name="ErrorInfo")
    @app.route(route="error-info")
    async def error_info(req: func.HttpRequest) -> func.HttpResponse:
        return func.HttpResponse(
            json.dumps(error_details),
            status_code=200,
            mimetype="application/json"
        )