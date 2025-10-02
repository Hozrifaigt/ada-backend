"""
Azure Functions wrapper for the FastAPI application.
"""
import azure.functions as func
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.main import app as fastapi_app
    logger.info("Successfully imported FastAPI app")

    # Create the Azure Functions app wrapper with ASGI
    app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
    logger.info("Successfully created AsgiFunctionApp")

except Exception as e:
    logger.error(f"Failed to import FastAPI app: {str(e)}")
    logger.error(f"Import error details: {type(e).__name__}")

    # Fallback function for debugging
    app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

    @app.function_name(name="HttpTrigger")
    @app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
        logger.info(f'HTTP trigger processed a request to {req.url}')

        error_response = {
            "error": "FastAPI app failed to load",
            "details": str(e),
            "path": req.url,
            "method": req.method
        }

        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

# Add a test endpoint to verify deployment
@app.function_name(name="Test")
@app.route(route="test")
async def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Test function triggered')
    return func.HttpResponse(
        json.dumps({"status": "Azure Functions is working", "message": "FastAPI integration pending"}),
        status_code=200,
        mimetype="application/json"
    )