import azure.functions as func
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

startup_error_message = ""
startup_error_type = ""

try:
    from app.main import app as fastapi_app
    logger.info("Successfully imported FastAPI app")
    app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
    logger.info("Created AsgiFunctionApp successfully")

except Exception as e:
    startup_error_message = str(e)
    startup_error_type = type(e).__name__
    
    logger.error(f"Failed to load FastAPI: [{startup_error_type}] {startup_error_message}")

    app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

    @app.route(route="hello")
    def hello(req: func.HttpRequest) -> func.HttpResponse:
        return func.HttpResponse("Hello from fallback.")

    @app.route(route="fastapi-error")
    def fastapi_error(req: func.HttpRequest) -> func.HttpResponse:
        # Now, this function has no dependency on the original 'e' object
        return func.HttpResponse(
            json.dumps({
                "error": "FastAPI failed to load during application startup.",
                "type": startup_error_type,
                "message": startup_error_message[:500] 
            }),
            status_code=200, # Return 200 so you can see the payload
            mimetype="application/json"
        )