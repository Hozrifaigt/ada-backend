import azure.functions as func
import json
import os
import sys

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
def hello(req: func.HttpRequest) -> func.HttpResponse:
    """Simple test endpoint"""
    return func.HttpResponse(
        json.dumps({
            "message": "Hello from Azure Functions",
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
        }),
        status_code=200,
        mimetype="application/json"
    )

# Try to load FastAPI
try:
    from app.main import app as fastapi_app

    # Add FastAPI wrapper
    asgi_app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)

    # Merge the apps - this is the key!
    for function_name in asgi_app.get_functions():
        app.register_functions(asgi_app.get_functions())

except Exception as e:
    # Add error endpoint if FastAPI fails
    @app.route(route="fastapi-error")
    def fastapi_error(req: func.HttpRequest) -> func.HttpResponse:
        return func.HttpResponse(
            json.dumps({
                "error": "FastAPI failed to load",
                "type": type(e).__name__,
                "message": str(e)
            }),
            status_code=200,
            mimetype="application/json"
        )