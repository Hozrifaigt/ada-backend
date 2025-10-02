"""
Azure Functions wrapper for the FastAPI application.
"""
import azure.functions as func
from app.main import app as fastapi_app

# Create the Azure Functions app wrapper
app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
