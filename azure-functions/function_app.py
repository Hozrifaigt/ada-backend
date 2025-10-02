"""
Azure Functions wrapper for the FastAPI application.
This file serves as the entry point for Azure Functions while keeping
the actual FastAPI app in app/main.py unchanged for local development.
"""
import sys
import os
from pathlib import Path

# Add the parent directory to Python path so we can import the app
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import azure.functions as func
from app.main import app as fastapi_app

# Create the Azure Functions app wrapper
app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
