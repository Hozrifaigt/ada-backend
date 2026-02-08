"""Debug endpoints for testing Azure deployment"""
import os
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/debug/env")
async def check_environment() -> Dict[str, Any]:
    """Check which environment variables are set (masks sensitive values)"""
    env_vars = {
        # Core settings
        "HOST_MODE": os.getenv("HOST_MODE", "NOT_SET"),
        "DATA_PATH": os.getenv("DATA_PATH", "NOT_SET"),
        "MAX_WORKERS": os.getenv("MAX_WORKERS", "NOT_SET"),
        "BATCH_SIZE": os.getenv("BATCH_SIZE", "NOT_SET"),
        "LLM_CHUNK_SIZE": os.getenv("LLM_CHUNK_SIZE", "NOT_SET"),
        "LLM_CHUNK_OVERLAP": os.getenv("LLM_CHUNK_OVERLAP", "NOT_SET"),

        # Azure OpenAI
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "NOT_SET"),
        "AZURE_OPENAI_KEY": "***" if os.getenv("AZURE_OPENAI_KEY") else "NOT_SET",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "NOT_SET"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "NOT_SET"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "NOT_SET"),
        "AZURE_OPENAI_EMBEDDING_DIMENSIONS": os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "NOT_SET"),

        # Azure Search
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT", "NOT_SET"),
        "AZURE_SEARCH_KEY": "***" if os.getenv("AZURE_SEARCH_KEY") else "NOT_SET",
        "AZURE_SEARCH_INDEX_NAME": os.getenv("AZURE_SEARCH_INDEX_NAME", "NOT_SET"),

        # Azure Storage
        "AZURE_STORAGE_CONNECTION_STRING": "***" if os.getenv("AZURE_STORAGE_CONNECTION_STRING") else "NOT_SET",
        "AZURE_STORAGE_CONTAINER_NAME": os.getenv("AZURE_STORAGE_CONTAINER_NAME", "NOT_SET"),
        "AZURE_TABLE_STORAGE_CONNECTION_STRING": "***" if os.getenv("AZURE_TABLE_STORAGE_CONNECTION_STRING") else "NOT_SET",
        "AZURE_TABLE_NAME": os.getenv("AZURE_TABLE_NAME", "NOT_SET"),

        # Azure AD / Auth
        "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID", "NOT_SET"),
        "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID", "NOT_SET"),
        "AZURE_CLIENT_SECRET": "***" if os.getenv("AZURE_CLIENT_SECRET") else "NOT_SET",

        # App settings
        "PROMPT_MANAGER_PATH": os.getenv("PROMPT_MANAGER_PATH", "NOT_SET"),
        "ADMIN_API_KEY": "***" if os.getenv("ADMIN_API_KEY") else "NOT_SET",
        "ALLOWED_EMAIL_DOMAINS": os.getenv("ALLOWED_EMAIL_DOMAINS", "NOT_SET"),
        "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "NOT_SET"),
    }
    return {"environment_variables": env_vars}

@router.get("/debug/test-imports")
async def test_imports() -> Dict[str, Any]:
    """Test if all modules can be imported"""
    results = {}

    try:
        import azure.search.documents
        results["azure_search"] = "OK"
    except ImportError as e:
        results["azure_search"] = f"Failed: {str(e)}"

    try:
        from langchain_openai import AzureChatOpenAI
        results["langchain_openai"] = "OK"
    except ImportError as e:
        results["langchain_openai"] = f"Failed: {str(e)}"

    try:
        from app.services.draft_service import DraftService
        results["draft_service"] = "OK"
    except Exception as e:
        results["draft_service"] = f"Failed: {str(e)}"

    return {"imports": results}

@router.get("/debug/test-connections")
async def test_connections() -> Dict[str, Any]:
    """Test connections to Azure services"""
    results = {}

    # Test Azure OpenAI
    try:
        from azure.identity import DefaultAzureCredential
        from openai import AzureOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_KEY")

        if endpoint and key:
            results["azure_openai"] = "Credentials present"
        else:
            results["azure_openai"] = "Missing credentials"
    except Exception as e:
        results["azure_openai"] = f"Error: {str(e)}"

    # Test Azure Search
    try:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")

        if endpoint and key:
            results["azure_search"] = "Credentials present"
        else:
            results["azure_search"] = "Missing credentials"
    except Exception as e:
        results["azure_search"] = f"Error: {str(e)}"

    return {"connections": results}