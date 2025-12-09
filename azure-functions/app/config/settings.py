from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT: str
    AZURE_OPENAI_API_VERSION: str = "2025-01-01-preview"
    AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = 1536


    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str
    AZURE_SEARCH_INDEX_NAME: str
    AZURE_SEARCH_REGULATIONS_INDEX_NAME: str = "regulations"

    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str

    AZURE_TABLE_STORAGE_CONNECTION_STRING: str
    AZURE_TABLE_NAME: str = "Policies"

    AZURE_TENANT_ID: str  
    AZURE_CLIENT_ID: str
    AZURE_CLIENT_SECRET: str

    JWT_ALGORITHM: str = "RS256"
    TOKEN_CACHE_TTL_HOURS: int = 1

    ALLOWED_EMAIL_DOMAINS: str = "ae.gt.com"

    ADMIN_API_KEY: str 
    PROMPT_MANAGER_PATH: str

    DATA_PATH: str
    MAX_WORKERS: int = 1
    BATCH_SIZE: int = 12
    LLM_CHUNK_SIZE: int = 10000
    LLM_CHUNK_OVERLAP: int = 300
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Regulations path - defaults to Docker path, can be overridden
    REGULATIONS_PATH: str = "/code/app/regulations"

    # Regulation Search Settings
    REGULATION_SEARCH_MIN_SCORE: float = 0.55
    REGULATION_SEARCH_TOP_K: int = 5

    # Rate Limiting Settings
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60  # Max requests per minute per user (across all endpoints)
    RATE_LIMIT_PER_HOUR: int = 1000  # Max requests per hour per user (across all endpoints)

settings = Settings()
