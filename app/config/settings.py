from enum import Enum
from typing import Literal
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
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str

    ADMIN_API_KEY: str 
    PROMPT_MANAGER_PATH: str

    DATA_PATH: str
    MAX_WORKERS: int = 1
    BATCH_SIZE: int = 12
    LLM_CHUNK_SIZE: int = 10000
    LLM_CHUNK_OVERLAP: int = 300


settings = Settings()
