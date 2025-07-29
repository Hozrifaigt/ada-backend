from enum import Enum
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT: str
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = 1536  # ADA model's vector size
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str
    AZURE_SEARCH_INDEX_NAME: str
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str

    DATA_PATH: str
    MAX_WORKERS: int = 1
    BATCH_SIZE: int = 32

    class Config:
        env_file = "env.dev"
        env_file_encoding = 'utf-8'


settings = Settings()
