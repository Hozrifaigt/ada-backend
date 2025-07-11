from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):

    HOST_MODE: Literal["local", "azure"] = "local"

    QDRANT_URL: str = "http://localhost:6666"
    POLICY_CONTENT_COLLECTION_NAME: str = "policy_content_index"
    POLICY_SEARCH_COLLECTION_NAME: str = "policy_search_index"

    OLLAMA_HOST: str = "http://localhost:11444"
    LLM_MODEL: str = "qwen3:8b"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    VECTOR_SIZE: int = 768

    DATA_PATH: str = "data"
    MAX_WORKERS: int = 1
    BATCH_SIZE: int = 32

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
