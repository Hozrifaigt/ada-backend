from functools import lru_cache
from app.config.settings import settings
from app.core.rag_system import RAGSystem
from app.services.local_rag import LocalRAGSystem
from app.services.azure_rag import AzureRAGSystem


@lru_cache()
def get_rag_system() -> RAGSystem:
    """
    Returns an instance of the RAG system based on the configuration.
    """
    if settings.HOST_MODE == "local":
        return LocalRAGSystem()
    elif settings.HOST_MODE == "azure":
        return AzureRAGSystem()
    else:
        raise ValueError(f"Unsupported RAG system: {settings.HOST_MODE}")