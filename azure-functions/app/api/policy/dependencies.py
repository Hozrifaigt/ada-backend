import logging

from fastapi import HTTPException, status


from app.config.settings import settings
from app.prompts.manager import PromptManager
from app.pipelines.components.llm import AzureLLM
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)

try:
    embedder = AzureEmbedder()
    vector_store = AzureVectorStore(embedder=embedder)
    llm = AzureLLM(prompt_manager=prompt_manager)
    logger.info("Azure components initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Azure components: {e}", exc_info=True)
    embedder = None
    vector_store = None
    llm = None

def get_vector_store() -> AzureVectorStore:
    """FastAPI dependency to provide a vector store client instance."""
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not available due to an initialization error."
        )
    return vector_store

def get_llm() -> AzureLLM:
    """FastAPI dependency to provide an LLM instance."""
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM is not available due to an initialization error."
        )
    return llm
