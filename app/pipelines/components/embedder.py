import logging
import asyncio
import time

from typing import List
from app.config.settings import settings
from langchain_openai import AzureOpenAIEmbeddings
from openai import APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)

class AzureEmbedder:
    """Creates embeddings using Azure OpenAI - REAL embeddings for production."""
    
    def __init__(self):
        self.client = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            request_timeout=120,  # 2 minutes timeout
            max_retries=5,  # Increase retry attempts
        )
        logger.info("AzureEmbedder initialized with Azure OpenAI embeddings")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"Generating embeddings for {len(texts)} texts")

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries + 1):
            try:
                embeddings = self.client.embed_documents(texts)
                logger.info(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

            except (APIConnectionError, APITimeoutError) as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection/timeout error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to generate embeddings after {max_retries + 1} attempts: {e}")
                    raise RuntimeError("Embedding generation failed after retries") from e

            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                raise RuntimeError("Embedding generation failed") from e
    
    def embed_query(self, text: str) -> List[float]:
        logger.debug(f"Generating embedding for query: '{text[:50]}...'")

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries + 1):
            try:
                embedding = self.client.embed_query(text)
                logger.debug("Successfully generated query embedding")
                return embedding

            except (APIConnectionError, APITimeoutError) as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection/timeout error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to generate embedding after {max_retries + 1} attempts: {e}")
                    raise RuntimeError("Embedding generation failed after retries") from e