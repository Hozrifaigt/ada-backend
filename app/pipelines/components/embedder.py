import logging

from typing import List
from app.config.settings import settings
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

class AzureEmbedder:
    """Creates embeddings using Azure OpenAI - REAL embeddings for production."""
    
    def __init__(self):
        self.client = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
        )
        logger.info("AzureEmbedder initialized with Azure OpenAI embeddings")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.client.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError("Embedding generation failed") from e
    
    def embed_query(self, text: str) -> List[float]:
        logger.debug(f"Generating embedding for query: '{text[:50]}...'")
        
        try:
            embedding = self.client.embed_query(text)
            logger.debug("Successfully generated query embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError("Embedding generation failed") from e