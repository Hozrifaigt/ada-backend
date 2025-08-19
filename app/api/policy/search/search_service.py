import logging

from typing import List, Dict, Any
from fastapi import HTTPException, status
from azure.search.documents.models import VectorizedQuery

from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)


class SearchService:
    """Service for general search operations."""
    
    @staticmethod
    async def search_chunks_by_policy_and_topic(
        policy_id: str,
        topic_id: str,
        store: AzureVectorStore
    ) -> List[Dict[str, Any]]:
        """Search for chunks by policy and topic."""
        try:
            filter_string = f"record_type eq 'chunk' and id eq '{topic_id}'" # source eq '{policy_id}' and
            
            logger.debug(f"Searching chunks with filter: {filter_string}")
            
            results = list(store.search_client.search(
                search_text="*",
                filter=filter_string,
                select="topic,content,source",
                top=10
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to search chunks: {str(e)}"
            )
    
    @staticmethod
    async def semantic_search_within_policy(
        policy_id: str,
        query: str,
        store: AzureVectorStore,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search within a specific policy."""
        try:
            query_vector = store.embedder.embed_query(query)
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = list(store.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=f"record_type eq 'chunk'",
                select="topic,content,source",
                top=top_k
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Semantic search failed: {str(e)}"
            )
