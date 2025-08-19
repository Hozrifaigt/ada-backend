import logging
from fastapi import HTTPException, status

from app.models.policy_models import ChunkInfo
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

class ChunkService:
    """Service for chunk management operations."""
    
    @staticmethod
    async def get_chunk_by_id(chunk_id: str, store: AzureVectorStore) -> ChunkInfo:
        """Get a specific chunk by its unique chunk ID - simplified response."""
        logger.info(f"Getting chunk with ID: '{chunk_id}'")
        
        try:
            chunk = store.search_client.get_document(key=chunk_id)
            
            if chunk and chunk.get("record_type") == "chunk":
                logger.info(f"Successfully retrieved chunk '{chunk_id}' for topic '{chunk['topic']}'")
                return ChunkInfo(chunk_id=chunk["id"], chunk_content=chunk.get("content", ""))
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chunk with ID '{chunk_id}' not found"
                )
                
        except ImportError:
            logger.warning("get_document method not available, falling back to search")
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chunk with ID '{chunk_id}' not found"
                )
            logger.warning(f"get_document failed: {e}, falling back to search")
            
        # Fallback to search if direct lookup fails
        try:
            results = list(store.search_client.search(
                search_text="*",
                filter=f"record_type eq 'chunk' and id eq '{chunk_id}'",
                select="id",
                top=1
            ))
            
            if results:
                chunk = results[0]
                return ChunkInfo(chunk_id=chunk["id"])
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chunk with ID '{chunk_id}' not found"
                )
                
        except Exception as e:
            logger.error(f"Failed to retrieve chunk by search: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve chunk: {str(e)}"
            )
