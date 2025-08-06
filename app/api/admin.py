import logging

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, status

from app.config.settings import settings
from app.pipelines.components.index_manager import AzureSearchIndexManager

logger = logging.getLogger(__name__)

class AdminResponse(BaseModel):
    message: str
    index_name: str

router = APIRouter()

def _validate_admin_api_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    
    admin_password = settings.ADMIN_API_KEY
    
    if not admin_password:
        logger.error("ADMIN_API_KEY not set in environment variables")
        return False
    
    return api_key == admin_password

@router.delete("/reset-index", response_model=AdminResponse)
def reset_search_index(x_admin_api_key: Optional[str] = Header(None)):
    """
    Deletes and recreates the entire Azure AI Search index with proper RAG schema.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for index reset")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )
    
    index_name = settings.AZURE_SEARCH_INDEX_NAME
    logger.warning(f"CRITICAL: Authorized request to RESET search index '{index_name}'.")
    
    try:        
        index_manager = AzureSearchIndexManager()
        
        logger.info(f"Attempting to reset index '{index_name}'")
        success = index_manager.recreate_index()        
        if success:
            logger.warning(f"Successfully RESET index '{index_name}'.")
            stats = index_manager.get_index_stats()
            
            return AdminResponse(
                message=f"Search index reset successfully."
                       f"Index now has {stats['field_count'] if stats else 'unknown'} fields.",
                index_name=index_name
            )
        else:
            logger.error(f"Failed to reset index '{index_name}'")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset the search index. Check server logs for details."
            )
        
    except Exception as e:
        logger.error(f"Failed to reset index '{index_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while trying to reset the index: {e}"
        )

