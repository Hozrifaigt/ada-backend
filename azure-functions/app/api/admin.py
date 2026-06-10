import logging

from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks

from app.config.settings import settings
from app.pipelines.components.index_manager import AzureSearchIndexManager
from app.pipelines.pipeline_trigger import run_indexing_pipeline

logger = logging.getLogger(__name__)

class AdminResponse(BaseModel):
    message: str
    details: Optional[Dict[str, Any]] = None

class IndexStatsResponse(BaseModel):
    index_name: str
    field_count: int
    vector_search_enabled: bool
    semantic_search_enabled: bool
    fields: list

router = APIRouter(prefix="/admin")

def _validate_admin_api_key(api_key: Optional[str]) -> bool:
    """Validate admin API key from header."""
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
    Deletes and recreates the entire Azure AI Search index with the new RAG schema.
    This will remove all indexed documents and create a fresh index.
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
            logger.warning(f"Successfully RESET INDEX '{index_name}'.")
            stats = index_manager.get_index_stats()
            
            return AdminResponse(
                message=f"Search index '{index_name}' reset successfully. Ready for new document ingestion.",
                details={
                    "index_name": index_name,
                    "field_count": stats['field_count'] if stats else 0,
                    "vector_search_enabled": stats['vector_search_enabled'] if stats else False,
                    "semantic_search_enabled": stats['semantic_search_enabled'] if stats else False
                }
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
            detail=f"An error occurred while trying to reset the index: {str(e)}"
        )

@router.get("/index-stats", response_model=IndexStatsResponse)
def get_index_stats(x_admin_api_key: Optional[str] = Header(None)):
    """
    Get statistics and schema information for the current search index.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for index stats")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )
    
    try:
        index_manager = AzureSearchIndexManager()
        stats = index_manager.get_index_stats()
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index '{settings.AZURE_SEARCH_INDEX_NAME}' not found"
            )
        
        return IndexStatsResponse(
            index_name=stats['name'],
            field_count=stats['field_count'],
            vector_search_enabled=stats['vector_search_enabled'],
            semantic_search_enabled=stats['semantic_search_enabled'],
            fields=stats['fields']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve index statistics: {str(e)}"
        )

@router.post("/reindex-all", response_model=AdminResponse)
async def reindex_all_documents(
    background_tasks: BackgroundTasks,
    x_admin_api_key: Optional[str] = Header(None)
):
    """
    Trigger a complete reindexing of all documents.
    This will process all documents in the data directory and update the index.
    Runs as a background task.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for reindexing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )
    
    logger.info("Admin triggered complete reindexing")
    
    # Add the indexing pipeline to background tasks
    background_tasks.add_task(run_indexing_pipeline)
    
    return AdminResponse(
        message="Reindexing job started successfully. The process is running in the background.",
        details={
            "data_path": settings.DATA_PATH,
            "index_name": settings.AZURE_SEARCH_INDEX_NAME,
            "status": "Processing started - check server logs for progress"
        }
    )

@router.delete("/clear-index", response_model=AdminResponse)
def clear_index_data(x_admin_api_key: Optional[str] = Header(None)):
    """
    Clear all documents from the index without deleting the index structure.
    This is faster than reset-index if you just want to remove documents.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for index clear")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )
    
    try:
        from app.pipelines.components.vector_store import AzureVectorStore
        from app.pipelines.components.embedder import AzureEmbedder
        
        logger.info("Clearing all documents from index")
        
        # Initialize components
        embedder = AzureEmbedder()
        vector_store = AzureVectorStore(embedder=embedder)
        
        # Get all documents and delete them
        # Note: In production, you might want to implement a bulk delete operation
        search_results = vector_store.search_client.search(
            search_text="*",
            select="id",
            top=1000  # Adjust based on your expected document count
        )
        
        document_ids = [doc["id"] for doc in search_results]
        
        if document_ids:
            # Delete in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(document_ids), batch_size):
                batch = document_ids[i:i + batch_size]
                delete_batch = [{"id": doc_id} for doc_id in batch]
                
                result = vector_store.search_client.delete_documents(documents=delete_batch)
                deleted_count += sum(1 for r in result if r.succeeded)
            
            logger.info(f"Deleted {deleted_count} documents from index")
            
            return AdminResponse(
                message=f"Successfully cleared {deleted_count} documents from index",
                details={
                    "documents_deleted": deleted_count,
                    "index_name": settings.AZURE_SEARCH_INDEX_NAME
                }
            )
        else:
            return AdminResponse(
                message="Index is already empty",
                details={
                    "documents_deleted": 0,
                    "index_name": settings.AZURE_SEARCH_INDEX_NAME
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to clear index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear index: {str(e)}"
        )

@router.post("/process-and-index", response_model=AdminResponse)
async def process_and_index_documents(
    background_tasks: BackgroundTasks,
    reset_first: bool = False,
    x_admin_api_key: Optional[str] = Header(None)
):
    """
    Complete pipeline: optionally reset index, then process and index all documents.
    
    Args:
        reset_first: If true, will reset the index before processing documents
        x_admin_api_key: Admin API key for authentication
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for process-and-index")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )
    
    try:
        if reset_first:
            logger.info("Resetting index before processing documents")
            index_manager = AzureSearchIndexManager()
            
            if not index_manager.recreate_index():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to reset index before processing"
                )
            
            logger.info("Index reset successful, starting document processing")
        
        # Add the indexing pipeline to background tasks
        background_tasks.add_task(run_indexing_pipeline)
        
        return AdminResponse(
            message=f"Document processing started successfully{' (index was reset first)' if reset_first else ''}",
            details={
                "index_reset": reset_first,
                "data_path": settings.DATA_PATH,
                "index_name": settings.AZURE_SEARCH_INDEX_NAME,
                "status": "Processing started - check server logs for progress"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start process-and-index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )

@router.delete("/sync-sharepoint/delta", response_model=AdminResponse)
def reset_sharepoint_delta(x_admin_api_key: Optional[str] = Header(None)):
    """
    Clear the saved delta token so the next sync-sharepoint call does a full
    re-scan of the entire SharePoint library instead of only fetching changes.
    Use this after clearing the search index.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )

    if not settings.SHAREPOINT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SharePoint integration is disabled."
        )

    from app.services.sync_state_storage import SyncStateStorage
    SyncStateStorage().reset_delta()

    return AdminResponse(
        message="Delta token cleared. The next sync-sharepoint call will re-ingest all files.",
        details={"delta_token_present": False}
    )


@router.get("/sync-sharepoint/status", response_model=AdminResponse)
def get_sharepoint_sync_status(x_admin_api_key: Optional[str] = Header(None)):
    """
    Return the last SharePoint sync time and whether a delta token is stored.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )

    if not settings.SHAREPOINT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SharePoint integration is disabled."
        )

    from app.services.sync_state_storage import SyncStateStorage
    state = SyncStateStorage().get_status()

    return AdminResponse(
        message="SharePoint sync status retrieved",
        details={
            "sharepoint_enabled": settings.SHAREPOINT_ENABLED,
            "site_url": settings.SHAREPOINT_SITE_URL,
            "drive": settings.SHAREPOINT_DRIVE_NAME,
            "folder": settings.SHAREPOINT_FOLDER_PATH or "(root)",
            **state,
        }
    )


@router.post("/sync-sharepoint", response_model=AdminResponse)
async def sync_sharepoint_library(
    background_tasks: BackgroundTasks,
    x_admin_api_key: Optional[str] = Header(None)
):
    """
    Trigger a SharePoint CDC sync: fetch only the changed .docx files from the
    configured SharePoint library and update the Azure Search index accordingly.
    Uses the Microsoft Graph delta API so only new/modified/deleted files are processed.
    Runs as a background task.
    """
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for SharePoint sync")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )

    if not settings.SHAREPOINT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SharePoint integration is disabled. Set SHAREPOINT_ENABLED=true in config."
        )

    logger.info("Admin triggered SharePoint CDC sync")

    from app.services.sharepoint_sync_service import SharePointSyncService
    sync_service = SharePointSyncService()
    background_tasks.add_task(sync_service.run_sync)

    return AdminResponse(
        message="SharePoint sync started. Processing changes in the background.",
        details={
            "site_url": settings.SHAREPOINT_SITE_URL,
            "drive": settings.SHAREPOINT_DRIVE_NAME,
            "folder": settings.SHAREPOINT_FOLDER_PATH or "(root)",
            "status": "Running — check server logs for progress",
        }
    )


# Excel template endpoints removed - no longer using Excel sheets for policy templates