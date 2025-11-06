import logging

from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks, Depends

from fastapi import File, UploadFile
from app.config.settings import settings
from app.auth.entra_service import get_current_user
from app.pipelines.components.index_manager import AzureSearchIndexManager
from app.pipelines.pipeline_trigger import run_indexing_pipeline
from app.services.excel_parser_service import ExcelParserService
from app.services.policy_template_storage import PolicyTemplateStorageService
from app.models.policy_template_models import UploadPolicyTemplatesResponse

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
            logger.warning(f"Successfully RESET index '{index_name}'.")
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

@router.post("/upload-policy-templates", response_model=UploadPolicyTemplatesResponse)
async def upload_policy_templates(
    file: UploadFile = File(...),
    x_admin_api_key: Optional[str] = Header(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload policy templates from Excel file.

    Expected Excel format:
    - Columns: Function, Policies, Topics, Topic Content, Subtopics, Subtopic Content
    - Empty cells mean "same as above" (merged cell pattern)
    """
    # Check if user is admin (for now, hardcoded email)
    admin_emails = ["hozayfa.rifai@ae.gt.com"]
    user_email = current_user.get("email", "").lower()

    if user_email not in admin_emails:
        logger.warning(f"SECURITY: Non-admin user {user_email} attempted to upload templates")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource"
        )

    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an Excel file (.xlsx or .xls)"
        )

    try:
        # Read file content
        file_content = await file.read()

        # Parse Excel
        parser = ExcelParserService()
        templates, errors = await parser.parse_excel(file_content)

        if errors and not templates:
            return UploadPolicyTemplatesResponse(
                success=False,
                templates_imported=0,
                functions=[],
                policy_types=[],
                message="Failed to parse Excel file",
                errors=errors
            )

        # Store templates
        storage = PolicyTemplateStorageService(
            connection_string=settings.AZURE_TABLE_STORAGE_CONNECTION_STRING,
            table_name=settings.AZURE_TABLE_NAME
        )

        imported_count = 0
        functions = set()
        policy_types = set()

        for template in templates:
            if storage.create_template(template):
                imported_count += 1
                functions.add(template.function)
                policy_types.add(template.policy_type)

        logger.info(f"Successfully imported {imported_count}/{len(templates)} policy templates")

        return UploadPolicyTemplatesResponse(
            success=True,
            templates_imported=imported_count,
            functions=list(functions),
            policy_types=list(policy_types),
            message=f"Successfully imported {imported_count} policy templates",
            errors=errors if errors else []
        )

    except Exception as e:
        logger.error(f"Failed to upload policy templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload templates: {str(e)}"
        )

@router.delete("/policy-templates", response_model=AdminResponse)
def delete_all_policy_templates(x_admin_api_key: Optional[str] = Header(None)):
    """Delete all policy templates from storage."""
    if not _validate_admin_api_key(x_admin_api_key):
        logger.warning("SECURITY: Invalid or missing X-Admin-API-Key for template deletion")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header"
        )

    try:
        storage = PolicyTemplateStorageService(
            connection_string=settings.AZURE_TABLE_STORAGE_CONNECTION_STRING,
            table_name=settings.AZURE_TABLE_NAME
        )

        deleted_count = storage.delete_all_templates()

        logger.info(f"Deleted {deleted_count} policy templates")

        return AdminResponse(
            message=f"Successfully deleted {deleted_count} policy templates",
            details={"deleted_count": deleted_count}
        )

    except Exception as e:
        logger.error(f"Failed to delete policy templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete templates: {str(e)}"
        )