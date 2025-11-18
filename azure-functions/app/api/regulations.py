"""
API endpoints for regulation management.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.pipelines.regulations_processor import RegulationsProcessor
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore
from app.models.regulation_models import RegulationsIndexStatus
from app.api.auth import get_current_user
from app.config.settings import settings

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
)
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/regulations", tags=["Regulations"])


@router.post("/index", response_model=RegulationsIndexStatus)
async def index_regulations(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process and index all regulation PDFs into Azure AI Search.

    This endpoint:
    1. Reads all PDFs from the regulations directory
    2. Extracts text and detects articles
    3. Creates intelligent chunks
    4. Generates embeddings
    5. Indexes into Azure AI Search

    Returns:
        RegulationsIndexStatus with processing results for each regulation
    """
    try:
        logger.info(f"Regulation indexing triggered by user: {current_user.get('email')}")

        # Initialize components
        embedder = AzureEmbedder()
        vector_store = AzureVectorStore(embedder=embedder)

        # Initialize regulations processor
        regulations_dir = settings.REGULATIONS_PATH
        logger.info(f"Using regulations directory: {regulations_dir}")

        processor = RegulationsProcessor(
            regulations_dir=regulations_dir,
            embedder=embedder,
            vector_store=vector_store
        )

        # Process all regulations
        logger.info("Starting regulation processing...")
        index_status = await processor.process_all_regulations()

        # Log results
        successful = sum(1 for r in index_status.regulations if r.success)
        failed = sum(1 for r in index_status.regulations if not r.success)

        logger.info(
            f"Regulation indexing complete: {successful} successful, {failed} failed, "
            f"{index_status.total_chunks} total chunks"
        )

        return index_status

    except FileNotFoundError as e:
        logger.error(f"Regulations directory not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Regulations directory not found. Please ensure PDFs are in the correct location."
        )
    except Exception as e:
        logger.error(f"Failed to index regulations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index regulations: {str(e)}"
        )


@router.get("/index/status")
async def get_index_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the current status of regulation indexing.

    This endpoint checks how many regulation documents are currently indexed.

    Returns:
        Dictionary with index statistics
    """
    try:
        embedder = AzureEmbedder()
        vector_store = AzureVectorStore(embedder=embedder)

        # Query for regulation documents in the regulations index
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from app.config.settings import settings

        search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        )

        # Count regulation documents
        results = search_client.search(
            search_text="*",
            filter="record_type eq 'regulation'",
            select="id",
            include_total_count=True,
            top=0  # We only want the count
        )

        # Get the count (this triggers the search)
        _ = list(results)
        total_count = results.get_count()

        return {
            "indexed": total_count > 0,
            "total_chunks": total_count,
            "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            "message": f"Found {total_count} regulation chunks indexed in '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'" if total_count > 0
                      else f"No regulations indexed yet in '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'"
        }

    except Exception as e:
        logger.error(f"Failed to get index status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index status: {str(e)}"
        )


@router.delete("/index")
async def clear_regulation_index(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Clear all regulation documents from the regulations index.

    WARNING: This will delete all indexed regulations.
    Use with caution - typically only for testing or re-indexing.

    Returns:
        Dictionary with number of documents deleted
    """
    try:
        logger.warning(f"Regulation index clearing triggered by user: {current_user.get('email')}")

        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from app.config.settings import settings

        search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        )

        # Find all regulation documents
        results = search_client.search(
            search_text="*",
            filter="record_type eq 'regulation'",
            select="id",
            top=1000  # Assuming max 1000 regulation chunks
        )

        # Collect IDs to delete
        ids_to_delete = [{"id": doc["id"]} for doc in results]

        if not ids_to_delete:
            return {
                "deleted": 0,
                "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
                "message": f"No regulation documents found to delete in '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'"
            }

        # Delete documents
        delete_result = search_client.delete_documents(ids_to_delete)

        deleted_count = sum(1 for item in delete_result if item.succeeded)
        failed_count = len(delete_result) - deleted_count

        logger.warning(f"Deleted {deleted_count} regulation chunks from '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}', {failed_count} failed")

        return {
            "deleted": deleted_count,
            "failed": failed_count,
            "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            "message": f"Deleted {deleted_count} regulation documents from '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'"
        }

    except Exception as e:
        logger.error(f"Failed to clear regulation index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear regulation index: {str(e)}"
        )


@router.delete("/index/delete-index")
async def delete_regulations_index(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete the regulations index completely.

    ⚠️ WARNING: This will delete the entire regulations index and all its data!

    Use this to fix schema issues or start fresh.
    After deletion, recreate the index with POST /index/schema.

    Returns:
        Dictionary with deletion status
    """
    try:
        logger.warning(f"Regulations index deletion triggered by user: {current_user.get('email')}")

        from app.pipelines.components.index_manager import AzureSearchIndexManager

        # Initialize index manager
        index_manager = AzureSearchIndexManager()

        logger.info(f"Deleting regulations index: {settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}")

        # Check if index exists
        if not index_manager.check_index_exists(settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME):
            return {
                "deleted": False,
                "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
                "message": f"Regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}' does not exist"
            }

        # Delete the index
        success = index_manager.delete_index(settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'"
            )

        logger.warning(f"Successfully deleted regulations index: {settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}")

        return {
            "deleted": True,
            "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            "message": f"Successfully deleted regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'. You can now recreate it with POST /index/schema"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete regulations index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete regulations index: {str(e)}"
        )


@router.post("/index/schema")
async def create_regulations_index(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a dedicated Azure AI Search index for regulations.

    This endpoint creates a new index specifically for regulation documents.
    Run this BEFORE indexing regulations for the first time.

    Index created: 'regulations' (configurable via AZURE_SEARCH_REGULATIONS_INDEX_NAME)

    Fields in the index:
    - id, record_type
    - content, content_vector (regulation text and embeddings)
    - regulation_name, article_number, chapter
    - regulation_type, effective_date, superseded_by
    - topics, references

    Returns:
        Dictionary with creation status
    """
    try:
        logger.info(f"Regulations index creation triggered by user: {current_user.get('email')}")

        from app.pipelines.components.index_manager import AzureSearchIndexManager

        # Initialize index manager
        index_manager = AzureSearchIndexManager()

        logger.info(f"Connecting to Azure Search: {settings.AZURE_SEARCH_ENDPOINT}")
        logger.info(f"Creating regulations index: {settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}")

        # Check if index already exists
        if index_manager.check_index_exists(settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME):
            logger.info(f"Regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}' already exists")

            # Get index stats
            stats = index_manager.get_index_stats(settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME)

            return {
                "created": False,
                "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
                "message": f"Regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}' already exists",
                "field_count": stats.get("field_count", 0) if stats else 0,
                "vector_search_enabled": stats.get("vector_search_enabled", False) if stats else False
            }

        # Create the regulations index
        success = index_manager.create_regulations_index()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'"
            )

        # Get stats of newly created index
        stats = index_manager.get_index_stats(settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME)

        logger.info(f"Successfully created regulations index: {settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}")

        return {
            "created": True,
            "index_name": settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME,
            "message": f"Successfully created regulations index '{settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME}'",
            "field_count": stats.get("field_count", 0) if stats else 0,
            "vector_search_enabled": stats.get("vector_search_enabled", False) if stats else False,
            "semantic_search_enabled": stats.get("semantic_search_enabled", False) if stats else False
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create regulations index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create regulations index: {str(e)}"
        )
