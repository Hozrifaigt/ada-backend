import logging

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse

from app.config.settings import settings
from app.auth.entra_service import get_current_user

from app.services.draft_service import DraftService
from app.services.draft_storage import DraftStorageService

from app.api.policy.dependencies import get_llm, get_vector_store as get_store
from app.api.policy.search.similarity_service import SimilarityService

from app.models.draft_models import (
    InitializeDraftRequest,
    InitializeDraftResponse,
    UpdateTOCRequest,
    GenerateContentRequest,
    UpdateContentRequest,
    RegenerateRequest,
    FinalizeTopicRequest,
    ExportRequest,
    DraftSummary,
    DraftListResponse,
    DraftProgress,
    TopicContext,
    ContentGenerationResponse
)


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/drafts", tags=["drafts"])

def get_draft_service():
    """Dependency to get draft service instance."""
    storage = DraftStorageService(
        connection_string=settings.AZURE_TABLE_STORAGE_CONNECTION_STRING,
        table_name=settings.AZURE_TABLE_NAME
    )
    return DraftService(storage)

def get_similarity_service():
    """Dependency to get similarity service."""
    return SimilarityService

def get_vector_store():
    """Dependency to get vector store."""
    return get_store()

def get_llm_service():
    """Dependency to get LLM service."""
    return get_llm()


@router.post("/initialize", response_model=InitializeDraftResponse)
async def initialize_draft(
    request: InitializeDraftRequest,
    draft_service: DraftService = Depends(get_draft_service),
    similarity_service = Depends(get_similarity_service),
    store = Depends(get_vector_store),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Initialize a new draft from description.
    Searches for similar policies and suggests TOC.
    """
    logger.info(f"Draft initialization requested by user: {current_user.get('email')}")
    
    try:
        result = await draft_service.initialize_draft_from_description(
            description=request.description,
            title=request.title,
            user_id=current_user.get('user_id', current_user.get('email')),
            similarity_service=similarity_service,
            store=store,
            client_metadata=request.client_metadata
        )
        
        if result:
            return InitializeDraftResponse(
                draft_id=result["draft_id"],
                similar_policies_found=result["similar_policies_found"],
                suggested_toc=result["draft"]["toc"],
                message="Draft initialized successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize draft"
            )
            
    except Exception as e:
        logger.error(f"Failed to initialize draft: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=DraftListResponse)
async def list_drafts(
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all drafts (collaborative - all users see all drafts)."""
    logger.info(f"Draft list requested by user: {current_user.get('email')}")
    
    try:
        drafts = draft_service.list_drafts()
        
        draft_summaries = [
            DraftSummary(**draft) for draft in drafts
        ]
        
        return DraftListResponse(
            drafts=draft_summaries,
            total=len(draft_summaries)
        )
        
    except Exception as e:
        logger.error(f"Failed to list drafts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve drafts"
        )

@router.get("/{draft_id}")
async def get_draft(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get complete draft details."""
    logger.info(f"Draft {draft_id} requested by user: {current_user.get('email')}")
    
    try:
        draft = draft_service.get_draft(draft_id)
        
        if draft:
            return draft
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get draft: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve draft"
        )

@router.put("/{draft_id}/toc")
async def update_toc(
    draft_id: str,
    request: UpdateTOCRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update/finalize the TOC structure before content generation."""
    logger.info(f"TOC update for draft {draft_id} by user: {current_user.get('email')}")
    
    try:
        success = draft_service.update_toc(draft_id, request.toc)
        
        if success:
            return {"message": "TOC updated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found or update failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update TOC: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update TOC"
        )

@router.delete("/{draft_id}")
async def delete_draft(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a draft from storage."""
    logger.info(f"Draft deletion for {draft_id} by user: {current_user.get('email')}")
    
    try:
        success = draft_service.delete_draft(draft_id)
        
        if success:
            return {"message": "Draft deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete draft: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete draft"
        )

@router.delete("/")
async def delete_all_drafts(
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete all drafts from storage."""
    logger.info(f"Delete all drafts requested by user: {current_user.get('email')}")
    
    try:
        result = draft_service.delete_all_drafts()
        
        if result["deleted_count"] == 0 and result["total_count"] == 0:
            return {
                "message": result["message"],
                "deleted_count": 0,
                "total_count": 0
            }
        elif result["failed_count"] > 0:
            return {
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "total_count": result["total_count"],
                "failed_count": result["failed_count"],
                "failed_draft_ids": result["failed_draft_ids"]
            }
        else:
            return {
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "total_count": result["total_count"]
            }
            
    except Exception as e:
        logger.error(f"Failed to delete all drafts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete all drafts"
        )

@router.post("/{draft_id}/topics/{topic_id}/generate", response_model=ContentGenerationResponse)
async def generate_topic_content(
    draft_id: str,
    topic_id: str,
    request: GenerateContentRequest,
    draft_service: DraftService = Depends(get_draft_service),
    store = Depends(get_vector_store),
    llm_service = Depends(get_llm_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate content for a topic or subtopic.
    Uses RAG to find relevant content and generates with context.
    """
    logger.info(f"Content generation for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")
    
    try:
        # This will be implemented with the actual generation logic
        # For now, returning placeholder
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
        
        # Build context
        context = draft_service.build_sequential_context(
            draft, 
            topic_id, 
            request.subtopic_id
        )
        
        # Placeholder for actual generation
        generated_content = f"Generated content for topic with prompt: {request.user_prompt[:100]}..."
        
        # Update the content
        success = draft_service.update_topic_content(
            draft_id,
            topic_id,
            generated_content,
            request.subtopic_id
        )
        
        if success:
            return ContentGenerationResponse(
                content=generated_content,
                context_used=str(context)[:500],  # Preview of context
                message="Content generated successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save generated content"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content generation failed"
        )

@router.put("/{draft_id}/topics/{topic_id}/content")
async def update_topic_content(
    draft_id: str,
    topic_id: str,
    request: UpdateContentRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Manually update topic or subtopic content."""
    logger.info(f"Manual content update for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")
    
    try:
        success = draft_service.update_topic_content(
            draft_id,
            topic_id,
            request.content,
            request.subtopic_id
        )
        
        if success:
            return {"message": "Content updated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft or topic not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update content"
        )

@router.post("/{draft_id}/topics/{topic_id}/regenerate", response_model=ContentGenerationResponse)
async def regenerate_topic_content(
    draft_id: str,
    topic_id: str,
    request: RegenerateRequest,
    draft_service: DraftService = Depends(get_draft_service),
    store = Depends(get_vector_store),
    llm_service = Depends(get_llm_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Regenerate content with a new prompt."""
    logger.info(f"Content regeneration for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")
    
    # Similar to generate but replaces existing content
    return await generate_topic_content(
        draft_id=draft_id,
        topic_id=topic_id,
        request=GenerateContentRequest(
            user_prompt=request.user_prompt,
            subtopic_id=request.subtopic_id
        ),
        draft_service=draft_service,
        store=store,
        llm_service=llm_service,
        current_user=current_user
    )

@router.post("/{draft_id}/topics/{topic_id}/finalize")
async def finalize_topic(
    draft_id: str,
    topic_id: str,
    request: FinalizeTopicRequest,
    draft_service: DraftService = Depends(get_draft_service),
    llm_service = Depends(get_llm_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Mark topic as complete and optionally generate summary.
    Summary is generated by LLM if content > 150 words.
    """
    logger.info(f"Finalizing topic {topic_id} in draft {draft_id} by user: {current_user.get('email')}")
    
    try:
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
        
        topic = draft_service.get_topic_by_id(draft, topic_id)
        if not topic:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic {topic_id} not found"
            )
        
        # Check if summary needed
        if topic.get("summary") == "NEEDS_SUMMARY" and request.generate_summary:
            # Here you would call LLM to generate actual summary
            # For now, just marking as finalized
            pass
        
        return {"message": "Topic finalized successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to finalize topic: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to finalize topic"
        )


@router.get("/{draft_id}/progress", response_model=DraftProgress)
async def get_draft_progress(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get generation progress for a draft."""
    logger.info(f"Progress check for draft {draft_id} by user: {current_user.get('email')}")
    
    try:
        progress = draft_service.get_draft_progress(draft_id)
        
        if "error" in progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=progress["error"]
            )
        
        return DraftProgress(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve progress"
        )

@router.get("/{draft_id}/context/{topic_id}", response_model=TopicContext)
async def get_topic_context(
    draft_id: str,
    topic_id: str,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get context for a specific topic (used for generation)."""
    logger.info(f"Context request for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")
    
    try:
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
        
        context = draft_service.build_sequential_context(
            draft,
            topic_id,
            subtopic_id
        )
        
        return TopicContext(**context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve context"
        )


@router.post("/{draft_id}/export/word")
async def export_draft_to_word(
    draft_id: str,
    request: ExportRequest = ExportRequest(),
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Export draft to Word document.
    By default, deletes the draft after export (temporary storage).
    """
    logger.info(f"Word export for draft {draft_id} by user: {current_user.get('email')}")
    
    try:
        # Get the draft
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )
        
        # Import here to avoid circular imports
        from app.services.word_export_service import WordExportService
        
        export_service = WordExportService()
        word_doc = export_service.export_draft_to_word(draft)
        filename = export_service.get_filename(draft)
        
        # Delete draft after export if requested
        if request.delete_after_export:
            draft_service.delete_draft(draft_id)
            logger.info(f"Draft {draft_id} deleted after export")
        
        # Return Word document
        return StreamingResponse(
            word_doc,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export draft: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export draft"
        )