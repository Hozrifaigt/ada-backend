import logging

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse

from app.config.settings import settings
from app.auth.entra_service import get_current_user
from app.pipelines.components.llm import AzureLLM

from app.services.draft_service import DraftService
from app.services.draft_storage import DraftStorageService

from app.api.policy.dependencies import get_llm, get_vector_store as get_store

from app.models.draft_models import (
    InitializeDraftRequest,
    InitializeDraftResponse,
    ValidateDraftRequest,
    ValidateDraftResponse,
    UpdateTOCRequest,
    UpdateDraftMetadataRequest,
    GenerateContentRequest,
    UpdateContentRequest,
    ExportRequest,
    DraftSummary,
    DraftListResponse,
    DraftProgress,
    TopicContext,
    ContentGenerationResponse,
    CritiqueResponse
)
from app.models.toc_chat_models import (
    TOCChatMessage,
    TOCChatResponse,
    TOCConfirmRequest,
    TOCConfirmResponse,
    TOCOperation
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

def get_vector_store():
    """Dependency to get vector store."""
    return get_store()

def get_llm_service():
    """Dependency to get LLM service."""
    return get_llm()


@router.post("/validate", response_model=ValidateDraftResponse)
async def validate_draft(
    request: ValidateDraftRequest,
    llm: AzureLLM = Depends(get_llm),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Validate draft creation parameters using LLM.
    The LLM evaluates if the description is sufficient to create a meaningful policy.
    """
    logger.info(f"Draft validation requested by user: {current_user.get('email')}")
    logger.info(f"Validating description: {request.description[:100]}...")

    try:
        from app.services.draft_validation_service import DraftValidationService
        from app.prompts.manager import PromptManager
        from app.config.settings import settings

        # Initialize validation service with LLM
        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        validation_service = DraftValidationService(llm, prompt_manager)

        # Validate using LLM
        validation_result = await validation_service.validate_draft_request(
            title=request.title,
            description=request.description,
            client_metadata=request.client_metadata.dict()
        )

        logger.info(f"Validation result: is_valid={validation_result.get('is_valid')}, score={validation_result.get('description_quality_score')}")

        return ValidateDraftResponse(**validation_result)

    except Exception as e:
        logger.error(f"Error validating draft: {e}")
        return ValidateDraftResponse(
            is_valid=False,
            issues=[f"Validation error: {str(e)}"],
            suggestions=["Please try again or contact support if the issue persists"],
            description_quality_score=0,
            improved_description=None
        )


@router.post("/initialize", response_model=InitializeDraftResponse)
async def initialize_draft_endpoint(
    request: InitializeDraftRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Initialize a new draft with AI-generated TOC based on policy information.
    TOC is generated from title and context - description is optional.
    """
    logger.info(f"Draft initialization requested by user: {current_user.get('email')}")

    try:
        # Extract clean name (split on dash and take first part)
        raw_name = current_user.get('name', current_user.get('email'))
        clean_name = raw_name.split('-')[0].strip() if raw_name and '-' in raw_name else raw_name

        # Use the new initialize_draft method (no description required)
        result = await draft_service.initialize_draft(
            title=request.title,
            user_id=clean_name,
            client_metadata=request.client_metadata,
            function=request.function,
            llm_service=get_llm(),
            client_specific_requests=request.client_specific_requests,
            sector_specific_comments=request.sector_specific_comments,
            regulations=request.regulations,
            detail_level=request.detail_level
        )

        if result:
            return InitializeDraftResponse(
                draft_id=result["draft_id"],
                similar_policies_found=result["similar_policies_found"],
                suggested_toc=result["draft"]["toc"],
                message="Draft initialized successfully",
                toc_source=result.get("toc_source", "ai_generated")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize draft"
            )

    except Exception as e:
        logger.error(f"Failed to initialize draft: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=DraftListResponse)
async def list_drafts(
    title: Optional[str] = None,
    country: Optional[str] = None,
    city: Optional[str] = None,
    created_by: Optional[str] = None,
    industry: Optional[str] = None,
    function: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all drafts with optional filtering (collaborative - all users see all drafts)."""
    logger.info(f"Draft list requested by user: {current_user.get('email')} with filters: title={title}, country={country}, city={city}, created_by={created_by}, industry={industry}, function={function}")

    try:
        drafts = draft_service.list_drafts(
            title=title,
            country=country,
            city=city,
            created_by=created_by,
            industry=industry,
            function=function
        )

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


# Policy type endpoints removed - no longer using Excel templates

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

@router.put("/{draft_id}/metadata")
async def update_draft_metadata(
    draft_id: str,
    request: UpdateDraftMetadataRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update draft metadata."""
    logger.info(f"Metadata update for draft {draft_id} by user: {current_user.get('email')}")

    try:
        success = draft_service.update_draft_metadata(draft_id, request)

        if success:
            return {"message": "Draft metadata updated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found or update failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update draft metadata"
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
    Includes query classification, conversation history, and sequential context.
    """
    logger.info(f"Content generation for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")

    try:
        # Use the complete generation service with all features
        result = await draft_service.generate_content_for_topic(
            draft_id=draft_id,
            topic_id=topic_id,
            user_prompt=request.user_prompt,
            subtopic_id=request.subtopic_id,
            vector_store=store,
            llm_service=llm_service
        )

        if result:
            return ContentGenerationResponse(
                content=result.get("content", ""),
                sources_used=result.get("sources_used", []),
                summary=result.get("summary", ""),
                word_count=result.get("word_count", 0),
                message="Content generated successfully",
                is_chat_response=result.get("is_chat_response", False),
                critique=result.get("critique")
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
    llm_service: AzureLLM = Depends(get_llm_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Manually update topic or subtopic content."""
    logger.info(f"Manual content update for draft {draft_id}, topic {topic_id} by user: {current_user.get('email')}")

    try:
        success = await draft_service.update_topic_content(
            draft_id,
            topic_id,
            request.content,
            request.subtopic_id,
            llm_service,
            request.conversation_history
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


@router.post("/{draft_id}/export/pdf")
async def export_draft_to_pdf(
    draft_id: str,
    request: ExportRequest = ExportRequest(),
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Export draft to PDF document.
    By default, deletes the draft after export (temporary storage).
    """
    logger.info(f"PDF export for draft {draft_id} by user: {current_user.get('email')}")

    try:
        # Get the draft
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )

        # Import here to avoid circular imports
        from app.services.pdf_export_service import PDFExportService

        export_service = PDFExportService()
        pdf_doc = export_service.export_draft_to_pdf(draft)
        filename = export_service.get_filename(draft)

        # Delete draft after export if requested
        if request.delete_after_export:
            draft_service.delete_draft(draft_id)
            logger.info(f"Draft {draft_id} deleted after export")

        # Return PDF document
        return StreamingResponse(
            pdf_doc,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export draft to PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export draft to PDF"
        )


# TOC Chat Endpoints

@router.post("/{draft_id}/toc/chat")
async def chat_modify_toc(
    draft_id: str,
    message: TOCChatMessage,
    draft_service: DraftService = Depends(get_draft_service),
    llm: AzureLLM = Depends(get_llm),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TOCChatResponse:
    """
    Process a natural language command to modify the Table of Contents.

    This endpoint:
    1. Parses the user's natural language command
    2. Determines the intended TOC modification
    3. Returns a preview of the changes
    4. May require confirmation for destructive operations
    """
    logger.info(f"TOC chat for draft {draft_id}: '{message.message[:50]}...' by user: {current_user.get('email')}")

    try:
        # Get the draft
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )

        # Initialize TOC manipulation service
        from app.services.toc_manipulation_service import TOCManipulationService
        from app.prompts.manager import PromptManager
        from app.config.settings import settings

        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        toc_service = TOCManipulationService(llm, prompt_manager)

        # Prepare draft context - draft is a dictionary
        metadata = draft.get('metadata', {})
        client_metadata = metadata.get('client_metadata', {})

        draft_context = {
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'function': metadata.get('function', ''),
            'industry': client_metadata.get('industry', ''),
            'company_name': client_metadata.get('name', ''),
            'city': client_metadata.get('city', ''),
            'country': client_metadata.get('country', ''),
            'client_specific_requests': metadata.get('client_specific_requests', ''),
            'sector_specific_comments': metadata.get('sector_specific_comments', ''),
            'regulations': metadata.get('regulations', ''),
            'detail_level': metadata.get('detail_level', 3)
        }

        # Parse the command
        parsed_operation = await toc_service.parse_toc_command(
            user_command=message.message,
            current_toc=draft.get('toc', []),
            conversation_history=message.conversation_history,
            draft_context=draft_context
        )

        # Check for errors
        if parsed_operation.get('action') == 'error':
            return TOCChatResponse(
                success=False,
                operation=None,
                preview_toc=None,
                current_toc=draft.get('toc', []),
                message=parsed_operation.get('error', 'Could not understand the command'),
                follow_up_question=None
            )

        # Check if clarification is needed
        if parsed_operation.get('action') == 'clarification_needed':
            return TOCChatResponse(
                success=True,
                operation=None,
                preview_toc=None,
                current_toc=draft.get('toc', []),
                message="I need more information to complete this action.",
                follow_up_question=parsed_operation.get('follow_up_question')
            )

        # Create operation object
        operation = TOCOperation(
            action=parsed_operation['action'],
            parameters=parsed_operation.get('parameters', {}),
            interpretation=parsed_operation.get('interpretation', ''),
            requires_confirmation=parsed_operation.get('requires_confirmation', False) or parsed_operation['action'] == 'suggest_topics'
        )

        # Handle different action types
        preview_toc = None
        success_message = ""
        ai_suggestions = None

        try:
            if parsed_operation['action'] == 'suggest_topics':
                # Generate AI-powered suggestions
                params = parsed_operation.get('parameters', {})
                suggestions_data = await toc_service.generate_topic_suggestions(
                    user_query=params.get('query', message.message),
                    current_toc=draft.get('toc', []),
                    draft_context=draft_context
                )

                # Format suggestions into a message with better readability
                if suggestions_data.get('suggestions'):
                    success_message = f"📋 **Here are my suggestions:**\n\n"
                    for i, suggestion in enumerate(suggestions_data['suggestions'], 1):
                        # Header with title and type
                        success_message += f"**{i}. {suggestion['title']}**\n"
                        success_message += f"   *Type:* {suggestion['type'].title()}\n\n"

                        # Description
                        success_message += f"   {suggestion['description']}\n\n"

                        # Where to add it
                        if suggestion.get('proposed_position'):
                            success_message += f"   📍 **Where to add:** {suggestion['proposed_position']}\n\n"

                        # Subtopics if any
                        if suggestion.get('subtopics'):
                            success_message += f"   📂 **Subtopics:**\n"
                            for subtopic in suggestion['subtopics']:
                                success_message += f"      • {subtopic}\n"
                            success_message += "\n"

                        # Add spacing between suggestions for better readability
                        if i < len(suggestions_data['suggestions']):  # Not the last one
                            success_message += "\n"

                    # Rationale at the end
                    if suggestions_data.get('rationale'):
                        success_message += f"💡 **Why these suggestions?**\n{suggestions_data['rationale']}\n\n"

                    # Clear call to action
                    success_message += "ℹ️ *To implement these suggestions, use specific commands like:*\n"
                    success_message += "   • \"Add topic [name]\" - to add a new topic\n"
                    success_message += "   • \"Remove topic 3\" - to delete by position number\n"
                    success_message += "   • \"Move topic 5 to position 2\" - to reorder topics\n"
                    success_message += "   • \"Merge topics 3 and 4\" - to combine topics"

                    ai_suggestions = suggestions_data['suggestions']
                else:
                    success_message = "I couldn't generate suggestions based on your request."

                # For suggestions, store them in the operation for later application
                operation.parameters['suggestions'] = ai_suggestions

                # Generate preview of what the TOC will look like with suggestions applied
                if ai_suggestions:
                    preview_toc, _ = toc_service.execute_toc_operation(
                        operation={'action': 'suggest_topics', 'parameters': {'suggestions': ai_suggestions}},
                        current_toc=draft.get('toc', []),
                        preserve_content=True
                    )
                else:
                    preview_toc = draft.get('toc', [])
            else:
                # Execute normal TOC modification
                preview_toc, success_message = toc_service.execute_toc_operation(
                    operation=parsed_operation,
                    current_toc=draft.get('toc', []),
                    preserve_content=True
                )
        except Exception as e:
            logger.error(f"Error executing TOC operation: {e}")
            return TOCChatResponse(
                success=False,
                operation=operation,
                preview_toc=None,
                current_toc=draft.get('toc', []),
                message=f"Error: {str(e)}",
                follow_up_question=None
            )

        # Generate follow-up suggestions
        suggestions = []
        if parsed_operation['action'] == 'suggest_topics' and ai_suggestions:
            # For AI suggestions, offer to add the suggested topics
            suggestions.append("Would you like me to add any of these suggested topics to your document?")
            suggestions.append("You can say something like 'Add topic 1 and 3' or 'Add all suggestions'")
        elif parsed_operation['action'] == 'add_topic' and 'subtopics' not in parsed_operation.get('parameters', {}):
            suggestions.append("Would you like me to suggest subtopics for this new topic?")
        elif parsed_operation['action'] in ['remove_topic', 'remove_subtopic']:
            suggestions.append("Would you like to move the content somewhere else first?")

        return TOCChatResponse(
            success=True,
            operation=operation,
            preview_toc=preview_toc,
            current_toc=draft.get('toc', []),
            message=success_message,
            follow_up_question=parsed_operation.get('follow_up_question'),
            suggestions=suggestions if suggestions else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process TOC chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process TOC modification: {str(e)}"
        )


@router.post("/{draft_id}/toc/confirm")
async def confirm_toc_modification(
    draft_id: str,
    request: TOCConfirmRequest,
    draft_service: DraftService = Depends(get_draft_service),
    llm: AzureLLM = Depends(get_llm),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TOCConfirmResponse:
    """
    Confirm and apply a TOC modification after preview.

    This endpoint applies the previously parsed operation to actually
    modify the draft's Table of Contents.
    """
    logger.info(f"Confirming TOC modification for draft {draft_id}: {request.operation.action} by user: {current_user.get('email')}")

    try:
        # Get the draft
        draft = draft_service.get_draft(draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found"
            )

        # Initialize TOC manipulation service
        from app.services.toc_manipulation_service import TOCManipulationService
        from app.prompts.manager import PromptManager
        from app.config.settings import settings

        # Initialize prompt manager (llm is already injected via dependency)
        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        toc_service = TOCManipulationService(llm, prompt_manager)

        # Execute the operation
        try:
            updated_toc, success_message = toc_service.execute_toc_operation(
                operation=request.operation.dict(),
                current_toc=request.current_toc,
                preserve_content=True
            )
        except Exception as e:
            logger.error(f"Error executing TOC operation: {e}")
            return TOCConfirmResponse(
                success=False,
                updated_toc=draft.get('toc', []),
                message=f"Failed to apply changes: {str(e)}",
                undo_available=False
            )

        # Update the draft with new TOC
        success = draft_service.update_toc(draft_id, updated_toc)

        if not success:
            logger.error(f"Failed to update TOC for draft {draft_id}")
            return TOCConfirmResponse(
                success=False,
                updated_toc=draft.get('toc', []),
                message="Failed to save TOC changes to database",
                undo_available=False
            )

        logger.info(f"Successfully applied TOC modification to draft {draft_id}")

        return TOCConfirmResponse(
            success=True,
            updated_toc=updated_toc,
            message=success_message,
            undo_available=True  # We could implement undo by storing previous TOC
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to confirm TOC modification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply TOC modification: {str(e)}"
        )


# Template endpoint removed - no longer using Excel templates

@router.post("/test/regulations-retrieval")
async def test_regulations_retrieval(
    query_text: str,
    regulation_name: Optional[str] = None,
    top_k: int = 7,
    min_score: float = 0.75,
    store = Depends(get_store),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Test endpoint to retrieve regulation chunks.
    
    This endpoint searches for regulation chunks based on a query and returns
    the results for testing and validation purposes.
    
    Args:
        query_text: Search query text (required)
        regulation_name: Optional filter by regulation name
        top_k: Number of results to return (default: 7, max: 20)
        min_score: Minimum similarity score (default: 0.75)
    
    Returns:
        List of matching regulation chunks with scores
    """
    logger.info(f"Test regulations retrieval requested by {current_user.get('email')}")
    logger.info(f"Query: {query_text}, Regulation: {regulation_name}, top_k: {top_k}, min_score: {min_score}")
    
    try:
        # Call the search method from vector store
        regulation_chunks = await store.search_relevant_regulations(
            query_text=query_text,
            regulation_name=regulation_name,
            top_k=top_k,
            min_score=min_score,
            context={}
        )
        
        logger.info(f"Retrieved {len(regulation_chunks)} regulation chunks")
        
        # Format response
        return {
            "status": "success",
            "total_results": len(regulation_chunks),
            "query": query_text,
            "filters": {
                "regulation_name": regulation_name,
                "top_k": top_k,
                "min_score": min_score
            },
            "chunks": [chunk.model_dump() for chunk in regulation_chunks],
            "message": f"Successfully retrieved {len(regulation_chunks)} regulation chunks"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve regulations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve regulations: {str(e)}"
        )