import logging

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
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
    ApplyPolicyRequest,
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
    CritiqueResponse,
    SimilarPolicy,
    GapReportResponse,
    ApplyGapRequest,
    ApplyGapResponse,
    AssessBatchRequest,
    GapSummary,
    ConsistencyReport,
    DraftVersion,
    AuditEntry,
    SaveVersionRequest,
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
            vector_store=get_store(),
            description=request.description,
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
                toc_source=result.get("toc_source", "ai_generated"),
                needs_policy_selection=result.get("needs_policy_selection", False),
                similar_policies=[SimilarPolicy(**p) for p in result.get("similar_policies", [])]
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

@router.post("/initialize-from-upload", response_model=InitializeDraftResponse)
async def initialize_draft_from_upload_endpoint(
    title: str = Form(...),
    function: str = Form(...),
    client_name: str = Form(...),
    client_country: str = Form(...),
    client_city: str = Form(...),
    client_industry: str = Form(...),
    client_specific_requests: Optional[str] = Form(None),
    sector_specific_comments: Optional[str] = Form(None),
    regulations: str = Form(""),
    detail_level: int = Form(3),
    policy_file: UploadFile = File(...),
    previous_policy_file: Optional[UploadFile] = File(None),
    regulations_files: Optional[List[UploadFile]] = File(None),
    benchmark_file: Optional[UploadFile] = File(None),
    benchmark_policy_id: Optional[str] = Form(None),
    review_intensity: str = Form("preserve"),
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Initialize a new draft from an uploaded .docx policy file.
    The document's structure (TOC and content) becomes the starting point for the draft.
    """
    logger.info(f"Upload-based draft initialization requested by user: {current_user.get('email')}")

    if not policy_file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .docx files are supported"
        )

    try:
        from app.models.draft_models import ClientMetadata

        raw_name = current_user.get('name', current_user.get('email'))
        clean_name = raw_name.split('-')[0].strip() if raw_name and '-' in raw_name else raw_name

        file_bytes = await policy_file.read()

        client_metadata = ClientMetadata(
            name=client_name,
            country=client_country,
            city=client_city,
            industry=client_industry
        )

        previous_policy_bytes = await previous_policy_file.read() if previous_policy_file else None
        regulations_bytes_list = []
        if regulations_files:
            for rf in regulations_files:
                regulations_bytes_list.append((rf.filename, await rf.read()))

        benchmark_bytes = await benchmark_file.read() if benchmark_file else None

        result = await draft_service.initialize_draft_from_upload(
            file_bytes=file_bytes,
            filename=policy_file.filename,
            title=title,
            user_id=clean_name,
            client_metadata=client_metadata,
            function=function,
            client_specific_requests=client_specific_requests,
            sector_specific_comments=sector_specific_comments,
            regulations=regulations,
            detail_level=detail_level,
            previous_policy_bytes=previous_policy_bytes,
            previous_policy_filename=previous_policy_file.filename if previous_policy_file else None,
            regulations_files=regulations_bytes_list,
            benchmark_file_bytes=benchmark_bytes,
            benchmark_filename=benchmark_file.filename if benchmark_file else None,
            selected_benchmark_policy_id=benchmark_policy_id,
            vector_store=get_store(),
            llm_service=get_llm(),
            review_intensity=review_intensity,
        )

        if result:
            draft_service.log_audit(result["draft_id"], current_user.get('email', ''), "created",
                                    summary=f"Review draft from upload: {policy_file.filename}")
            return InitializeDraftResponse(
                draft_id=result["draft_id"],
                similar_policies_found=0,
                suggested_toc=result["draft"]["toc"],
                message="Draft initialized from uploaded policy",
                toc_source="uploaded_policy"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize draft from uploaded file"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize draft from upload: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{draft_id}/apply-policy")
async def apply_policy_to_draft(
    draft_id: str,
    request: ApplyPolicyRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Apply a selected similar policy's TOC to a draft, or generate the TOC with AI.
    Called after the user picks a policy from the selection screen.
    Pass policy_id=null to skip selection and generate with AI instead.
    """
    success = await draft_service.apply_policy_toc(
        draft_id=draft_id,
        policy_id=request.policy_id,
        vector_store=get_store(),
        llm_service=get_llm()
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply policy TOC to draft"
        )
    return {"message": "TOC applied successfully", "draft_id": draft_id}


@router.post("/{draft_id}/topics/{topic_id}/gap-report", response_model=GapReportResponse)
async def generate_topic_gap_report(
    draft_id: str,
    topic_id: str,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate a gap report for one topic/subtopic, comparing the client's content to the benchmark + regulations."""
    logger.info(f"Gap report requested for draft {draft_id} topic {topic_id} subtopic {subtopic_id} by {current_user.get('email')}")
    report = await draft_service.generate_gap_report_for_topic(
        draft_id=draft_id,
        topic_id=topic_id,
        vector_store=get_store(),
        llm_service=get_llm(),
        subtopic_id=subtopic_id,
    )
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate gap report"
        )
    return GapReportResponse(gap_report=report, message="Gap report generated")


@router.post("/{draft_id}/topics/{topic_id}/apply-gap", response_model=ApplyGapResponse)
async def apply_topic_gap(
    draft_id: str,
    topic_id: str,
    request: ApplyGapRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Apply the reviewer-confirmed gap findings to a topic, preserving the baseline."""
    logger.info(f"Apply-gap requested for draft {draft_id} topic {topic_id} by {current_user.get('email')}")
    result = await draft_service.apply_gap_for_topic(
        draft_id=draft_id,
        topic_id=topic_id,
        request=request,
        vector_store=get_store(),
        llm_service=get_llm()
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply gap changes"
        )
    if not request.preview:  # previews don't persist, so don't audit them
        target = request.subtopic_id or topic_id
        draft_service.log_audit(draft_id, current_user.get('email', ''), "gap_applied",
                                target_id=target, summary=f"{result.get('word_count', 0)} words")
    return ApplyGapResponse(**result)


@router.post("/{draft_id}/assess-batch")
async def assess_units_batch(
    draft_id: str,
    request: AssessBatchRequest,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Assess several topics/subtopics in one request — loads once, runs the LLM reports in
    parallel (asyncio.gather), and writes once."""
    logger.info(f"Assess-batch ({len(request.units)} units) for draft {draft_id} by {current_user.get('email')}")
    results = await draft_service.assess_batch(
        draft_id=draft_id,
        units=[u.model_dump() for u in request.units],
        vector_store=get_store(),
        llm_service=get_llm(),
    )
    if results is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return {"results": results}


@router.post("/{draft_id}/consistency-check", response_model=ConsistencyReport)
async def run_consistency_check(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Run a whole-policy consistency pass (contradictions, terminology, duplicates, etc.)."""
    report = await draft_service.check_consistency(draft_id, llm_service=get_llm())
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "consistency_check",
                            summary=f"{len(report['issues'])} issues")
    return ConsistencyReport(**report)


@router.get("/{draft_id}/gap-summary", response_model=GapSummary)
async def get_gap_summary(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Aggregate the persisted gap data into a client-facing summary (rollup + narrative)."""
    summary = await draft_service.build_gap_summary(draft_id, llm_service=get_llm())
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return GapSummary(**summary)


@router.post("/{draft_id}/export/gap-summary/word")
async def export_gap_summary_word(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Export the gap-assessment summary as a Word document."""
    summary = await draft_service.build_gap_summary(draft_id, llm_service=get_llm())
    draft = draft_service.get_draft(draft_id)
    if summary is None or draft is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    from app.services.word_export_service import WordExportService
    stream = WordExportService().export_gap_summary(summary, draft["metadata"])
    draft_service.log_audit(draft_id, current_user.get('email', ''), "exported", summary="Gap summary (Word)")
    safe = "".join(c for c in draft["metadata"]["title"] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:50]
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={safe}_gap_summary.docx"}
    )


@router.post("/{draft_id}/versions")
async def save_draft_version(
    draft_id: str,
    request: SaveVersionRequest = SaveVersionRequest(),
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Snapshot the current draft as a restorable version."""
    actor = current_user.get('email', '')
    version_id = draft_service.save_version(
        draft_id, label=request.label or "manual", created_by=actor, reason=request.reason or ""
    )
    if not version_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    draft_service.log_audit(draft_id, actor, "version_saved", target_id=version_id,
                            summary=request.label or "manual")
    return {"version_id": version_id, "message": "Version saved"}


@router.get("/{draft_id}/versions", response_model=List[DraftVersion])
async def list_draft_versions(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List saved versions (metadata only), newest first."""
    return [DraftVersion(**v) for v in draft_service.list_versions(draft_id)]


@router.post("/{draft_id}/versions/{version_id}/restore")
async def restore_draft_version(
    draft_id: str,
    version_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Restore a version (auto-snapshots the current draft first)."""
    actor = current_user.get('email', '')
    result = draft_service.restore_version(draft_id, version_id, restored_by=actor)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version or draft not found")
    draft_service.log_audit(draft_id, actor, "version_restored", target_id=version_id)
    return {"message": "Version restored", "draft": result}


@router.get("/{draft_id}/audit", response_model=List[AuditEntry])
async def get_draft_audit(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Return the draft's audit trail, newest first."""
    return [AuditEntry(**a) for a in draft_service.list_audit(draft_id)]


@router.get("/library-policies")
async def list_library_policies(
    function: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List library policies for the benchmark picker, optionally filtered by function."""
    try:
        store = get_store()
        return {"policies": store.list_library_policies(function=function)}
    except Exception as e:
        logger.error(f"Failed to list library policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list library policies"
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
        draft_service.log_audit(draft_id, current_user.get('email', ''), "exported", summary="Word")

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
        draft_service.log_audit(draft_id, current_user.get('email', ''), "exported", summary="PDF")

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

        # The chatbot operates on the dropdown-selected TOC when provided, else the working TOC.
        active_toc = message.current_toc if message.current_toc is not None else draft.get('toc', [])

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

        # When editing the Good TOC, make the bot aware of the Current + Extracted TOCs so it can
        # propose a better structure based on BOTH (plus the user's query). We surface them through the
        # description the prompt already consumes.
        def _titles_block(label: str, toc_list) -> str:
            lines = []
            for i, t in enumerate(toc_list or [], 1):
                title = (t.get('topic') or t.get('title') or t.get('name') or '').strip()
                if not title:
                    continue
                lines.append(f"{i}. {title}")
                for s in (t.get('subtopics') or []):
                    st = (s if isinstance(s, str) else (s.get('topic') or s.get('name') or s.get('title') or '')).strip()
                    if st:
                        lines.append(f"   - {st}")
            return f"\n\n{label}:\n" + "\n".join(lines) if lines else ""

        reference_block = _titles_block("CLIENT'S CURRENT TOC (reference)", message.client_toc) \
            + _titles_block("BENCHMARK / EXTRACTED TOC (reference)", message.benchmark_toc)
        if reference_block:
            draft_context['description'] = (draft_context.get('description') or '') + reference_block + \
                "\n\nUse both reference TOCs above, together with the user's request, to propose the best structure."

        # Parse the command
        parsed_operation = await toc_service.parse_toc_command(
            user_command=message.message,
            current_toc=active_toc,
            conversation_history=message.conversation_history,
            draft_context=draft_context
        )

        # Check for errors
        if parsed_operation.get('action') == 'error':
            return TOCChatResponse(
                success=False,
                operation=None,
                preview_toc=None,
                current_toc=active_toc,
                message=parsed_operation.get('error', 'Could not understand the command'),
                follow_up_question=None
            )

        # Handle conversational responses
        if parsed_operation.get('action') == 'chat_response':
            chat_msg = parsed_operation.get('parameters', {}).get('message', '') or parsed_operation.get('interpretation', '')
            return TOCChatResponse(
                success=True,
                operation=None,
                preview_toc=None,
                current_toc=active_toc,
                message=chat_msg,
                follow_up_question=parsed_operation.get('follow_up_question')
            )

        # Check if clarification is needed
        if parsed_operation.get('action') == 'clarification_needed':
            return TOCChatResponse(
                success=True,
                operation=None,
                preview_toc=None,
                current_toc=active_toc,
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
                    current_toc=active_toc,
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
                        current_toc=active_toc,
                        preserve_content=True
                    )
                else:
                    preview_toc = active_toc
            else:
                # Execute normal TOC modification
                preview_toc, success_message = toc_service.execute_toc_operation(
                    operation=parsed_operation,
                    current_toc=active_toc,
                    preserve_content=True
                )
        except Exception as e:
            logger.error(f"Error executing TOC operation: {e}")
            return TOCChatResponse(
                success=False,
                operation=operation,
                preview_toc=None,
                current_toc=active_toc,
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
            current_toc=active_toc,
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

        # Persist to the dropdown-selected TOC: client/benchmark write to their reference snapshot,
        # good (or unset) writes the working TOC.
        target = (request.target or "good").strip()
        if target in ("client", "benchmark"):
            success = draft_service.save_toc_snapshot(draft_id, target, updated_toc) is not None
        else:
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


@router.put("/{draft_id}/toc/chat-history")
async def update_toc_chat_history(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Save the TOC chat conversation history."""
    logger.info(f"Updating TOC chat history for draft {draft_id} by user: {current_user.get('email')}")
    conversation_history = request.get("conversation_history", [])
    success = draft_service.update_toc_chat_history(draft_id, conversation_history)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save TOC chat history"
        )
    return {"success": True}


# Perfect-TOC-first stage (v2): extract / adopt / approve the working "good" TOC

@router.post("/{draft_id}/toc/extract-from-file")
async def extract_toc_from_file_endpoint(
    draft_id: str,
    file: UploadFile = File(...),
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Parse an uploaded .docx and return its TOC structure so the reviewer can adopt it as the good TOC."""
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .docx files are supported")
    try:
        file_bytes = await file.read()
        preview = draft_service.extract_toc_from_file(file_bytes)
    except Exception as e:
        logger.error(f"Failed to extract TOC from file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not extract TOC from file")
    return {"preview_toc": preview, "message": f"Extracted {len(preview)} sections from {file.filename}"}


@router.post("/{draft_id}/toc/from-text")
async def toc_from_text_endpoint(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    llm: AzureLLM = Depends(get_llm),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Parse a pasted/described TOC into a structured preview the reviewer can adopt as the good TOC."""
    draft = draft_service.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")
    current_titles = "\n".join(f"{i+1}. {t.get('topic','')}" for i, t in enumerate(draft.get("toc", [])))
    toc = await draft_service.generation_service.generate_toc_from_text(
        request.get("text", ""), current_titles, llm
    )
    return {"preview_toc": toc, "message": f"Parsed {len(toc)} sections"}


@router.post("/{draft_id}/toc/use")
async def use_toc_endpoint(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Adopt a provided TOC structure as the working good TOC, redistributing the client's existing
    content into it by title match."""
    result = draft_service.apply_toc_structure(draft_id, request.get("toc", []))
    if result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not apply the provided TOC")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "toc_adopted",
                            summary=f"{len(request.get('toc', []))} sections")
    return {"success": True, "draft": result}


@router.post("/{draft_id}/toc/snapshot")
async def save_toc_snapshot_endpoint(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Persist an edited titles-only TOC for one reference column: which = client | benchmark | good.
    'good' is the working TOC (content redistributed by title); client/benchmark update the read-only
    reference snapshots shown in the TOC tab."""
    which = (request.get("which") or "").strip()
    if which not in ("client", "benchmark", "good"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="which must be one of: client, benchmark, good")
    result = draft_service.save_toc_snapshot(draft_id, which, request.get("toc", []))
    if result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not save the TOC")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "toc_snapshot_saved",
                            summary=f"{which}: {len(request.get('toc', []))} sections")
    return {"success": True, "draft": result}


@router.post("/{draft_id}/toc/regenerate-good")
async def regenerate_good_toc_endpoint(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Re-derive a fresh Good TOC by reconciling the current Current + Extracted TOCs (after the
    reviewer edits those). Preserves existing content by section title."""
    result = await draft_service.regenerate_good_toc(draft_id, llm_service=get_llm())
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "toc_good_regenerated")
    return {"success": True, "draft": result}


@router.post("/{draft_id}/toc/approve")
async def approve_toc_endpoint(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Lock the good TOC as the agreed structure (the gate before per-section content work)."""
    result = draft_service.approve_toc(draft_id, actor=current_user.get('email', ''))
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "toc_approved")
    return {"success": True, "draft": result}


@router.post("/{draft_id}/topics/{topic_id}/consolidate")
async def consolidate_theme_endpoint(
    draft_id: str,
    topic_id: str,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Consolidate everything about this section's theme from across the whole policy into one section
    (preview; persist via the normal content-update path after the reviewer confirms)."""
    result = await draft_service.consolidate_theme_for_topic(
        draft_id=draft_id, topic_id=topic_id,
        vector_store=get_store(), llm_service=get_llm(), subtopic_id=subtopic_id,
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or topic not found")
    return result


# Content redistribution: map the client's original content into the current Good TOC (by meaning)

@router.post("/{draft_id}/redistribute/propose")
async def propose_redistribution_endpoint(
    draft_id: str,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Propose a mapping of the client's original content into the current Good TOC sections (by meaning,
    with an embedding pre-filter + LLM assignment). Read-only — the reviewer confirms via /redistribute/apply."""
    result = await draft_service.propose_redistribution(
        draft_id=draft_id, vector_store=get_store(), llm_service=get_llm()
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")
    return result


@router.post("/{draft_id}/redistribute/apply")
async def apply_redistribution_endpoint(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Commit the reviewer-confirmed redistribution. Body: {sections: [{topic_id, subtopic_id?, content}]}.
    Sets both content and baseline_content for each section to the assigned client text."""
    result = draft_service.apply_redistribution(draft_id, request.get("sections", []))
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "content_redistributed",
                            summary=f"{len(request.get('sections', []))} sections")
    return {"success": True, "draft": result}


@router.post("/{draft_id}/redistribute/commit")
async def commit_redistribution_endpoint(
    draft_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Rebuild the Good TOC from the redistribution editor (renames/adds/removes) and set content+baseline.
    Body: {sections: [{level, id, title, content, include}]} (full ordered final structure)."""
    result = draft_service.commit_final_structure(draft_id, request.get("sections", []))
    if result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not save the structure")
    draft_service.log_audit(draft_id, current_user.get('email', ''), "redistribution_committed",
                            summary=f"{len(request.get('sections', []))} sections")
    return {"success": True, "draft": result}


# Per-section review pipeline (review_mode drafts): benchmark → law → close

@router.post("/{draft_id}/topics/{topic_id}/review/benchmark")
async def review_benchmark_endpoint(
    draft_id: str,
    topic_id: str,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Step 1: functional benchmark review for one section. Returns baseline + benchmark references
    and two suggestions (tweak-to-fix-gaps and use-benchmark-as-is). Read-only."""
    result = await draft_service.assess_benchmark_for_unit(
        draft_id=draft_id, topic_id=topic_id,
        vector_store=get_store(), llm_service=get_llm(), subtopic_id=subtopic_id,
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or section not found")
    return result


@router.post("/{draft_id}/topics/{topic_id}/review/regulation")
async def review_regulation_endpoint(
    draft_id: str,
    topic_id: str,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Step 2: law/regulation check for one section via RAG over the draft's regulations.
    Returns {applies, extract, suggestion}. Read-only."""
    result = await draft_service.check_regulation_for_unit(
        draft_id=draft_id, topic_id=topic_id,
        vector_store=get_store(), llm_service=get_llm(), subtopic_id=subtopic_id,
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or section not found")
    return result


@router.post("/{draft_id}/topics/{topic_id}/review/apply")
async def review_apply_endpoint(
    draft_id: str,
    topic_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Persist reviewer-approved content for a section and advance its review_step
    (benchmark | regulation | review). Body: {content, step?, subtopic_id?}."""
    content = request.get("content")
    if content is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="content is required")
    result = await draft_service.apply_section_content(
        draft_id=draft_id, topic_id=topic_id, content=content,
        subtopic_id=request.get("subtopic_id"), step=(request.get("step") or "review"),
        llm_service=get_llm(),
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or section not found")
    return result


@router.post("/{draft_id}/topics/{topic_id}/review/close")
async def review_close_endpoint(
    draft_id: str,
    topic_id: str,
    request: dict = None,
    subtopic_id: Optional[str] = None,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Bank a content-ready section (review_step='closed'), or reopen it. Body: {subtopic_id?, reopen?}."""
    body = request or {}
    result = draft_service.close_section(
        draft_id=draft_id, topic_id=topic_id,
        subtopic_id=body.get("subtopic_id") or subtopic_id, reopen=bool(body.get("reopen", False)),
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or section not found")
    return result


@router.post("/{draft_id}/topics/{topic_id}/review/chat")
async def review_chat_endpoint(
    draft_id: str,
    topic_id: str,
    request: dict,
    draft_service: DraftService = Depends(get_draft_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Per-section review chatbot. Body: {message, subtopic_id?}. Returns {reply, proposed_content}
    where proposed_content (when present) is previewed and applied via the review/apply endpoint."""
    message = request.get("message")
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message is required")
    result = await draft_service.review_chat_for_unit(
        draft_id=draft_id, topic_id=topic_id, message=message,
        vector_store=get_store(), llm_service=get_llm(), subtopic_id=request.get("subtopic_id"),
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft or section not found")
    return result


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