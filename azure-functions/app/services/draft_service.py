import uuid
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


from app.models.draft_models import (
    ClientMetadata, DraftMetadata, CompleteDraft, TOCTopic,
    GapReport, GapFinding, ApplyGapRequest,
)
from app.services.generation_service import ContentGenerationService
from app.services.toc_generation_service import TOCGenerationService
from app.services.reflection_agent import ReflectionAgent
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DraftService:
    """Business logic service for draft management."""

    def __init__(self, storage_service):
        """Initialize draft service with storage dependency."""
        self.storage = storage_service

        # Initialize reflection agent for content critique
        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        llm_service = AzureLLM(prompt_manager)
        reflection_agent = ReflectionAgent(llm_service)
        self.generation_service = ContentGenerationService(reflection_agent=reflection_agent)
        # text -> embedding cache for benchmark matching (lives for the life of this instance,
        # i.e. one request — so a batch reuses each name's embedding instead of re-embedding it).
        self._embed_cache: Dict[str, List[float]] = {}
    
    async def initialize_draft(
        self,
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        llm_service=None,
        vector_store=None,
        description: Optional[str] = None,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3
    ) -> Dict[str, Any]:
        """
        Initialize a new draft.
        If similar policies exist (>70% threshold), return them for user selection
        instead of immediately generating a TOC. Otherwise generate TOC with AI.
        """
        try:
            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=description,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source="pending_selection",
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level
            )

            draft_to_save = CompleteDraft(
                id=draft_id,
                metadata=draft_metadata,
                toc=[]
            )

            # Search for similar policies (top 3, threshold 0.7)
            similar_policies = []
            if vector_store:
                try:
                    description = draft_metadata.description or ""
                    search_query = f"{title} {description}".strip()
                    embedding = vector_store.embedder.embed_query(search_query)
                    matches = vector_store.search_similar_policies(
                        query_vector=embedding,
                        threshold=0.75,
                        top_k=3,
                        function=function
                    )
                    similar_policies = [
                        {
                            "policy_id": m["id"],
                            "filename": m["filename"],
                            "description": m["description"],
                            "web_url": m.get("web_url", ""),
                            "similarity_score": round(m["similarity_score"], 3),
                        }
                        for m in matches
                    ]
                    logger.info(f"Found {len(similar_policies)} similar policies above cosine 0.75 for '{title}'")
                except Exception as e:
                    import traceback
                    logger.error(f"Similar policy search FAILED (falling back to AI TOC): {e}")
                    logger.error(traceback.format_exc())

            # If similar policies found, save draft with empty TOC and return for user selection
            if similar_policies:
                draft_to_save.metadata.toc_source = "pending_selection"
                success = self.storage.create_draft(draft_to_save)
                if not success:
                    logger.error(f"Failed to save draft {draft_id}")
                    return None
                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_to_save.model_dump(),
                    "similar_policies_found": len(similar_policies),
                    "most_similar_policy_id": similar_policies[0]["policy_id"],
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": "pending_selection",
                    "needs_policy_selection": True,
                    "similar_policies": similar_policies,
                }

            # No similar policies — generate TOC with AI
            toc_source = "ai_generated"
            logger.info(f"No similar policies found, generating AI TOC for '{title}'")
            if llm_service:
                toc_generator = TOCGenerationService(llm_service)
                generated_toc = await toc_generator.generate_toc(
                    title=title,
                    client_metadata=client_metadata,
                    function=function,
                    client_specific_requests=client_specific_requests,
                    sector_specific_comments=sector_specific_comments,
                    regulations=regulations,
                    detail_level=detail_level
                )
                draft_to_save.toc = self._transform_ai_generated_toc(generated_toc)
                logger.info(f"Generated TOC with {len(draft_to_save.toc)} topics using AI")
            else:
                logger.warning("No LLM service available, using basic template")
                toc_source = "basic_template"

            draft_to_save.metadata.toc_source = toc_source
            if not draft_to_save.toc:
                draft_to_save.toc = self._get_basic_toc_template()

            success = self.storage.create_draft(draft_to_save)
            if not success:
                logger.error(f"Failed to save draft {draft_to_save.id}")
                return None

            return {
                "draft_id": draft_to_save.id,
                "draft": draft_to_save.model_dump(),
                "similar_policies_found": 0,
                "most_similar_policy_id": None,
                "client_metadata": client_metadata.model_dump(),
                "toc_source": toc_source,
                "needs_policy_selection": False,
                "similar_policies": [],
            }

        except Exception as e:
            logger.error(f"Failed to initialize draft: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    async def apply_policy_toc(
        self,
        draft_id: str,
        policy_id: Optional[str],
        vector_store=None,
        llm_service=None
    ) -> bool:
        """
        Apply a selected policy's TOC to an existing draft, or generate with AI.
        Called after the user picks a similar policy from the selection screen.
        """
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found for apply_policy_toc")
                return False

            if policy_id and vector_store:
                # Get topics for the selected policy from the vector store
                topics = vector_store.get_topics_by_policy(policy_id)
                if topics:
                    toc_input = self._build_toc_from_vector_topics(topics)
                    draft.toc = self._transform_toc_for_draft(toc_input)
                    draft.metadata.toc_source = "similar_policy"
                    draft.metadata.most_similar_policy_id = policy_id
                    logger.info(f"Applied policy {policy_id} TOC ({len(draft.toc)} topics) to draft {draft_id}")
                else:
                    logger.warning(f"Policy {policy_id} has no topics, falling back to AI")
                    policy_id = None

            if not policy_id:
                # Generate TOC with AI
                if llm_service:
                    toc_generator = TOCGenerationService(llm_service)
                    generated_toc = await toc_generator.generate_toc(
                        title=draft.metadata.title,
                        client_metadata=draft.metadata.client_metadata,
                        function=draft.metadata.function,
                        client_specific_requests=draft.metadata.client_specific_requests,
                        sector_specific_comments=draft.metadata.sector_specific_comments,
                        regulations=draft.metadata.regulations,
                        detail_level=draft.metadata.detail_level
                    )
                    draft.toc = self._transform_ai_generated_toc(generated_toc)
                else:
                    draft.toc = self._get_basic_toc_template()
                draft.metadata.toc_source = "ai_generated"
                logger.info(f"Generated AI TOC ({len(draft.toc)} topics) for draft {draft_id}")

            draft.metadata.modified_at = datetime.now(timezone.utc).isoformat()
            return self.storage.update_draft(draft_id, draft)

        except Exception as e:
            logger.error(f"Failed to apply policy TOC to draft {draft_id}: {e}")
            return False

    def _build_toc_from_vector_topics(self, topics: List[Dict]) -> List[Dict]:
        """Map vector store topic documents into the format _transform_toc_for_draft expects."""
        import json as _json
        result = []
        for idx, t in enumerate(topics, 1):
            raw_subs = t.get("subtopics", [])
            if isinstance(raw_subs, str):
                raw_subs = _json.loads(raw_subs)
            subtopics = []
            for sub_idx, sub in enumerate(raw_subs, 1):
                subtopics.append({
                    "subtopic_id": sub.get("id", sub.get("subtopic_id", str(uuid.uuid4()))),
                    "topic": sub.get("name", sub.get("topic", "")),
                    "order": sub_idx,
                })
            result.append({
                "topic_id": t.get("id", t.get("topic_id", str(uuid.uuid4()))),
                "topic": t.get("topic", ""),
                "order": idx,
                "subtopics": subtopics,
            })
        return result

    async def initialize_draft_from_upload(
        self,
        file_bytes: bytes,
        filename: str,
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3,
        previous_policy_bytes: Optional[bytes] = None,
        previous_policy_filename: Optional[str] = None,
        regulations_files: Optional[List[tuple]] = None,  # list of (filename, bytes)
        benchmark_file_bytes: Optional[bytes] = None,
        benchmark_filename: Optional[str] = None,
        selected_benchmark_policy_id: Optional[str] = None,
        vector_store=None,
        llm_service=None,
        review_intensity: str = "preserve",
    ) -> Dict[str, Any]:
        """
        Initialize a new draft from an uploaded .docx file (Review Policy / gap-review flow).
        Parses the client document into the TOC baseline, resolves a good-standard benchmark
        (uploaded file or best library match), and auto-adds benchmark sections the client lacks.
        """
        try:
            from io import BytesIO
            from app.pipelines.document_processor import DocumentProcessor
            from azure.storage.blob import BlobServiceClient

            toc_source = "uploaded_policy"
            logger.info(f"Initializing draft '{title}' from uploaded file '{filename}'")

            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            # Upload files to Azure Blob Storage
            try:
                blob_service = BlobServiceClient.from_connection_string(
                    settings.AZURE_STORAGE_CONNECTION_STRING
                )
                container_client = blob_service.get_container_client(
                    settings.AZURE_STORAGE_CONTAINER_NAME
                )
                container_client.upload_blob(
                    name=f"uploaded-policies/{draft_id}/{filename}",
                    data=file_bytes, overwrite=True
                )
                if previous_policy_bytes and previous_policy_filename:
                    container_client.upload_blob(
                        name=f"uploaded-policies/{draft_id}/previous/{previous_policy_filename}",
                        data=previous_policy_bytes, overwrite=True
                    )
                if regulations_files:
                    for reg_filename, reg_bytes in regulations_files:
                        container_client.upload_blob(
                            name=f"uploaded-policies/{draft_id}/regulations/{reg_filename}",
                            data=reg_bytes, overwrite=True
                        )
                if benchmark_file_bytes and benchmark_filename:
                    container_client.upload_blob(
                        name=f"uploaded-policies/{draft_id}/benchmark/{benchmark_filename}",
                        data=benchmark_file_bytes, overwrite=True
                    )
                logger.info(f"Uploaded policy files to blob for draft {draft_id}")
            except Exception as e:
                logger.warning(f"Failed to upload files to blob storage (non-fatal): {e}")

            # Parse the main (client) policy document → TOC baseline. keep_empty_sections=True so the
            # client's empty sections are preserved as gaps to review (not pruned away).
            processor = DocumentProcessor()
            processed = processor.process(BytesIO(file_bytes), keep_empty_sections=True)
            topics = processed.get("topics", [])
            logger.info(f"Parsed client document: {len(topics)} topics "
                        f"(confidence: {(processed.get('metadata') or {}).get('confidence')})")

            # Low-confidence extraction → LLM leveling pass: send the doc's first 300 words (the TOC,
            # if any) + the detected headings; the LLM fixes the nesting. Content stays verbatim
            # (reused from deterministic slicing). Deterministic result is kept on any failure.
            conf = (processed.get("metadata") or {}).get("confidence")
            needs_manual_toc = (conf == "empty")  # nothing to level — consultant builds the TOC
            if llm_service and conf not in ("high", "empty", None):
                try:
                    updates, drops = await self.generation_service.infer_heading_levels(
                        processor.toc_context_for_llm(300), processor.get_heading_hints(), llm_service
                    )
                    if updates or drops:
                        rebuilt = processor.rebuild_with_levels(updates, drops)
                        if rebuilt.get("topics"):  # guard: don't let a degenerate result wipe the TOC
                            processed = rebuilt
                            topics = rebuilt.get("topics", [])
                            logger.info(f"LLM leveling applied (was {conf}): {len(topics)} topics")
                        else:
                            logger.warning("LLM leveling produced no topics; keeping deterministic")
                except Exception as e:
                    logger.warning(f"LLM leveling failed (non-fatal, using deterministic): {e}")

            # Extract text from uploaded regulations files (.docx only)
            uploaded_regulations_text = None
            if regulations_files:
                reg_texts = []
                for reg_filename, reg_bytes in regulations_files:
                    if reg_filename.endswith(".docx"):
                        try:
                            reg_processed = processor.process(BytesIO(reg_bytes))
                            reg_text = reg_processed.get("full_text", "").strip()
                            if reg_text:
                                reg_texts.append(f"[{reg_filename}]\n{reg_text}")
                        except Exception as e:
                            logger.warning(f"Could not parse regulation file '{reg_filename}': {e}")
                if reg_texts:
                    uploaded_regulations_text = "\n\n---\n\n".join(reg_texts)
                    logger.info(f"Extracted text from {len(reg_texts)} regulation file(s)")

            # Extract previous client policy text (continuity reference)
            previous_policy_text = None
            if previous_policy_bytes and (previous_policy_filename or "").endswith(".docx"):
                try:
                    prev_processed = processor.process(BytesIO(previous_policy_bytes))
                    previous_policy_text = (prev_processed.get("full_text", "") or "").strip() or None
                except Exception as e:
                    logger.warning(f"Could not parse previous policy '{previous_policy_filename}': {e}")

            # Resolve the good-standard benchmark: uploaded file OR best library match
            benchmark_source = None
            benchmark_policy_id = None
            benchmark_topics_json = None
            benchmark_topics_for_toc: List[Dict[str, Any]] = []
            if benchmark_file_bytes:
                try:
                    bench_processed = processor.process(BytesIO(benchmark_file_bytes))
                    benchmark_topics_for_toc = bench_processed.get("topics", []) or []
                    benchmark_topics_json = json.dumps(benchmark_topics_for_toc)
                    benchmark_source = "uploaded"
                    logger.info(f"Benchmark (uploaded) parsed: {len(benchmark_topics_for_toc)} topics")
                except Exception as e:
                    logger.warning(f"Could not parse uploaded benchmark '{benchmark_filename}': {e}")
            elif selected_benchmark_policy_id and vector_store:
                try:
                    benchmark_policy_id = selected_benchmark_policy_id
                    benchmark_source = "library"
                    benchmark_topics_for_toc = vector_store.get_topics_by_policy(benchmark_policy_id) or []
                    logger.info(f"Benchmark (library, user-selected) policy {benchmark_policy_id} "
                                f"with {len(benchmark_topics_for_toc)} topics")
                except Exception as e:
                    logger.warning(f"Could not load selected benchmark policy {selected_benchmark_policy_id}: {e}")
            elif vector_store:
                try:
                    embedding = vector_store.embedder.embed_query(title)
                    matches = vector_store.search_similar_policies(
                        query_vector=embedding, threshold=0.75, top_k=1, function=function
                    )
                    if matches:
                        benchmark_policy_id = matches[0]["id"]
                        benchmark_source = "library"
                        benchmark_topics_for_toc = vector_store.get_topics_by_policy(benchmark_policy_id) or []
                        logger.info(f"Benchmark (library) matched policy {benchmark_policy_id} "
                                    f"with {len(benchmark_topics_for_toc)} topics")
                    else:
                        logger.info("No library benchmark found above threshold for review draft")
                except Exception as e:
                    logger.warning(f"Library benchmark search failed (non-fatal): {e}")

            # Build the client TOC baseline
            toc = self._transform_uploaded_doc_toc(topics)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=None,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source=toc_source,
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level,
                uploaded_regulations_text=uploaded_regulations_text,
                previous_policy_text=previous_policy_text,
                review_mode=True,
                benchmark_policy_id=benchmark_policy_id,
                benchmark_source=benchmark_source,
                benchmark_topics_json=benchmark_topics_json,
                review_intensity=(review_intensity or "preserve"),
                needs_manual_toc=needs_manual_toc,
            )

            # Reconcile the client TOC against the benchmark: the LLM decides the final TOC
            # and the client<->benchmark mapping (persisted per topic for the gap-report step).
            if benchmark_topics_for_toc:
                if llm_service:
                    toc, recon_note = await self._build_reconciled_toc(
                        client_toc=toc,
                        benchmark_topics=benchmark_topics_for_toc,
                        draft_metadata=draft_metadata,
                        llm_service=llm_service,
                    )
                    draft_metadata.toc_reconciliation_note = recon_note
                else:
                    toc = self._naive_benchmark_union(toc, benchmark_topics_for_toc)

            draft_to_save = CompleteDraft(
                id=draft_id,
                metadata=draft_metadata,
                toc=toc
            )

            success = self.storage.create_draft(draft_to_save)

            if success:
                # Snapshot the original as version 0 — powers safe restore + "% preserved" in the gap summary.
                try:
                    self.storage.create_version(draft_id, draft_to_save, label="original", created_by=user_id)
                except Exception as e:
                    logger.warning(f"Could not snapshot original version (non-fatal): {e}")
                logger.info(f"Initialized draft {draft_id} from upload with {len(draft_to_save.toc)} topics")
                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_to_save.model_dump(),
                    "similar_policies_found": 0,
                    "most_similar_policy_id": benchmark_policy_id,
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": toc_source,
                    "review_mode": True,
                    "benchmark_source": benchmark_source,
                }
            else:
                logger.error(f"Failed to save uploaded draft {draft_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to initialize draft from upload: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _transform_uploaded_doc_toc(self, topics: List[Dict[str, Any]]) -> List:
        """
        Transform DocumentProcessor output topics into draft TOCTopic models.
        Content from the uploaded document is pre-filled into each topic/subtopic.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for idx, topic_data in enumerate(topics, 1):
            subtopics = []
            for sub_idx, sub in enumerate(topic_data.get("subtopics", []), 1):
                subtopic = TOCSubtopic(
                    subtopic_id=sub.get("id", str(uuid.uuid4())),
                    topic=sub.get("name", sub.get("topic", "")),
                    order=sub_idx,
                    content=sub.get("content", ""),
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            topic = TOCTopic(
                topic_id=topic_data.get("id", str(uuid.uuid4())),
                topic=topic_data.get("topic", ""),
                order=idx,
                content=topic_data.get("topic_content", ""),
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _naive_benchmark_union(self, client_toc: List, benchmark_topics: List[Dict[str, Any]]) -> List:
        """Fallback TOC: keep the client topics and append benchmark topics with no exact-name match."""
        from app.models.draft_models import TOCTopic
        toc = list(client_toc)
        client_titles = {(t.topic or "").strip().lower() for t in toc}
        next_order = len(toc)
        for bt in benchmark_topics:
            bt_name = (bt.get("topic") or bt.get("name") or "").strip()
            if bt_name and bt_name.lower() not in client_titles:
                next_order += 1
                toc.append(TOCTopic(
                    topic_id=str(uuid.uuid4()),
                    topic=bt_name,
                    order=next_order,
                    content="",
                    summary="",
                    conversation_history=[],
                    subtopics=[],
                    added_from_benchmark=True,
                    gap_status="pending",
                    benchmark_match=bt_name,
                ))
        return toc

    async def _build_reconciled_toc(self, client_toc: List, benchmark_topics: List[Dict[str, Any]],
                                    draft_metadata, llm_service):
        """
        Ask the LLM to decide the final TOC and the client<->benchmark mapping.
        Reuses each client TOCTopic (preserving content/subtopics) and appends benchmark-only
        topics. Returns (toc, reconciliation_note). Falls back to a naive union on failure.
        """
        from app.models.draft_models import TOCTopic
        review_intensity = getattr(draft_metadata, "review_intensity", "preserve") or "preserve"
        try:
            client_dicts = [{"topic": t.topic, "content": t.content} for t in client_toc]
            data = await self.generation_service.reconcile_toc(
                client_topics=client_dicts,
                benchmark_topics=benchmark_topics,
                draft_metadata=draft_metadata,
                llm_service=llm_service,
            )
        except Exception as e:
            logger.warning(f"TOC reconciliation failed, falling back to naive union: {e}")
            return self._naive_benchmark_union(client_toc, benchmark_topics), None

        client_by_title = {(t.topic or "").strip().lower(): t for t in client_toc}
        final_toc: List = []
        used_client = set()
        for entry in data.get("final_toc", []):
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            source = (entry.get("source") or "").strip().lower()
            client_match = (entry.get("client_match") or "").strip()
            benchmark_match = (entry.get("benchmark_match") or "").strip() or None
            note = (entry.get("note") or "").strip() or None
            cm_key = client_match.lower()
            existing = client_by_title.get(cm_key)
            if existing is not None and cm_key not in used_client:
                # Reuse the client topic, preserving its content/subtopics; apply mapping + optional rename
                used_client.add(cm_key)
                if title and title != existing.topic:
                    existing.reconcile_note = note or f"Renamed from '{existing.topic}'"
                    existing.topic = title
                else:
                    existing.reconcile_note = note
                existing.benchmark_match = benchmark_match
                final_toc.append(existing)
            elif source == "client":
                # LLM claimed a client topic but its client_match didn't resolve (likely hallucinated);
                # skip — the safety loop below re-adds the genuine client topic.
                continue
            else:
                final_toc.append(TOCTopic(
                    topic_id=str(uuid.uuid4()),
                    topic=title,
                    content="",
                    summary="",
                    conversation_history=[],
                    subtopics=[],
                    added_from_benchmark=True,
                    gap_status="pending",
                    benchmark_match=benchmark_match or title,
                    reconcile_note=note,
                    order=0,
                ))

        # Safety: in preserve mode never silently drop a client topic the LLM omitted.
        if review_intensity != "rebuild":
            for t in client_toc:
                key = (t.topic or "").strip().lower()
                if key and key not in used_client:
                    final_toc.append(t)

        if not final_toc:
            return self._naive_benchmark_union(client_toc, benchmark_topics), None

        # Assign final 1-based ordering.
        for idx, t in enumerate(final_toc, 1):
            t.order = idx

        assessment = data.get("assessment") or {}
        recon_note = (assessment.get("summary") or "").strip() or None
        logger.info(f"Reconciled TOC: {len(final_toc)} topics "
                    f"({len(used_client)} client matched, intensity={review_intensity})")
        return final_toc, recon_note

    async def initialize_draft_from_description(
        self,
        description: Optional[str],
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        llm_service=None,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3
    ) -> Dict[str, Any]:
        """
        Initialize a new draft using AI-generated TOC based on policy information.
        TOC is always generated by AI using the policy title, description, and function.

        DEPRECATED: Use initialize_draft() instead for new implementations.
        """
        try:
            # Determine if we have a description
            has_description = description and description.strip()
            toc_source = "ai_generated"

            logger.info(f"Initializing draft with AI-generated TOC for function='{function}'")

            # Create draft structure
            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=description,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source=toc_source,
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level
            )

            draft_to_save = CompleteDraft(
                id=str(uuid.uuid4()),
                metadata=draft_metadata,
                toc=[]
            )

            # Generate TOC using AI (always use AI generation)
            if llm_service:
                # Initialize TOC generation service
                toc_generator = TOCGenerationService(llm_service)

                # Generate TOC based on policy information
                generated_toc = await toc_generator.generate_toc_from_description(
                    description=description if has_description else f"A policy for {function}",
                    title=title,
                    client_metadata=client_metadata,
                    function=function,
                    client_specific_requests=client_specific_requests,
                    sector_specific_comments=sector_specific_comments,
                    regulations=regulations,
                    detail_level=detail_level
                )

                # Transform to draft format
                draft_to_save.toc = self._transform_ai_generated_toc(generated_toc)
                logger.info(f"Generated TOC with {len(draft_to_save.toc)} topics using AI")
            else:
                # Fallback to basic structure if no LLM service available
                logger.warning("No LLM service available for TOC generation, using basic template")
                toc_source = "basic_template"
                draft_to_save.metadata.toc_source = toc_source
                draft_to_save.toc = self._get_basic_toc_template()

            success = self.storage.create_draft(draft_to_save)

            if success:
                logger.info(f"Initialized draft {draft_to_save.id} with {len(draft_to_save.toc)} suggested topics (source: {toc_source})")
                draft_dict = draft_to_save.model_dump()

                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_dict,
                    "similar_policies_found": 0,
                    "most_similar_policy_id": None,
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": toc_source
                }
            else:
                logger.error(f"Failed to save initialized draft {draft_to_save.id}")
                return None

        except Exception as e:
            logger.error(f"Failed to initialize draft: {e}")
            return None

    def _transform_toc_for_draft(self, toc_from_policy: List[Dict[str, Any]]) -> List:
        """
        Transform TOC from similar policy to draft format.
        Creates new IDs for draft while preserving source IDs from vector DB.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for topic_item in toc_from_policy:
            # Create subtopics with new IDs but preserve source IDs
            subtopics = []
            for sub in topic_item.get('subtopics', []):
                subtopic = TOCSubtopic(
                    subtopic_id=str(uuid.uuid4()),  # New ID for draft
                    topic=sub.get('topic', ''),
                    order=sub.get('order', 0),
                    source_subtopic_id=sub.get('subtopic_id'),  # Preserve vector DB ID
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            # Create topic with new ID but preserve source ID
            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),  # New ID for draft
                topic=topic_item.get('topic', ''),
                order=topic_item.get('order', 0),
                source_topic_id=topic_item.get('topic_id'),  # Preserve vector DB ID
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _transform_ai_generated_toc(self, ai_toc: List[Dict[str, Any]]) -> List:
        """
        Transform AI-generated TOC to draft format.
        AI-generated topics don't have source_topic_id or source_subtopic_id.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for topic_item in ai_toc:
            # Create subtopics without source IDs
            subtopics = []
            for sub in topic_item.get('subtopics', []):
                subtopic = TOCSubtopic(
                    subtopic_id=sub.get('subtopic_id', str(uuid.uuid4())),
                    topic=sub.get('topic', ''),
                    order=sub.get('order', 0),
                    # No source_subtopic_id for AI-generated
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            # Create topic without source ID
            topic = TOCTopic(
                topic_id=topic_item.get('topic_id', str(uuid.uuid4())),
                topic=topic_item.get('topic', ''),
                order=topic_item.get('order', 0),
                # No source_topic_id for AI-generated
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _get_basic_toc_template(self) -> List:
        """
        Get a basic TOC template when no similar policies and no LLM available.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        basic_template = [
            {
                "topic": "Purpose and Scope",
                "subtopics": [
                    {"topic": "Policy Objectives"},
                    {"topic": "Applicability"}
                ]
            },
            {
                "topic": "Definitions and Terms",
                "subtopics": [
                    {"topic": "Key Terminology"},
                    {"topic": "Acronyms"}
                ]
            },
            {
                "topic": "Policy Requirements",
                "subtopics": [
                    {"topic": "Core Requirements"},
                    {"topic": "Implementation Guidelines"}
                ]
            },
            {
                "topic": "Compliance and Monitoring",
                "subtopics": [
                    {"topic": "Monitoring Procedures"},
                    {"topic": "Enforcement"}
                ]
            }
        ]

        transformed_toc = []
        for idx, template_item in enumerate(basic_template, 1):
            subtopics = []
            for sub_idx, sub in enumerate(template_item['subtopics'], 1):
                subtopic = TOCSubtopic(
                    subtopic_id=str(uuid.uuid4()),
                    topic=sub['topic'],
                    order=sub_idx,
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),
                topic=template_item['topic'],
                order=idx,
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _merge_tocs_from_policies(self, policies: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract and merge TOCs from similar policies.
        Prioritize topics that appear in multiple policies.
        """
        topic_frequency = {}
        topic_details = {}
        
        print(f"Number of policies: {len(policies)}")

        for i, policy in enumerate(policies):

            # if i != 0: break  # let's just keep only the first policy
        
            if not policy.toc:
                continue
                
            for topic in policy.toc:
                topic_key = topic.get('topic', '').lower().strip()
                if not topic_key:
                    continue
                
                if topic_key not in topic_frequency:
                    topic_frequency[topic_key] = 0
                    topic_details[topic_key] = {
                        "id": str(uuid.uuid4()),
                        "topic": topic.get('topic', ''),
                        "order": len(topic_details) + 1,
                        "source_topic_id": topic.get('topic_id', ''),  # Use topic_id from vector DB as source
                        "source_policy_id": policy.policy_id,  # PolicyMatch is a Pydantic model, use attribute access
                        "subtopics": []
                    }
                topic_frequency[topic_key] += 1
                
                # Merge subtopics if they exist
                if 'subtopics' in topic and topic['subtopics']:
                    for subtopic in topic['subtopics']:
                        subsection = {
                            "id": str(uuid.uuid4()),
                            "topic": subtopic.get('topic', ''),
                            "order": len(topic_details[topic_key]["subtopics"]) + 1,
                            "source_subtopic_id": subtopic.get('subtopic_id', '')  # Use subtopic_id from vector DB as source
                        }
                        # Avoid duplicate subtopics
                        existing_titles = [s['topic'].lower() for s in topic_details[topic_key]["subtopics"]]
                        if subsection['topic'].lower() not in existing_titles:
                            topic_details[topic_key]["subtopics"].append(subsection)
        
        # Sort by frequency and return as list
        sorted_topics = sorted(
            topic_details.items(),
            key=lambda x: topic_frequency[x[0]],
            reverse=True
        )
        
        # Reorder and return
        toc = []
        for idx, (_, topic) in enumerate(sorted_topics, 1):
            topic["order"] = idx
            toc.append(topic)
        
        return toc
    
    def _format_toc_for_model(self, raw_toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform raw TOC data to match Pydantic model structure.
        Handles different input formats and ensures required fields are present.
        """
        formatted_toc = []

        for idx, item in enumerate(raw_toc, 1):
            # Handle different possible input formats
            if isinstance(item, dict):
                # Extract title from 'topic' field or 'title' field
                title = item.get('topic', item.get('title', f'Topic {idx}'))

                formatted_topic = {
                    'topic_id': item.get('topic_id') or item.get('id') or str(uuid.uuid4()),  # Check both 'topic_id' and 'id'
                    'title': title.strip(),
                    'order': item.get('order', idx),
                    'source_topic_id': item.get('source_topic_id'),  # Preserve source_topic_id if it exists
                    'subtopics': []
                }

                # Handle subtopics if they exist
                subtopics = item.get('subtopics', [])
                if subtopics:
                    for sub_idx, subtopic in enumerate(subtopics, 1):
                        if isinstance(subtopic, dict):
                            sub_title = subtopic.get('title') or subtopic.get('topic', f'Subtopic {sub_idx}')

                            formatted_subtopic = {
                                'id': subtopic.get('subtopic_id') or subtopic.get('id') or str(uuid.uuid4()),  # Check both 'subtopic_id' and 'id'
                                'title': sub_title.strip(),
                                'order': subtopic.get('order', sub_idx),
                                'source_subtopic_id': subtopic.get('source_subtopic_id')  # Preserve source_subtopic_id if it exists
                            }
                            formatted_topic['subtopics'].append(formatted_subtopic)

                formatted_toc.append(formatted_topic)

        return formatted_toc
    
    def update_toc(self, draft_id: str, updated_toc: List[Dict[str, Any]]) -> bool:
        """
        Update the TOC structure of a draft from frontend.
        """
        try:
            logger.info(f"Starting TOC update for draft {draft_id}")

            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found for TOC update")
                return False

            logger.info(f"Successfully retrieved draft {draft_id}")

            # Transform TOC data to match Pydantic model structure
            formatted_toc = self._format_toc_for_model(updated_toc)

            # Convert formatted TOC to TOCTopic models
            from app.models.draft_models import TOCTopic, TOCSubtopic

            toc_topics = []
            for toc_item in formatted_toc:
                # Find existing topic to preserve content and conversation history
                existing_topic = None
                for existing in draft_model.toc:
                    if existing.topic_id == toc_item['topic_id']:
                        existing_topic = existing
                        break

                # Convert subtopics
                subtopics = []
                for sub_item in toc_item.get('subtopics', []):
                    # Find existing subtopic to preserve content and conversation history
                    existing_subtopic = None
                    if existing_topic:
                        for existing_sub in existing_topic.subtopics:
                            if existing_sub.subtopic_id == sub_item['id']:
                                existing_subtopic = existing_sub
                                break

                    subtopic = TOCSubtopic(
                        subtopic_id=sub_item['id'],
                        topic=sub_item['title'],
                        order=sub_item['order'],
                        source_subtopic_id=sub_item.get('source_subtopic_id'),
                        content=existing_subtopic.content if existing_subtopic else "",
                        summary=existing_subtopic.summary if existing_subtopic else "",
                        conversation_history=existing_subtopic.conversation_history if existing_subtopic else [],
                        gap_report_json=existing_subtopic.gap_report_json if existing_subtopic else None,
                        gap_status=existing_subtopic.gap_status if existing_subtopic else "pending",
                        benchmark_match=existing_subtopic.benchmark_match if existing_subtopic else None,
                    )
                    subtopics.append(subtopic)

                # Convert main topic (preserve gap-assessment fields across TOC edits)
                topic = TOCTopic(
                    topic_id=toc_item['topic_id'],  # Use topic_id, not id
                    topic=toc_item['title'],
                    order=toc_item['order'],
                    source_topic_id=toc_item.get('source_topic_id'),
                    content=existing_topic.content if existing_topic else "",
                    summary=existing_topic.summary if existing_topic else "",
                    conversation_history=existing_topic.conversation_history if existing_topic else [],
                    subtopics=subtopics,
                    gap_report_json=existing_topic.gap_report_json if existing_topic else None,
                    gap_status=existing_topic.gap_status if existing_topic else "pending",
                    added_from_benchmark=existing_topic.added_from_benchmark if existing_topic else False,
                    benchmark_match=existing_topic.benchmark_match if existing_topic else None,
                    reconcile_note=existing_topic.reconcile_note if existing_topic else None,
                )
                toc_topics.append(topic)

            # Update TOC in the model
            draft_model.toc = toc_topics
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            logger.info(f"Updated TOC with {len(toc_topics)} topics")

            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully updated TOC for draft {draft_id} with {len(toc_topics)} topics")
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update TOC for draft {draft_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    
    def update_toc_chat_history(self, draft_id: str, conversation_history: List[Dict[str, Any]]) -> bool:
        """Update the TOC chat conversation history for a draft."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found for TOC chat history update")
                return False

            from app.models.draft_models import ConversationEntry
            entries = [ConversationEntry(**entry) for entry in conversation_history]
            draft_model.toc_chat_history = entries

            success = self.storage.update_draft(draft_id, draft_model)
            if success:
                logger.info(f"Updated TOC chat history for draft {draft_id} with {len(entries)} entries")
            return success
        except Exception as e:
            logger.error(f"Failed to update TOC chat history for draft {draft_id}: {e}")
            return False

    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete draft by ID."""
        draft_model = self.storage.get_draft(draft_id)
        if draft_model:
            return draft_model.model_dump()
        return None
    
    def list_drafts(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        created_by: Optional[str] = None,
        industry: Optional[str] = None,
        function: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all drafts with optional filtering.
        user_id parameter kept for compatibility but all drafts are visible to all users.
        """
        return self.storage.list_all_drafts(
            title=title,
            country=country,
            city=city,
            created_by=created_by,
            industry=industry,
            function=function
        )
    
    async def update_topic_content(
        self,
        draft_id: str,
        topic_id: str,
        content: str,
        subtopic_id: Optional[str] = None,
        llm_service = None,
        conversation_history: Optional[List] = None
    ) -> bool:
        """
        Update content for a topic or subtopic using typed models.
        """
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.warning(f"Draft {draft_id} not found")
                return False

            # Find and update in the TOC structure
            topic_found = False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    topic_found = True

                    if subtopic_id:
                        # Update subtopic content
                        subtopic_found = False
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.content = content
                                # Update conversation history if provided
                                if conversation_history is not None:
                                    toc_subtopic.conversation_history = conversation_history
                                    logger.info(f"Updated subtopic {subtopic_id} conversation history with {len(conversation_history)} entries")
                                # Generate LLM summary if content > 150 words and LLM service available
                                if llm_service:
                                    toc_subtopic.summary = await self.generation_service.generate_summary(
                                        content=content,
                                        title=toc_subtopic.topic,
                                        llm_service=llm_service
                                    )
                                else:
                                    toc_subtopic.summary = self._create_simple_summary_for_context(content)
                                subtopic_found = True
                                logger.info(f"Updated subtopic {subtopic_id} content")
                                break

                        if not subtopic_found:
                            logger.warning(f"Subtopic {subtopic_id} not found in topic {topic_id}")
                            return False
                    else:
                        # Update main topic content
                        toc_topic.content = content
                        # Update conversation history if provided
                        if conversation_history is not None:
                            toc_topic.conversation_history = conversation_history
                            logger.info(f"Updated topic {topic_id} conversation history with {len(conversation_history)} entries")
                        # Generate LLM summary if content > 150 words and LLM service available
                        if llm_service:
                            toc_topic.summary = await self.generation_service.generate_summary(
                                content=content,
                                title=toc_topic.topic,
                                llm_service=llm_service
                            )
                        else:
                            toc_topic.summary = self._create_simple_summary_for_context(content)
                        logger.info(f"Updated topic {topic_id} content")
                    break

            if not topic_found:
                logger.warning(f"Topic {topic_id} not found in draft {draft_id}")
                return False

            # Update modified timestamp
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            # Save the updated model
            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully saved content for topic {topic_id}" +
                           (f", subtopic {subtopic_id}" if subtopic_id else ""))
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update topic content: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _create_simple_summary_for_context(self, content: str) -> str:
        """
        Create a simple summary for context building by truncating content > 150 words.
        This is used for context building when full content would be too long.
        """
        if not content:
            return ""

        word_count = len(content.split())
        if word_count > 150:
            # Take first 150 words as summary for context
            words = content.split()
            return " ".join(words[:150]) + "..."

        return ""  # No summary needed for short content

    def get_toc_item_by_id(self, draft: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
        """Get a TOC item (topic) by its ID."""
        # Convert to model if needed for compatibility
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        for toc_topic in draft_model.toc:
            if toc_topic.topic_id == item_id:
                return {
                    "id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "title": toc_topic.topic,  # For backward compatibility
                    "order": toc_topic.order,
                    "subtopics": [
                        {
                            "id": sub.subtopic_id,
                            "topic": sub.topic,
                            "title": sub.topic  # For backward compatibility
                        }
                        for sub in toc_topic.subtopics
                    ]
                }
        return None
    
    def build_sequential_context(
        self,
        draft: Dict[str, Any],
        target_topic_id: str,
        target_subtopic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context from all previous topics up to the target.
        Returns structured context for LLM consumption.
        Note: Still accepts dict for API compatibility but works with typed structure internally.
        """
        # Convert to model if needed
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        context_parts = {
            "document_context": {
                "title": draft_model.metadata.title,
                "description": draft_model.metadata.description,
                "client_metadata": draft_model.metadata.client_metadata.model_dump() if draft_model.metadata.client_metadata else {}
            },
            "previous_topics": [],
            "current_topic_toc": None
        }

        # Find target position in TOC
        target_found = False

        for toc_topic in draft_model.toc:
            if target_found:
                break

            if toc_topic.topic_id == target_topic_id:
                context_parts["current_topic_toc"] = {
                    "topic_id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "order": toc_topic.order
                }

                if target_subtopic_id:
                    # Include main topic content if generating subtopic
                    if toc_topic.content:
                        context_parts["current_topic_content"] = {
                            "title": toc_topic.topic,
                            "content": toc_topic.content if not toc_topic.summary else toc_topic.summary
                        }

                    # Include previous subtopics of current topic
                    for toc_subtopic in toc_topic.subtopics:
                        if toc_subtopic.subtopic_id == target_subtopic_id:
                            break

                        if toc_subtopic.content:
                            context_parts["previous_topics"].append({
                                "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                                "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
                                "is_subtopic": True
                            })

                target_found = True
            else:
                # Add all content from topics before target
                if toc_topic.content:
                    context_parts["previous_topics"].append({
                        "title": toc_topic.topic,
                        "content": toc_topic.content if not toc_topic.summary else toc_topic.summary,
                        "is_subtopic": False
                    })

                # Add subtopics
                for toc_subtopic in toc_topic.subtopics:
                    if toc_subtopic.content:
                        context_parts["previous_topics"].append({
                            "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                            "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
                            "is_subtopic": True
                        })

        return context_parts
    
    def update_draft_metadata(self, draft_id: str, metadata_request) -> bool:
        """
        Update draft metadata (title, description, client info, etc.).
        """
        try:
            # Get the existing draft
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found")
                return False

            # Update metadata fields
            draft_model.metadata.title = metadata_request.title
            draft_model.metadata.description = metadata_request.description
            draft_model.metadata.client_metadata = metadata_request.client_metadata
            draft_model.metadata.function = metadata_request.function
            draft_model.metadata.client_specific_requests = metadata_request.client_specific_requests
            draft_model.metadata.sector_specific_comments = metadata_request.sector_specific_comments
            draft_model.metadata.regulations = metadata_request.regulations
            draft_model.metadata.detail_level = metadata_request.detail_level

            # Update modified timestamp
            from datetime import datetime, timezone
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            # Save the updated draft
            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully updated metadata for draft {draft_id}")
            else:
                logger.error(f"Failed to update metadata for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating draft metadata: {e}")
            return False

    def delete_draft(self, draft_id: str) -> bool:
        """Delete a draft from storage."""
        try:
            success = self.storage.delete_draft(draft_id)
            if success:
                logger.info(f"Deleted draft {draft_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete draft {draft_id}: {e}")
            return False
    
    def delete_all_drafts(self) -> Dict[str, Any]:
        """Delete all drafts from storage."""
        try:
            # Get all drafts first
            all_drafts = self.storage.list_all_drafts()
            total_count = len(all_drafts)
            
            if total_count == 0:
                logger.info("No drafts found to delete")
                return {
                    "deleted_count": 0,
                    "total_count": 0,
                    "message": "No drafts found to delete"
                }
            
            # Delete each draft
            deleted_count = 0
            failed_deletions = []
            
            for draft in all_drafts:
                # The list_all_drafts returns 'draft_id' not 'id'
                draft_id = draft.get("draft_id")
                if draft_id:
                    try:
                        if self.storage.delete_draft(draft_id):
                            deleted_count += 1
                        else:
                            failed_deletions.append(draft_id)
                    except Exception as e:
                        logger.error(f"Failed to delete draft {draft_id}: {e}")
                        failed_deletions.append(draft_id)
            
            logger.info(f"Deleted {deleted_count}/{total_count} drafts")
            
            return {
                "deleted_count": deleted_count,
                "total_count": total_count,
                "failed_count": len(failed_deletions),
                "failed_draft_ids": failed_deletions,
                "message": f"Successfully deleted {deleted_count} out of {total_count} drafts"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete all drafts: {e}")
            return {
                "deleted_count": 0,
                "total_count": 0,
                "failed_count": 0,
                "failed_draft_ids": [],
                "message": f"Failed to delete all drafts: {str(e)}"
            }
    
    async def generate_content_for_topic(
        self,
        draft_id: str,
        topic_id: str,
        user_prompt: str,
        subtopic_id: Optional[str],
        vector_store,
        llm_service
    ) -> Dict[str, Any]:
        """
        Generate content for a topic or subtopic using the generation service.
        
        Returns:
            Dict with content and sources used
        """
        try:
            # Get the draft
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found")
                return None
            

            # Log conversation history BEFORE generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"BEFORE generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"BEFORE generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Generate content using the generation service
            content, source_ids, is_chat_response, critique = await self.generation_service.generate_topic_content(
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                vector_store=vector_store,
                llm_service=llm_service
            )

            # Log conversation history AFTER generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"AFTER generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"AFTER generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Don't save content automatically - let the frontend handle saving
            # The conversation history is already updated by the generation service

            # Generate summary for response but don't save it
            word_count = len(content.split())
            summary = ""
            if word_count > 150 and not is_chat_response:
                # Get the topic title for summary generation
                toc_item = self.get_toc_item_by_id(draft.model_dump(), topic_id)

                if subtopic_id:
                    # Find subtopic title
                    for sub in toc_item.get("subtopics", []):
                        if sub["id"] == subtopic_id:
                            title = sub["title"]
                            break
                else:
                    title = toc_item["title"]

                summary = await self.generation_service.generate_summary(
                    content=content,
                    title=title,
                    llm_service=llm_service
                )
                # Don't save the summary - let the frontend handle it when saving content

            # Log conversation history state before saving (for debugging)
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"Saving subtopic {subtopic_id} with {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"Saving topic {topic_id} with {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Save the modified draft (only conversation history is updated)
            success = self.storage.update_draft(draft_id, draft)

            if success:
                # Build critique dict if available
                critique_dict = None
                if critique:
                    critique_dict = {
                        "score": critique.score,
                        "alignment_score": critique.alignment_score,
                        "regulation_compliance_score": critique.regulation_compliance_score,
                        "strengths": critique.strengths,
                        "weaknesses": critique.weaknesses,
                        "improvements": critique.improvements,
                        "compliance_gaps": critique.compliance_gaps,
                        "recommendations": critique.recommendations,
                        "requires_revision": critique.requires_revision
                    }
                
                return {
                    "content": content,
                    "sources_used": source_ids,
                    "summary": summary,
                    "word_count": word_count,
                    "is_chat_response": is_chat_response,
                    "critique": critique_dict
                }
            else:
                logger.error(f"Failed to save generated content for topic {topic_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    # ============================================================
    # GAP ASSESSMENT ORCHESTRATION (review_mode drafts)
    # ============================================================

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def _get_benchmark_topics(self, draft, vector_store) -> List[Dict[str, Any]]:
        """Return the benchmark's per-topic content (uploaded JSON or the library policy)."""
        meta = draft.metadata
        if getattr(meta, "benchmark_topics_json", None):
            try:
                return json.loads(meta.benchmark_topics_json) or []
            except Exception:
                return []
        if getattr(meta, "benchmark_policy_id", None) and vector_store:
            try:
                return vector_store.get_topics_by_policy(meta.benchmark_policy_id) or []
            except Exception as e:
                logger.warning(f"Could not load benchmark topics for policy {meta.benchmark_policy_id}: {e}")
        return []

    def _benchmark_topic_text(self, bt: Dict[str, Any]) -> str:
        """Flatten a benchmark topic (and its subtopics) into a single text block."""
        parts = []
        content = bt.get("topic_content") or bt.get("content") or ""
        if content:
            parts.append(content)
        subs = bt.get("subtopics") or []
        if isinstance(subs, str):
            try:
                subs = json.loads(subs)
            except Exception:
                subs = []
        for s in subs:
            s_name = s.get("name") or s.get("topic") or ""
            s_content = s.get("content") or s.get("topic_content") or ""
            if s_content:
                parts.append(f"{s_name}\n{s_content}" if s_name else s_content)
        return "\n\n".join(parts).strip()

    def _embed(self, text: str, vector_store) -> Optional[List[float]]:
        """Embed text with a per-instance cache so identical strings aren't re-embedded."""
        key = (text or "").strip()
        if not key or not vector_store:
            return None
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        vec = vector_store.embedder.embed_query(key)
        self._embed_cache[key] = vec
        return vec

    def _candidate_vector(self, bt: Dict[str, Any], vector_store) -> Optional[List[float]]:
        """Reuse a benchmark candidate's stored vector (library topics carry `topic_vector`);
        fall back to embedding its name (cached) when no stored vector exists (uploaded benchmarks,
        and all subtopics — which are stored as JSON without their own vector)."""
        for k in ("topic_vector", "subtopic_vector", "content_vector", "vector"):
            v = bt.get(k)
            if isinstance(v, list) and v:
                return v
        return self._embed(bt.get("topic") or bt.get("name") or "", vector_store)

    def _match_benchmark_topic(self, client_title: str, benchmark_topics: List[Dict[str, Any]], vector_store) -> Optional[Dict[str, Any]]:
        """Match a client topic to a benchmark topic: tier 1 by name, tier 2 by embedding cosine
        (reusing stored vectors where available, else cached embeddings)."""
        if not benchmark_topics:
            return None
        target = (client_title or "").strip().lower()
        for bt in benchmark_topics:
            name = (bt.get("topic") or bt.get("name") or "").strip().lower()
            if name and (name == target or name in target or target in name):
                return bt
        if not vector_store:
            return None
        try:
            q = self._embed(client_title, vector_store)
            if q is None:
                return None
            best, best_score = None, 0.0
            for bt in benchmark_topics:
                cand = self._candidate_vector(bt, vector_store)
                if cand is None:
                    continue
                score = self._cosine(q, cand)
                if score > best_score:
                    best, best_score = bt, score
            if best is not None and best_score >= 0.80:
                return best
        except Exception as e:
            logger.warning(f"Benchmark embedding match failed: {e}")
        return None

    def _benchmark_for_topic(self, topic, benchmark_topics: List[Dict[str, Any]], vector_store) -> Optional[Dict[str, Any]]:
        """Resolve the benchmark topic for a client topic: prefer the persisted benchmark_match
        (set at TOC reconciliation), else fall back to fuzzy/embedding matching (old drafts)."""
        if not benchmark_topics:
            return None
        match_name = (getattr(topic, "benchmark_match", None) or "").strip().lower()
        if match_name:
            for bt in benchmark_topics:
                name = (bt.get("topic") or bt.get("name") or "").strip().lower()
                if name and name == match_name:
                    return bt
        return self._match_benchmark_topic(topic.topic, benchmark_topics, vector_store)

    def _benchmark_topic_intro(self, bt: Dict[str, Any]) -> str:
        """Just the benchmark topic's own intro text (no subtopics)."""
        return (bt.get("topic_content") or bt.get("content") or "").strip()

    def _benchmark_subtopics(self, bt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalized list of a benchmark topic's subtopics."""
        subs = bt.get("subtopics") or []
        if isinstance(subs, str):
            try:
                subs = json.loads(subs)
            except Exception:
                subs = []
        return subs or []

    def _benchmark_content_for_unit(self, topic, subtopic, benchmark_topics, vector_store):
        """Return (benchmark_content, benchmark_name) for a gap unit (topic or subtopic).
        Subtopic → matched benchmark subtopic (fallback: topic intro); topic-with-subtopics →
        intro only; topic-without-subtopics → full flatten (legacy)."""
        bt = self._benchmark_for_topic(topic, benchmark_topics, vector_store)
        if not bt:
            return None, None
        topic_name = bt.get("topic") or bt.get("name")
        if subtopic is not None:
            subs = self._benchmark_subtopics(bt)
            msub = None
            match_name = (getattr(subtopic, "benchmark_match", None) or "").strip().lower()
            if match_name:
                msub = next((s for s in subs
                             if (s.get("name") or s.get("topic") or "").strip().lower() == match_name), None)
            if msub is None:
                msub = self._match_benchmark_topic(subtopic.topic, subs, vector_store)
            if msub is not None:
                name = msub.get("name") or msub.get("topic")
                content = (msub.get("content") or msub.get("topic_content") or "").strip() or None
                return content, name
            return (self._benchmark_topic_intro(bt) or None), topic_name
        if getattr(topic, "subtopics", None):
            return (self._benchmark_topic_intro(bt) or None), topic_name
        return (self._benchmark_topic_text(bt) or None), topic_name

    def _surrounding_context_for_unit(self, topic, subtopic) -> str:
        """Read-only context so a unit isn't assessed in isolation: parent intro + sibling subsections
        (for a subtopic), or the list of subsections (for a topic that has them)."""
        parts = []
        intro = (topic.content or "").strip()
        if subtopic is not None:
            if intro:
                parts.append(f"[Parent topic '{topic.topic}' intro]\n{intro[:1500]}")
            for s in topic.subtopics:
                if s.subtopic_id == subtopic.subtopic_id:
                    continue
                c = (s.content or "").strip()
                parts.append(f"[Sibling subsection '{s.topic}']\n{c[:800]}" if c
                             else f"[Sibling subsection '{s.topic}'] (planned, no content yet)")
        else:
            for s in topic.subtopics:
                has = "has content" if (s.content or "").strip() else "planned"
                parts.append(f"[Subsection '{s.topic}'] ({has}) — assessed separately")
        return "\n\n".join(parts).strip()

    async def generate_gap_report_for_topic(self, draft_id: str, topic_id: str, vector_store, llm_service, subtopic_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate (and persist) a gap report for one topic, comparing it to the benchmark + regulations."""
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found")
                return None
            topic = next((t for t in draft.toc if t.topic_id == topic_id), None)
            if not topic:
                logger.error(f"Topic {topic_id} not found in draft {draft_id}")
                return None
            # The gap unit is the subtopic when subtopic_id is given, else the topic itself.
            subtopic = None
            if subtopic_id:
                subtopic = next((s for s in topic.subtopics if s.subtopic_id == subtopic_id), None)
                if not subtopic:
                    logger.error(f"Subtopic {subtopic_id} not found in topic {topic_id}")
                    return None
            unit = subtopic or topic

            benchmark_topics = self._get_benchmark_topics(draft, vector_store)
            benchmark_content, benchmark_name = self._benchmark_content_for_unit(
                topic, subtopic, benchmark_topics, vector_store
            )

            regulation_chunks = []
            try:
                regulation_chunks = await self.generation_service._search_relevant_regulations(
                    topic_title=unit.topic, user_prompt="", draft_metadata=draft.metadata, vector_store=vector_store
                )
            except Exception as e:
                logger.warning(f"Regulation search failed for gap report (non-fatal): {e}")

            previous_excerpt = (getattr(draft.metadata, "previous_policy_text", None) or "")[:4000]

            report = await self.generation_service.generate_gap_report(
                topic_id=subtopic_id or topic_id,
                topic_title=unit.topic,
                existing_content=unit.content or "",
                benchmark_content=benchmark_content,
                draft=draft,
                previous_policy_excerpt=previous_excerpt,
                llm_service=llm_service,
                regulation_chunks=regulation_chunks,
                benchmark_topic_name=benchmark_name,
                surrounding_context=self._surrounding_context_for_unit(topic, subtopic),
            )

            unit.gap_report_json = report.model_dump_json()
            unit.gap_status = "assessed"
            if subtopic is not None and benchmark_name:
                subtopic.benchmark_match = benchmark_name  # persist so apply reuses it
            self.storage.update_draft(draft_id, draft)
            return report.model_dump()
        except Exception as e:
            logger.error(f"Failed to generate gap report for topic {topic_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def assess_batch(self, draft_id: str, units: List[Dict[str, Any]], vector_store, llm_service) -> Optional[List[Dict[str, Any]]]:
        """Assess several units in ONE request: load the draft once, run the LLM reports
        concurrently (asyncio.gather), then persist the draft once (avoids the per-request
        whole-draft read-modify-write that would clobber on parallel calls)."""
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                return None
            benchmark_topics = self._get_benchmark_topics(draft, vector_store)

            # Resolve the requested (topic, subtopic) targets against the loaded draft.
            targets = []
            for u in units:
                topic = next((t for t in draft.toc if t.topic_id == u.get("topic_id")), None)
                if not topic:
                    continue
                sub = None
                sid = u.get("subtopic_id")
                if sid:
                    sub = next((s for s in topic.subtopics if s.subtopic_id == sid), None)
                    if not sub:
                        continue
                targets.append((topic, sub))

            previous_excerpt = (getattr(draft.metadata, "previous_policy_text", None) or "")[:4000]

            # Phase 1 — prep SEQUENTIALLY: benchmark matching (cached embeddings → each distinct name
            # embedded once) + regulation lookup. Cheap, and avoids concurrent re-embedding.
            prepared = []
            for topic, sub in targets:
                unit = sub or topic
                benchmark_content, benchmark_name = self._benchmark_content_for_unit(topic, sub, benchmark_topics, vector_store)
                regulation_chunks = []
                try:
                    regulation_chunks = await self.generation_service._search_relevant_regulations(
                        topic_title=unit.topic, user_prompt="", draft_metadata=draft.metadata, vector_store=vector_store
                    )
                except Exception as e:
                    logger.warning(f"Regulation search failed in batch (non-fatal): {e}")
                prepared.append((topic, sub, benchmark_content, benchmark_name, regulation_chunks,
                                 self._surrounding_context_for_unit(topic, sub)))

            # Phase 2 — run ONLY the LLM calls concurrently (the expensive, truly-async part).
            async def gen(topic, sub, benchmark_content, benchmark_name, regulation_chunks, surrounding):
                unit = sub or topic
                report = await self.generation_service.generate_gap_report(
                    topic_id=(sub.subtopic_id if sub else topic.topic_id),
                    topic_title=unit.topic,
                    existing_content=unit.content or "",
                    benchmark_content=benchmark_content,
                    draft=draft,
                    previous_policy_excerpt=previous_excerpt,
                    llm_service=llm_service,
                    regulation_chunks=regulation_chunks,
                    benchmark_topic_name=benchmark_name,
                    surrounding_context=surrounding,
                )
                return topic, sub, report, benchmark_name

            results = await asyncio.gather(*[gen(*p) for p in prepared], return_exceptions=True)

            out = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Batch unit assessment failed (non-fatal): {r}")
                    continue
                topic, sub, report, benchmark_name = r
                unit = sub or topic
                unit.gap_report_json = report.model_dump_json()
                unit.gap_status = "assessed"
                if sub is not None and benchmark_name:
                    sub.benchmark_match = benchmark_name
                out.append({
                    "topic_id": topic.topic_id,
                    "subtopic_id": (sub.subtopic_id if sub else None),
                    "gap_report": report.model_dump(),
                })

            self.storage.update_draft(draft_id, draft)  # single write for the whole batch
            return out
        except Exception as e:
            logger.error(f"Failed to assess batch for draft {draft_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def apply_gap_for_topic(self, draft_id: str, topic_id: str, request: ApplyGapRequest, vector_store, llm_service) -> Optional[Dict[str, Any]]:
        """Apply the reviewer-confirmed gap findings to a topic, preserving the baseline."""
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                return None
            topic = next((t for t in draft.toc if t.topic_id == topic_id), None)
            if not topic:
                return None
            # The gap unit is the subtopic when subtopic_id is given, else the topic itself.
            subtopic = None
            if request.subtopic_id:
                subtopic = next((s for s in topic.subtopics if s.subtopic_id == request.subtopic_id), None)
                if not subtopic:
                    return None
            unit = subtopic or topic

            # CONFIRM step: reviewer approved (and possibly edited) the previewed content → just save it.
            if request.final_content is not None:
                final = request.final_content
                word_count = len(final.split())
                summary = ""
                if word_count > 150:
                    try:
                        summary = await self.generation_service.generate_summary(
                            content=final, title=unit.topic, llm_service=llm_service
                        )
                    except Exception as e:
                        logger.warning(f"Summary generation failed (non-fatal): {e}")
                unit.content = final
                unit.summary = summary
                unit.gap_status = "applied"
                self.storage.update_draft(draft_id, draft)
                return {"content": final, "summary": summary, "word_count": word_count, "message": "Gap changes confirmed"}

            # Resolve confirmed findings from the stored report, overlaying reviewer edits
            stored = []
            if unit.gap_report_json:
                try:
                    stored = json.loads(unit.gap_report_json).get("findings", [])
                except Exception:
                    stored = []
            edited_by_id = {f.id: f.model_dump() for f in (request.edited_findings or [])}
            confirmed = []
            for f in stored:
                fid = f.get("id")
                if fid in request.confirmed_finding_ids:
                    confirmed.append(edited_by_id.get(fid, f))
            for fid, f in edited_by_id.items():
                if fid in request.confirmed_finding_ids and not any(c.get("id") == fid for c in confirmed):
                    confirmed.append(f)

            benchmark_topics = self._get_benchmark_topics(draft, vector_store)
            benchmark_content, _ = self._benchmark_content_for_unit(
                topic, subtopic, benchmark_topics, vector_store
            )

            regulation_chunks = []
            try:
                regulation_chunks = await self.generation_service._search_relevant_regulations(
                    topic_title=unit.topic, user_prompt="", draft_metadata=draft.metadata, vector_store=vector_store
                )
            except Exception as e:
                logger.warning(f"Regulation search failed for apply-gap (non-fatal): {e}")

            # Benchmark-only topic with no baseline → seed from the benchmark reference
            existing = unit.content or ""
            if subtopic is None and topic.added_from_benchmark and not existing.strip():
                existing = benchmark_content or ""

            new_content = await self.generation_service.apply_gap_to_content(
                existing_content=existing,
                benchmark_content=benchmark_content,
                confirmed_findings=confirmed,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=request.subtopic_id,
                conversation_history="",
                llm_service=llm_service,
                regulation_chunks=regulation_chunks,
            )

            word_count = len(new_content.split())

            # PREVIEW step: return the proposed rewrite WITHOUT persisting (status stays 'assessed').
            if request.preview:
                return {"content": new_content, "summary": "", "word_count": word_count,
                        "message": "Proposed changes generated"}

            summary = ""
            if word_count > 150:
                try:
                    summary = await self.generation_service.generate_summary(
                        content=new_content, title=unit.topic, llm_service=llm_service
                    )
                except Exception as e:
                    logger.warning(f"Summary generation failed (non-fatal): {e}")
            unit.content = new_content
            unit.summary = summary
            unit.gap_status = "applied"
            self.storage.update_draft(draft_id, draft)

            return {"content": new_content, "summary": summary, "word_count": word_count, "message": "Gap changes applied"}
        except Exception as e:
            logger.error(f"Failed to apply gap for topic {topic_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # ===== Version history & audit trail =====

    def save_version(self, draft_id: str, label: str = "", created_by: str = "", reason: str = "") -> Optional[str]:
        """Snapshot the current draft as a named version."""
        draft = self.storage.get_draft(draft_id)
        if not draft:
            return None
        return self.storage.create_version(draft_id, draft, label=label, created_by=created_by, reason=reason)

    def list_versions(self, draft_id: str) -> List[Dict[str, Any]]:
        return self.storage.list_versions(draft_id)

    def restore_version(self, draft_id: str, version_id: str, restored_by: str = "") -> Optional[Dict[str, Any]]:
        restored = self.storage.restore_version(draft_id, version_id, restored_by=restored_by)
        return restored.model_dump() if restored else None

    def list_audit(self, draft_id: str) -> List[Dict[str, Any]]:
        return self.storage.list_audit(draft_id)

    def log_audit(self, draft_id: str, actor: str, action: str,
                  target_id: str = "", target_title: str = "", summary: str = "") -> None:
        self.storage.append_audit(draft_id, actor, action,
                                  target_id=target_id, target_title=target_title, summary=summary)

    async def check_consistency(self, draft_id: str, llm_service) -> Optional[Dict[str, Any]]:
        """Run a whole-policy consistency pass, persist the result, and return it."""
        from app.models.draft_models import ConsistencyReport, ConsistencyIssue
        draft = self.storage.get_draft(draft_id)
        if not draft:
            return None
        data = await self.generation_service.run_consistency_check(draft, llm_service)
        issues = []
        for it in data.get("issues", []):
            try:
                issues.append(ConsistencyIssue(
                    type=str(it.get("type", "contradiction")),
                    severity=str(it.get("severity", "medium")),
                    description=str(it.get("description", "")).strip(),
                    locations=[str(x) for x in (it.get("locations") or [])],
                    suggestion=str(it.get("suggestion", "")).strip(),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed consistency issue: {e}")
        report = ConsistencyReport(
            issues=issues,
            summary=str(data.get("summary", "")).strip(),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        draft.metadata.consistency_report_json = report.model_dump_json()
        self.storage.update_draft(draft_id, draft)
        return report.model_dump()

    async def build_gap_summary(self, draft_id: str, llm_service) -> Optional[Dict[str, Any]]:
        """Aggregate the persisted gap data into a client-facing summary (rollup + narrative)."""
        from difflib import SequenceMatcher
        from app.models.draft_models import GapSummary, GapSummaryTopic, GapFinding
        draft = self.storage.get_draft(draft_id)
        if not draft:
            return None

        # Original snapshot → enables "% baseline preserved" on applied units.
        orig_by_id: Dict[str, str] = {}
        original = self.storage.get_original_version(draft_id)
        if original:
            for t in original.toc:
                orig_by_id[t.topic_id] = t.content or ""
                for s in t.subtopics:
                    orig_by_id[s.subtopic_id] = s.content or ""

        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_source: Dict[str, int] = {}

        def _bump(d, k):
            if k:
                d[k] = d.get(k, 0) + 1

        # Flatten units (topic + subtopics).
        units = []
        for t in draft.toc:
            units.append((t.topic, t.topic_id, t.content or "", t.gap_status, getattr(t, "benchmark_match", None), t.gap_report_json))
            for s in t.subtopics:
                units.append((f"{t.topic} › {s.topic}", s.subtopic_id, s.content or "", s.gap_status,
                              getattr(s, "benchmark_match", None), s.gap_report_json))

        topics_out = []
        total_units = applied = assessed = pending = benchmark_coverage = total_findings = 0
        preserved = []
        for title, uid, content, status, bmatch, gj in units:
            total_units += 1
            st = status or "pending"
            if st == "applied":
                applied += 1
            elif st == "assessed":
                assessed += 1
            else:
                pending += 1
            if bmatch:
                benchmark_coverage += 1

            findings_out = []
            overall = ""
            if gj:
                try:
                    data = json.loads(gj)
                    overall = data.get("overall_summary", "") or ""
                    for f in data.get("findings", []):
                        _bump(by_severity, f.get("severity"))
                        _bump(by_type, f.get("type"))
                        _bump(by_source, f.get("source"))
                        findings_out.append(GapFinding(
                            id=f.get("id") or str(uuid.uuid4()),
                            type=str(f.get("type", "enhancement")),
                            source=str(f.get("source", "benchmark")),
                            severity=str(f.get("severity", "medium")),
                            description=str(f.get("description", "")),
                            suggested_change=str(f.get("suggested_change", "")),
                        ))
                except Exception:
                    pass
            total_findings += len(findings_out)

            if st == "applied" and uid in orig_by_id:
                orig = orig_by_id[uid]
                if orig.strip() or content.strip():
                    preserved.append(SequenceMatcher(None, orig.split(), content.split()).ratio())

            topics_out.append(GapSummaryTopic(
                title=title, status=st, findings_count=len(findings_out),
                overall_summary=overall, findings=findings_out,
            ))

        pct_preserved = round(100 * sum(preserved) / len(preserved), 1) if preserved else None

        summary = GapSummary(
            total_units=total_units, assessed=assessed, applied=applied, pending=pending,
            benchmark_coverage=benchmark_coverage, pct_preserved=pct_preserved,
            total_findings=total_findings, by_severity=by_severity, by_type=by_type,
            by_source=by_source, topics=topics_out,
        )
        try:
            summary.narrative = await self.generation_service.generate_gap_summary_narrative(draft, summary, llm_service)
        except Exception as e:
            logger.warning(f"Gap summary narrative failed (non-fatal): {e}")
        return summary.model_dump()

    def _update_summary(
        self,
        draft_id: str,
        topic_id: str,
        subtopic_id: Optional[str],
        summary: str
    ) -> bool:
        """Update the summary for a topic or subtopic."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.summary = summary
                                break
                    else:
                        toc_topic.summary = summary

                    return self.storage.update_draft(draft_id, draft_model)

            return False

        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
            return False
    
    def get_draft_progress(self, draft_id: str) -> Dict[str, Any]:
        """Calculate draft completion progress."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return {"error": "Draft not found"}

            total_items = 0
            completed_items = 0

            for toc_topic in draft_model.toc:
                total_items += 1
                if toc_topic.content:
                    completed_items += 1

                for toc_subtopic in toc_topic.subtopics:
                    total_items += 1
                    if toc_subtopic.content:
                        completed_items += 1

            return {
                "total": total_items,
                "completed": completed_items,
                "percentage": (completed_items / total_items * 100) if total_items > 0 else 0,
                "remaining": total_items - completed_items
            }

        except Exception as e:
            logger.error(f"Failed to calculate progress for draft {draft_id}: {e}")
            return {"error": "Failed to calculate progress"}
